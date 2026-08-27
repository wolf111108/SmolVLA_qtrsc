"""
Model wrapper for VLA-TCS2.

Ported to the opt-qt contract (verified working reference:
/home/zyzhao/lfw_opt/opt-qt/quant/model_wrapper.py + opt_wrapper.py).

Responsibilities
----------------
1. Load the pretrained SmolVLA policy.
2. Replace selected Linear modules with QuantizedLinear
   (constructor + weight copy + set_layer_info, like opt-qt).
3. Inject QuantizedMatMul into SmolVLA attention (QK/PV) via
   monkey-patching eager_attention_forward.
4. switch_quantization_mode_all(model, mode) toggles .mode on every
   QuantizedLinear / QuantizedMatMul.

Scale persistence follows opt-qt's pickle scheme:
    {layer_name}_{w|a|o}_scale_{layer_idx}.p
managed by QuantStatManager (quant/stat_manager.py).
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul
from vla_tcs2.quant.stat_manager import QuantStatManager


# =============================================================================
# Quantized module factories (mirror opt-qt opt_wrapper.create_quantized_linear)
# =============================================================================


def resolve_calibration_policy(
    quant_config: dict[str, Any],
    layer_type: str,
    layer_idx: int,
) -> str:
    cp = quant_config.get("calibration_policy") or {}
    default_policy = cp.get("default") or "recalibrate"
    layer_policy = cp.get("layer_policy") or {}
    per_layer_policy = cp.get("per_layer_policy") or {}

    key = f"{layer_type}_{layer_idx}"
    policy = per_layer_policy.get(key, layer_policy.get(layer_type, default_policy))
    policy = str(policy).lower()

    valid = {"auto", "reuse", "recalibrate"}
    if policy not in valid:
        raise ValueError(
            f"Invalid calibration policy '{policy}' for {key}, valid: {sorted(valid)}"
        )
    return policy


def create_quantized_linear(
    original_layer: nn.Linear,
    layer_type: str,
    layer_idx: int,
    quant_config: dict[str, Any],
    mode: str = "scale_inspection",
) -> QuantizedLinear:
    """
    Build a QuantizedLinear from an nn.Linear (opt-qt style).

    layer_config = quant_config[<layer_type>] provides a_bit/w_bit/o_bit/
    d_bit/p/outlier_ratio; top-level quant_config provides
    dynamic_activation / mixed_precision / scale_dir.
    """
    layer_config = quant_config.get(layer_type, {})

    quant_layer = QuantizedLinear(
        in_features=original_layer.in_features,
        out_features=original_layer.out_features,
        bias=original_layer.bias is not None,
        mode=mode,
        a_bit=layer_config.get("a_bit", 8),
        w_bit=layer_config.get("w_bit", 8),
        o_bit=layer_config.get("o_bit", 8),
        d_bit=layer_config.get("d_bit", 4),
        p=layer_config.get("p", 4),
        outlier_ratio=layer_config.get(
            "outlier_ratio",
            quant_config.get("outlier_ratio", 0.0),
        ),
        dynamic_activation=quant_config.get("dynamic_activation", False),
        mixed_precision=quant_config.get("mixed_precision", False),
        mp_high_ratio=quant_config.get("mp_high_ratio", 0.2),
        mp_low_ratio=quant_config.get("mp_low_ratio", 0.3),
        scale_root_str=quant_config.get("scale_dir", ""),
    )

    quant_layer.calibration_policy = resolve_calibration_policy(
        quant_config, layer_type, layer_idx
    )

    # Copy weights (defensive device/dtype alignment first).
    _ow = original_layer.weight
    if _ow.device.type != "meta":
        quant_layer = quant_layer.to(
            device=_ow.device,
            dtype=_ow.dtype,
        )
    quant_layer.weight.data = original_layer.weight.data.clone()
    if original_layer.bias is not None:
        quant_layer.bias.data = original_layer.bias.data.clone()

    quant_layer.set_layer_info(layer_type, layer_idx)

    return quant_layer


def create_quantized_matmul(
    layer_type: str,
    layer_idx: int,
    quant_config: dict[str, Any],
    mode: str = "scale_inspection",
) -> QuantizedMatMul:
    """Build a QuantizedMatMul (opt-qt style)."""
    layer_config = quant_config.get(layer_type, {})

    quant_matmul = QuantizedMatMul(
        mode=mode,
        A_bit=layer_config.get("A_bit", 8),
        B_bit=layer_config.get("B_bit", 8),
        O_bit=layer_config.get("O_bit", 10),
        scale_root_str=quant_config.get("scale_dir", ""),
        d_bit=layer_config.get("d_bit", 4),
        p=layer_config.get("p", 4),
        outlier_ratio=layer_config.get(
            "outlier_ratio",
            quant_config.get("outlier_ratio", 0.0),
        ),
    )

    quant_matmul.calibration_policy = resolve_calibration_policy(
        quant_config, layer_type, layer_idx
    )

    quant_matmul.set_layer_info(layer_type, layer_idx)

    return quant_matmul


def switch_quantization_mode_all(model: nn.Module, mode: str) -> nn.Module:
    """Toggle every QuantizedLinear / QuantizedMatMul to the given mode."""
    valid_modes = {"raw", "scale_inspection", "quant_forward"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid quantization mode: {mode}")

    count = 0
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module.mode = mode
            count += 1

    print(f"Switched {count} quantized modules to mode={mode}")
    return model


# =============================================================================
# SmolVLA wrapping
# =============================================================================

SMOLVLA_ATTN_LINEAR_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]
SMOLVLA_MLP_LINEAR_NAMES = ["gate_proj", "up_proj", "down_proj"]


def _setattr_path(obj: nn.Module, dotted: str, value) -> None:
    parts = dotted.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _should_wrap(
    module_name: str,
    quant_config: dict[str, Any],
) -> bool:
    linear_cfg = quant_config.get("linear", {})
    include_patterns = linear_cfg.get("include", ["*"])
    exclude_patterns = linear_cfg.get("exclude", [])

    included = any(fnmatch(module_name, p) for p in include_patterns)
    excluded = any(fnmatch(module_name, p) for p in exclude_patterns)

    return included and not excluded


def _wrap_smolvla_linear_layers(
    model: SmolVLAPolicy,
    quant_config: dict[str, Any],
    mode: str,
    stat_manager: QuantStatManager | None,
) -> int:
    """
    Replace Linear modules inside the SmolVLA policy.

    Targets (relative to model.model = SmolVLMWithExpertModel):
      - vlm.model.text_model.layers.<i>.self_attn.{q,k,v,o}_proj
      - vlm.model.text_model.layers.<i>.mlp.{gate,up,down}_proj
      - lm_expert.layers.<i>.self_attn.{q,k,v,o}_proj
      - lm_expert.layers.<i>.mlp.{gate,up,down}_proj
      - any other nn.Linear under model.model (action head, etc.),
        excluding vlm/lm_expert subtrees to avoid double-wrapping.

    Selection honors quantization.linear.include/exclude globs.
    """
    replaced = 0
    model_obj = model.model  # SmolVLMWithExpertModel

    text_model = model_obj.get_vlm_model().text_model

    def wrap_layer_group(group, prefix, layer_idx):
        nonlocal replaced
        attn = group.self_attn
        mlp = getattr(group, "mlp", None)

        for name in SMOLVLA_ATTN_LINEAR_NAMES:
            mod = getattr(attn, name, None)
            if mod is None or not isinstance(mod, nn.Linear):
                continue
            if not _should_wrap(f"{prefix}.self_attn.{name}", quant_config):
                continue
            ql = create_quantized_linear(mod, name, layer_idx, quant_config, mode)
            ql._stat_manager = stat_manager
            setattr(attn, name, ql)
            replaced += 1
            if stat_manager is not None:
                stat_manager.register_layer(name, layer_idx)

        if mlp is not None:
            for name in SMOLVLA_MLP_LINEAR_NAMES:
                mod = getattr(mlp, name, None)
                if mod is None or not isinstance(mod, nn.Linear):
                    continue
                if not _should_wrap(f"{prefix}.mlp.{name}", quant_config):
                    continue
                ql = create_quantized_linear(mod, name, layer_idx, quant_config, mode)
                ql._stat_manager = stat_manager
                setattr(mlp, name, ql)
                replaced += 1
                if stat_manager is not None:
                    stat_manager.register_layer(name, layer_idx)

    # VLM text layers
    for i, layer in enumerate(text_model.layers):
        wrap_layer_group(layer, f"vlm.text_model.layers.{i}", i)

    # LM expert layers
    expert = model_obj.lm_expert
    for i, layer in enumerate(expert.layers):
        wrap_layer_group(layer, f"lm_expert.layers.{i}", i)

    # Action head / misc Linear (exclude vlm + lm_expert subtrees)
    for name, mod in list(model_obj.named_modules()):
        if name.startswith(("vlm", "lm_expert")):
            continue
        if not isinstance(mod, nn.Linear):
            continue
        full_name = f"model.model.{name}"
        if not _should_wrap(full_name, quant_config):
            continue
        ql = create_quantized_linear(mod, name, 0, quant_config, mode)
        ql._stat_manager = stat_manager
        _setattr_path(model_obj, name, ql)
        replaced += 1
        if stat_manager is not None:
            stat_manager.register_layer(name, 0)

    return replaced


def _inject_smolvla_quantized_matmul(
    attention_module,
    layer_idx: int,
    quant_config: dict[str, Any],
    mode: str,
    stat_manager: QuantStatManager | None,
):
    """
    Monkey-patch SmolVLMWithExpertModel.get_attention_interface so the
    QK^T and PV matmuls in eager attention go through QuantizedMatMul.

    The eager_attention_forward receives (attention_mask, batch_size,
    head_dim, query_states, key_states, value_states) and internally does:

        att_weights = Q @ K^T            -> qk_matmul
        att_output  = softmax(...) @ V   -> pv_matmul
    """
    if not quant_config.get("quantize_matmul", False):
        return

    qk_matmul = create_quantized_matmul("qk_matmul", layer_idx, quant_config, mode)
    pv_matmul = create_quantized_matmul("pv_matmul", layer_idx, quant_config, mode)

    qk_matmul._stat_manager = stat_manager
    pv_matmul._stat_manager = stat_manager

    attention_module.qk_matmul = qk_matmul
    attention_module.pv_matmul = pv_matmul
    attention_module.stat_manager = stat_manager

    if stat_manager is not None:
        stat_manager.register_layer("qk_matmul", layer_idx)
        stat_manager.register_layer("pv_matmul", layer_idx)

    # Keep the original eager implementation as fallback reference.
    original_forward = attention_module.get_attention_interface()
    attention_module._original_attention_forward = original_forward

    def quantized_eager_attention_forward(
        attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = attention_module.num_attention_heads
        num_key_value_heads = attention_module.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = attention_module.qk_matmul(
            query_states, key_states.transpose(2, 3)
        )
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = attention_module.pv_matmul(
            probs, value_states.permute(0, 2, 1, 3)
        )

        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )

        return att_output

    attention_module.get_attention_interface = lambda: quantized_eager_attention_forward


# =============================================================================
# ModelWrapper (public class used by main.py)
# =============================================================================


class ModelWrapper:
    """
    Build and modify a VLA policy for quantization experiments.

    Expected config structure (opt-qt compatible):

        model:
            type: smolvla
            path: lerobot/smolvla_libero
            device: cuda
            overrides:
                n_action_steps: 1
                num_steps: 10

        quantization:
            enabled: true
            scale_dir: scales/int8
            quantize_matmul: false
            linear:
                enabled: true
                include: ["*"]
                exclude: []
            calibration_policy:
                default: recalibrate
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.quant_cfg = config.get("quantization", {})
        self.model: SmolVLAPolicy | None = None
        self.stat_manager: QuantStatManager | None = None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def build(self, mode: str = "scale_inspection") -> nn.Module:
        """
        Load the pretrained policy and wrap it with quantized modules.

        Returns the wrapped SmolVLAPolicy.
        """
        self.model = self._load_model()

        if self.quant_cfg.get("enabled", False):
            scale_dir = self.quant_cfg.get("scale_dir", "scales/default")
            self.stat_manager = QuantStatManager(scale_dir)

            self._wrap_smolvla()

            switch_quantization_mode_all(self.model, mode)

        self._print_summary()

        return self.model

    def set_mode(self, mode: str) -> None:
        """Switch all quantized modules to the requested mode."""
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build() first.")
        switch_quantization_mode_all(self.model, mode)

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        model_type = str(self.model_cfg.get("type", "smolvla")).lower()
        if model_type == "smolvla":
            return self._load_smolvla()
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    def _load_smolvla(self) -> SmolVLAPolicy:
        model_path = self.model_cfg.get("path")
        if not model_path:
            raise ValueError("Missing config entry: model.path")

        revision = self.model_cfg.get("revision", None)

        policy_cfg = SmolVLAConfig.from_pretrained(model_path, revision=revision)

        # Record the checkpoint path so downstream processor construction can
        # reload normalizer stats from the checkpoint (mirrors lerobot-eval's
        # `cfg.policy.pretrained_path = Path(policy_path)`).
        policy_cfg.pretrained_path = Path(model_path)

        if "device" in self.model_cfg:
            policy_cfg.device = self.model_cfg["device"]

        overrides = self.model_cfg.get("overrides", {})
        for name, value in overrides.items():
            if not hasattr(policy_cfg, name):
                raise ValueError(f"Unknown SmolVLA config override: {name}")
            setattr(policy_cfg, name, value)

        print(f"Loading SmolVLA from: {model_path}")

        policy = SmolVLAPolicy.from_pretrained(
            model_path,
            config=policy_cfg,
            revision=revision,
        )

        policy.eval()
        return policy

    # -------------------------------------------------------------------------
    # Wrapping
    # -------------------------------------------------------------------------

    def _wrap_smolvla(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been built.")

        mode = "scale_inspection"

        n_linear = _wrap_smolvla_linear_layers(
            self.model,
            self.quant_cfg,
            mode,
            self.stat_manager,
        )
        print(f"Replaced {n_linear} Linear modules.")

        if self.quant_cfg.get("quantize_matmul", False):
            model_obj = self.model.model
            num_layers = model_obj.num_vlm_layers
            for layer_idx in range(num_layers):
                _inject_smolvla_quantized_matmul(
                    model_obj,
                    layer_idx,
                    self.quant_cfg,
                    mode,
                    self.stat_manager,
                )
            print(f"Injected SmolVLA QuantizedMatMul for {num_layers} layers")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _print_summary(self) -> None:
        if self.model is None:
            return

        total_params = sum(p.numel() for p in self.model.parameters())

        n_quant = sum(
            1
            for m in self.model.modules()
            if isinstance(m, (QuantizedLinear, QuantizedMatMul))
        )

        print()
        print("Model wrapper summary")
        print("---------------------")
        print(f"Model type       : {self.model_cfg.get('type', 'smolvla')}")
        print(f"Total parameters : {total_params:,}")
        print(f"Quantization     : {self.quant_cfg.get('enabled', False)}")
        print(f"Quantized modules: {n_quant}")
        print()
