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

from contextvars import ContextVar
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

    # Pluggable quantization method name (quant/scale_methods.py +
    # quant/quant_methods.py, paired by the same key). Per-layer config
    # overrides the top-level quantization.method.
    quant_layer.method = layer_config.get(
        "method",
        quant_config.get("method", "per_tensor"),
    )

    # Pluggable test-forward method name (quant/test_methods.py), used when
    # mode == "test_forward". Per-layer config overrides the top-level
    # quantization.test_method.
    quant_layer.test_method = layer_config.get(
        "test_method",
        quant_config.get("test_method", "raw"),
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

    # Pluggable quantization method name — same registry keys as
    # QuantizedLinear (dispatched to the matmul_* entries of
    # scale_methods/quant_methods). Per-layer config overrides the
    # top-level quantization.method.
    quant_matmul.method = layer_config.get(
        "method",
        quant_config.get("method", "per_tensor"),
    )

    # Pluggable test-forward method name (quant/test_methods.py), used when
    # mode == "test_forward". Per-layer config overrides the top-level
    # quantization.test_method.
    quant_matmul.test_method = layer_config.get(
        "test_method",
        quant_config.get("test_method", "raw"),
    )

    quant_matmul.set_layer_info(layer_type, layer_idx)

    return quant_matmul


# =============================================================================
# MatMul site routing (fine-grained quant plan §13-§18)
# =============================================================================

# Holds the currently-executing attention site as (component, layer_idx).
# `get_attention_interface()` is a model-level method with no layer_idx
# parameter, so we thread the site through a ContextVar instead of modifying
# lerobot source. component ∈ {"vlm", "expert", "joint"}.
_CURRENT_ATTN_SITE: ContextVar = ContextVar("smolvla_attn_site", default=None)


def resolve_matmul_scale_group(
    component: str,
    layer_idx: int,
    op: str,
    granularity: str,
) -> tuple[str, int]:
    """Map a physical MatMul site to its scale-sharing (name, idx) pair.

    granularity ∈ {global, per_component, per_layer, per_site}:
      - global        : every qk/pv shares one scale (qk/pv only).
      - per_component : vlm vs expert split (qk/pv per component).
      - per_layer     : shared across vlm/expert per layer index.
      - per_site      : every physical MatMul has its own scale (default).
    """
    if granularity == "global":
        return f"{op}_matmul", 0
    if granularity == "per_component":
        return f"{component}_{op}_matmul", 0
    if granularity == "per_layer":
        return f"layer_{op}_matmul", layer_idx
    if granularity == "per_site":
        return f"{component}_{op}_matmul", layer_idx
    raise ValueError(f"Unknown matmul_scale_granularity: {granularity}")


def resolve_linear_scale_group(
    component: str,
    layer_idx: int,
    name: str,
    granularity: str,
) -> tuple[str, int]:
    """Map a physical Linear site to its scale-sharing (name, idx) pair.

    Mirrors resolve_matmul_scale_group. `name` is the Linear op name
    (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj).

    granularity ∈ {global, per_component, per_layer, per_site}:
      - global        : every layer/component shares one scale per op name.
      - per_component : vlm vs expert split (op scale per component).
      - per_layer     : shared across vlm/expert per layer index.
      - per_site      : every physical Linear has its own scale (default).
    """
    if granularity == "global":
        return f"{name}", 0
    if granularity == "per_component":
        return f"{component}_{name}", 0
    if granularity == "per_layer":
        return f"layer_{name}", layer_idx
    if granularity == "per_site":
        return f"{component}_{name}", layer_idx
    raise ValueError(f"Unknown linear_scale_granularity: {granularity}")


def switch_quantization_mode_all(model: nn.Module, mode: str) -> nn.Module:
    """Toggle every QuantizedLinear / QuantizedMatMul to the given mode."""
    valid_modes = {"raw", "scale_inspection", "quant_forward", "test_forward"}
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
# Sensitivity targeting (fine-grained quant plan §19-§20)
# =============================================================================

def _matches_target(module_id: str, target: dict[str, Any]) -> bool:
    """Return True if a module's `module_id` matches the sensitivity target.

    Supported selectors (ANDed when multiple are present):

      - `module_id`:  exact string or glob (fnmatch) against module_id
      - `module_ids`: list of exact/glob patterns (OR)
      - `component`:  "vlm" / "expert"
      - `layer`:      int, or list[int], matching the layer index
      - `operator`:   q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
                      or qk/pv

    When `target` is empty or lacks any selector, nothing matches (safety).
    """
    if not target:
        return False

    # module_id / module_ids act as filters: when specified, a module that
    # does not match them must be excluded (previously the code fell through
    # to `return True`, silently matching ALL modules).
    id_matched = None  # None = no id selector present
    ids = target.get("module_ids")
    if ids:
        id_matched = any(fnmatch(module_id, str(p)) for p in ids)

    mid = target.get("module_id")
    if mid:
        m = fnmatch(module_id, str(mid))
        id_matched = m if id_matched is None else (id_matched or m)

    if id_matched is False:
        return False

    component = target.get("component")
    if component:
        comp = module_id.split(".")[0]
        if comp != component:
            return False

    layer = target.get("layer")
    if layer is not None:
        # module_id forms:
        #   Linear : {comp}.layers.{i}.{self_attn|mlp}.{name}
        #   MatMul : {comp}.layer.{i}.{op}
        parts = module_id.split(".")
        idx = None
        for i, p in enumerate(parts):
            if p in ("layers", "layer"):
                idx = int(parts[i + 1])
                break
        if idx is None:
            return False
        if isinstance(layer, list):
            if idx not in layer:
                return False
        elif idx != int(layer):
            return False

    operator = target.get("operator")
    if operator:
        last = module_id.split(".")[-1]
        if last != operator:
            return False

    return True


def _apply_test_params(
    module: nn.Module,
    method: str,
    test_cfg: dict[str, Any],
    group_cfg: dict[str, Any],
) -> None:
    """Set test_forward params on a module, with per-group overrides.

    Resolution order (group wins over global `test.*` defaults).
    """
    merged = {**test_cfg, **{k: v for k, v in group_cfg.items() if k != "target"}}
    module.mode = "test_forward"
    module.test_method = str(merged.get("method", "gaussian_rms_output"))
    module.test_noise_alpha = float(merged.get("alpha", 0.03))
    module.test_noise_seed = int(merged.get("seed", 0))
    module.test_site = str(merged.get("site", "output"))
    module.test_use_outlier_protection = bool(
        merged.get("use_outlier_protection", False)
    )
    module.test_residual_lambda = float(merged.get("residual_lambda", 1.0))
    module.test_outlier_ratio = float(merged.get("outlier_ratio", 0.01))


def apply_sensitivity_target(
    model: nn.Module,
    config: dict[str, Any],
) -> int:
    """Apply a sensitivity-test target to the wrapped model.

    Reads the top-level `test` config block:

        test:
          enabled: true
          method: gaussian_rms_output   # or gaussian_rms_*/quant_residual_*
          alpha: 0.03
          seed: 0
          site: output                  # input|weight|output (Linear) / A|B|output (MatMul)
          use_outlier_protection: false
          residual_lambda: 1.0
          outlier_ratio: 0.01
          target:
            module_id: expert.layer.7.qk   # or component/layer/operator selectors

    OR, for multi-group experiments (e.g. separate alpha per component):

        test:
          enabled: true
          targets:
            - target: {component: vlm}
              alpha: 0.01
            - target: {component: expert}
              alpha: 0.05

    Every QuantizedLinear / QuantizedMatMul matching a group's `target` is
    switched to mode="test_forward" with that group's test method + params
    (group-level keys override the global `test.*` defaults); all others are
    switched to "raw".

    Returns the number of targeted modules.
    """
    test_cfg = config.get("test", {})
    if not test_cfg.get("enabled", False):
        return 0

    method = str(test_cfg.get("method", "gaussian_rms_output"))

    groups = test_cfg.get("targets")
    if groups is None:
        groups = [{"target": test_cfg.get("target", {})}]
    if not groups:
        groups = [{"target": {}}]

    n_target = 0
    for module in model.modules():
        if not isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            continue

        mid = getattr(module, "module_id", "") or (
            f"{getattr(module, 'layer_name', '')}_"
            f"{getattr(module, 'layer_idx', 0)}"
        )

        matched_group = None
        for g in groups:
            if _matches_target(mid, g.get("target", g)):
                matched_group = g
                break

        if matched_group is not None:
            _apply_test_params(module, method, test_cfg, matched_group)
            n_target += 1
        else:
            module.mode = "raw"

    group_desc = [
        {"target": g.get("target", g), **{k: v for k, v in g.items() if k != "target"}}
        for g in groups
    ]
    print(
        f"[sensitivity] method={method} targets={group_desc} "
        f"-> {n_target} modules in test_forward, rest raw"
    )
    return n_target


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
    model_obj = model.model  # VLAFlowMatching

    granularity = str(
        quant_config.get("linear_scale_granularity", "per_site")
    ).lower()

    # SmolVLMWithExpertModel lives under .vlm_with_expert
    vlm_expert = model_obj.vlm_with_expert
    text_model = vlm_expert.get_vlm_model().text_model

    def wrap_layer_group(group, component, layer_idx):
        nonlocal replaced
        attn = group.self_attn
        mlp = getattr(group, "mlp", None)

        for name in SMOLVLA_ATTN_LINEAR_NAMES:
            mod = getattr(attn, name, None)
            if mod is None or not isinstance(mod, nn.Linear):
                continue
            module_id = f"{component}.layers.{layer_idx}.self_attn.{name}"
            if not _should_wrap(module_id, quant_config):
                continue
            ql = create_quantized_linear(mod, name, layer_idx, quant_config, mode)
            ql._stat_manager = stat_manager
            ql.set_layer_info(name, layer_idx, module_id=module_id)
            sg_name, sg_idx = resolve_linear_scale_group(
                component, layer_idx, name, granularity
            )
            ql.set_scale_group(sg_name, sg_idx)
            setattr(attn, name, ql)
            replaced += 1
            if stat_manager is not None:
                stat_manager.register_layer(sg_name, sg_idx)

        if mlp is not None:
            for name in SMOLVLA_MLP_LINEAR_NAMES:
                mod = getattr(mlp, name, None)
                if mod is None or not isinstance(mod, nn.Linear):
                    continue
                module_id = f"{component}.layers.{layer_idx}.mlp.{name}"
                if not _should_wrap(module_id, quant_config):
                    continue
                ql = create_quantized_linear(mod, name, layer_idx, quant_config, mode)
                ql._stat_manager = stat_manager
                ql.set_layer_info(name, layer_idx, module_id=module_id)
                sg_name, sg_idx = resolve_linear_scale_group(
                    component, layer_idx, name, granularity
                )
                ql.set_scale_group(sg_name, sg_idx)
                setattr(mlp, name, ql)
                replaced += 1
                if stat_manager is not None:
                    stat_manager.register_layer(sg_name, sg_idx)

    # VLM text layers
    for i, layer in enumerate(text_model.layers):
        wrap_layer_group(layer, "vlm", i)

    # LM expert layers
    expert = vlm_expert.lm_expert
    for i, layer in enumerate(expert.layers):
        wrap_layer_group(layer, "expert", i)

    # Action head / misc Linear (exclude vlm + lm_expert subtrees)
    for name, mod in list(vlm_expert.named_modules()):
        if name.startswith(("vlm", "lm_expert")):
            continue
        if not isinstance(mod, nn.Linear):
            continue
        full_name = f"model.model.{name}"
        if not _should_wrap(full_name, quant_config):
            continue
        ql = create_quantized_linear(mod, name, 0, quant_config, mode)
        ql._stat_manager = stat_manager
        ql.set_layer_info(name, 0, module_id=full_name)
        ql.set_scale_group(f"head_{name}", 0)
        _setattr_path(vlm_expert, name, ql)
        replaced += 1
        if stat_manager is not None:
            stat_manager.register_layer(f"head_{name}", 0)

    return replaced


def _resolve_attn_component(inputs_embeds: list) -> str:
    """Infer the attention component from the inputs_embeds list.

    inputs_embeds[0] is the VLM prefix, inputs_embeds[1] the expert suffix.
    Eval paths pass exactly one of them (prefix prefill -> vlm, denoise ->
    expert). When both are present (joint/training) we return "joint" and the
    dispatcher falls back to raw matmul.
    """
    has_prefix = bool(inputs_embeds) and inputs_embeds[0] is not None
    has_suffix = len(inputs_embeds) > 1 and inputs_embeds[1] is not None
    if has_prefix and not has_suffix:
        return "vlm"
    if not has_prefix and has_suffix:
        return "expert"
    return "joint"


def _inject_smolvla_quantized_matmul(
    attention_module,
    quant_config: dict[str, Any],
    mode: str,
    stat_manager: QuantStatManager | None,
):
    """
    Create 64 physical QuantizedMatMul objects (2 components × 16 layers ×
    qk/pv) and route the eager attention QK^T / PV matmuls to the correct one
    via a ContextVar, without modifying lerobot source.

    `get_attention_interface()` is a model-level method with no layer_idx, so
    we (a) wrap forward_attn_layer / forward_cross_attn_layer to set the
    current (component, layer_idx) site, and (b) make the attention interface
    dispatch to `quant_matmuls[f"{component}_{op}_{layer_idx}"]`.

    Each MatMul has a unique `module_id` (physical identity) and a
    `scale_group` resolved by `matmul_scale_granularity` (scale sharing).
    """
    if not quant_config.get("quantize_matmul", False):
        return

    granularity = str(
        quant_config.get("matmul_scale_granularity", "per_site")
    ).lower()

    num_vlm_layers = attention_module.num_vlm_layers
    num_expert_layers = attention_module.num_expert_layers

    attention_module.quant_matmuls = nn.ModuleDict()
    attention_module.stat_manager = stat_manager

    for component, n_layers in (
        ("vlm", num_vlm_layers),
        ("expert", num_expert_layers),
    ):
        for layer_idx in range(n_layers):
            for op in ("qk", "pv"):
                layer_type = f"{op}_matmul"
                mm = create_quantized_matmul(
                    layer_type, layer_idx, quant_config, mode
                )
                mm._stat_manager = stat_manager
                mm.set_layer_info(
                    layer_type,
                    layer_idx,
                    module_id=f"{component}.layer.{layer_idx}.{op}",
                )
                sg_name, sg_idx = resolve_matmul_scale_group(
                    component, layer_idx, op, granularity
                )
                mm.set_scale_group(sg_name, sg_idx)
                attention_module.quant_matmuls[
                    f"{component}_{op}_{layer_idx}"
                ] = mm
                if stat_manager is not None:
                    stat_manager.register_layer(sg_name, sg_idx)

    # Keep the original eager implementation as fallback reference.
    original_forward = attention_module.get_attention_interface()
    attention_module._original_attention_forward = original_forward

    # Wrap the two layer-level forward methods to set the current site.
    original_forward_attn_layer = attention_module.forward_attn_layer
    original_forward_cross_attn_layer = attention_module.forward_cross_attn_layer

    def wrapped_forward_attn_layer(
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache=True,
        past_key_values=None,
    ):
        token = _CURRENT_ATTN_SITE.set(
            (_resolve_attn_component(inputs_embeds), layer_idx)
        )
        try:
            return original_forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        finally:
            _CURRENT_ATTN_SITE.reset(token)

    def wrapped_forward_cross_attn_layer(
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache=True,
        past_key_values=None,
    ):
        token = _CURRENT_ATTN_SITE.set(
            (_resolve_attn_component(inputs_embeds), layer_idx)
        )
        try:
            return original_forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        finally:
            _CURRENT_ATTN_SITE.reset(token)

    attention_module.forward_attn_layer = wrapped_forward_attn_layer
    attention_module.forward_cross_attn_layer = wrapped_forward_cross_attn_layer

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

        site = _CURRENT_ATTN_SITE.get()
        if site is not None and site[0] in ("vlm", "expert"):
            component, layer_idx = site
            qk_matmul = attention_module.quant_matmuls[
                f"{component}_qk_{layer_idx}"
            ]
            pv_matmul = attention_module.quant_matmuls[
                f"{component}_pv_{layer_idx}"
            ]
        else:
            # joint / unknown site -> raw matmul fallback (training boundary)
            qk_matmul = None
            pv_matmul = None

        if qk_matmul is not None:
            att_weights = qk_matmul(
                query_states, key_states.transpose(2, 3)
            )
        else:
            att_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            )
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        if pv_matmul is not None:
            att_output = pv_matmul(
                probs, value_states.permute(0, 2, 1, 3)
            )
        else:
            att_output = torch.matmul(
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
            # get_attention_interface is a MODEL-level method on
            # SmolVLMWithExpertModel, so per-layer routing is done via a
            # ContextVar set inside wrapped forward_attn_layer /
            # forward_cross_attn_layer (see _inject_smolvla_quantized_matmul).
            # 64 physical MatMul objects (vlm/expert × 16 layers × qk/pv) are
            # created; scale sharing is controlled by matmul_scale_granularity.
            model_obj = self.model.model.vlm_with_expert
            _inject_smolvla_quantized_matmul(
                model_obj,
                self.quant_cfg,
                mode,
                self.stat_manager,
            )
            n_mm = len(getattr(model_obj, "quant_matmuls", {}))
            print(
                "Injected SmolVLA QuantizedMatMul "
                f"({n_mm} physical qk/pv objects)"
            )

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
