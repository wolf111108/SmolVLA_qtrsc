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
from vla_tcs2.runtime_context import (
    CURRENT_PHASE,
    CURRENT_FLOW_STEP,
    CURRENT_ATTN_KIND,
    CURRENT_GENERATION_ID,
)


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
    module_id: str | None = None,
) -> QuantizedLinear:
    """
    Build a QuantizedLinear from an nn.Linear (opt-qt style).

    layer_config = quant_config[<layer_type>] provides a_bit/w_bit/o_bit/
    d_bit/p/outlier_ratio; top-level quant_config provides
    dynamic_activation / mixed_precision / scale_dir.

    When `module_id` is given, component/module-aware `linear.overrides`
    are resolved on top of the operator base config (G6 upgrade).
    """
    physical_id = module_id or f"{layer_type}_{layer_idx}"

    layer_config, matched_overrides = resolve_linear_quant_config(
        quant_config=quant_config,
        layer_type=layer_type,
        module_id=physical_id,
    )

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

    # Intra-weight-tensor scale granularity (G2). ORTHOGONAL to
    # linear_scale_granularity (site-level sharing). When set to
    # per_output_channel and the layer method is pot_ao_outlier, the
    # method is switched to the per-channel implementation
    # (pot_ao_outlier_channel) which pairs a matching scale/forward.
    weight_granularity = quant_config.get("weight_quant_granularity", None)
    if weight_granularity == "per_output_channel":
        if quant_layer.method == "pot_ao_outlier":
            quant_layer.method = "pot_ao_outlier_channel"
        else:
            raise ValueError(
                "weight_quant_granularity=per_output_channel currently "
                "supports method=pot_ao_outlier only (got "
                f"{quant_layer.method!r})"
            )
    elif weight_granularity in (None, "per_tensor"):
        pass  # legacy scalar weight scale path
    else:
        raise ValueError(
            f"Unsupported weight_quant_granularity: {weight_granularity!r} "
            "(per_output_channel is implemented; groupwise pending Gate 2)"
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

    # Debug / manifest only (not read by forward).
    quant_layer.quant_override_names = matched_overrides
    quant_layer.effective_quant_config = dict(layer_config)

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


# =============================================================================
# Component-aware Linear precision routing (G6 upgrade)
# =============================================================================

# Fields an override is allowed to patch on a Linear's effective quant
# config. Anything else is a typo and must fail fast (manual §3.3).
_ALLOWED_LINEAR_OVERRIDE_KEYS = {
    "a_bit",
    "w_bit",
    "o_bit",
    "d_bit",
    "p",
    "outlier_ratio",
    "method",
    "test_method",
}

# Selectors an override `target` may use (mirrors _matches_target). A target
# with an unknown key (e.g. "componet" typo) would otherwise silently match
# ALL modules, so validate strictly (manual §3.3, audit §8).
_ALLOWED_LINEAR_TARGET_KEYS = {
    "module_id",
    "module_ids",
    "component",
    "layer",
    "operator",
}


def resolve_linear_quant_config(
    quant_config: dict[str, Any],
    layer_type: str,
    module_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve effective quant config for one physical Linear.

    Precedence (manual §3.4):
        operator base config
        -> matching linear.overrides in list order (later wins)

    Returns:
        resolved_config
        matched_override_names
    """
    base = dict(quant_config.get(layer_type, {}) or {})

    linear_cfg = quant_config.get("linear", {}) or {}
    overrides = linear_cfg.get("overrides", []) or []

    if not isinstance(overrides, list):
        raise TypeError("quantization.linear.overrides must be a list")

    matched_names: list[str] = []

    for idx, override in enumerate(overrides):
        if not isinstance(override, dict):
            raise TypeError(f"linear.overrides[{idx}] must be a dict")

        target = override.get("target")
        patch = override.get("config")

        if not isinstance(target, dict) or not target:
            raise ValueError(
                f"linear.overrides[{idx}].target must be a non-empty dict"
            )

        unknown_target = set(target) - _ALLOWED_LINEAR_TARGET_KEYS
        if unknown_target:
            raise ValueError(
                f"Unsupported linear override target fields at index "
                f"{idx}: {sorted(unknown_target)}"
            )

        if not any(k in target for k in _ALLOWED_LINEAR_TARGET_KEYS):
            raise ValueError(
                f"linear.overrides[{idx}].target has no supported selector"
            )

        if not isinstance(patch, dict):
            raise ValueError(
                f"linear.overrides[{idx}].config must be a dict"
            )

        unknown = set(patch) - _ALLOWED_LINEAR_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                f"Unsupported linear override fields at index {idx}: "
                f"{sorted(unknown)}"
            )

        if _matches_target(module_id, target):
            base.update(patch)
            matched_names.append(
                str(override.get("name", f"override_{idx}"))
            )

    return base, matched_names


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

# Vision encoder (transformers SmolVLM) Linear names. NOTE: vision attention
# uses "out_proj" (NOT "o_proj" like VLM text), MLP uses fc1/fc2.
VISION_ATTN_LINEAR_NAMES = ["q_proj", "k_proj", "v_proj", "out_proj"]
VISION_MLP_LINEAR_NAMES = ["fc1", "fc2"]


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

    # linear.enabled=false must disable Linear wrapping (audit §11).
    if not linear_cfg.get("enabled", True):
        return False

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

    # Component/module-aware overrides change per-module precision, which
    # must never share calibration scales with a differently-quantized
    # module. v1 forces per_site (manual §4.4).
    overrides = (
        (quant_config.get("linear", {}) or {}).get("overrides", []) or []
    )
    if overrides and granularity != "per_site":
        raise ValueError(
            "component/module-aware linear overrides currently require "
            "linear_scale_granularity=per_site to prevent mixed-precision "
            "modules from sharing calibration scales."
        )

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
            ql = create_quantized_linear(
                mod, name, layer_idx, quant_config, mode, module_id=module_id
            )
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
                ql = create_quantized_linear(
                    mod, name, layer_idx, quant_config, mode, module_id=module_id
                )
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
        ql = create_quantized_linear(
            mod, name, 0, quant_config, mode, module_id=full_name
        )
        ql._stat_manager = stat_manager
        ql.set_layer_info(name, 0, module_id=full_name)
        ql.set_scale_group(f"head_{name}", 0)
        _setattr_path(vlm_expert, name, ql)
        replaced += 1
        if stat_manager is not None:
            stat_manager.register_layer(f"head_{name}", 0)

    return replaced


def _wrap_smolvlm_vision_linear_layers(
    model: SmolVLAPolicy,
    quant_config: dict[str, Any],
    mode: str,
    stat_manager: QuantStatManager | None,
) -> int:
    """
    Opt-in wrapping of the SmolVLM vision encoder Linear + connector
    projection (Phase I vision quantization).

    Default-off (backward-compat gate): when ``quantization.vision.enabled``
    and ``quantization.connector.enabled`` are both false (or absent), this
    returns 0 and leaves the legacy VLM/Expert wrapping (224 Linear) untouched.

    Targets (transformers SmolVLM, accessed via ``vlm.model``):
      - vision_model.encoder.layers.<i>.self_attn.{q,k,v,out}_proj
      - vision_model.encoder.layers.<i>.mlp.{fc1,fc2}
      - connector.modality_projection.proj
        (module_id ``connector.layer.0.connector_proj``)

    Operator-group selection (Phase I manual V1/V2/V3, er.md P0):
      ``vision.enabled`` alone does NOT wrap anything. The sub-groups
      ``vision.linear.mlp`` and ``vision.linear.attn_proj`` independently
      gate wrapping, so the experiment stages map exactly to counts:

        V1  connector only            -> +1   (vision.linear both false)
        V2  vision MLP only           -> +24  (mlp=true,  attn_proj=false)
        V3  vision MLP + attn proj    -> +72  (mlp=true,  attn_proj=true)

    Both sub-groups default to false (explicit opt-in; never wrap the whole
    Vision encoder just by setting ``vision.enabled``).

    Precision is resolved through the SAME ``quantization.linear.overrides``
    as VLM/Expert (component / module_id selectors); ``linear.enabled`` /
    ``include`` / ``exclude`` also apply via ``_should_wrap()``.
    """
    vision_cfg = quant_config.get("vision", {}) or {}
    connector_cfg = quant_config.get("connector", {}) or {}
    vision_enabled = bool(vision_cfg.get("enabled", False))
    connector_enabled = bool(connector_cfg.get("enabled", False))

    if not vision_enabled and not connector_enabled:
        return 0

    # Operator-group gates (default false; explicit opt-in per er.md P0).
    vision_linear_cfg = vision_cfg.get("linear", {}) or {}
    wrap_vision_mlp = vision_enabled and bool(
        vision_linear_cfg.get("mlp", False)
    )
    wrap_vision_attn_proj = vision_enabled and bool(
        vision_linear_cfg.get("attn_proj", False)
    )

    # Vision/Connector first join must recalibrate (their scale files are
    # component-specific and may not exist yet; er.md P1 calibration).
    vision_cal_policy = str(
        vision_cfg.get("calibration_policy", "recalibrate")
    ).lower()
    connector_cal_policy = str(
        connector_cfg.get("calibration_policy", "recalibrate")
    ).lower()

    granularity = str(
        quant_config.get("linear_scale_granularity", "per_site")
    ).lower()

    model_obj = model.model
    vlm_expert = model_obj.vlm_with_expert
    vlm_model = vlm_expert.get_vlm_model()

    replaced = 0

    def _replace_linear(
        mod,
        component,
        layer_idx,
        attr_name,
        op_name,
        submodule,
        module_id,
        calibration_policy,
    ):
        nonlocal replaced
        if mod is None or not isinstance(mod, nn.Linear):
            return
        # Honor the global linear.enabled / include / exclude contract
        # (er.md P0.5).
        if not _should_wrap(module_id, quant_config):
            return
        ql = create_quantized_linear(
            mod, op_name, layer_idx, quant_config, mode, module_id=module_id
        )
        ql._stat_manager = stat_manager
        ql.set_layer_info(op_name, layer_idx, module_id=module_id)
        sg_name, sg_idx = resolve_linear_scale_group(
            component, layer_idx, op_name, granularity
        )
        ql.set_scale_group(sg_name, sg_idx)
        # Component-specific calibration policy (er.md P1): vision/connector
        # scale files are keyed by component, so force recalibrate on first
        # join instead of inheriting a shared `layer_policy`/`per_layer_policy`
        # that does not know about the vision component.
        ql.calibration_policy = calibration_policy
        setattr(submodule, attr_name, ql)
        replaced += 1
        if stat_manager is not None:
            stat_manager.register_layer(sg_name, sg_idx)

    # Vision encoder (12 layers × 6 Linear, gated by sub-group).
    if vision_enabled:
        vision_model = vlm_model.vision_model
        encoder = vision_model.encoder
        for layer_idx, layer in enumerate(encoder.layers):
            if wrap_vision_attn_proj:
                for name in VISION_ATTN_LINEAR_NAMES:
                    _replace_linear(
                        getattr(layer.self_attn, name, None),
                        "vision", layer_idx, name, name, layer.self_attn,
                        f"vision.layers.{layer_idx}.self_attn.{name}",
                        vision_cal_policy,
                    )
            if wrap_vision_mlp:
                for name in VISION_MLP_LINEAR_NAMES:
                    _replace_linear(
                        getattr(layer.mlp, name, None),
                        "vision", layer_idx, name, name, layer.mlp,
                        f"vision.layers.{layer_idx}.mlp.{name}",
                        vision_cal_policy,
                    )

    # Connector projection (single Linear; attr is "proj").
    if connector_enabled:
        modality_projection = vlm_model.connector.modality_projection
        _replace_linear(
            getattr(modality_projection, "proj", None),
            "connector", 0, "proj", "connector_proj", modality_projection,
            "connector.layer.0.connector_proj",
            connector_cal_policy,
        )

    return replaced


def _inject_smolvlm_vision_quantized_matmul(model, quant_config, mode, stat_manager):
    """Vision's independent opt-in switch; legacy quantize_matmul covers VLM/Expert."""
    vision_cfg = quant_config.get("vision", {}) or {}
    cfg = vision_cfg.get("matmul", {}) or {}
    if not vision_cfg.get("enabled", False) or not cfg.get("enabled", False):
        return 0
    from vla_tcs2.vision_attention import attach_vision_attention

    policy = str(cfg.get("calibration_policy", "recalibrate")).lower()
    if policy not in ("recalibrate", "reuse"):
        raise ValueError("vision.matmul.calibration_policy must be recalibrate or reuse")
    granularity = str(quant_config.get("matmul_scale_granularity", "per_site")).lower()
    if granularity not in ("per_site", "per_component"):
        raise ValueError("Vision MatMul requires component-isolated per_site/per_component scales")
    layers = model.model.vlm_with_expert.get_vlm_model().vision_model.encoder.layers
    for idx, layer in enumerate(layers):
        modules = []
        for op in ("qk", "pv"):
            mm = create_quantized_matmul(f"{op}_matmul", idx, quant_config, mode)
            mm.set_layer_info(f"{op}_matmul", idx, module_id=f"vision.layer.{idx}.{op}")
            name, scale_idx = resolve_matmul_scale_group("vision", idx, op, granularity)
            mm.set_scale_group(name, scale_idx)
            mm.calibration_policy = policy
            mm._stat_manager = stat_manager
            modules.append(mm)
            if stat_manager is not None:
                stat_manager.register_layer(name, scale_idx)
        attach_vision_attention(layer.self_attn, *modules)
    return 2 * len(layers)


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
        site_token = _CURRENT_ATTN_SITE.set(
            (_resolve_attn_component(inputs_embeds), layer_idx)
        )
        # attention_kind from the real call site (never layer parity).
        kind_token = CURRENT_ATTN_KIND.set("self")
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
            CURRENT_ATTN_KIND.reset(kind_token)
            _CURRENT_ATTN_SITE.reset(site_token)

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
        site_token = _CURRENT_ATTN_SITE.set(
            (_resolve_attn_component(inputs_embeds), layer_idx)
        )
        kind_token = CURRENT_ATTN_KIND.set("cross")
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
            CURRENT_ATTN_KIND.reset(kind_token)
            _CURRENT_ATTN_SITE.reset(site_token)

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
# Runtime context hooks (phase / flow_step auto instrumentation)
# =============================================================================

_generation_counter = [0]


def install_runtime_hooks(model: SmolVLAPolicy) -> bool:
    """
    Monkey-patch VLAFlowMatching.sample_actions / denoise_step so the
    runtime context (phase, flow_step, generation_id) is tagged
    automatically during evaluation — no runner-side set_phase needed.

    Mapping (research manual §12-§16):
      sample_actions entry : phase="prefill", flow_step=-1,
                             generation_id += 1 (denoise counter reset)
      denoise_step entry   : phase="denoise", flow_step=0..num_steps-1

    Safe to call multiple times: already-patched models are detected via
    the `_runtime_hooks_installed` marker and re-installation is a no-op.

    Returns True if hooks were installed by this call.
    """
    flow_model = model.model  # VLAFlowMatching
    if getattr(flow_model, "_runtime_hooks_installed", False):
        return False

    original_sample_actions = flow_model.sample_actions
    original_denoise_step = flow_model.denoise_step

    # Per-model denoise step counter. CURRENT_FLOW_STEP is a *context tag*
    # (the current step being executed), NOT a persistent loop accumulator:
    # resetting it in the `finally` restores the pre-entry value (-1), so
    # `CURRENT_FLOW_STEP.get() + 1` would yield 0 on every step. The actual
    # 0..num_steps-1 sequence must live in a separate bookkeeping cell.
    denoise_step_counter = [-1]

    def hooked_sample_actions(*args, **kwargs):
        _generation_counter[0] += 1
        denoise_step_counter[0] = -1
        gen_token = CURRENT_GENERATION_ID.set(_generation_counter[0])
        phase_token = CURRENT_PHASE.set("prefill")
        step_token = CURRENT_FLOW_STEP.set(-1)
        try:
            return original_sample_actions(*args, **kwargs)
        finally:
            CURRENT_FLOW_STEP.reset(step_token)
            CURRENT_PHASE.reset(phase_token)
            CURRENT_GENERATION_ID.reset(gen_token)

    def hooked_denoise_step(*args, **kwargs):
        # flow_step 0..num_steps-1: euler_integrate calls denoise_step in
        # order, once per step, inside sample_actions.
        denoise_step_counter[0] += 1
        step = denoise_step_counter[0]
        phase_token = CURRENT_PHASE.set("denoise")
        step_token = CURRENT_FLOW_STEP.set(step)
        try:
            return original_denoise_step(*args, **kwargs)
        finally:
            CURRENT_FLOW_STEP.reset(step_token)
            CURRENT_PHASE.reset(phase_token)

    flow_model.sample_actions = hooked_sample_actions
    flow_model.denoise_step = hooked_denoise_step
    flow_model._runtime_hooks_installed = True
    return True


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

        # Auto phase/flow-step instrumentation for sparsity & workload
        # stats (no-op if quantization disabled; context is only read by
        # opt-in collectors).
        install_runtime_hooks(self.model)

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

        # Phase I vision quantization (default-off opt-in).
        n_vision = _wrap_smolvlm_vision_linear_layers(
            self.model,
            self.quant_cfg,
            mode,
            self.stat_manager,
        )
        if n_vision > 0:
            print(f"Replaced {n_vision} vision/connector Linear modules.")

        n_vision_mm = _inject_smolvlm_vision_quantized_matmul(
            self.model, self.quant_cfg, mode, self.stat_manager
        )
        if n_vision_mm:
            print(f"Injected Vision QuantizedMatMul ({n_vision_mm} physical qk/pv objects)")

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
