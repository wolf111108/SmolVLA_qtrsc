"""
Model wrapper for VLA-TCS2.

Responsibilities
----------------
ModelWrapper is responsible for:

1. Loading the original pretrained VLA policy.
2. Replacing selected floating-point modules with quantized modules.
3. Switching quantized modules between execution modes.

ModelWrapper does NOT:
- run calibration data,
- calculate quantization scales,
- save/load calibration statistics,
- implement quantization arithmetic,
- run LIBERO evaluation.

Those responsibilities belong to other modules.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul


# =============================================================================
# ModelWrapper
# =============================================================================


class ModelWrapper:
    """
    Build and modify a VLA policy for quantization experiments.

    Expected top-level config structure:

        model:
            type: smolvla
            path: lerobot/smolvla_libero
            device: cuda
            revision: null

            overrides:
                n_action_steps: 1
                num_steps: 10

        quantization:
            enabled: true
            scale_dir: scales/int8

            linear:
                enabled: true
                include:
                    - "*"
                exclude: []

            matmul:
                enabled: false
    """

    VALID_MODES = {
        "raw",
        "scale_inspection",
        "quant_forward",
    }

    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

        self.model_cfg = config.get(
            "model",
            {},
        )

        self.quant_cfg = config.get(
            "quantization",
            {},
        )

        self.model: nn.Module | None = None

        # Keep track of modules replaced by this wrapper.
        self.replaced_linear_modules: list[str] = []
        self.replaced_matmul_modules: list[str] = []


    # =========================================================================
    # Public interface
    # =========================================================================


    def build(self) -> nn.Module:
        """
        Build the complete policy used by the experiment.

        Flow:

            load pretrained policy
                    ↓
            replace Linear modules
                    ↓
            inject quantized MatMul modules
                    ↓
            return wrapped policy
        """

        self.model = self._load_model()

        if self.quant_cfg.get("enabled", False):

            self._wrap_linear_modules()

            self._wrap_matmul_modules()

        self._print_summary()

        return self.model


    def set_mode(
        self,
        mode: str,
    ) -> None:
        """
        Switch all quantized modules to the requested execution mode.

        Modes
        -----
        raw:
            Floating-point forward.

        scale_inspection:
            Calibration forward.
            Quantized modules inspect tensor ranges and collect/save scales.

        quant_forward:
            Quantized inference using calibrated scales.
        """

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported quantization mode: {mode}. "
                f"Expected one of {sorted(self.VALID_MODES)}"
            )

        if self.model is None:
            raise RuntimeError(
                "Model has not been built. "
                "Call wrapper.build() before set_mode()."
            )

        changed = 0

        for module in self.model.modules():

            if isinstance(
                module,
                (QuantizedLinear, QuantizedMatMul),
            ):
                module.set_mode(mode)
                changed += 1

        print(
            f"Quantization mode: {mode} "
            f"({changed} modules)"
        )


    # =========================================================================
    # Model loading
    # =========================================================================


    def _load_model(self) -> nn.Module:
        """
        Load the original pretrained VLA model.

        Model-family-specific loading is isolated here so additional
        VLA families can be added later.
        """

        model_type = str(
            self.model_cfg.get(
                "type",
                "smolvla",
            )
        ).lower()

        if model_type == "smolvla":
            return self._load_smolvla()

        raise NotImplementedError(
            f"Unsupported model type: {model_type}"
        )


    def _load_smolvla(self) -> SmolVLAPolicy:
        """
        Load a pretrained SmolVLA policy.

        The checkpoint configuration is loaded first, then optional
        experiment-specific overrides are applied before model weights
        are loaded.
        """

        model_path = self.model_cfg.get(
            "path"
        )

        if not model_path:
            raise ValueError(
                "Missing config entry: model.path"
            )

        revision = self.model_cfg.get(
            "revision",
            None,
        )

        # ---------------------------------------------------------------------
        # 1. Load checkpoint configuration
        # ---------------------------------------------------------------------

        policy_cfg = SmolVLAConfig.from_pretrained(
            model_path,
            revision=revision,
        )

        # ---------------------------------------------------------------------
        # 2. Apply explicit device override
        # ---------------------------------------------------------------------

        if "device" in self.model_cfg:
            policy_cfg.device = self.model_cfg["device"]

        # ---------------------------------------------------------------------
        # 3. Apply inference-time config overrides
        # ---------------------------------------------------------------------

        overrides = self.model_cfg.get(
            "overrides",
            {},
        )

        for name, value in overrides.items():

            if not hasattr(policy_cfg, name):
                raise ValueError(
                    f"Unknown SmolVLA config override: {name}"
                )

            setattr(
                policy_cfg,
                name,
                value,
            )

        # ---------------------------------------------------------------------
        # 4. Load pretrained weights
        # ---------------------------------------------------------------------

        print(
            f"Loading SmolVLA from: {model_path}"
        )

        policy = SmolVLAPolicy.from_pretrained(
            model_path,
            config=policy_cfg,
            revision=revision,
        )

        policy.eval()

        return policy


    # =========================================================================
    # Linear replacement
    # =========================================================================


    def _wrap_linear_modules(self) -> None:
        """
        Replace selected nn.Linear modules with QuantizedLinear.

        Selection is controlled by:

            quantization.linear.enabled
            quantization.linear.include
            quantization.linear.exclude

        Pattern matching uses the full PyTorch module name.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        linear_cfg = self.quant_cfg.get(
            "linear",
            {},
        )

        if not linear_cfg.get(
            "enabled",
            True,
        ):
            print(
                "Linear quantization disabled."
            )
            return

        # Take a snapshot because modules will be modified during traversal.
        modules = list(
            self.model.named_modules()
        )

        for module_name, module in modules:

            if not isinstance(
                module,
                nn.Linear,
            ):
                continue

            if not self._should_wrap(
                module_name=module_name,
                module_config=linear_cfg,
            ):
                continue

            quantized_module = QuantizedLinear.from_float(
                module=module,
                layer_name=module_name,
                config=self.quant_cfg,
            )

            self._replace_module(
                module_name=module_name,
                new_module=quantized_module,
            )

            self.replaced_linear_modules.append(
                module_name
            )

        print(
            f"Replaced "
            f"{len(self.replaced_linear_modules)} "
            f"Linear modules."
        )


    # =========================================================================
    # MatMul replacement
    # =========================================================================


    def _wrap_matmul_modules(self) -> None:
        """
        Inject QuantizedMatMul into selected model operations.

        Unlike nn.Linear, torch.matmul operations do not appear as modules in
        model.named_modules(). Therefore MatMul replacement requires
        model-family-specific forward modification.

        SmolVLA MatMul injection will be implemented separately.

        For now:
            quantization.matmul.enabled = false

        allows Linear-only quantization experiments to run normally.
        """

        matmul_cfg = self.quant_cfg.get(
            "matmul",
            {},
        )

        if not matmul_cfg.get(
            "enabled",
            False,
        ):
            return

        model_type = str(
            self.model_cfg.get(
                "type",
                "smolvla",
            )
        ).lower()

        if model_type == "smolvla":
            self._wrap_smolvla_matmul(
                matmul_cfg
            )
            return

        raise NotImplementedError(
            f"MatMul wrapping is not implemented "
            f"for model type: {model_type}"
        )


    def _wrap_smolvla_matmul(
        self,
        matmul_cfg: dict[str, Any],
    ) -> None:
        """
        Inject QuantizedMatMul into SmolVLA attention operations.

        This will later replace / intercept operations such as:

            Q @ K^T
            P @ V

        The implementation is intentionally left for quant_matmul support.
        """

        raise NotImplementedError(
            "SmolVLA MatMul quantization has not been implemented yet. "
            "Set quantization.matmul.enabled=false for Linear-only experiments."
        )


    # =========================================================================
    # Module selection
    # =========================================================================


    @staticmethod
    def _should_wrap(
        module_name: str,
        module_config: dict[str, Any],
    ) -> bool:
        """
        Decide whether a module should be quantized.

        include:
            If specified, the module must match at least one pattern.

        exclude:
            If the module matches any exclude pattern, it is not quantized.

        Examples
        --------

        include:
            - "*"

        exclude:
            - "*vision_model*"
            - "*action_out_proj"

        or:

        include:
            - "*lm_expert*"
        """

        include_patterns = module_config.get(
            "include",
            ["*"],
        )

        exclude_patterns = module_config.get(
            "exclude",
            [],
        )

        included = any(
            fnmatch(
                module_name,
                pattern,
            )
            for pattern in include_patterns
        )

        excluded = any(
            fnmatch(
                module_name,
                pattern,
            )
            for pattern in exclude_patterns
        )

        return included and not excluded


    # =========================================================================
    # Module replacement utility
    # =========================================================================


    def _replace_module(
        self,
        module_name: str,
        new_module: nn.Module,
    ) -> None:
        """
        Replace a PyTorch submodule using its full module name.

        Example:

            model.model.action_in_proj

        module_name:

            "model.action_in_proj"
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        if not module_name:
            raise ValueError(
                "Cannot replace the root model."
            )

        parts = module_name.split(".")

        parent_name = ".".join(
            parts[:-1]
        )

        child_name = parts[-1]

        if parent_name:
            parent_module = self.model.get_submodule(
                parent_name
            )
        else:
            parent_module = self.model

        # PyTorch containers such as ModuleList use numeric names
        # stored directly in _modules.
        if child_name in parent_module._modules:

            parent_module._modules[
                child_name
            ] = new_module

        else:

            setattr(
                parent_module,
                child_name,
                new_module,
            )


    # =========================================================================
    # Summary
    # =========================================================================


    def _print_summary(self) -> None:
        """
        Print model wrapping summary.
        """

        if self.model is None:
            return

        total_params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print()
        print("Model wrapper summary")
        print("---------------------")
        print(
            f"Model type       : "
            f"{self.model_cfg.get('type', 'smolvla')}"
        )
        print(
            f"Total parameters : "
            f"{total_params:,}"
        )
        print(
            f"Quantization     : "
            f"{self.quant_cfg.get('enabled', False)}"
        )
        print(
            f"Quantized Linear : "
            f"{len(self.replaced_linear_modules)}"
        )
        print(
            f"Quantized MatMul : "
            f"{len(self.replaced_matmul_modules)}"
        )
