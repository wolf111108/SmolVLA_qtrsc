"""Runtime adapter. Descriptors only: never copy tensors or synchronize CUDA."""
from math import prod
from .schema import TensorDesc, OperatorEvent, broadcast_shape


def describe(tensor, spec=None):
    dtype = str(tensor.dtype).removeprefix("torch.")
    # Disabled QuantSpec is a pass-through in this repository; use actual dtype.
    if spec is None or not getattr(spec, "enabled", False) or spec.kind == "none":
        fmt, bits = dtype, tensor.element_size() * 8
    elif spec.kind == "int":
        fmt, bits = f"int{spec.bits}", spec.bits
    elif spec.kind == "fp":
        fmt = spec.fmt
        widths = {"e2m1": 4, "e4m3": 8, "e4m3fn": 8, "e5m2": 8, "e5m10": 16}
        if fmt not in widths:
            raise ValueError(f"Unknown logical float format: {fmt}")
        bits = widths[fmt]
    elif spec.kind == "bf":
        fmt, bits = "bfloat16", 16
    else:
        raise ValueError(f"Unknown quantization kind: {spec.kind}")
    return TensorDesc(tuple(tensor.shape), dtype, fmt, bits)


def component_of(module_id):
    if module_id.split(".")[0] in {"vision", "vlm", "expert", "connector"}:
        return module_id.split(".")[0]
    if "vision_model" in module_id:
        return "vision"
    if "connector" in module_id:
        return "connector"
    if "lm_expert" in module_id:
        return "expert"
    if "text_model" in module_id:
        return "vlm"
    if any(s in module_id for s in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")):
        return "action_head"
    return "unknown"


def make_event(index, module_id, op_type, module, args, kwargs, output, context):
    mode = getattr(module, "mode", "raw")
    quantized = mode == "quant_forward"
    def operand(i, names):
        if len(args) > i:
            return args[i]
        for name in names:
            if name in kwargs:
                return kwargs[name]
        raise ValueError(f"Missing operand {i} at {module_id}")
    a_tensor = operand(0, ("x", "input", "A"))
    b_tensor = module.weight if op_type == "linear" else operand(1, ("B",))
    specs = ("a_spec", "w_spec", "o_spec") if op_type == "linear" else ("A_spec", "B_spec", "O_spec")
    a, b, o = [describe(tensor, getattr(module, spec, None) if quantized else None)
               for tensor, spec in zip((a_tensor, b_tensor, output), specs)]
    if op_type == "linear":
        m, k, n, batch, batch_shape = prod(a.shape[:-1]), a.shape[-1], b.shape[0], 1, ()
    else:
        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError("Vector matmul is outside this adapter's coverage")
        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        batch_shape = broadcast_shape(a.shape[:-2], b.shape[:-2])
        batch = prod(batch_shape)
    intervals = ("a_interval", "w_interval", "o_interval") if op_type == "linear" else ("A_interval", "B_interval", "O_interval")
    scale_shapes = {}
    for role, name in zip(("a", "b", "output"), intervals):
        value = getattr(module, name, None)
        scale_shapes[role] = None if value is None else list(getattr(value, "shape", ()))
    return OperatorEvent(
        event_id=index, module_id=module_id, component=component_of(module_id), op_type=op_type,
        a=a, b=b, output=o, m=m, n=n, k=k, batch=batch, batch_shape=batch_shape,
        phase=context.get("phase", "unknown"), flow_step=context.get("flow_step", -1),
        generation_id=context.get("generation_id", -1), attention_kind=context.get("attention_kind", "unknown"),
        mode=mode, method=getattr(module, "method", "raw") if quantized else "raw",
        weight_id=f"{module_id}:weight" if op_type == "linear" else None,
        metadata={"scale_shapes": scale_shapes, "outlier_ratio": getattr(module, "outlier_ratio", 0),
                  "is_bitnet": bool(getattr(module, "is_bitnet", False)),
                  "mixed_precision": bool(getattr(module, "mixed_precision", False)),
                  "has_bias": getattr(module, "bias", None) is not None,
                  "b_transposed": op_type == "linear", "features": ["metadata"]},
    )
