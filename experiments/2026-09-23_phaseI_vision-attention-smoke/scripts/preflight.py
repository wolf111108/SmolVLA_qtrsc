"""Checkpoint-backed routing and raw SDPA/eager equivalence gate; no rollout."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import yaml
from vla_tcs2.model_wrapper import ModelWrapper, _inject_smolvlm_vision_quantized_matmul
from vla_tcs2.quant_matmul import QuantizedMatMul
from vla_tcs2.quant_linear import QuantizedLinear

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
torch.manual_seed(1000)
control = copy.deepcopy(cfg)
control['quantization']['vision']['matmul']['enabled'] = False
wrapper = ModelWrapper(control)
model = wrapper.build(mode='raw').eval()
assert not any(isinstance(m, (QuantizedLinear, QuantizedMatMul)) for m in model.modules())
vision = model.model.vlm_with_expert.get_vlm_model().vision_model
assert _inject_smolvlm_vision_quantized_matmul(model, cfg['quantization'], 'raw', wrapper.stat_manager) == 24
modules = [m for m in model.modules() if isinstance(m, QuantizedMatMul)]
expected_ids = {f'vision.layer.{i}.{op}' for i in range(12) for op in ('qk', 'pv')}
assert {m.module_id for m in modules} == expected_ids
assert len({m._scale_identity() for m in modules}) == 24
calls = dict.fromkeys(expected_ids, 0)
def hook(module, args, result):
    calls[module.module_id] += 1
handles = [m.register_forward_hook(hook) for m in modules]
records = []
try:
    with torch.no_grad():
        for i, layer in enumerate(vision.encoder.layers):
            attn = layer.self_attn
            weight = attn.q_proj.weight
            x = torch.randn(1, 1024, attn.embed_dim, device=weight.device, dtype=weight.dtype)
            # Test both no mask and a partial padding mask; never mask all keys.
            for masked in (False, True):
                mask = None
                if masked:
                    mask = torch.zeros(1, 1, 1024, 1024, device=x.device, dtype=x.dtype)
                    mask[..., -16:] = torch.finfo(x.dtype).min
                expected = attn._original_vision_forward(x, attention_mask=mask)[0]
                actual = attn(x, attention_mask=mask)[0]
                atol, rtol = ((1e-5, 1e-4) if x.dtype == torch.float32 else (5e-3, 5e-2))
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                records.append({'layer': i, 'masked': masked, 'dtype': str(x.dtype),
                                'max_abs_error': (actual-expected).abs().max().item(),
                                'mean_abs_error': (actual-expected).abs().mean().item(),
                                'atol': atol, 'rtol': rtol})
finally:
    for h in handles: h.remove()
assert all(n == 2 for n in calls.values()), calls
out = Path(cfg['output_dir']).parent / 'preflight.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'status': 'PASS', 'calls': calls, 'comparisons': records}, indent=2))
print(f'PASS: 24 vision MatMul sites, raw equivalence; {out}')
