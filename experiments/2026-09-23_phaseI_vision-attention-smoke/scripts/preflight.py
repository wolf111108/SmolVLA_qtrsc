"""Three-way checkpoint attention diagnostic; write evidence before failing.

No tolerance relaxation: adapter/eager correctness is a hard gate.
Finite backend drift is diagnostic only when explicitly enabled by the smoke config.
"""
import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import yaml
from vla_tcs2.model_wrapper import ModelWrapper, _inject_smolvlm_vision_quantized_matmul
from vla_tcs2.quant_matmul import QuantizedMatMul
from vla_tcs2.quant_linear import QuantizedLinear


def compare(actual, expected, atol, rtol):
    # Measure in FP32, rather than rounding errors/thresholds back into BF16.
    a, b = actual.detach().float(), expected.detach().float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    error = (a - b).abs()
    limit = atol + rtol * b.abs()
    mismatched = (~finite) | (error > limit)
    valid_errors = error[finite]
    flat_index = int(torch.where(finite, error, torch.full_like(error, float('inf'))).flatten().argmax())
    index = []
    for size in reversed(error.shape):
        index.append(flat_index % size)
        flat_index //= size
    return {
        'passed': not bool(mismatched.any()),
        'numel': a.numel(), 'mismatched': int(mismatched.sum()),
        'nonfinite': int((~finite).sum()),
        'max_abs_error': float(valid_errors.max()) if valid_errors.numel() else None,
        'mean_abs_error': float(valid_errors.mean()) if valid_errors.numel() else None,
        'max_error_index': list(reversed(index)),
        'atol': atol, 'rtol': rtol,
        'criterion': 'abs(actual-expected) <= atol + rtol*abs(expected)',
    }


def original_forward(attn, backend, x, mask):
    # The saved bound method uses attn.config. Assign an isolated copy to this
    # instance; never mutate the shared vision config or global backend registry.
    shared_config = attn.config
    local_config = copy.deepcopy(shared_config)
    local_config._attn_implementation = backend
    attn.config = local_config
    try:
        return attn._original_vision_forward(
            x, attention_mask=mask, output_attentions=False
        )[0]
    finally:
        attn.config = shared_config


def main():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    require_backend = cfg.get('preflight', {}).get('require_backend_equivalence', True)
    if not isinstance(require_backend, bool):
        raise ValueError('preflight.require_backend_equivalence must be a YAML boolean')
    out = Path(cfg['output_dir']).parent / 'preflight.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f'Archive previous evidence before rerunning: {out}')
    report = {'status': 'RUNNING', 'seed': 1000, 'calls': {}, 'comparisons': [],
              'errors': [], 'gates': {}, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'schema_version': 2,
              'require_backend_equivalence': require_backend,
              'eligible_for_smoke': False, 'warnings': []}
    handles = []

    def save():
        temporary = out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(out)

    save()
    try:
        report['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['transformers'] = importlib.metadata.version('transformers')
        torch.manual_seed(1000)
        control = copy.deepcopy(cfg)
        control['quantization']['vision']['matmul']['enabled'] = False
        wrapper = ModelWrapper(control)
        model = wrapper.build(mode='raw').eval()
        if any(isinstance(m, (QuantizedLinear, QuantizedMatMul)) for m in model.modules()):
            raise RuntimeError('Control must contain no quantized modules')
        vision = model.model.vlm_with_expert.get_vlm_model().vision_model
        count = _inject_smolvlm_vision_quantized_matmul(
            model, cfg['quantization'], 'raw', wrapper.stat_manager)
        modules = [m for m in model.modules() if isinstance(m, QuantizedMatMul)]
        expected_ids = {f'vision.layer.{i}.{op}' for i in range(12) for op in ('qk', 'pv')}
        if (count != 24 or len(modules) != 24 or
                {m.module_id for m in modules} != expected_ids or
                len({m._scale_identity() for m in modules}) != 24 or
                any(m.mode != 'raw' for m in modules)):
            raise RuntimeError('Expected 24 unique raw vision MatMul sites and scale groups')
        report['calls'] = dict.fromkeys(sorted(expected_ids), 0)

        def hook(module, args, result):
            report['calls'][module.module_id] += 1
        handles = [m.register_forward_hook(hook) for m in modules]
        with torch.no_grad():
            for i, layer in enumerate(vision.encoder.layers):
                attn = layer.self_attn
                weight = attn.q_proj.weight
                for masked in (False, True):
                    record = {'layer': i, 'masked': masked, 'dtype': str(weight.dtype),
                              'device': str(weight.device),
                              'original_backend': attn.config._attn_implementation}
                    report['comparisons'].append(record)
                    try:
                        x = torch.randn(1, 1024, attn.embed_dim,
                                        device=weight.device, dtype=weight.dtype)
                        mask = None
                        if masked:
                            mask = torch.zeros(1, 1, 1024, 1024, device=x.device, dtype=x.dtype)
                            mask[..., -16:] = torch.finfo(x.dtype).min
                        # Identical checkpoint weights, input and mask for all three paths.
                        sdpa = original_forward(attn, 'sdpa', x, mask)
                        eager = original_forward(attn, 'eager', x, mask)
                        adapted = attn(x, attention_mask=mask, output_attentions=False)[0]
                        backend_tol = (1e-5, 1e-4) if x.dtype == torch.float32 else (5e-3, 5e-2)
                        record['adapter_vs_eager'] = compare(adapted, eager, 1e-6, 1e-5)
                        record['eager_vs_sdpa'] = compare(eager, sdpa, *backend_tol)
                        record['adapter_vs_sdpa'] = compare(adapted, sdpa, *backend_tol)
                        print(f"layer={i} masked={masked}: " + ', '.join(
                            f"{name} mismatched={record[name]['mismatched']}"
                            for name in ('adapter_vs_eager', 'eager_vs_sdpa', 'adapter_vs_sdpa')))
                    except Exception:
                        record['error'] = traceback.format_exc()
                        report['errors'].append({'layer': i, 'masked': masked, 'error': record['error']})
                    save()  # Persist each case, including failures; inspect remaining layers.
        complete = len(report['comparisons']) == 24 and not report['errors']
        report['gates'] = {
            'complete': complete,
            'coverage': all(n == 2 for n in report['calls'].values()),
            **{name: complete and all(r[name]['passed'] for r in report['comparisons'])
               for name in ('adapter_vs_eager', 'eager_vs_sdpa', 'adapter_vs_sdpa')},
        }
        pairs = ('adapter_vs_eager', 'eager_vs_sdpa', 'adapter_vs_sdpa')
        report['gates']['finite'] = complete and all(
            r[name]['nonfinite'] == 0 for r in report['comparisons'] for name in pairs)
        hard_names = ('complete', 'coverage', 'finite', 'adapter_vs_eager')
        report['hard_gates'] = {name: report['gates'][name] for name in hard_names}
        backend_ok = all(report['gates'][name] for name in pairs[1:])
        report['backend_status'] = 'PASS' if backend_ok else 'DIFFERENT'
        eligible = all(report['hard_gates'].values()) and (backend_ok or not require_backend)
        report['eligible_for_smoke'] = eligible
        if not eligible:
            report['status'] = 'FAIL'
        elif backend_ok:
            report['status'] = 'PASS'
        else:
            report['status'] = 'PASS_WITH_BACKEND_DRIFT'
            report['warnings'].append(
                'Finite SDPA/eager drift retained at unchanged tolerances. '
                'Eligible for engineering smoke only; no closed-loop equivalence claim.')
    except Exception:
        report['errors'].append({'error': traceback.format_exc()})
        report['status'] = 'ERROR'
        report['eligible_for_smoke'] = False
    finally:
        for handle in handles:
            handle.remove()
        save()
    print(f"{report['status']}: {out}; gates={report['gates']}")
    return 0 if report['status'] in ('PASS', 'PASS_WITH_BACKEND_DRIFT') else 1


if __name__ == '__main__':
    raise SystemExit(main())
