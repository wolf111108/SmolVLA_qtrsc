"""Validate joint vision coverage and export counter-weighted summaries."""
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
out = Path(cfg['output_dir'])
scales = Path(cfg['quantization']['scale_dir'])
pre = json.loads((out.parent / 'preflight.json').read_text())
assert pre.get('schema_version') == 2 and pre.get('eligible_for_smoke') is True
assert pre['status'] in ('PASS', 'PASS_WITH_BACKEND_DRIFT')
assert all(pre['hard_gates'].get(k) is True for k in
           ('complete', 'coverage', 'finite', 'adapter_vs_eager', 'linear_raw'))
linear_ids = {f'vision.layers.{i}.{group}.{op}' for i in range(12)
              for group, ops in [('self_attn', ('q_proj','k_proj','v_proj','out_proj')),
                                 ('mlp', ('fc1','fc2'))] for op in ops}
mm_ids = {f'vision.layer.{i}.{op}' for i in range(12) for op in ('qk','pv')}
paths = {scales / f'vision_{op}_{role}_scale_{i}.p' for i in range(12)
         for op in ('q_proj','k_proj','v_proj','out_proj','fc1','fc2') for role in ('a','w','o')}
paths |= {scales / f'vision_{op}_matmul_{role}_scale_{i}.p' for i in range(12)
          for op in ('qk','pv') for role in ('A','B','O')}
assert set(scales.glob('*.p')) == paths and len(paths) == 288
scale_report = []
for path in sorted(paths):
    with path.open('rb') as f:
        value = float(pickle.load(f))  # Only this run's locally generated trusted scales.
    assert math.isfinite(value) and value > 0, path
    assert math.frexp(value)[0] == 0.5, f'Not PoT: {path}: {value}'
    scale_report.append({'file':path.name, 'value':value,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(out / 'scale_audit.json').write_text(json.dumps(scale_report, indent=2))

def read(name):
    with (out / 'sparsity' / name).open() as f:
        return list(csv.DictReader(f))
manifest = read('quantization_manifest.csv')
assert len(manifest) == 96 and {r['module_id'] for r in manifest} == linear_ids | mm_ids
for r in manifest:
    assert r['component'] == 'vision' and r['method'] == 'pot_fp8_outlier'
    assert r['op_type'] == ('linear' if r['module_id'] in linear_ids else 'matmul')
    assert all(r[k] == 'e4m3' for k in ('a_or_A_kind','w_or_B_kind','o_or_O_kind'))
    assert float(r['outlier_ratio']) == 0.01
rows = read('module_sparsity.csv')
expected = {(m,r) for m in linear_ids for r in ('activation','output')}
expected |= {(m,r) for m in mm_ids for r in ('A','B','O')}
assert len(rows) == 216 and {(r['module_id'],r['tensor_role']) for r in rows} == expected
assert all(r['component'] == 'vision' and r['phase'] == 'prefill' for r in rows)
assert all(r['attention_kind'] == 'self' for r in rows if r['module_id'] in mm_ids)
weights = read('weight_sparsity_static.csv')
assert len(weights) == 72 and {r['module_id'] for r in weights} == linear_ids
assert all(r['component'] == 'vision' and r['weight_spec'] == 'e4m3' for r in weights)

def summary(items, native):
    suffix = '_native' if native else ''
    totals = {}
    for num, den in [('zero_elements','total_elements'), ('sparse_bits','total_bits')]:
        n, d = num+suffix, den+suffix
        for row in items:
            assert 0 <= int(row[n]) <= int(row[d]) and int(row[d]) > 0
        totals[num], totals[den] = sum(int(r[n]) for r in items), sum(int(r[d]) for r in items)
    totals['element_sparsity'] = totals['zero_elements']/totals['total_elements']
    totals['bit_sparsity'] = totals['sparse_bits']/totals['total_bits']
    return totals
result = json.loads((out / 'result.json').read_text())
assert result['overall']['n_episodes'] == 1
assert len(result['per_task']) == 1
assert result['per_task'][0]['task_group'] == 'libero_goal' and result['per_task'][0]['task_id'] == 0
assert len(result['per_task'][0]['metrics']['successes']) == 1
report = {'status':'PASS', 'scope':'vision_joint_engineering_smoke_only',
          'linear_sites':72, 'matmul_sites':24, 'scale_files':288,
          'runtime_rows':216, 'weight_rows':72, 'fp_bit_metric':'S|MMM', 'fp_bit_metric_version':1,
          'preflight_status':pre['status'], 'backend_status':pre['backend_status'],
          'episode_success':result['per_task'][0]['metrics']['successes'][0],
          'runtime_linear':summary([r for r in rows if r['module_id'] in linear_ids], True),
          'runtime_matmul':summary([r for r in rows if r['module_id'] in mm_ids], True),
          'runtime_pooled':summary(rows, True), 'static_weight':summary(weights, False)}
(out / 'coverage_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
