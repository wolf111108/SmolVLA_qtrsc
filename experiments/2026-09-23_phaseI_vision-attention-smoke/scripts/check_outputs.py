"""Fail closed on missing scales, sites, operand roles or native counters."""
import csv
import json
import pickle
from pathlib import Path
import sys
import math
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
out = Path(cfg['output_dir'])
scales = Path(cfg['quantization']['scale_dir'])
expected = {f'vision.layer.{i}.{op}' for i in range(12) for op in ('qk', 'pv')}
paths = {scales / f'vision_{op}_matmul_{role}_scale_{i}.p'
         for i in range(12) for op in ('qk', 'pv') for role in ('A', 'B', 'O')}
assert set(scales.glob('*.p')) == paths, 'Expected exactly 72 independent scale files'
for path in paths:
    with path.open('rb') as f: value = float(pickle.load(f))
    assert math.isfinite(value) and value > 0, path

def read(name):
    with (out / 'sparsity' / name).open() as f:
        return list(csv.DictReader(f))
manifest = read('quantization_manifest.csv')
assert len(manifest) == 24 and {r['module_id'] for r in manifest} == expected
rows = read('module_sparsity.csv')
assert len(rows) == 72
assert {(r['module_id'], r['tensor_role']) for r in rows} == {(m, r) for m in expected for r in ('A','B','O')}
assert all(r['component'] == 'vision' and r['phase'] == 'prefill' and
           r['attention_kind'] == 'self' for r in rows)
for r in rows:
    for num, den in [('zero_elements_native','total_elements_native'),
                     ('sparse_bits_native','total_bits_native')]:
        assert 0 <= int(r[num]) <= int(r[den]) and int(r[den]) > 0
summary = {'status': 'PASS', 'sites': 24, 'scale_files': 72, 'runtime_rows': 72,
           'fp_bit_metric': 'S|MMM', 'fp_bit_metric_version': 1}
for label, num, den in [('element_sparsity','zero_elements_native','total_elements_native'),
                        ('bit_sparsity','sparse_bits_native','total_bits_native')]:
    summary[label] = sum(int(r[num]) for r in rows) / sum(int(r[den]) for r in rows)
(out / 'coverage_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
