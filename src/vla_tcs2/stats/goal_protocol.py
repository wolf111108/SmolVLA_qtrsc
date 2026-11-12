"""Shared Goal INT8 protocol validation; no Torch import at module load."""
import hashlib
import json
import math
from pathlib import Path
import pickle


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_ids():
    linear, mm = {}, {}
    for comp in ('vlm', 'expert'):
        for i in range(16):
            for group, ops in (('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
                               ('mlp', ('gate_proj', 'up_proj', 'down_proj'))):
                for op in ops:
                    linear[f'{comp}.layers.{i}.{group}.{op}'] = comp
            for op in ('qk', 'pv'):
                mm[f'{comp}.layer.{i}.{op}'] = comp
    return linear, mm


def audit_scales(modules, scale_dir):
    entries = []
    for m in sorted(modules, key=lambda x: x.module_id):
        for filename in m._scale_file_paths():
            path = Path(filename)
            with path.open('rb') as f:
                value = float(pickle.load(f))
            require(math.isfinite(value) and value > 0, f'Invalid scale: {path}')
            entries.append({'file': path.name, 'module_id': m.module_id,
                            'value': value, 'sha256': digest(path)})
    names = {e['file'] for e in entries}
    require(len(entries) == len(names) == 864, 'Expected 864 independent scales')
    require({p.name for p in scale_dir.glob('*.p')} == names, 'Missing/extra scale files')
    return entries
