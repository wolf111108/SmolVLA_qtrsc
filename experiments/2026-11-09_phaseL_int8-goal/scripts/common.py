"""Experiment constants and lightweight validation (no Torch dependency)."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from vla_tcs2.stats.goal_protocol import digest, expected_ids, require, write
EXP = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / EXP.name
ARMS = ('baseline', 'a8w8', 'a8w4_expert')
QUANT_ARMS = ARMS[1:]


def source_fingerprint():
    files = [ROOT / 'pyproject.toml']
    for directory, pattern in ((ROOT / 'src', '*.py'),
                               (ROOT / 'lerobot_current/src', '*.py'),
                               (EXP / 'scripts', '*.py'), (EXP / 'configs', '*.yaml')):
        files.extend(directory.rglob(pattern))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def weight_bits(arm, comp):
    return 4 if arm == 'a8w4_expert' and comp == 'expert' else 8


def load_configs():
    configs = {arm: yaml.safe_load((EXP / f'configs/{arm}.yaml').read_text()) for arm in ARMS}
    # All differences must be intentional: mode/stat collection, W4, output/scale paths.
    reference = None
    for arm, cfg in configs.items():
        normalized = json.loads(json.dumps(cfg))
        normalized.pop('output_dir')
        q = normalized['quantization']
        q.pop('scale_dir')
        require(q['mode'] == ('raw' if arm == 'baseline' else 'quant_forward'), 'Wrong mode')
        q['mode'] = 'raw'
        normalized['sparsity']['enabled'] = False
        for rule in q['linear']['overrides']:
            comp = rule['target']['component']
            require(rule['config']['w_bit'] == weight_bits(arm, comp), 'Wrong weight precision')
            rule['config']['w_bit'] = 8
        if reference is None:
            reference = normalized
        require(normalized == reference, f'Uncontrolled arm difference: {arm}')
        require(q['enabled'] and q['quantize_matmul'] and q['method'] == 'outlier', 'INT QK/PV required')
        for op in ('qk_matmul', 'pv_matmul'):
            require(tuple(q[op][k] for k in ('A_bit', 'B_bit', 'O_bit')) == (8, 8, 8), 'QK/PV must be INT8')
        require(cfg['evaluation']['env']['task_ids'] == list(range(10)), 'All Goal tasks required')
        require(cfg['evaluation']['n_episodes'] == 10, '10 episodes per task required')
    require(len({c['quantization']['scale_dir'] for c in configs.values()}) == 3, 'Separate scale directories required')
    return configs
