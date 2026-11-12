"""Pinned, isolated calibration and Goal evaluation for three matched arms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from common import (ARMS, QUANT_ARMS, ROOT, OUT, digest, expected_ids, load_configs,
                    require, source_fingerprint, weight_bits, write)

sys.path.insert(0, str(ROOT / 'src'))
from vla_tcs2.stats.goal_protocol import audit_scales


def prepare():
    from huggingface_hub import snapshot_download
    require(not (OUT / 'prepared.json').exists(), 'Already prepared; archive outputs/scales for a new run')
    configs = load_configs()
    mc = configs['baseline']['model']
    checkpoint_repo = mc['path']
    snapshot = Path(snapshot_download(repo_id=mc['path'], revision=mc['revision'])).resolve()
    require(snapshot.name == mc['revision'], 'Checkpoint revision mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, cfg in configs.items():
        cfg['model'].update(path=str(snapshot), revision=None)
        (OUT / f'{arm}_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    write(OUT / 'prepared.json', {
        'checkpoint_repo': checkpoint_repo, 'snapshot': str(snapshot), 'snapshot_revision': snapshot.name,
        'source_fingerprint': source_fingerprint(),
        'resolved_configs': {a: digest(OUT / f'{a}_resolved.yaml') for a in ARMS},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    })


def check_prepared():
    prepared = json.loads((OUT / 'prepared.json').read_text())
    require(source_fingerprint() == prepared['source_fingerprint'], 'Source/config changed since prepare')
    require({a: digest(OUT / f'{a}_resolved.yaml') for a in ARMS} == prepared['resolved_configs'],
            'Resolved configs changed since prepare')
    return prepared


def run(stage, arm):
    require(stage != 'calibrate' or arm in QUANT_ARMS, 'Baseline needs no calibration')
    prepared = check_prepared()
    from lerobot.utils.random_utils import set_seed
    from vla_tcs2.model_wrapper import ModelWrapper
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant_matmul import QuantizedMatMul
    from vla_tcs2.calibration import calibrate
    from vla_tcs2.eval import evaluate
    from summarize import check_quant, results

    cfg = yaml.safe_load((OUT / f'{arm}_resolved.yaml').read_text())
    dest = OUT / stage / arm
    require(not dest.exists(), f'Existing stage output; archive before retry: {dest}')
    if stage == 'smoke':
        cfg['evaluation']['env']['task_ids'] = [0]
        cfg['evaluation']['n_episodes'] = 1
    cfg['output_dir'] = str(dest)
    set_seed(cfg['calibration']['seed'] if stage == 'calibrate' else cfg['evaluation']['seed'])
    wrapper = ModelWrapper(cfg)
    model = wrapper.build(mode='raw').eval()
    modules = [m for m in model.modules() if isinstance(m, (QuantizedLinear, QuantizedMatMul))]
    linear, mm = expected_ids()
    require(len(modules) == 288 and {m.module_id for m in modules} == set(linear) | set(mm), 'Wrong module coverage')
    for m in modules:
        comp = m.module_id.split('.')[0]
        if isinstance(m, QuantizedLinear):
            actual = (m.a_spec.name(), m.w_spec.name(), m.o_spec.name())
            expected = ('int8', f'int{weight_bits(arm, comp)}', 'int8')
        else:
            actual = (m.A_spec.name(), m.B_spec.name(), m.O_spec.name())
            expected = ('int8', 'int8', 'int8')
        require(actual == expected and m.method == 'outlier' and m.outlier_ratio == 0.01,
                f'Wrong precision/method: {m.module_id}')
    dest.mkdir(parents=True)
    (dest / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    scale_dir = Path(cfg['quantization']['scale_dir'])
    if stage == 'calibrate':
        require(not any(scale_dir.glob('*.p')), 'Fresh independent calibration required')
        wrapper.set_mode('scale_inspection')
        calibrate(model, cfg)
        require(all(m._stat_manager is wrapper.stat_manager for m in modules), 'Stat manager binding lost')
    if arm in QUANT_ARMS:
        audit = audit_scales(modules, scale_dir)
        if stage != 'calibrate':
            calibration = OUT / 'calibrate' / arm
            require((calibration / 'completed.json').exists(), 'Calibration incomplete')
            require(audit == json.loads((calibration / 'scale_audit.json').read_text()), 'Scales changed after calibration')
        write(dest / 'scale_audit.json', audit)
    if stage != 'calibrate':
        sm = wrapper.stat_manager
        mode = 'raw' if arm == 'baseline' else 'quant_forward'
        wrapper.set_mode(mode)
        require(all(m.mode == mode for m in modules), 'Wrong runtime mode')
        if arm in QUANT_ARMS:
            sm.enable_sparsity(enable=True, chunk_size=cfg['sparsity']['chunk_size'])
            sm.configure_unit_sparsity(enable=False)
        write(dest / 'result.json', evaluate(model, cfg, dest))
        write(dest / 'execution.json', {'arm': arm, 'mode': mode, 'linear_sites': 224,
              'matmul_sites': 64, 'vlm_expert_backend': 'injected_eager', 'vision': 'raw',
              'prepared_sha256': digest(OUT / 'prepared.json'), 'snapshot_revision': prepared['snapshot_revision']})
        if arm in QUANT_ARMS:
            require(sm.collect_model_weight_sparsity(model) == 224, 'Wrong static weight coverage')
            sp = dest / 'sparsity'
            sp.mkdir()
            sm.export_quantization_manifest_csv(model, str(sp / 'quantization_manifest.csv'))
            sm.export_module_sparsity_csv(str(sp / 'module_sparsity.csv'))
            sm.export_per_layer_weight_sparsity_csv(str(sp / 'weight_sparsity_static.csv'))
            sm.export_outlier_sidepath_csv(str(sp / 'outlier_sidepath.csv'))
            require(audit_scales(modules, scale_dir) == audit, 'Evaluation modified scales')
            report = check_quant(dest, cfg['evaluation']['n_episodes'], arm)
        else:
            report = results(dest, cfg['evaluation']['n_episodes'])
        write(dest / 'coverage_summary.json', report)
    write(dest / 'completed.json', {'status': 'PASS', 'stage': stage, 'arm': arm})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'prepare', 'calibrate', 'smoke', 'eval'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == 'preflight':
        load_configs()
        print('PASS: matched Goal protocol and INT precision configuration')
    elif args.stage == 'prepare':
        prepare()
    else:
        if args.arm is None:
            parser.error('--arm is required for calibrate/smoke/eval')
        run(args.stage, args.arm)
