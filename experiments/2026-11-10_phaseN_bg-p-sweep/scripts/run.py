"""One Goal task/episode, all seven p values in the same quantized rollout."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/2026-11-09_phaseL_int8-goal'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(BASE / 'scripts'))
from common import expected_ids, require, digest, write
from run_arm import audit_scales
from vla_tcs2.stats.bit_group import BitGroupCollector, P_VALUES, observe_modules


def load_config():
    cfg = yaml.safe_load((EXP / 'configs/a8w4_expert_task0.yaml').read_text())
    old = yaml.safe_load((BASE / 'configs/a8w4_expert.yaml').read_text())
    normalized = json.loads(json.dumps(cfg))
    probe = normalized.pop('bit_group_probe')
    require(probe == {'p_values': list(P_VALUES), 'magnitude_bits': 8, 'group_bits': 2}, 'Wrong BG protocol')
    require(cfg['evaluation']['env']['task_ids'] == [0] and cfg['evaluation']['n_episodes'] == 1,
            'Expected one Goal task / episode')
    for value in (normalized, old):
        value.pop('output_dir')
        value['quantization'].pop('scale_dir')
        value['sparsity']['enabled'] = False
        value['evaluation']['env']['task_ids'] = [0]
        value['evaluation']['n_episodes'] = 1
    require(normalized == old, 'Quantization/model/calibration changed from latest INT experiment')
    return cfg


def check_environment():
    import torch
    import lerobot
    import libero
    import huggingface_hub
    require(torch.cuda.is_available(), 'A CUDA GPU and the smolvla_eval environment are required')
    return {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'huggingface_hub')}


def validate_reuse(previous, cfg):
    """Accept only the matching, completed Phase L calibration with its audit."""
    previous = previous.resolve()
    done = previous / 'calibrate/a8w4_expert'
    prepared = json.loads((previous / 'prepared.json').read_text())
    require(json.loads((done / 'completed.json').read_text()) ==
            {'status': 'PASS', 'stage': 'calibrate', 'arm': 'a8w4_expert'}, 'Incomplete calibration')
    old = yaml.safe_load((done / 'config.yaml').read_text())
    require(prepared['snapshot_revision'] == cfg['model']['revision'], 'Checkpoint revision mismatch')
    require(old['model']['overrides'] == cfg['model']['overrides'], 'Model override mismatch')
    require(old['calibration'] == cfg['calibration'], 'Calibration protocol mismatch')
    for field in ('vision', 'connector', 'linear', 'qk_matmul', 'pv_matmul', 'method',
                  'outlier_ratio', 'matmul_scale_granularity', 'linear_scale_granularity',
                  'weight_quant_granularity', 'weight_group_size', 'quantize_matmul'):
        require(old['quantization'][field] == cfg['quantization'][field], f'Quantization mismatch: {field}')
    # Check the code that actually generates scales. quant_methods.py only runs
    # in quant_forward; this experiment adds observation there, not calibration.
    for relative in ('calibration.py', 'model_wrapper.py', 'quant_linear.py', 'quant_matmul.py',
                     'quant/scale_methods.py', 'quant/quant_spec.py', 'quant/utils.py', 'quant/stat_manager.py'):
        path = f'src/vla_tcs2/{relative}'
        require(prepared['source_fingerprint'][path] == digest(ROOT / path), f'Calibration source changed: {path}')
    lerobot_files = {path: sha for path, sha in prepared['source_fingerprint'].items()
                     if path.startswith('lerobot_current/src/')}
    require(lerobot_files, 'Missing LeRobot calibration source fingerprint')
    for path, sha in lerobot_files.items():
        require((ROOT / path).is_file() and digest(ROOT / path) == sha, f'LeRobot source changed: {path}')
    source = Path(old['quantization']['scale_dir'])
    if not source.is_absolute():
        source = ROOT / source
    entries = json.loads((done / 'scale_audit.json').read_text())
    require(len(entries) == 864 and len({x['file'] for x in entries}) == 864, 'Expected 864 scales')
    for item in entries:
        require(Path(item['file']).name == item['file'], 'Invalid scale filename')
        require(digest(source / item['file']) == item['sha256'], 'Scale audit hash mismatch')
    return source, entries


def check_coverage(collector):
    linear, mm = expected_ids()
    expected = set()
    for mid in linear | mm:
        comp = mid.split('.')[0]
        for step in ([-1] if comp == 'vlm' else range(10)):
            for role in (['activation'] if mid in linear else ['A', 'B']):
                expected.add((mid, 'prefill' if comp == 'vlm' else 'denoise', step, role))
    observed = {(key[1], key[2], key[3], key[5]) for key in collector.observed_calls}
    require(observed == expected, f'Activation coverage mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}')
    return {'sites': len(linear) + len(mm), 'runtime_role_keys': len(observed),
            'calls': sum(collector.observed_calls.values()),
            'shapes': [{'module_id': k[1], 'phase': k[2], 'flow_step': k[3], 'role': k[5],
                        'shapes': sorted(v)} for k, v in sorted(collector.shapes.items())]}


def run(args):
    cfg = load_config()
    if args.preflight:
        print('PASS: latest A8W8O8/A8W4O8 config, QK/PV INT8, Goal task 0 x 1 episode')
        return
    versions = check_environment()  # Fail before downloading or creating outputs.
    if args.check_env:
        print(json.dumps(versions, indent=2))
        return
    from huggingface_hub import snapshot_download
    from lerobot.utils.random_utils import set_seed
    from vla_tcs2.model_wrapper import ModelWrapper
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant_matmul import QuantizedMatMul
    from vla_tcs2.calibration import calibrate
    from vla_tcs2.eval import evaluate

    previous = args.reuse_from or ROOT / 'outputs/2026-11-09_phaseL_int8-goal'
    reuse = None
    if not args.fresh_calibration and (args.reuse_from or (previous / 'calibrate/a8w4_expert/completed.json').exists()):
        reuse = validate_reuse(previous, cfg)
    out = (args.output_dir or ROOT / cfg['output_dir']).resolve()
    require(not out.exists(), f'Output exists; choose --output-dir for a new run: {out}')
    snapshot = Path(snapshot_download(repo_id=cfg['model']['path'], revision=cfg['model']['revision'])).resolve()
    require(snapshot.name == cfg['model']['revision'], 'Checkpoint revision mismatch')
    cfg['model'].update(path=str(snapshot), revision=None)
    scale_dir = out / 'scales'
    cfg['quantization']['scale_dir'] = str(scale_dir)
    cfg['quantization']['calibration_policy'] = {'default': 'reuse' if reuse else 'recalibrate'}
    cfg['output_dir'] = str(out)
    out.mkdir(parents=True)
    scale_dir.mkdir()
    if reuse:
        for item in reuse[1]:
            shutil.copy2(reuse[0] / item['file'], scale_dir / item['file'])
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_files = (sorted((ROOT / 'src').rglob('*.py')) + sorted((EXP / 'scripts').glob('*'))
                    + sorted((BASE / 'scripts').glob('*.py'))
                    + sorted((ROOT / 'lerobot_current/src').rglob('*.py')))
    meta = {'status': 'running', 'versions': versions, 'snapshot_revision': snapshot.name,
            'calibration': 'validated Phase L reuse' if reuse else 'fresh, matched 8 demonstrations',
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in source_files if p.is_file()},
            'config_sha256': digest(out / 'config.yaml')}
    write(out / 'execution.json', meta)
    try:
        set_seed(cfg['calibration']['seed'])
        wrapper = ModelWrapper(cfg)
        model = wrapper.build(mode='raw').eval()
        modules = [m for m in model.modules() if isinstance(m, (QuantizedLinear, QuantizedMatMul))]
        linear, mm = expected_ids()
        require(len(modules) == 288 and {m.module_id for m in modules} == set(linear) | set(mm), 'Wrong model coverage')
        for m in modules:
            expected_w = 'int4' if m.module_id.startswith('expert.') else 'int8'
            actual = ((m.a_spec.name(), m.w_spec.name(), m.o_spec.name()) if isinstance(m, QuantizedLinear)
                      else (m.A_spec.name(), m.B_spec.name(), m.O_spec.name()))
            expected = ('int8', expected_w, 'int8') if isinstance(m, QuantizedLinear) else ('int8',)*3
            require(actual == expected and m.method == 'outlier' and m.outlier_ratio == 0.01,
                    f'Wrong precision/method: {m.module_id}')
        if not reuse:
            wrapper.set_mode('scale_inspection')
            calibrate(model, cfg)
        audit = audit_scales(modules, scale_dir)
        if reuse:
            require(audit == reuse[1], 'Copied scale audit differs')
        write(out / 'scale_audit.json', audit)
        wrapper.set_mode('quant_forward')
        require(all(m.mode == 'quant_forward' for m in modules), 'Wrong runtime mode')
        model.reset()
        collector = BitGroupCollector(cfg['bit_group_probe']['p_values'])
        with observe_modules(modules, collector):
            result = evaluate(model, cfg, out / 'eval')
        coverage = check_coverage(collector)
        require(result['overall']['n_episodes'] == 1, 'Expected exactly one episode')
        require(audit_scales(modules, scale_dir) == audit, 'Runtime changed scales')
        collector.export(out / 'bit_groups')
        write(out / 'coverage.json', coverage)
        meta['status'] = 'PASS'
        write(out / 'execution.json', meta)
        write(out / 'completed.json', {'status': 'PASS'})
        print((out / 'bit_groups/summary.md').read_text())
    except Exception as exc:
        write(out / 'execution.json', dict(meta, status='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='CPU-only configuration check')
    mode.add_argument('--check-env', action='store_true', help='Check CUDA / model environment only')
    parser.add_argument('--output-dir', type=Path)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--reuse-from', type=Path, help='Completed Phase L output root with matching scales/audit')
    calibration.add_argument('--fresh-calibration', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['VLA_TENSOR_DUMP_DIR'] = ''
    run(args)
