"""Experiment-local runner: identical wrappers/backends for raw and mixed PTQ."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import yaml

EXP = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / EXP.name
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def expected_ids():
    linear, mm = {}, {}
    for comp, n in [('vision',12),('vlm',16),('expert',16)]:
        groups = [('self_attn', ('q_proj','k_proj','v_proj','out_proj' if comp=='vision' else 'o_proj')),
                  ('mlp', ('fc1','fc2') if comp=='vision' else ('gate_proj','up_proj','down_proj'))]
        for i in range(n):
            for group, ops in groups:
                for op in ops: linear[f'{comp}.layers.{i}.{group}.{op}'] = comp
            for op in ('qk','pv'): mm[f'{comp}.layer.{i}.{op}'] = comp
    return linear, mm


def source_fingerprint():
    paths = list((ROOT/'src').rglob('*.py')) + list((EXP/'scripts').glob('*.py')) + list((EXP/'scripts').glob('*.sh')) + list((EXP/'configs').glob('*.yaml'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def prepare():
    from huggingface_hub import snapshot_download
    if (OUT / 'prepared.json').exists():
        raise FileExistsError('Archive the existing experiment output before preparing again')
    configs = {a:yaml.safe_load((EXP/f'configs/{a}.yaml').read_text()) for a in ('baseline','quant')}
    assert configs['baseline']['model']==configs['quant']['model']
    assert configs['baseline']['evaluation']==configs['quant']['evaluation']
    assert configs['quant']['evaluation']['n_episodes']==10
    assert configs['quant']['evaluation']['env']['task_ids']==list(range(10))
    # Pin the entire checkpoint directory, including processors and normalizers.
    model_cfg = configs['quant']['model']
    snapshot = Path(snapshot_download(repo_id=model_cfg['path'], revision=model_cfg.get('revision'))).resolve()
    for arm, cfg in configs.items():
        cfg['model']['path'] = str(snapshot)
        cfg['model']['revision'] = None
        (OUT/f'{arm}_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write(OUT/'prepared.json', {'checkpoint_repo':model_cfg['path'], 'snapshot':str(snapshot),
                               'snapshot_revision':snapshot.name, 'sources':source_fingerprint(),
                               'resolved_hashes':{a:hashlib.sha256((OUT/f'{a}_resolved.yaml').read_bytes()).hexdigest() for a in configs}})


def run(stage, suite):
    prepared=json.loads((OUT/'prepared.json').read_text())
    assert source_fingerprint()==prepared['sources'], 'Source/config changed; start a new run'
    for arm, digest in prepared['resolved_hashes'].items():
        assert hashlib.sha256((OUT/f'{arm}_resolved.yaml').read_bytes()).hexdigest()==digest
    from lerobot.utils.random_utils import set_seed
    from vla_tcs2.model_wrapper import ModelWrapper
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant_matmul import QuantizedMatMul
    from vla_tcs2.calibration import calibrate
    from vla_tcs2.eval import evaluate
    arm = 'baseline' if stage == 'baseline' else 'quant'
    cfg = yaml.safe_load((OUT/f'{arm}_resolved.yaml').read_text())
    dest = OUT / stage if stage in ('calibrate','smoke') else OUT / suite / stage
    if dest.exists(): raise FileExistsError(f'Archive previous stage output: {dest}')
    dest.mkdir(parents=True)
    cfg['evaluation']['env']['task'] = suite
    if stage == 'smoke':
        cfg['evaluation']['env']['task_ids']=[0]
        cfg['evaluation']['n_episodes']=1
    cfg['output_dir']=str(dest)
    (dest/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    set_seed(cfg['evaluation']['seed'])
    wrapper=ModelWrapper(cfg)
    model=wrapper.build(mode='raw').eval()
    assert model.config.n_action_steps==10 and model.config.num_steps==10 and model.config.chunk_size==50
    modules=[m for m in model.modules() if isinstance(m,(QuantizedLinear,QuantizedMatMul))]
    linear_ids, mm_ids=expected_ids()
    assert len(modules)==384 and {m.module_id for m in modules}==set(linear_ids)|set(mm_ids)
    for m in modules:
        comp=m.module_id.split('.')[0]
        if isinstance(m,QuantizedLinear):
            assert m.module_id in linear_ids and m.method=='pot_ao_outlier'
            assert (m.a_spec.name(),m.w_spec.name(),m.o_spec.name())==('e4m3','int4' if comp=='expert' else 'int8','e4m3')
        else:
            assert m.module_id in mm_ids and m.method=='pot_fp8_outlier'
            assert (m.A_spec.name(),m.B_spec.name(),m.O_spec.name())==('e4m3','e4m3','e4m3')
    if stage=='calibrate':
        scale_dir=Path(cfg['quantization']['scale_dir'])
        if any(scale_dir.glob('*.p')): raise FileExistsError('Fresh independent calibration required')
        wrapper.set_mode('scale_inspection')
        calibrate(model,cfg)
        assert all(m._stat_manager is wrapper.stat_manager for m in modules)
    if arm=='quant':
        entries=[]
        for m in modules:
            for idx, filename in enumerate(m._scale_file_paths()):
                path=Path(filename)
                with path.open('rb') as f: value=float(pickle.load(f))
                pot=not (isinstance(m,QuantizedLinear) and idx==0)
                assert math.isfinite(value) and value>0
                if pot: assert math.frexp(value)[0]==0.5, path
                entries.append({'file':path.name,'module_id':m.module_id,'value':value,'pot_required':pot,
                                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        assert len(entries)==1152 and len({x['file'] for x in entries})==1152
        assert {p.name for p in Path(cfg['quantization']['scale_dir']).glob('*.p')}=={x['file'] for x in entries}
        if stage!='calibrate':
            previous=json.loads((OUT/'calibrate/scale_audit.json').read_text())
            assert entries==previous, 'Scales changed since calibration'
        write(dest/'scale_audit.json',entries)
    if stage=='calibrate':
        write(dest/'completed.json',{'status':'PASS','sites':384,'scales':1152})
        return
    sm=wrapper.stat_manager
    wrapper.set_mode('raw' if arm=='baseline' else 'quant_forward')
    if arm=='quant':
        sm.enable_sparsity(enable=True,chunk_size=cfg['sparsity']['chunk_size'])
        sm.configure_unit_sparsity(enable=False)
    assert all(m.mode==('raw' if arm=='baseline' else 'quant_forward') for m in modules)
    from compute import ComputeCounter
    with ComputeCounter(model) as counter:
        result=evaluate(model,cfg,dest)
    counter.export(dest, expected_ids())
    write(dest/'result.json',result)
    write(dest/'execution.json',{'mode':modules[0].mode,'linear_sites':296,'matmul_sites':88,
                                'vision_backend':'explicit_eager','vlm_expert_backend':'injected_eager'})
    if arm=='quant':
        assert sm.collect_model_weight_sparsity(model)==296
        sp=dest/'sparsity';sp.mkdir()
        sm.export_quantization_manifest_csv(model,str(sp/'quantization_manifest.csv'))
        sm.export_module_sparsity_csv(str(sp/'module_sparsity.csv'))
        sm.export_per_layer_weight_sparsity_csv(str(sp/'weight_sparsity_static.csv'))
        sm.export_outlier_sidepath_csv(str(sp/'outlier_sidepath.csv'))
        sm.export_workload_csv(model,str(sp/'workload.csv'))
    from summarize import check_compute, results, read
    episodes = 1 if stage=='smoke' else 10
    results(dest, episodes, suite)
    check_compute(dest, read(dest/'sparsity/module_sparsity.csv') if arm=='quant' else None)
    # Reuse exactly the same verification logic for smoke and formal quant.
    if arm=='quant':
        from summarize import check_quant
        write(dest/'coverage_summary.json',check_quant(dest, episodes, suite))
    write(dest/'completed.json',{'status':'PASS','stage':stage,'suite':suite,'episodes':episodes})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','calibrate','smoke','baseline','quant'])
    parser.add_argument('--suite', choices=SUITES)
    args=parser.parse_args()
    if args.stage in ('baseline','quant') and args.suite is None:
        parser.error('--suite is required for formal evaluation')
    if args.stage in ('prepare','calibrate','smoke') and args.suite is not None:
        parser.error('prepare/calibrate/smoke use the fixed Goal protocol')
    os.chdir(ROOT)
    if args.stage=='prepare':
        OUT.mkdir(parents=True,exist_ok=True)
        prepare()
    else: run(args.stage, args.suite or 'libero_goal')
