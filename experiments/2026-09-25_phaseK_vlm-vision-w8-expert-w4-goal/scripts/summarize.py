"""Validate complete task/site coverage and aggregate counts, never percentages."""
import csv
import json
from pathlib import Path
from run_arm import OUT, expected_ids, write


def results(path, episodes):
    data=json.loads((path/'result.json').read_text())
    expected={0} if episodes==1 else set(range(10))
    tasks={}
    for row in data['per_task']:
        tid=row['task_id'];values=row['metrics']['successes']
        assert row['task_group']=='libero_goal' and tid not in tasks and tid in expected
        assert len(values)==episodes and all(isinstance(v,bool) for v in values)
        tasks[tid]={'successes':sum(values),'episodes':len(values),'sr':sum(values)/len(values)}
    assert set(tasks)==expected
    total=sum(t['episodes'] for t in tasks.values())
    successes=sum(t['successes'] for t in tasks.values())
    assert data['overall']['n_episodes']==total
    assert abs(data['overall']['pc_success']-100*successes/total)<1e-6
    return {'successes':successes,'episodes':total,'sr':successes/total,'tasks':tasks}


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def aggregate(rows,native):
    assert rows
    suffix='_native' if native else ''
    out={}
    for num,den in [('zero_elements','total_elements'),('sparse_bits','total_bits')]:
        for r in rows: assert 0<=int(r[num+suffix])<=int(r[den+suffix]) and int(r[den+suffix])>0
        out[num]=sum(int(r[num+suffix]) for r in rows)
        out[den]=sum(int(r[den+suffix]) for r in rows)
    out['element_sparsity']=out['zero_elements']/out['total_elements']
    out['bit_sparsity']=out['sparse_bits']/out['total_bits']
    return out


def check_quant(path,episodes):
    linear,mm=expected_ids();sp=path/'sparsity'
    manifest=read(sp/'quantization_manifest.csv')
    assert len(manifest)==384 and {r['module_id'] for r in manifest}==set(linear)|set(mm)
    for r in manifest:
        comp=r['module_id'].split('.')[0]
        assert r['component']==comp and r['a_or_A_kind']==r['o_or_O_kind']=='e4m3'
        if r['module_id'] in linear:
            assert r['op_type']=='linear' and r['method']=='pot_ao_outlier'
            assert r['w_or_B_kind']==('int4' if comp=='expert' else 'int8')
        else:
            assert r['op_type']=='matmul' and r['method']=='pot_fp8_outlier' and r['w_or_B_kind']=='e4m3'
    rows=read(sp/'module_sparsity.csv')
    expected=set()
    for ids,roles in [(linear,('activation','output')),(mm,('A','B','O'))]:
        for mid,comp in ids.items():
            for step in (range(10) if comp=='expert' else [-1]):
                for role in roles:expected.add((mid,'denoise' if comp=='expert' else 'prefill',step,role))
    actual={(r['module_id'],r['phase'],int(r['flow_step']),r['tensor_role']) for r in rows}
    assert actual==expected and len(rows)==len(expected)==3736
    assert all(r['component']==r['module_id'].split('.')[0] for r in rows)
    assert all(int(r['total_bits_native'])==4*int(r['total_elements_native']) for r in rows)
    weights=read(sp/'weight_sparsity_static.csv')
    assert len(weights)==296 and {r['module_id'] for r in weights}==set(linear)
    for r in weights:
        comp=linear[r['module_id']]
        assert r['component']==comp and r['weight_spec']==('int4' if comp=='expert' else 'int8')
    groups={'vision':['vision'],'vlm':['vlm'],'vision_vlm':['vision','vlm'],'expert':['expert']}
    report={'status':'PASS','result':results(path,episodes),'sites':384,'runtime_rows':len(rows),
            'weight_rows':len(weights),'runtime_bit_metric':'S|MMM v1',
            'weight_bit_metric':'integer sign-aware; INT8 and INT4 reported separately','groups':{}}
    for name,comps in groups.items():
        rr=[r for r in rows if r['component'] in comps]
        ww=[r for r in weights if r['component'] in comps]
        report['groups'][name]={'runtime':aggregate(rr,True),'static_weight':aggregate(ww,False),
                               'runtime_linear':aggregate([r for r in rr if r['module_id'] in linear],True),
                               'runtime_matmul':aggregate([r for r in rr if r['module_id'] in mm],True)}
    return report


if __name__=='__main__':
    baseline=results(OUT/'baseline',10)
    quant=check_quant(OUT/'quant',10)
    write(OUT/'summary.json',{'baseline':baseline,'quant':quant,
                            'delta_sr_pp':100*(quant['result']['sr']-baseline['sr'])})
    with (OUT/'task_success.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task_id','episodes','baseline_successes','quant_successes','delta_pp'])
        for i in range(10):
            b,q=baseline['tasks'][i],quant['result']['tasks'][i]
            w.writerow([i,10,b['successes'],q['successes'],100*(q['sr']-b['sr'])])
    print('PASS: summary.json and task_success.csv written')
