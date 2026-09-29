"""Validate Goal SR and precision coverage without importing torch."""
import csv
import json
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
    total=sum(t['episodes'] for t in tasks.values());successes=sum(t['successes'] for t in tasks.values())
    assert data['overall']['n_episodes']==total
    assert abs(data['overall']['pc_success']-100*successes/total)<1e-6
    return {'successes':successes,'episodes':total,'sr':successes/total,'tasks':tasks}


def summarize():
    dest=OUT/'quant';metrics=results(dest,10)
    completed=json.loads((dest/'completed.json').read_text())
    assert completed['status']=='PASS' and completed['result']['successes']==metrics['successes']
    linear,mm=expected_ids()
    with (dest/'quantization_manifest.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==384 and {r['module_id'] for r in rows}==set(linear)|set(mm)
    for r in rows:
        comp=r['module_id'].split('.')[0]
        assert r['component']==comp and r['quantized']=='true'
        assert r['a_or_A_kind']==r['o_or_O_kind']=='e4m3'
        assert float(r['outlier_ratio'])==0.01
        if r['module_id'] in linear:
            assert r['op_type']=='linear' and r['method']=='pot_ao_outlier'
            assert r['w_or_B_kind']==('int8' if comp=='vlm' else 'int4')
            assert r['weight_quant_granularity']=='per_tensor'
        else:
            assert r['op_type']=='matmul' and r['method']=='pot_fp8_outlier' and r['w_or_B_kind']=='e4m3'
    execution=json.loads((dest/'execution.json').read_text())
    assert execution['mode']=='quant_forward' and not execution['sparsity_enabled']
    assert set(execution['calls'])==set(linear)|set(mm) and all(n>0 for n in execution['calls'].values())
    assert json.loads((dest/'scale_audit.json').read_text())==json.loads((OUT/'calibrate/scale_audit.json').read_text())
    write(OUT/'summary.json',{'status':'PASS','suite':'libero_goal','result':metrics,
                            'precision':{'vision':'WINT4','vlm':'WINT8','expert':'WINT4',
                                         'linear_AO_matmul_ABO':'E4M3 PoT','integer_weight_scale':'continuous'},
                            'sites':384,'baseline':'not rerun; historical results are separate references'})
    with (OUT/'task_success.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task_id','successes','episodes','sr'])
        for tid,t in sorted(metrics['tasks'].items()):w.writerow([tid,t['successes'],t['episodes'],t['sr']])
    print(f"PASS: Goal SR={metrics['successes']}/{metrics['episodes']} ({100*metrics['sr']:.1f}%)")


if __name__=='__main__':summarize()
