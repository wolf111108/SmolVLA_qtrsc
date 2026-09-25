"""Validate complete task/site coverage and aggregate counts, never percentages."""
import csv
import json
from pathlib import Path
from run_arm import OUT, SUITES, expected_ids, write


def results(path, episodes, suite):
    data=json.loads((path/'result.json').read_text())
    expected={0} if episodes==1 else set(range(10))
    tasks={}
    for row in data['per_task']:
        tid=row['task_id'];values=row['metrics']['successes']
        assert row['task_group']==suite and tid not in tasks and tid in expected
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


def check_quant(path,episodes,suite):
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
    report={'status':'PASS','result':results(path,episodes,suite),'sites':384,'runtime_rows':len(rows),
            'weight_rows':len(weights),'runtime_bit_metric':'S|MMM v1',
            'weight_bit_metric':'integer sign-aware; INT8 and INT4 reported separately','groups':{}}
    for name,comps in groups.items():
        rr=[r for r in rows if r['component'] in comps]
        ww=[r for r in weights if r['component'] in comps]
        report['groups'][name]={'runtime':aggregate(rr,True),'static_weight':aggregate(ww,False),
                               'runtime_linear':aggregate([r for r in rr if r['module_id'] in linear],True),
                               'runtime_matmul':aggregate([r for r in rr if r['module_id'] in mm],True)}
    return report


def csv_write(path, rows):
    assert rows
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def check_compute(path, runtime=None):
    report=json.loads((path/'compute_summary.json').read_text())
    rows=read(path/'compute.csv')
    assert report['status']=='PASS' and report['wrapped_sites']==384 and report['generations']>0
    keys={(r['module_id'],r['phase'],int(r['flow_step'])) for r in rows}
    assert len(keys)==len(rows)
    assert all(int(r['flops'])==2*int(r['macs'])>0 and int(r['calls'])>0 for r in rows)
    assert sum(int(r['macs']) for r in rows)==report['groups']['all']['macs']
    if runtime is not None:
        # One compute event per invocation, not one per A/B/O or activation/output.
        wrapped={(r['module_id'],r['phase'],int(r['flow_step'])):r
                 for r in rows if r['wrapped']=='True'}
        assert {r['module_id'] for r in wrapped.values()}==set(expected_ids()[0])|set(expected_ids()[1])
        for r in runtime:
            event=wrapped[(r['module_id'],r['phase'],int(r['flow_step']))]
            assert int(r['calls'])==int(event['calls'])
    return report


def summarize():
    suite_reports={};success_rows=[];task_rows=[];sparse_rows=[];compute_rows=[];outlier_rows=[]
    all_runtime=[];all_compute={'baseline':[], 'quant':[]};all_outliers=[];reference_weights=None
    reference_audit=json.loads((OUT/'calibrate/scale_audit.json').read_text())
    for suite in SUITES:
        bp,qp=OUT/suite/'baseline',OUT/suite/'quant'
        baseline=results(bp,10,suite);quant=check_quant(qp,10,suite)
        # results() also verifies task names; no Goal-only grouping is allowed.
        suite_reports[suite]={'baseline':baseline,'quant':quant,
                             'delta_sr_pp':100*(quant['result']['sr']-baseline['sr'])}
        runtime=read(qp/'sparsity/module_sparsity.csv');all_runtime.extend(runtime)
        weights=read(qp/'sparsity/weight_sparsity_static.csv')
        weights=sorted(weights,key=lambda r:r['module_id'])
        if reference_weights is None:reference_weights=weights
        else:assert weights==reference_weights, 'Static weights must agree across suites'
        assert json.loads((qp/'scale_audit.json').read_text())==reference_audit
        for arm,path,metrics in [('baseline',bp,baseline),('quant',qp,quant['result'])]:
            success_rows.append(dict(suite=suite,arm=arm,episodes=metrics['episodes'],
                                     successes=metrics['successes'],success_rate=metrics['sr']))
            comp=check_compute(path,runtime if arm=='quant' else None)
            all_compute[arm].append(comp)
            for group,counts in comp['groups'].items():
                compute_rows.append(dict(suite=suite,arm=arm,component=group,
                                         generations=comp['generations'],episodes=metrics['episodes'],
                                         **counts,flops_per_episode=counts['flops']/metrics['episodes'],
                                         flops_share=counts['flops']/comp['groups']['all']['flops']))
        braw=json.loads((bp/'result.json').read_text())
        qraw=json.loads((qp/'result.json').read_text())
        bd={r['task_id']:r['metrics']['successes'] for r in braw['per_task']}
        qd={r['task_id']:r['metrics']['successes'] for r in qraw['per_task']}
        for tid in range(10):
            b,q=bd[tid],qd[tid]
            task_rows.append(dict(suite=suite,task_id=tid,episodes=10,baseline_successes=sum(b),
                                  quant_successes=sum(q),delta_pp=10*(sum(q)-sum(b)),
                                  success_to_failure=sum(x and not y for x,y in zip(b,q)),
                                  failure_to_success=sum(not x and y for x,y in zip(b,q)),
                                  identical_episode_outcomes=b==q))
        for group,data in quant['groups'].items():
            for kind,counts in data.items():
                sparse_rows.append(dict(suite=suite,component=group,kind=kind,**counts))
        side=read(qp/'sparsity/outlier_sidepath.csv');all_outliers.extend(side)
        append_outliers(outlier_rows,suite,side)
        bmeta=json.loads((bp/'execution.json').read_text());qmeta=json.loads((qp/'execution.json').read_text())
        assert bmeta.pop('mode')=='raw' and qmeta.pop('mode')=='quant_forward' and bmeta==qmeta
    pooled={}
    for arm in ('baseline','quant'):
        selected=[r for r in success_rows if r['arm']==arm]
        episodes=sum(r['episodes'] for r in selected);successes=sum(r['successes'] for r in selected)
        assert episodes==400
        pooled[arm]={'episodes':episodes,'successes':successes,'sr':successes/episodes}
        success_rows.append(dict(suite='all',arm=arm,episodes=episodes,successes=successes,success_rate=successes/episodes))
        generations=sum(r['generations'] for r in all_compute[arm])
        for group in all_compute[arm][0]['groups']:
            macs=sum(r['groups'][group]['macs'] for r in all_compute[arm])
            compute_rows.append(dict(suite='all',arm=arm,component=group,generations=generations,
                                     episodes=episodes,macs=macs,flops=2*macs,
                                     flops_per_generation=2*macs/generations,flops_per_episode=2*macs/episodes,
                                     flops_share=2*macs/sum(r['groups']['all']['flops'] for r in all_compute[arm])))
    linear,mm=expected_ids()
    for group,comps in {'vision':['vision'],'vlm':['vlm'],'vision_vlm':['vision','vlm'],'expert':['expert']}.items():
        runtime=[r for r in all_runtime if r['component'] in comps]
        weights=[r for r in reference_weights if r['component'] in comps]
        for kind,rows,native in [('runtime',runtime,True),('runtime_linear',[r for r in runtime if r['module_id'] in linear],True),
                                 ('runtime_matmul',[r for r in runtime if r['module_id'] in mm],True),('static_weight',weights,False)]:
            sparse_rows.append(dict(suite='all',component=group,kind=kind,**aggregate(rows,native)))
    append_outliers(outlier_rows,'all',all_outliers)
    pooled['delta_sr_pp']=100*(pooled['quant']['sr']-pooled['baseline']['sr'])
    write(OUT/'summary.json',{'status':'PASS','suites':suite_reports,'pooled':pooled,
                             'compute_metric':'dense_equivalent_linear_conv2d_attention_v1',
                             'runtime_bit_metric':'S|MMM sign+mantissa; exponent/hidden-1 excluded',
                             'static_weight_aggregation':'one copy; identical across suites',
                             'limitations':'Single seed, 10 episodes/task; FLOPs exclude non-MAC ops and quantization overhead.'})
    for name,rows in [('success_summary',success_rows),('task_success',task_rows),('sparsity_summary',sparse_rows),
                       ('compute_summary',compute_rows),('outlier_summary',outlier_rows)]:
        csv_write(OUT/f'{name}.csv',rows)
    print('PASS: four suites, 400 baseline + 400 quant episodes; sparsity/compute summaries written')


def append_outliers(output,suite,rows):
    groups={}
    for r in rows:
        key=(r['module_id'].split('.')[0],r['tensor_role'])
        group=groups.setdefault(key,[0,0])
        total=int(r['total_elements']);protected=int(r['protected_elements'])
        assert 0<=protected<=total
        group[0]+=protected;group[1]+=total
    for (component,role),(protected,total) in sorted(groups.items()):
        assert total>0
        output.append(dict(suite=suite,component=component,tensor_role=role,protected_elements=protected,
                           total_elements=total,fp_sidepath_ratio=protected/total))


if __name__=='__main__':
    summarize()
