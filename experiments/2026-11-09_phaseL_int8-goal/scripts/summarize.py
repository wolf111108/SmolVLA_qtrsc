"""Validate three complete runs; aggregate integer counters, never row percentages."""
import csv
import json

import yaml

from common import (ARMS, QUANT_ARMS, OUT, digest, expected_ids, require,
                    weight_bits, write)


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def results(path, episodes):
    data = json.loads((path / 'result.json').read_text())
    expected = {0} if episodes == 1 else set(range(10))
    tasks = {}
    for row in data['per_task']:
        tid, values = row['task_id'], row['metrics']['successes']
        require(row['task_group'] == 'libero_goal' and tid not in tasks and tid in expected,
                'Wrong/duplicate Goal task')
        require(len(values) == episodes and all(type(v) is bool for v in values), 'Incomplete episode successes')
        tasks[tid] = {'successes': sum(values), 'episodes': len(values), 'sr': sum(values) / len(values)}
    require(set(tasks) == expected, 'Missing Goal tasks')
    total, successes = sum(t['episodes'] for t in tasks.values()), sum(t['successes'] for t in tasks.values())
    require(data['overall']['n_episodes'] == total, 'Episode count mismatch')
    require(abs(data['overall']['pc_success'] - 100 * successes / total) < 1e-6, 'Success rate mismatch')
    return {'successes': successes, 'episodes': total, 'sr': successes / total, 'tasks': tasks}


def aggregate(rows, native):
    require(bool(rows), 'Empty sparsity group')
    suffix = '_native' if native else ''
    out = {}
    for num, den in (('zero_elements', 'total_elements'), ('sparse_bits', 'total_bits')):
        require(all(0 <= int(r[num + suffix]) <= int(r[den + suffix]) and int(r[den + suffix]) > 0
                    for r in rows), f'Invalid {num}/{den} counts')
        out[num] = sum(int(r[num + suffix]) for r in rows)
        out[den] = sum(int(r[den + suffix]) for r in rows)
    out['element_sparsity'] = out['zero_elements'] / out['total_elements']
    out['bit_sparsity'] = out['sparse_bits'] / out['total_bits']
    if native:
        out['protected_elements'] = sum(int(r['protected_elements']) for r in rows)
        out['total_elements_reported'] = sum(int(r['total_elements_reported']) for r in rows)
        out['fp_sidepath_ratio'] = out['protected_elements'] / out['total_elements_reported']
    return out


def expected_runtime_keys():
    linear, mm = expected_ids()
    expected = set()
    for ids, roles in ((linear, ('activation', 'output')), (mm, ('A', 'B', 'O'))):
        for mid, comp in ids.items():
            for step in (range(10) if comp == 'expert' else [-1]):
                for role in roles:
                    expected.add((mid, 'denoise' if comp == 'expert' else 'prefill', step, role))
    return expected


def partition_key(row):
    return tuple(row[k] for k in ('module_id', 'phase', 'flow_step', 'tensor_role', 'attention_kind'))


def check_quant(path, episodes, arm):
    require(arm in QUANT_ARMS, 'Expected quantized arm')
    linear, mm = expected_ids()
    sp = path / 'sparsity'
    manifest = read(sp / 'quantization_manifest.csv')
    require(len(manifest) == 288 and {r['module_id'] for r in manifest} == set(linear) | set(mm), 'Wrong manifest sites')
    for r in manifest:
        comp, mid = r['component'], r['module_id']
        require(comp == mid.split('.')[0] and r['a_or_A_kind'] == r['o_or_O_kind'] == 'int8'
                and r['method'] == 'outlier', 'Wrong manifest precision/method')
        require(r['op_type'] == ('linear' if mid in linear else 'matmul'), 'Wrong operator type')
        require(r['w_or_B_kind'] == (f'int{weight_bits(arm, comp)}' if mid in linear else 'int8'), 'Wrong W/B precision')
    rows = read(sp / 'module_sparsity.csv')
    actual = {(r['module_id'], r['phase'], int(r['flow_step']), r['tensor_role']) for r in rows}
    require(actual == expected_runtime_keys() and len(rows) == 3520, 'Incomplete/duplicate runtime coverage')
    sidepath = read(sp / 'outlier_sidepath.csv')
    partitions = {partition_key(r): r for r in sidepath}
    require(len(partitions) == len(sidepath), 'Duplicate outlier partition')
    for r in rows:
        require(r['component'] == r['module_id'].split('.')[0] and int(r['calls']) > 0, 'Wrong runtime component/calls')
        total, protected = int(r['total_elements_reported']), int(r['protected_elements'])
        require(0 < protected < total and int(r['total_elements_native']) == total - protected, 'Invalid native partition')
        partition = partitions.get(partition_key(r))
        require(partition is not None, 'Missing outlier partition')
        require(int(partition['calls']) == int(r['calls']) and int(partition['total_elements']) == total
                and int(partition['protected_elements']) == protected, 'Outlier partition counts mismatch')
        require(int(r['total_bits_native']) == 8 * int(r['total_elements_native']), 'Runtime must use 8-bit codes')
        require(int(r['sparse_bits_native']) == int(r['sparse_bits_reported']) - 8 * protected, 'Wrong native sparse numerator')
    # Validate output counters too, but exclude output/O from all primary activation metrics.
    aggregate(rows, True)
    weights = read(sp / 'weight_sparsity_static.csv')
    require(len(weights) == 224 and {r['module_id'] for r in weights} == set(linear), 'Wrong static weight sites')
    for r in weights:
        comp = linear[r['module_id']]
        bits = weight_bits(arm, comp)
        require(r['component'] == comp and r['weight_spec'] == f'int{bits}', 'Wrong weight component/spec')
        require(r['stat_semantics'] == 'full_weight_quantized_no_dynamic_outlier_mask', 'Wrong static weight semantics')
        require(int(r['total_bits']) == bits * int(r['total_elements']), 'Wrong static weight bit denominator')
    report = {'status': 'PASS', 'result': results(path, episodes), 'sites': 288,
              'runtime_rows': len(rows), 'weight_rows': len(weights),
              'bit_metric': 'integer sign-aware two-complement, including sign bit',
              'activation_semantics': 'native Linear input + QK/PV A and B; excludes output/O and FP sidepath',
              'weight_semantics': 'full_weight_quantized_no_dynamic_outlier_mask', 'groups': {}}
    for comp in ('vlm', 'expert'):
        rr = [r for r in rows if r['component'] == comp and r['tensor_role'] in ('activation', 'A', 'B')]
        group = {'activation': aggregate(rr, True),
                 'static_weight': aggregate([r for r in weights if r['component'] == comp], False)}
        for label, selected in (
            ('linear_input', [r for r in rr if r['module_id'] in linear]),
            ('qk_inputs', [r for r in rr if r['module_id'].endswith('.qk')]),
            ('pv_inputs', [r for r in rr if r['module_id'].endswith('.pv')]),
        ):
            group[label] = aggregate(selected, True)
        report['groups'][comp] = group
    return report


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    from run_arm import check_prepared
    check_prepared()
    reports = {}
    for arm in ARMS:
        path = OUT / 'eval' / arm
        require(json.loads((path / 'completed.json').read_text()) ==
                {'status': 'PASS', 'stage': 'eval', 'arm': arm}, f'Incomplete {arm}')
        execution = json.loads((path / 'execution.json').read_text())
        require(execution['prepared_sha256'] == digest(OUT / 'prepared.json') and execution['arm'] == arm,
                'Mismatched run provenance')
        require(execution['mode'] == ('raw' if arm == 'baseline' else 'quant_forward'), 'Wrong evaluated mode')
        cfg = yaml.safe_load((OUT / f'{arm}_resolved.yaml').read_text())
        cfg['output_dir'] = str(path)
        require(yaml.safe_load((path / 'config.yaml').read_text()) == cfg, 'Evaluation config mismatch')
        if arm == 'baseline':
            reports[arm] = {'result': results(path, 10), 'groups': None}
        else:
            audit = json.loads((path / 'scale_audit.json').read_text())
            require(audit == json.loads((OUT / 'calibrate' / arm / 'scale_audit.json').read_text()), 'Mismatched calibration')
            reports[arm] = check_quant(path, 10, arm)
    base = reports['baseline']['result']
    comparison, tasks, sparsity = [], [], []
    for arm, report in reports.items():
        result = report['result']
        report['delta_sr_pp'] = 100 * (result['sr'] - base['sr'])
        comparison.append({'arm': arm, 'successes': result['successes'], 'episodes': result['episodes'],
                           'success_rate_pct': 100 * result['sr'], 'delta_vs_baseline_pp': report['delta_sr_pp']})
        for tid, task in result['tasks'].items():
            tasks.append({'arm': arm, 'task_id': tid, 'successes': task['successes'], 'episodes': task['episodes'],
                          'success_rate_pct': 100 * task['sr'], 'delta_vs_baseline_pp': 100 * (task['sr'] - base['tasks'][tid]['sr'])})
        for comp in ('vlm', 'expert'):
            for metric in ('activation', 'linear_input', 'qk_inputs', 'pv_inputs', 'static_weight'):
                values = report['groups'][comp][metric] if report['groups'] else {}
                sparsity.append({'arm': arm, 'component': comp, 'metric': metric,
                    'bitwidth': weight_bits(arm, comp) if metric == 'static_weight' and values else (8 if values else 'N/A'),
                    **{key: values.get(key, 'N/A') for key in ('sparse_bits', 'total_bits', 'bit_sparsity',
                        'zero_elements', 'total_elements', 'element_sparsity', 'protected_elements',
                        'total_elements_reported', 'fp_sidepath_ratio')}})
    write(OUT / 'summary.json', reports)
    csv_write(OUT / 'comparison.csv', comparison)
    csv_write(OUT / 'task_success.csv', tasks)
    csv_write(OUT / 'sparsity_by_component.csv', sparsity)
    lines = ['# Goal INT8 comparison', '', '| Arm | Successes | SR | Δ baseline | VLM A bits | VLM W bits | Expert A bits | Expert W bits |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, report in reports.items():
        r = report['result']
        bits = [f"{100 * report['groups'][c][m]['bit_sparsity']:.4f}%" if report['groups'] else 'N/A'
                for c in ('vlm', 'expert') for m in ('activation', 'static_weight')]
        lines.append(f"| {arm} | {r['successes']}/{r['episodes']} | {100*r['sr']:.1f}% | {report['delta_sr_pp']:+.1f} pp | " + ' | '.join(bits) + ' |')
    lines += ['', 'A: native Linear inputs + QK/PV A/B, excluding output/O and protected FP positions.',
              'W: full static Linear weight codes, once per layer; dynamic outlier masks are not removed.',
              'Bit sparsity uses integer sign-aware counts including the sign bit; see sparsity_by_component.csv for numerators, denominators and FP sidepath ratios.',
              'Baseline integer-code sparsity is N/A. Vision/connector/action heads remain raw.']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(f'PASS: comparison and sparsity reports written to {OUT}')


if __name__ == '__main__':
    summarize()
