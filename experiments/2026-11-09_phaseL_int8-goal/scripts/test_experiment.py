"""CPU contract tests with synthetic counters, never experimental measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import common
import run_arm
import summarize as s


def fixture(path, arm, episodes=10):
    tasks = [0] if episodes == 1 else range(10)
    per_task = [{'task_group': 'libero_goal', 'task_id': i,
                 'metrics': {'successes': [True] + [False] * (episodes - 1)}} for i in tasks]
    common.write(path / 'result.json', {'per_task': per_task,
                 'overall': {'n_episodes': len(per_task) * episodes, 'pc_success': 100 / episodes}})
    if arm == 'baseline':
        return
    linear, mm = common.expected_ids()
    manifest = []
    for mid, comp in (linear | mm).items():
        manifest.append({'module_id': mid, 'component': comp, 'method': 'outlier',
                         'op_type': 'linear' if mid in linear else 'matmul',
                         'a_or_A_kind': 'int8', 'o_or_O_kind': 'int8',
                         'w_or_B_kind': f'int{common.weight_bits(arm, comp)}' if mid in linear else 'int8'})
    runtime = []
    for mid, phase, step, role in sorted(s.expected_runtime_keys()):
        # Output counts deliberately dominate: they must not affect activation metrics.
        n = {'activation': 10, 'A': 20, 'B': 40, 'output': 1000, 'O': 1000}[role]
        protected = 2
        sparse = 4 * n if role in ('activation', 'A', 'B') else 8 * n
        runtime.append({'module_id': mid, 'component': mid.split('.')[0], 'phase': phase,
                        'flow_step': step, 'tensor_role': role, 'attention_kind': '', 'calls': 1,
                        'total_elements_reported': n + protected, 'protected_elements': protected,
                        'total_elements_native': n, 'zero_elements_native': n // 2,
                        'total_bits_native': 8 * n, 'sparse_bits_native': sparse,
                        'sparse_bits_reported': sparse + 8 * protected})
    weights = [{'module_id': mid, 'component': comp, 'weight_spec': f'int{common.weight_bits(arm, comp)}',
                'stat_semantics': 'full_weight_quantized_no_dynamic_outlier_mask',
                'total_elements': 10, 'zero_elements': 2,
                'total_bits': 10 * common.weight_bits(arm, comp), 'sparse_bits': 5 * common.weight_bits(arm, comp)}
               for mid, comp in linear.items()]
    sp = path / 'sparsity'
    sp.mkdir()
    for name, rows in (('quantization_manifest', manifest), ('module_sparsity', runtime),
                       ('weight_sparsity_static', weights)):
        s.csv_write(sp / f'{name}.csv', rows)
    s.csv_write(sp / 'outlier_sidepath.csv', [
        {**{k: r[k] for k in ('module_id', 'phase', 'flow_step', 'tensor_role', 'attention_kind',
                              'calls', 'protected_elements')}, 'total_elements': r['total_elements_reported']}
        for r in runtime])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_matched_configs(self):
        configs = common.load_configs()
        self.assertEqual(len(common.expected_ids()[0]), 224)
        self.assertEqual(configs['a8w4_expert']['quantization']['linear']['overrides'][1]['config']['w_bit'], 4)

    def test_count_weighted_aggregation(self):
        rows = [{'total_elements': 10, 'zero_elements': 1, 'total_bits': 80, 'sparse_bits': 8},
                {'total_elements': 100, 'zero_elements': 50, 'total_bits': 800, 'sparse_bits': 400}]
        self.assertAlmostEqual(s.aggregate(rows, False)['bit_sparsity'], 408 / 880)

    def test_both_precisions_and_activation_roles(self):
        for arm in common.QUANT_ARMS:
            path = self.path / arm
            fixture(path, arm)
            report = s.check_quant(path, 10, arm)
            for comp, steps in (('vlm', 1), ('expert', 10)):
                group = report['groups'][comp]
                self.assertEqual(group['activation']['total_elements'], steps * (112 * 10 + 32 * 60))
                self.assertEqual(group['activation']['bit_sparsity'], 0.5)
                self.assertEqual(group['qk_inputs']['total_elements'], steps * 16 * 60)
                self.assertEqual(group['static_weight']['total_bits'], 112 * 10 * common.weight_bits(arm, comp))

    def test_missing_runtime_site_rejected(self):
        fixture(self.path, 'a8w8')
        file = self.path / 'sparsity/module_sparsity.csv'
        s.csv_write(file, s.read(file)[:-1])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            s.check_quant(self.path, 10, 'a8w8')

    def test_wrong_matmul_B_precision_rejected(self):
        fixture(self.path, 'a8w4_expert')
        file = self.path / 'sparsity/quantization_manifest.csv'
        rows = s.read(file)
        next(r for r in rows if r['module_id'] == 'expert.layer.0.qk')['w_or_B_kind'] = 'int4'
        s.csv_write(file, rows)
        with self.assertRaisesRegex(ValueError, 'W/B precision'):
            s.check_quant(self.path, 10, 'a8w4_expert')

    def test_invalid_native_partition_rejected(self):
        fixture(self.path, 'a8w8')
        file = self.path / 'sparsity/module_sparsity.csv'
        rows = s.read(file)
        rows[0]['sparse_bits_native'] = rows[0]['sparse_bits_reported']
        s.csv_write(file, rows)
        with self.assertRaisesRegex(ValueError, 'native sparse numerator'):
            s.check_quant(self.path, 10, 'a8w8')

    def test_incomplete_goal_rejected(self):
        fixture(self.path, 'baseline')
        data = json.loads((self.path / 'result.json').read_text())
        data['per_task'].pop()
        common.write(self.path / 'result.json', data)
        with self.assertRaisesRegex(ValueError, 'Missing Goal tasks'):
            s.results(self.path, 10)

    def test_missing_outlier_partition_rejected(self):
        fixture(self.path, 'a8w8')
        file = self.path / 'sparsity/outlier_sidepath.csv'
        s.csv_write(file, s.read(file)[:-1])
        with self.assertRaisesRegex(ValueError, 'Missing outlier partition'):
            s.check_quant(self.path, 10, 'a8w8')

    def test_wrong_overall_success_rejected(self):
        fixture(self.path, 'baseline')
        data = json.loads((self.path / 'result.json').read_text())
        data['overall']['pc_success'] = 99
        common.write(self.path / 'result.json', data)
        with self.assertRaisesRegex(ValueError, 'Success rate mismatch'):
            s.results(self.path, 10)

    def test_complete_report_and_baseline_na(self):
        configs = common.load_configs()
        common.write(self.path / 'prepared.json', {'fixture': True})
        for arm in common.ARMS:
            path = self.path / 'eval' / arm
            fixture(path, arm)
            cfg = copy.deepcopy(configs[arm])
            (self.path / f'{arm}_resolved.yaml').write_text(yaml.safe_dump(cfg))
            cfg['output_dir'] = str(path)
            (path / 'config.yaml').write_text(yaml.safe_dump(cfg))
            common.write(path / 'completed.json', {'status': 'PASS', 'stage': 'eval', 'arm': arm})
            common.write(path / 'execution.json', {'arm': arm, 'mode': 'raw' if arm == 'baseline' else 'quant_forward',
                         'prepared_sha256': common.digest(self.path / 'prepared.json')})
            if arm in common.QUANT_ARMS:
                common.write(path / 'scale_audit.json', [])
                common.write(self.path / 'calibrate' / arm / 'scale_audit.json', [])
        with patch.object(s, 'OUT', self.path), patch.object(run_arm, 'check_prepared'):
            s.summarize()
        comparison = s.read(self.path / 'comparison.csv')
        self.assertEqual(len(comparison), 3)
        self.assertTrue(all(float(r['delta_vs_baseline_pp']) == 0 for r in comparison))
        baseline = [r for r in s.read(self.path / 'sparsity_by_component.csv') if r['arm'] == 'baseline']
        self.assertEqual(len(baseline), 10)
        self.assertTrue(all(r['bit_sparsity'] == 'N/A' for r in baseline))
        self.assertTrue((self.path / 'summary.md').exists())


if __name__ == '__main__':
    unittest.main()
