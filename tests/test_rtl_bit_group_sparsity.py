"""Independent scalar oracle plus production-forward tests for Phase O."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vla_tcs2.stats.bit_group import count_bit_groups, observe_modules
from vla_tcs2.stats.rtl_bit_group import (
    COUNTERS, PA_PW_PAIRS, PA_VALUES, PW_VALUES, RTLBitGroupCollector,
    count_rtl_bit_groups, rates, rtl_coefficients,
)
from vla_tcs2.stats.bg_experiment import check_coverage, load_config
from vla_tcs2.stats.goal_protocol import expected_ids


def oracle(codes, mask, p, bits, source_bits, level):
    """Independent arithmetic implementation: magnitude minus one, no XOR."""
    matrices = codes.reshape(-1, *codes.shape[-2:])
    masks = np.broadcast_to(mask, codes.shape).reshape(matrices.shape)
    out = dict.fromkeys(COUNTERS, 0)
    out['calls'] = 1
    for matrix, protected in zip(matrices, masks):
        m, k = matrix.shape
        for start in range(0, m, p):
            lanes = list(range(start, min(start + p, m)))
            for channel in range(k):
                native = [i for i in lanes if not protected[i, channel]]
                coeffs = []
                for i in native:
                    value = int(matrix[i, channel])
                    unsigned = -value - 1 if value < 0 else value
                    coefficient = (unsigned // (2 ** (bits * level))) % (2 ** bits)
                    if value < 0 and level == 0:
                        coefficient += 1
                    coeffs.append(coefficient)
                zero, full, tail = not any(coeffs), len(native) == p, len(lanes) < p
                out['total_packs'] += 1
                out['normal_zero_packs'] += zero
                out['normal_issued_packs'] += not zero
                out['full_native_packs'] += full
                out['full_native_zero_packs'] += full and zero
                out['native_groups'] += len(native)
                out['native_zero_groups'] += sum(x == 0 for x in coeffs)
                out['native_skipped_groups'] += len(native) if zero else 0
                out['protected_groups'] += len(lanes) - len(native)
                out['tail_packs'] += tail
                out['tail_zero_packs'] += tail and zero
                out['negative_groups'] += sum(matrix[i, channel] < 0 for i in native)
                out['sign_extended_groups'] += len(native) if bits * level >= source_bits else 0
    return out


class CounterTests(unittest.TestCase):
    def test_exhaustive_signed_reconstruction(self):
        for source_bits in (4, 8):
            q = np.arange(-(2 ** (source_bits - 1)), 2 ** (source_bits - 1)).reshape(-1, 1)
            for bits in (2, 4):
                d = rtl_coefficients(q, bits, source_bits)
                rebuilt = (d * (2 ** (bits * np.arange(8 // bits)))).sum(-1) * np.where(q < 0, -1, 1)
                np.testing.assert_array_equal(rebuilt, q)
                self.assertTrue((d[..., 0][q < 0] > 0).all())
                self.assertLessEqual(int(d[..., 0].max()), 2 ** bits)
                self.assertLess(int(d[..., 1:].max()), 2 ** bits)
                if source_bits == 4:
                    self.assertTrue((d[..., 4 // bits:] == 0).all())

    def test_scalar_oracle_masks_heads_and_tails(self):
        rng = np.random.default_rng(31)
        for source_bits, ps, bits in ((8, PA_VALUES, 2), (8, PW_VALUES, 2),
                                     (8, PW_VALUES, 4), (4, PW_VALUES, 2), (4, PW_VALUES, 4)):
            for m in (1, 3, 50, 65, 1025):
                q = rng.integers(-(2 ** (source_bits - 1)), 2 ** (source_bits - 1),
                                 (2, 2, m, 3), dtype=np.int16)
                mask = rng.random(q.shape) < .2
                q[mask] = 0
                for row in count_rtl_bit_groups(q, mask, ps, group_bits=bits, source_bits=source_bits):
                    self.assertEqual({k: row[k] for k in COUNTERS},
                                     oracle(q, mask, row['p'], bits, source_bits, row['level']))

    def test_lowest_group_negative_compensation(self):
        q = np.array([[-4], [4]])
        self.assertEqual([r['normal_issued_packs'] for r in count_rtl_bit_groups(q, p_values=(2,))], [1, 1, 0, 0])
        self.assertEqual([r['normal_zero_packs'] for r in count_bit_groups(q, p_values=(2,))], [1, 0, 1, 1])
        np.testing.assert_array_equal(rtl_coefficients([[-128]])[0, 0], [4, 3, 3, 1])
        np.testing.assert_array_equal(rtl_coefficients([[-16]], 4)[0, 0], [16, 0])
        self.assertEqual(count_rtl_bit_groups([[0], [4], [8], [12]], p_values=(4,))[0]['normal_issued_packs'], 0)
        self.assertEqual(count_rtl_bit_groups([[-4], [4], [8], [12]], p_values=(4,))[0]['normal_issued_packs'], 1)

    def test_int4_padded_denominator_and_sparse_upper_groups(self):
        q = np.arange(-8, 8, dtype=np.int8).reshape(16, 1)
        for bits in (2, 4):
            rows = count_rtl_bit_groups(q, p_values=(32,), group_bits=bits, source_bits=4)
            self.assertEqual(sum(r['native_groups'] for r in rows), q.size * (8 // bits))
            for row in rows:
                if row['level'] >= 4 // bits:
                    self.assertEqual(row['native_skipped_groups'], q.size)
                    self.assertEqual(row['normal_issued_packs'], 0)
                    self.assertEqual(rates(row)['native_skip_rate'], 1)
            self.assertEqual(sum(r['native_skipped_groups'] for r in rows) / sum(r['native_groups'] for r in rows), .5)

    def test_chunk_equivalence_and_short_packs(self):
        q = np.random.default_rng(8).integers(-128, 128, (2, 3, 67, 11))
        self.assertEqual(count_rtl_bit_groups(q, chunk_elements=7), count_rtl_bit_groups(q))
        row = rates(count_rtl_bit_groups(np.zeros((50, 3)), p_values=(64,))[0])
        self.assertEqual(row['native_groups'], 150)
        self.assertEqual(row['native_skip_rate'], 1)
        self.assertIsNone(row['full_native_pack_zero_rate'])
        self.assertEqual(row['tail_pack_fraction'], 1)

    def test_fp_protected_zeros_excluded(self):
        row = rates(count_rtl_bit_groups([[0, 1], [0, 1]], [[True, False]], p_values=(2,))[0])
        self.assertEqual(row['normal_pack_zero_rate'], .5)
        self.assertEqual(row['native_skip_rate'], 0)
        self.assertEqual(row['native_groups'], 2)
        row = rates(count_rtl_bit_groups(np.zeros((2, 2)), np.ones((2, 2), bool), (2,))[0])
        self.assertIsNone(row['native_skip_rate'])
        self.assertIsNone(row['full_native_pack_zero_rate'])

    def test_axes_heads_not_flattened(self):
        q = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        self.assertEqual(count_rtl_bit_groups(q, p_values=(2,))[0]['normal_zero_packs'], 2)
        self.assertEqual(count_rtl_bit_groups(q, p_values=(2,), lane_axis=-1)[0]['normal_zero_packs'], 0)
        q = np.array([[[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]]])
        self.assertEqual(count_rtl_bit_groups(q, lane_axis=-1), count_rtl_bit_groups(q.swapaxes(-1, -2)))
        row = count_rtl_bit_groups(np.zeros((2, 3, 1)), p_values=(2,))[0]
        self.assertEqual(row['full_native_packs'], 2)
        self.assertEqual(row['tail_packs'], 2)

    def test_nested_pack_monotonicity(self):
        q = np.random.default_rng(9).integers(-12, 13, (256, 7))
        for g in range(4):
            skip = [rates(r)['native_skip_rate'] for r in count_rtl_bit_groups(q) if r['level'] == g]
            self.assertEqual(skip, sorted(skip, reverse=True))

    def test_invalid_inputs_rejected(self):
        for q in ([[128]], [[-129]], [[.5]], [[float('nan')]], [], [[True]]):
            with self.assertRaises(ValueError):
                count_rtl_bit_groups(q)
        for kwargs in ({'p_values': (0,)}, {'p_values': (2, 2)}, {'p_values': (True,)},
                       {'source_bits': 16}, {'group_bits': 3}, {'lane_axis': 0}, {'chunk_elements': 0}):
            with self.assertRaises(ValueError):
                count_rtl_bit_groups([[0]], **kwargs)
        with self.assertRaises(ValueError):
            count_rtl_bit_groups([[8]], source_bits=4)
        with self.assertRaises(ValueError):
            count_rtl_bit_groups([[1]], [[True]])


class ProtocolTests(unittest.TestCase):
    def test_pairs_and_unchanged_quantization(self):
        self.assertEqual(PA_VALUES, (2, 4, 8, 16, 32, 64))
        self.assertEqual(PW_VALUES, (1024, 512, 256, 128, 64, 32))
        self.assertTrue(all(a * w == 2048 for a, w in PA_PW_PAIRS))
        for name, protocol in (('2026-11-10_phaseN_bg-p-sweep', 'magnitude'),
                               ('2026-11-12_phaseO_rtl-bg-aw-sweep', 'rtl')):
            self.assertEqual(load_config(ROOT / 'experiments' / name, protocol)['evaluation']['n_episodes'], 1)

    def test_wrong_pairs_fail(self):
        exp = ROOT / 'experiments/2026-11-12_phaseO_rtl-bg-aw-sweep'
        cfg = yaml.safe_load((exp / 'configs/a8w4_expert_task0.yaml').read_text())
        cfg['bit_group_probe']['pa_pw_pairs'][0][1] = 512
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'configs').mkdir()
            (out / 'configs/a8w4_expert_task0.yaml').write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, 'BG protocol'):
                load_config(out, 'rtl')

    def test_static_weights_deduplicated(self):
        collector = RTLBitGroupCollector()
        mid = 'expert.layers.0.mlp.up_proj'
        collector.collect_weight_codes(mid, [[-8], [-1], [0], [7]], 4)
        with self.assertRaisesRegex(ValueError, 'already counted'):
            collector.collect_weight_codes(mid, [[0]], 4)
        with self.assertRaises(ValueError):
            collector.collect_weight_codes('vlm.layers.0.mlp.up_proj', [[0]], 4)
        self.assertEqual(len(collector.rows(weights=True)), 36)
        self.assertTrue(all(r['calls'] == 1 for r in collector.rows(weights=True)))
        with self.assertRaisesRegex(ValueError, 'Missing paired report'):
            collector.paired_rows()

    def test_coverage_requires_all_steps_and_weights(self):
        c = RTLBitGroupCollector()
        linear, mm = expected_ids()
        for mid in linear | mm:
            comp = mid.split('.')[0]
            for step in ([-1] if comp == 'vlm' else range(10)):
                for role in (['activation'] if mid in linear else ['A', 'B']):
                    key = (comp, mid, 'prefill' if comp == 'vlm' else 'denoise', step, 'self', role)
                    c.observed_calls[key] = 1
                    c.shapes[key].add((1, 50, 8))
        c.weight_manifest = {mid: {} for mid in linear}
        r = check_coverage(c)
        self.assertEqual(r['static_weight_sites'], 224)
        self.assertEqual(r['runtime_role_keys'], 1936)
        del c.weight_manifest[next(iter(linear))]
        with self.assertRaisesRegex(ValueError, 'Static weight coverage'):
            check_coverage(c)
        c.weight_manifest = {mid: {} for mid in linear}
        del c.observed_calls[next(k for k in c.observed_calls if k[3] == 9)]
        with self.assertRaisesRegex(ValueError, 'Activation coverage'):
            check_coverage(c)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class ForwardTests(unittest.TestCase):
    def test_real_forward_static_codes_and_reports(self):
        import torch
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant_matmul import QuantizedMatMul
        from vla_tcs2.quant.quant_spec import quant_awo
        from vla_tcs2.quant.scale_methods import get_outlier_mask_channel
        from vla_tcs2.runtime_context import CURRENT_PHASE, CURRENT_FLOW_STEP
        torch.manual_seed(2)
        tokens = CURRENT_PHASE.set('prefill'), CURRENT_FLOW_STEP.set(-1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                modules, c = [], RTLBitGroupCollector()
                for comp, wb in (('vlm', 8), ('expert', 4)):
                    layer = QuantizedLinear(8, 7, a_bit=8, w_bit=wb, o_bit=8,
                                            outlier_ratio=.25, scale_root_str=directory)
                    layer.set_layer_info(comp + '_up', 0, f'{comp}.layers.0.mlp.up_proj')
                    layer.a_interval = layer.w_interval = layer.o_interval = .125
                    layer.method, layer.mode = 'outlier', 'quant_forward'
                    layer.save_scales()
                    modules.append(layer)
                    CURRENT_PHASE.set('prefill' if comp == 'vlm' else 'denoise')
                    CURRENT_FLOW_STEP.set(-1 if comp == 'vlm' else 0)
                    x = torch.randn(1, 5, 8)
                    reference = layer(x)
                    captured = []
                    def callback(mod, role, code, mask):
                        captured.append((code.clone(), mask.clone()))
                        c(mod, role, code, mask)
                    with observe_modules([layer], callback):
                        actual = layer(x)
                    self.assertTrue(torch.equal(reference, actual))
                    self.assertFalse(hasattr(layer, '_bit_group_collector'))
                    mask = get_outlier_mask_channel(x, .25).view(1, 1, -1)
                    expected = quant_awo(x * (~mask), .125, layer.a_spec, out_dtype=torch.float32)
                    self.assertTrue(torch.equal(captured[0][0], expected))
                    self.assertTrue(torch.equal(captured[0][1], mask))
                c.collect_weights(modules)
                for layer in modules:
                    expected = torch.round(layer.weight.detach().float() / .125).clamp(
                        -(2 ** (layer.w_spec.bits - 1)), 2 ** (layer.w_spec.bits - 1) - 1)
                    direct = RTLBitGroupCollector()
                    direct.collect_weight_codes(layer.module_id, expected.numpy(), layer.w_spec.bits)
                    self.assertEqual([r for r in c.rows(weights=True) if r['module_id'] == layer.module_id],
                                     direct.rows(weights=True))
                before = c.rows(weights=True)
                CURRENT_PHASE.set('denoise')
                for step in range(10):
                    CURRENT_FLOW_STEP.set(step)
                    with observe_modules([modules[1]], c):
                        modules[1](x)
                self.assertEqual(c.rows(weights=True), before)
                with self.assertRaisesRegex(ValueError, 'already counted'):
                    c.collect_weights(modules)
                CURRENT_PHASE.set('prefill')
                CURRENT_FLOW_STEP.set(-1)
                for kind in ('qk', 'pv'):
                    mm = QuantizedMatMul(A_bit=8, B_bit=8, O_bit=8, outlier_ratio=.25,
                                         scale_root_str=directory)
                    mm.set_layer_info('vlm_' + kind, 0, f'vlm.layer.0.{kind}')
                    mm.A_interval = mm.B_interval = mm.O_interval = .125
                    mm.method, mm.mode = 'outlier', 'quant_forward'
                    mm.save_scales()
                    a, b = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 8, 6)
                    reference = mm(a, b)
                    with observe_modules([mm], c):
                        actual = mm(a, b)
                    self.assertTrue(torch.equal(reference, actual))
                report = Path(directory) / 'report'
                c.export(report)
                with (report / 'paired_summary.csv').open() as f:
                    pairs = list(csv.DictReader(f))
                self.assertEqual(len(pairs), 6)
                self.assertTrue(all(int(r['pa']) * int(r['pw']) == 2048 for r in pairs))
                self.assertTrue(all(float(r['expert_weight_bg2_native_skip_rate']) >= .5 for r in pairs))
                self.assertTrue(all(float(r['expert_weight_bg4_native_skip_rate']) >= .5 for r in pairs))
                self.assertEqual(len(json.loads((report / 'weight_manifest.json').read_text())), 2)
                self.assertEqual(pairs[0]['vlm_weight_bg2_full_native_pack_zero_rate'], '')
                self.assertIn('RTL-aligned', (report / 'summary.md').read_text())
        finally:
            CURRENT_PHASE.reset(tokens[0])
            CURRENT_FLOW_STEP.reset(tokens[1])


if __name__ == '__main__':
    unittest.main()
