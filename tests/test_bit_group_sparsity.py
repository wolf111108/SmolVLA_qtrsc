"""Independent scalar oracle plus real outlier-forward observer tests."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vla_tcs2.stats.bit_group import (
    BitGroupCollector, COUNTERS, P_VALUES, count_bit_groups, observe_modules, rates,
)


def scalar_oracle(codes, mask, p, level):
    batches = codes.reshape(-1, *codes.shape[-2:])
    masks = np.broadcast_to(mask, codes.shape).reshape(batches.shape)
    result = dict.fromkeys(COUNTERS, 0)
    result['calls'] = 1
    for matrix, protected in zip(batches, masks):
        m, k = matrix.shape
        for start in range(0, m, p):
            for channel in range(k):
                lanes = list(range(start, min(start+p, m)))
                native = [i for i in lanes if not protected[i, channel]]
                active = any((abs(int(matrix[i, channel])) // 4**level) % 4 != 0 for i in native)
                full_native = len(native) == p
                result['total_packs'] += 1
                result['normal_zero_packs'] += not active
                result['full_native_packs'] += full_native
                result['full_native_zero_packs'] += full_native and not active
                result['native_groups'] += len(native)
                result['native_skipped_groups'] += len(native) if not active else 0
                result['protected_groups'] += len(lanes) - len(native)
                result['tail_packs'] += len(lanes) < p
                result['minus128_elements'] += sum(matrix[i, channel] == -128 for i in lanes)
    return result


class CounterTests(unittest.TestCase):
    def test_scalar_oracle_heads_batches_tails_masks(self):
        rng = np.random.default_rng(31)
        for m in (1, 3, 50, 128, 131):
            codes = rng.integers(-128, 128, (2, 3, m, 5), dtype=np.int16)
            mask = rng.random(codes.shape) < .2
            codes[mask] = 0
            for row in count_bit_groups(codes, mask):
                expected = scalar_oracle(codes, mask, row['p'], row['level'])
                self.assertEqual({k: row[k] for k in COUNTERS}, expected)

    def test_sign_magnitude_and_minimum_integer(self):
        # +/-1 have identical nonzero low BG; -128 must not wrap or disappear.
        codes = np.array([[-1, -128], [1, -128]], dtype=np.int8)
        rows = count_bit_groups(codes, p_values=(2,))
        self.assertEqual([r['normal_zero_packs'] for r in rows], [1, 2, 2, 1])
        self.assertEqual(rows[0]['minus128_elements'], 2)

    def test_p_groups_tokens_not_channels(self):
        codes = np.array([[0, 1], [0, 1]])
        row = count_bit_groups(codes, p_values=(2,))[0]
        self.assertEqual(row['normal_zero_packs'], 1)

    def test_qk_b_transposed_and_no_cross_head_packing(self):
        code = np.array([[[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]]])
        self.assertEqual(count_bit_groups(code, token_axis=-1),
                         count_bit_groups(code.swapaxes(-1, -2)))
        rows = count_bit_groups(np.zeros((2, 3, 1)))
        self.assertEqual(rows[0]['full_native_packs'], 2)
        self.assertEqual(rows[0]['tail_packs'], 2)

    def test_padding_does_not_inflate_denominator(self):
        code = np.zeros((50, 1))
        code[0, 0] = 1
        row = rates(count_bit_groups(code, p_values=(128,))[0])
        self.assertEqual(row['native_groups'], 50)
        self.assertEqual(row['native_skip_rate'], 0)
        self.assertIsNone(row['full_native_pack_zero_rate'])
        self.assertEqual(row['tail_pack_fraction'], 1)

    def test_protected_zero_is_not_natural_sparsity(self):
        code = np.array([[0, 1], [0, 1]])
        row = rates(count_bit_groups(code, [[True, False]], p_values=(2,))[0])
        self.assertEqual(row['normal_pack_zero_rate'], .5)
        self.assertEqual(row['native_skip_rate'], 0)
        self.assertEqual(row['full_native_pack_zero_rate'], 0)
        all_fp = rates(count_bit_groups(np.zeros((2, 2)), np.ones((2, 2), bool), (2,))[0])
        self.assertIsNone(all_fp['native_skip_rate'])
        self.assertIsNone(all_fp['full_native_pack_zero_rate'])

    def test_monotonic_on_nested_groups(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(-10, 11, (256, 5))
        result = count_bit_groups(codes)
        for level in range(4):
            values = [rates(r)['native_skip_rate'] for r in result if r['level'] == level]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_invalid_codes_and_masks_rejected(self):
        for code in ([[128]], [[.5]], [[float('nan')]], []):
            with self.assertRaises(ValueError):
                count_bit_groups(code)
        with self.assertRaises(ValueError):
            count_bit_groups([[1]], [[True]])
        with self.assertRaises(ValueError):
            count_bit_groups([[0]], p_values=(3,))

    def test_binding_survives_worker_and_restores_on_error(self):
        from vla_tcs2.stats.bit_group import emit_activation_code
        module = SimpleNamespace()
        got = []
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with observe_modules([module], lambda *a: got.append(a)):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(emit_activation_code, module, 'A', 1, 0).result()
                raise RuntimeError('stop')
        self.assertEqual(len(got), 1)
        self.assertFalse(hasattr(module, '_bit_group_collector'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class ForwardTests(unittest.TestCase):
    def test_linear_and_matmul_outputs_and_exact_mask_capture(self):
        import torch
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant_matmul import QuantizedMatMul
        from vla_tcs2.quant.quant_methods import quant_forward_with_outlier, matmul_quant_forward_with_outlier
        from vla_tcs2.quant.quant_spec import quant_awo
        from vla_tcs2.quant.scale_methods import get_outlier_mask_channel, get_outlier_mask_1d, _matmul_B_channel_mask
        from vla_tcs2.runtime_context import CURRENT_PHASE, CURRENT_FLOW_STEP

        torch.manual_seed(0)
        collector = BitGroupCollector()
        tokens = CURRENT_PHASE.set('prefill'), CURRENT_FLOW_STEP.set(-1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                linear = QuantizedLinear(8, 6, a_bit=8, w_bit=8, o_bit=8,
                                         outlier_ratio=.25, scale_root_str=directory)
                linear.module_id = 'vlm.layers.0.mlp.up_proj'
                linear.a_interval = linear.w_interval = linear.o_interval = .125
                x = torch.randn(1, 5, 8)
                reference = quant_forward_with_outlier(linear, x)
                captured = []
                def callback(layer, role, code, mask):
                    captured.append((role, code.clone(), mask.clone()))
                    collector(layer, role, code, mask)
                with observe_modules([linear], callback):
                    actual = quant_forward_with_outlier(linear, x)
                self.assertTrue(torch.equal(reference, actual))
                mask = get_outlier_mask_channel(x, .25).view(1, 1, -1)
                expected = quant_awo(x * (~mask), .125, linear.a_spec, out_dtype=torch.float32)
                self.assertTrue(torch.equal(captured[0][1], expected))
                self.assertTrue(torch.equal(captured[0][2], mask))
                for kind in ('qk', 'pv'):
                    mm = QuantizedMatMul(A_bit=8, B_bit=8, O_bit=8,
                                         outlier_ratio=.25, scale_root_str=directory)
                    mm.module_id = f'vlm.layer.0.{kind}'
                    mm.A_interval = mm.B_interval = mm.O_interval = .125
                    A, B = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 8, 6)
                    reference = matmul_quant_forward_with_outlier(mm, A, B)
                    captured.clear()
                    with observe_modules([mm], callback):
                        actual = matmul_quant_forward_with_outlier(mm, A, B)
                    self.assertTrue(torch.equal(reference, actual))
                    self.assertEqual([c[0] for c in captured], ['A', 'B'])
                    amask = get_outlier_mask_channel(A, .25)
                    bmask = _matmul_B_channel_mask(B, amask, A) | get_outlier_mask_1d(B, .25)
                    self.assertTrue(torch.equal(captured[1][2], bmask))
                    self.assertTrue(torch.equal(captured[1][1], quant_awo(B*(~bmask), .125, mm.B_spec, out_dtype=B.dtype)))
                # Exercise expert context and the two-component report too.
                CURRENT_PHASE.set('denoise')
                CURRENT_FLOW_STEP.set(0)
                linear.module_id = 'expert.layers.0.mlp.up_proj'
                with observe_modules([linear], collector):
                    quant_forward_with_outlier(linear, x)
                collector.export(Path(directory) / 'report')
                self.assertTrue((Path(directory) / 'report/summary.md').exists())
        finally:
            CURRENT_PHASE.reset(tokens[0])
            CURRENT_FLOW_STEP.reset(tokens[1])


if __name__ == '__main__':
    unittest.main()
