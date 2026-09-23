"""Real transformers attention, small random weights; no checkpoint download."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
import torch
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionAttention
from vla_tcs2.vision_attention import attach_vision_attention
from vla_tcs2.quant_matmul import QuantizedMatMul
from vla_tcs2.quant.stat_manager import QuantStatManager
from vla_tcs2.runtime_context import CURRENT_ATTN_KIND, CURRENT_PHASE


def make_attention(backend='sdpa'):
    cfg = SmolVLMVisionConfig(hidden_size=32, intermediate_size=64,
                             num_attention_heads=4, num_hidden_layers=1,
                             attention_dropout=0.0)
    cfg._attn_implementation = backend
    return SmolVLMVisionAttention(cfg).eval()


def make_mm(tmp_path, op, mode='raw', sm=None):
    mm = QuantizedMatMul(mode=mode, A_bit='e4m3', B_bit='e4m3', O_bit='e4m3',
                        scale_root_str=str(tmp_path), outlier_ratio=0.01)
    mm.method = 'pot_fp8_outlier'
    mm.set_layer_info(f'{op}_matmul', 0, module_id=f'vision.layer.0.{op}')
    mm.set_scale_group(f'vision_{op}_matmul', 0)
    mm._stat_manager = sm
    return mm


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
@pytest.mark.parametrize('masked', [False, True])
def test_raw_equivalence(tmp_path, backend, masked):
    torch.manual_seed(17)
    reference = make_attention(backend)
    patched = copy.deepcopy(reference)
    attach_vision_attention(patched, make_mm(tmp_path, 'qk'), make_mm(tmp_path, 'pv'))
    x = torch.randn(2, 9, 32)
    mask = None
    if masked:
        mask = torch.zeros(2, 1, 9, 9)
        mask[..., -2:] = torch.finfo(x.dtype).min
    with torch.no_grad():
        expected = reference(x, mask)[0]
        actual, weights = patched(x, mask, output_attentions=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert weights.shape == (2, 4, 9, 9)
    if masked:
        assert torch.count_nonzero(weights[..., -2:]) == 0
    assert patched(x)[1] is None


def test_calibrate_reload_quantize_and_stats(tmp_path):
    torch.manual_seed(3)
    sm = QuantStatManager(str(tmp_path))
    sm.enable_sparsity(enable=True)
    sm.configure_unit_sparsity(enable=False)
    attn = make_attention()
    modules = [make_mm(tmp_path, op, 'scale_inspection', sm) for op in ('qk', 'pv')]
    attach_vision_attention(attn, *modules)
    x = torch.randn(2, 9, 32)
    phase = CURRENT_PHASE.set('prefill')
    kind = CURRENT_ATTN_KIND.set('cross')
    try:
        with torch.no_grad():
            raw = attn(x)[0]
            for mm in modules:
                mm.save_scales()
                mm.mode = 'quant_forward'
                mm.A_interval = mm.B_interval = mm.O_interval = None
            quant = attn(x)[0]
        assert CURRENT_ATTN_KIND.get() == 'cross'
        assert torch.isfinite(quant).all()
        assert not torch.equal(raw, quant)
        assert len(list(tmp_path.glob('*.p'))) == 6
        sm.export_module_sparsity_csv(str(tmp_path / 'stats.csv'))
        import csv
        rows = list(csv.DictReader((tmp_path / 'stats.csv').open()))
        assert {r['module_id'] for r in rows} == {'vision.layer.0.qk', 'vision.layer.0.pv'}
        assert len(rows) == 6
        assert all(r['phase'] == 'prefill' and r['attention_kind'] == 'self' for r in rows)
        assert all(int(r['total_elements_native']) > 0 for r in rows)
    finally:
        CURRENT_ATTN_KIND.reset(kind)
        CURRENT_PHASE.reset(phase)


def test_guards_and_exception_context(tmp_path):
    attn = make_attention()
    qk, pv = make_mm(tmp_path, 'qk'), make_mm(tmp_path, 'pv')
    attach_vision_attention(attn, qk, pv)
    with pytest.raises(RuntimeError, match='already'):
        attach_vision_attention(attn, qk, pv)
    with pytest.raises(ValueError, match='additive'):
        attn(torch.randn(1, 4, 32), torch.ones(1, 1, 4, 4, dtype=torch.bool))
    qk.mode = 'quant_forward'
    token = CURRENT_ATTN_KIND.set('cross')
    try:
        with pytest.raises(FileNotFoundError):
            attn(torch.randn(1, 4, 32))
        assert CURRENT_ATTN_KIND.get() == 'cross'
    finally:
        CURRENT_ATTN_KIND.reset(token)
    unsupported = make_attention('flash_attention_2')
    with pytest.raises(ValueError, match='backend'):
        attach_vision_attention(unsupported, qk, pv)
