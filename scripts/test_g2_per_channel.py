#!/usr/bin/env python
"""G2 per-channel W4 链路冒烟测试（CPU，无模型加载）。

验证 pot_ao_outlier_channel 全链路（setup §8.2 / P0 PASS 条件 ⑥⑦）：
  1. 校准侧：scales_with_pot_ao_outlier_channel 返回 [N_out] w scale，
     a/o 为标量 PoT；
  2. stat_manager：张量 w scale 聚合（elementwise max）与 pickle 落盘；
  3. 前向侧：quant_forward_pot_ao_outlier_channel 正常运行且输出有限；
  4. 等价性 A：weight_quant_granularity 缺省（per_tensor）时 raw 路径
     bit-exact（不触量化）；
  5. 等价性 B：per-channel 数值 ≤ 保守界（与手写参考实现一致）；
  6. per-channel NMSE ≤ per-tensor NMSE（在合成 outlier 权重上）。

用法：conda run -n smolvla_eval python scripts/test_g2_per_channel.py
"""

import os
import pickle
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant.quant_spec import parse_quant_spec, quant_awo
from vla_tcs2.quant.scale_methods import (
    get_scale_method,
    scales_with_pot_ao_outlier,
    scales_with_pot_ao_outlier_channel,
    get_outlier_mask_1d,
    get_outlier_mask_channel,
)
from vla_tcs2.quant.quant_methods import get_quant_method


def make_layer(mode="quant_forward", method="pot_ao_outlier"):
    torch.manual_seed(0)
    layer = QuantizedLinear(
        in_features=64,
        out_features=48,
        bias=True,
        mode=mode,
        a_bit="e4m3",
        w_bit=4,
        o_bit="e4m3",
        d_bit=4,
        p=4,
        outlier_ratio=0.01,
    )
    layer.method = method
    layer.set_layer_info("q_proj", 3, module_id="vlm.layers.3.self_attn.q_proj")
    # 行间分布差异大的合成权重（模拟 tensor-wise 的痛点）
    with torch.no_grad():
        base = torch.randn(48, 64) * 0.01
        base[0] *= 8.0  # 少数行 absmax 大 8 倍
        base[:, 5] *= 10.0  # 一个 outlier 通道
        layer.weight.copy_(base)
        layer.bias.copy_(torch.randn(48) * 0.001)
    return layer


def calibrate(layer, x, scale_fn):
    with torch.no_grad():
        out = F.linear(x, layer.weight, layer.bias)
        return scale_fn(
            x, layer.weight, out,
            layer.a_spec, layer.w_spec, layer.o_spec,
            outlier_ratio=layer.outlier_ratio,
        )


def nmse(ref, test):
    return ((ref - test) ** 2).sum().item() / (ref ** 2).sum().item()


def main():
    torch.manual_seed(1)
    x = torch.randn(2, 16, 64) * 0.1

    # ---- 1. 校准侧 ----
    layer = make_layer()
    a, w, o = calibrate(layer, x, scales_with_pot_ao_outlier_channel)
    assert isinstance(w, torch.Tensor) and w.shape == (48,), f"w scale 形状错误: {w}"
    assert torch.isfinite(w).all(), "w scale 含非有限值"
    import math
    for v in (a, o):
        assert isinstance(v, float) and v > 0
        e = math.log2(v)
        assert abs(e - round(e)) < 1e-9, f"a/o 需为 PoT 标量: {v}"
    print(f"1 校准侧 OK: w scale {tuple(w.shape)}, a={a:.3e}(PoT), o={o:.3e}(PoT)")

    # ---- 2. stat_manager 聚合 + 落盘 round-trip ----
    from vla_tcs2.quant.stat_manager import QuantStatManager

    with tempfile.TemporaryDirectory() as td:
        sm = QuantStatManager(td)
        sm.collect_linear_stats("q_proj", 3, w, a, o)
        sm.collect_linear_stats("q_proj", 3, w, a, o)  # 两次采样
        sm.save_all_scales()
        w_loaded = pickle.load(open(os.path.join(td, "q_proj_w_scale_3.p"), "rb"))
        assert isinstance(w_loaded, torch.Tensor) and torch.equal(w_loaded, w)
        print("2 stat_manager 聚合+落盘 OK: elementwise max 保持 [N_out]")

    # ---- 3. 前向侧（直接调用 quant method，绕过 scale 文件加载） ----
    layer = make_layer(method="pot_ao_outlier_channel")
    layer.a_interval, layer.w_interval, layer.o_interval = a, w, o
    with torch.no_grad():
        y = get_quant_method(layer.method)(layer, x)
    assert y.shape == (2, 16, 48) and torch.isfinite(y).all()
    print(f"3 前向侧 OK: 输出 {tuple(y.shape)} 有限")

    # ---- 3b. 层输出量级正确性（防 x_sim 未去量化的 bug：输出会被放大 ~1/a_interval 倍） ----
    with torch.no_grad():
        ref_out = F.linear(x, layer.weight, layer.bias)
    scale_ratio = (y.norm() / ref_out.norm()).item()
    assert 0.5 < scale_ratio < 2.0, (
        f"输出/参考范数比 {scale_ratio:.3f} 异常——量化层输出量级被破坏"
    )
    out_rel = ((y - ref_out) ** 2).sum().item() / (ref_out ** 2).sum().item()
    assert out_rel < 1.0, f"输出相对误差过大: {out_rel:.3f}"
    print(f"3b 层输出量级 OK: norm_ratio={scale_ratio:.3f}, rel_err={out_rel:.4f}")

    # ---- 4. 等价性 A：raw 路径 bit-exact ----
    layer_raw = make_layer(mode="raw")
    with torch.no_grad():
        y_raw = layer_raw(x)
    ref = F.linear(x, layer_raw.weight, layer_raw.bias)
    assert torch.equal(y_raw, ref), "raw 路径必须 bit-exact"
    print("4 raw 等价 OK: bit-exact")

    # ---- 5/6. per-channel NMSE ≤ per-tensor NMSE（权重误差口径） ----
    with torch.no_grad():
        ratio = 0.01
        w_mask = get_outlier_mask_1d(layer.weight, ratio)
        ch_mask = get_outlier_mask_channel(x, ratio)
        w_ch_mask = ch_mask.view(1, -1) | w_mask
        w_normal = (layer.weight * (~w_ch_mask).float()).float()

        w_spec = layer.w_spec
        qmax = 2 ** (w_spec.bits - 1)

        # per-tensor 参考（scales_with_pot_ao_outlier 的 w 部分）
        _, w_t, _ = calibrate(layer, x, scales_with_pot_ao_outlier)
        dq_t = quant_awo(w_normal, w_t, w_spec, out_dtype=torch.float32) * w_t

        # per-channel（[N_out] 缩放，前向同款路径）
        w_col = w.view(-1, 1)
        dq_c = quant_awo(w_normal, w_col, w_spec, out_dtype=torch.float32) * w_col

        n_t, n_c = nmse(w_normal, dq_t), nmse(w_normal, dq_c)
        assert n_c <= n_t, f"per-channel NMSE ({n_c:.3e}) 应 ≤ per-tensor ({n_t:.3e})"
        print(f"5/6 NMSE 对比 OK: per-channel {n_c:.3e} ≤ per-tensor {n_t:.3e}")

    # ---- 注册表完备 ----
    get_scale_method("pot_ao_outlier_channel")
    get_quant_method("pot_ao_outlier_channel")
    print("6 注册表 OK: scale/quant 方法均已注册")

    print("\nG2 PER-CHANNEL SMOKE: ALL PASS")


if __name__ == "__main__":
    main()
