"""Exact two's-complement BG semantics; no uniform sparsity estimates."""

import numpy as np


def digits(code, bits=8):
    a = np.asarray(code, dtype=np.int64)
    if bits % 2 or np.any(a < -(1 << (bits - 1))) or np.any(a >= 1 << (bits - 1)):
        raise ValueError("Invalid signed input code / width")
    neg = a < 0
    magnitude = np.where(neg, ~a, a)
    return [
        np.where(
            neg, -((magnitude >> (2 * lv)) & 3) - (lv == 0), (magnitude >> (2 * lv)) & 3
        )
        for lv in range(bits // 2)
    ]


def effective_mask(a, lv, p, bits=8, t=16):
    """Return [ceil(M/p),ceil(K/t),t] masks; tail lanes are inert."""
    d = digits(a, bits)[lv]
    m, k = d.shape
    padded = np.zeros(((m + p - 1) // p * p, (k + t - 1) // t * t), dtype=bool)
    padded[:m, :k] = d != 0
    return padded.reshape(-1, p, padded.shape[1] // t, t).any(axis=1)


def digit_gemm(a, b, a_bits=8, b_bits=8):
    """Independent integer reference: signed top weight digit, LSB correction."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for lv, d in enumerate(digits(a, a_bits)):
        for wlv in range(b_bits // 4):
            w = (b >> (4 * wlv)) & 15
            if wlv == b_bits // 4 - 1:
                w = np.where(w >= 8, w - 16, w)
            out += (d @ w) * (1 << (2 * lv + 4 * wlv))
    return out


def pack_int4(a):
    x = np.asarray(a, dtype=np.int64).ravel()
    if np.any(x < -8) or np.any(x > 7):
        raise ValueError("INT4 overflow")
    x = (x & 15).astype(np.uint8)
    if x.size % 2:
        x = np.append(x, np.uint8(0))
    return x[::2] | (x[1::2] << 4)


def unpack_int4(a, shape):
    x = np.asarray(a, dtype=np.uint8)
    y = np.stack((x & 15, x >> 4), axis=1).ravel().astype(np.int8)
    y[y >= 8] -= 16
    return y[: int(np.prod(shape))].reshape(shape)
