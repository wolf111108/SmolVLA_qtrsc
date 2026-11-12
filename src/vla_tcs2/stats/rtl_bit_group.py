"""RTL-compensation-aware activation/static-weight BG statistics for Phase O.

Signed INT4/INT8 codes are sign-extended to eight bits. Negative groups are
complemented; only BG0 has +1 compensation. A shared pack skips iff all valid
coefficients are zero. Weight packs span output channels at fixed input K.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PA_PW_PAIRS = ((2, 1024), (4, 512), (8, 256), (16, 128), (32, 64), (64, 32))
PA_VALUES = tuple(a for a, _ in PA_PW_PAIRS)
PW_VALUES = tuple(w for _, w in PA_PW_PAIRS)
COUNTERS = (
    'calls', 'total_packs', 'normal_zero_packs', 'normal_issued_packs',
    'full_native_packs', 'full_native_zero_packs', 'native_groups',
    'native_zero_groups', 'native_skipped_groups', 'protected_groups',
    'tail_packs', 'tail_zero_packs', 'negative_groups', 'sign_extended_groups',
)
KEYS = ('component', 'module_id', 'phase', 'flow_step', 'attention_kind',
        'role', 'source_bits', 'storage_bits', 'group_bits', 'p', 'level')
SUMMARY_KEYS = ('component', 'scope', 'source_bits', 'storage_bits', 'group_bits', 'p', 'level')


def _validate_codes(code, source_bits):
    if source_bits not in (4, 8):
        raise ValueError('source_bits must be 4 or 8')
    code = np.asarray(code)
    if (code.ndim < 2 or not code.size or code.dtype.kind not in 'iuf'
            or not np.isfinite(code).all()):
        raise ValueError('Expected nonempty finite numeric [..., lanes, K] codes')
    lo, hi = -(1 << (source_bits - 1)), (1 << (source_bits - 1)) - 1
    if np.any(code != np.rint(code)) or code.min() < lo or code.max() > hi:
        raise ValueError(f'Expected signed INT{source_bits} integer codes in [{lo}, {hi}]')
    return code


def rtl_coefficients(code, group_bits=2, source_bits=8):
    """Diagnostic D[g], low to high; D0 may be 4 (BG2) or 16 (BG4).

    q = (-1 if q < 0 else 1) * sum_g D[g] * 2**(group_bits*g).
    INT4 sign extension keeps numerical values; negative high bits become zero
    after complementing. No carry is propagated from D0 into the next group.
    """
    code = _validate_codes(code, source_bits).astype(np.int16)
    if group_bits not in (2, 4):
        raise ValueError('group_bits must be 2 or 4')
    negative = code < 0
    normalized = (code & 255) ^ np.where(negative, 255, 0)
    result = np.stack([(normalized >> (group_bits * g)) & ((1 << group_bits) - 1)
                       for g in range(8 // group_bits)], axis=-1)
    result[..., 0] += negative
    return result


def count_rtl_bit_groups(code, protected=None, p_values=PA_VALUES, *,
                         group_bits=2, source_bits=8, lane_axis=-2,
                         chunk_elements=1_048_576):
    """Integer counts with protected/tail lanes excluded from native denominators.

    normal_issued_packs is ordinary-path work, NOT elapsed cycles. A strict
    full-native pack has p real, unprotected lanes. K chunking preserves pack
    boundaries and never joins batches, heads or separate calls. No large
    artificial padding tensor is allocated, including when pw=1024.
    """
    code = _validate_codes(code, source_bits)
    p_values = tuple(p_values)
    if (not p_values or len(set(p_values)) != len(p_values)
            or any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer))
                   or p <= 0 for p in p_values)):
        raise ValueError('p_values must be distinct positive integers')
    if group_bits not in (2, 4) or lane_axis not in (-2, -1):
        raise ValueError('Expected group_bits=2/4 and lane_axis=-2/-1')
    if not isinstance(chunk_elements, int) or chunk_elements <= 0:
        raise ValueError('chunk_elements must be a positive integer')
    protected = (np.zeros(code.shape, dtype=bool) if protected is None
                 else np.broadcast_to(np.asarray(protected, dtype=bool), code.shape))
    if np.any(code[protected] != 0):
        raise ValueError('FP-protected positions must be zero in normal-path codes')
    q = np.moveaxis(code, lane_axis, -2).astype(np.int16)
    mask = np.moveaxis(protected, lane_axis, -2)
    m, k = q.shape[-2:]
    q = q.reshape(-1, m, k)
    mask = mask.reshape(q.shape)
    batches = q.shape[0]
    width = max(1, chunk_elements // (batches * m))
    rows = {(p, g): dict(p=int(p), level=g, group_bits=group_bits, source_bits=source_bits,
                         storage_bits=8, **dict.fromkeys(COUNTERS, 0))
            for p in p_values for g in range(8 // group_bits)}
    for row in rows.values():
        row['calls'] = 1
    for begin in range(0, k, width):
        end = min(begin + width, k)
        block = q[:, :, begin:end]
        native = ~mask[:, :, begin:end]
        neg = block < 0
        norm = (block & 255) ^ np.where(neg, 255, 0)
        groups = int(native.sum())
        negatives = int((native & neg).sum())
        protected_count = int(native.size - groups)
        # Reuse each significance-level predicate across all six p values.
        nonzeros = [(((norm >> (group_bits * g)) & ((1 << group_bits) - 1)) != 0)
                    | (neg if g == 0 else False) for g in range(8 // group_bits)]
        for p in p_values:
            starts = np.arange(0, m, p)
            lengths = np.minimum(p, m - starts)
            per_pack = np.add.reduceat(native, starts, axis=1, dtype=np.int64)
            full = per_pack == p
            tail = np.broadcast_to((lengths < p)[None, :, None], per_pack.shape)
            for level, nonzero in enumerate(nonzeros):
                active = np.logical_or.reduceat(nonzero & native, starts, axis=1)
                zero = ~active
                counts = dict(
                    total_packs=active.size, normal_zero_packs=int(zero.sum()),
                    normal_issued_packs=int(active.sum()), full_native_packs=int(full.sum()),
                    full_native_zero_packs=int((zero & full).sum()),
                    native_groups=groups, native_zero_groups=int((native & ~nonzero).sum()),
                    native_skipped_groups=int(per_pack[zero].sum()),
                    protected_groups=protected_count, tail_packs=int(tail.sum()),
                    tail_zero_packs=int((tail & zero).sum()), negative_groups=negatives,
                    sign_extended_groups=groups if group_bits * level >= source_bits else 0,
                )
                target = rows[p, level]
                for name, value in counts.items():
                    target[name] += int(value)
    return list(rows.values())


def rates(row):
    row = dict(row)
    for name, num, den in (
        ('native_skip_rate', 'native_skipped_groups', 'native_groups'),
        ('individual_bg_zero_rate', 'native_zero_groups', 'native_groups'),
        ('full_native_pack_zero_rate', 'full_native_zero_packs', 'full_native_packs'),
        ('normal_pack_zero_rate', 'normal_zero_packs', 'total_packs'),
        ('tail_pack_fraction', 'tail_packs', 'total_packs'),
        ('sign_extended_group_fraction', 'sign_extended_groups', 'native_groups'),
    ):
        row[name] = row[num] / row[den] if row[den] else None
    return row


def _csv(path, rows):
    if not rows:
        raise ValueError(f'Refusing empty report: {path.name}')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RTLBitGroupCollector:
    """Reuse actual activation hooks; scan each calibrated static Linear once.

    Static full-weight quantized codes exclude no runtime outlier locations.
    QK/PV B are runtime activations, never static parameters. Weight-side BG
    skipping and BG4 are analytical extensions, not implemented chipds speedups.
    """
    def __init__(self):
        self.counts = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
        self.shapes = defaultdict(set)
        self.observed_calls = defaultdict(int)
        self.weight_manifest = {}

    def _add(self, key, rows):
        for row in rows:
            dest = self.counts[(*key, row['source_bits'], 8, row['group_bits'], row['p'], row['level'])]
            for name in COUNTERS:
                dest[name] += row[name]

    def __call__(self, layer, role, code, protected):
        import torch
        from vla_tcs2.runtime_context import get_runtime_context
        mid = layer.module_id
        comp = mid.split('.')[0]
        if comp not in ('vlm', 'expert') or role not in ('activation', 'A', 'B'):
            raise ValueError(f'Unexpected activation site: {mid}/{role}')
        spec = layer.a_spec if role == 'activation' else getattr(layer, f'{role}_spec')
        if spec.name() != 'int8':
            raise ValueError(f'Expected INT8 activation: {mid}/{role}')
        ctx = get_runtime_context()
        phase, step = ctx['phase'], ctx['flow_step']
        if (comp == 'vlm' and (phase, step) != ('prefill', -1)) or (
                comp == 'expert' and (phase != 'denoise' or step not in range(10))):
            raise ValueError(f'Missing/wrong runtime context: {mid}: {ctx}')
        array = code.detach().to(device='cpu', dtype=torch.float32).numpy()
        mask = protected.detach().to(device='cpu').numpy()
        axis = -1 if role == 'B' and mid.endswith('.qk') else -2
        key = (comp, mid, phase, step, ctx['attention_kind'], role)
        self._add(key, count_rtl_bit_groups(array, mask, PA_VALUES, lane_axis=axis))
        self.observed_calls[key] += 1
        self.shapes[key].add(tuple(code.shape))

    def collect_weight_codes(self, module_id, codes, source_bits):
        """[out_features, in_features]; pw spans output channels at fixed K."""
        comp = module_id.split('.')[0]
        expected = 8 if comp == 'vlm' else 4
        if comp not in ('vlm', 'expert') or source_bits != expected:
            raise ValueError(f'Wrong weight precision: {module_id}/INT{source_bits}')
        if module_id in self.weight_manifest:
            raise ValueError(f'Weight already counted: {module_id}')
        codes = _validate_codes(codes, source_bits)
        if codes.ndim != 2:
            raise ValueError('Static Linear weights must have [out_features, in_features] shape')
        key = (comp, module_id, 'static', -1, 'none', 'weight')
        for bits in (2, 4):
            rows = count_rtl_bit_groups(codes, p_values=PW_VALUES, group_bits=bits, source_bits=source_bits)
            if source_bits == 4:
                for row in rows:
                    if row['level'] * bits >= 4 and row['normal_issued_packs'] != 0:
                        raise AssertionError('Sign-extended INT4 high BG is not sparse')
            self._add(key, rows)
        self.weight_manifest[module_id] = dict(
            module_id=module_id, component=comp, source_bits=source_bits, storage_bits=8,
            shape=list(codes.shape), elements=int(codes.size),
            codes_sha256=hashlib.sha256(codes.astype(np.int8).tobytes(order='C')).hexdigest(),
            semantics='full_weight_quantized_no_dynamic_outlier_mask',
        )

    def collect_weights(self, modules):
        """Use production quant_awo and audited scale files, once per Linear."""
        import torch
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant.quant_spec import quant_awo
        with torch.no_grad():
            for layer in modules:
                if not isinstance(layer, QuantizedLinear):
                    continue
                if layer.module_id in self.weight_manifest:
                    raise ValueError(f'Weight already counted: {layer.module_id}')
                layer._load_scales()
                codes = quant_awo(layer.weight.detach().to(torch.float32), layer.w_interval,
                                  layer.w_spec, out_dtype=torch.float32, chunk_size=1_048_576)
                self.collect_weight_codes(layer.module_id, codes.cpu().numpy(), layer.w_spec.bits)
                self.weight_manifest[layer.module_id]['scale'] = float(layer.w_interval)
                del codes

    def rows(self, weights=False):
        return [rates(dict(zip(KEYS, key), **value)) for key, value in sorted(self.counts.items())
                if (key[5] == 'weight') == weights]

    def summary(self):
        buckets = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
        for row in self.rows() + self.rows(weights=True):
            scopes = (('static_weight',) if row['role'] == 'weight' else
                      (('all_inputs', 'bsc_left') if row['role'] in ('activation', 'A') else ('all_inputs',)))
            for scope in scopes:
                for level in (str(row['level']), 'all'):
                    key = (row['component'], scope, row['source_bits'], 8, row['group_bits'], row['p'], level)
                    for name in COUNTERS:
                        buckets[key][name] += row[name]
        return [rates(dict(zip(SUMMARY_KEYS, key), **value)) for key, value in sorted(buckets.items())]

    def paired_rows(self):
        index = {(r['component'], r['scope'], r['group_bits'], r['p']): r
                 for r in self.summary() if r['level'] == 'all'}
        result = []
        for pa, pw in PA_PW_PAIRS:
            row = dict(pa=pa, pw=pw, pa_pw=pa * pw)
            for comp in ('vlm', 'expert'):
                for name, scope, bits, p in (('activation_bg2', 'bsc_left', 2, pa),
                                             ('weight_bg2', 'static_weight', 2, pw),
                                             ('weight_bg4', 'static_weight', 4, pw)):
                    source = index.get((comp, scope, bits, p))
                    if source is None:
                        raise ValueError(f'Missing paired report data: {comp}/{scope}/BG{bits}/p={p}')
                    for metric in (*COUNTERS, 'native_skip_rate', 'individual_bg_zero_rate',
                                   'full_native_pack_zero_rate', 'normal_pack_zero_rate',
                                   'tail_pack_fraction', 'sign_extended_group_fraction'):
                        row[f'{comp}_{name}_{metric}'] = source[metric]
            result.append(row)
        return result

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _csv(directory / 'activation_by_site.csv', self.rows())
        _csv(directory / 'weight_by_site.csv', self.rows(weights=True))
        _csv(directory / 'summary.csv', self.summary())
        pairs = self.paired_rows()
        _csv(directory / 'paired_summary.csv', pairs)
        (directory / 'weight_manifest.json').write_text(json.dumps(
            list(self.weight_manifest.values()), indent=2, allow_nan=False) + '\n')
        lines = [
            '# Phase O — RTL-aligned activation / padded-weight BG sparsity', '',
            '8-bit sign extension → negative BG complement → negative BG0 +1 only.',
            'Activation: Linear input + QK/PV A (bsc_left), BG2, grouped by pa tokens.',
            'Weight: each static Linear [out,in] once, BG2 and BG4, grouped by pw output channels.',
            'INT4 high sign-extension BGs count in the 8-bit denominator. No dynamic outlier weight mask.',
            'Native skip excludes FP-protected activation lanes and nonexistent tail lanes.',
            'Full-native pack ratios exclude tails/protected packs; missing denominators remain N/A.',
            'Weight-side skip / BG4 / large p are analysis extensions, not implemented RTL speedups.', '',
            '| pa | pw | VLM A BG2 | VLM W BG2 | VLM W BG4 | Expert A BG2 | Expert W BG2 | Expert W BG4 |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        names = [f'{comp}_{kind}_native_skip_rate' for comp in ('vlm', 'expert')
                 for kind in ('activation_bg2', 'weight_bg2', 'weight_bg4')]
        for r in pairs:
            vals = ['N/A' if r[n] is None else f'{100*r[n]:.4f}%' for n in names]
            lines.append(f"| {r['pa']} | {r['pw']} | " + ' | '.join(vals) + ' |')
        lines += ['', 'Detailed level counts: summary.csv. Full/tail counts: paired_summary.csv.',
                  'Counts are pooled before division; activation and static-weight denominators are never mixed.', '']
        (directory / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
