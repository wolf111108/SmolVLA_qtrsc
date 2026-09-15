#!/usr/bin/env python
"""Phase H — build the §14 result tables (experiment_setup.md §14) from raw CSVs.

Aggregation rule (experiment_setup.md §8): never average percentages, always
sum numerators / sum denominators.

Scope discipline (er.md audit, 2026-09-15):
  - S0 quantizes ALL Linear (`include: ['*']`), S1 only `expert.*`. Therefore
    the raw per-config aggregates are NOT the same workload scope. This script
    emits explicit *common-scope* comparisons for both runtime module sparsity
    and static weight sparsity so that S0-vs-S1 deltas are apples-to-apples.
  - Static weight CSVs are re-exported per task (10 identical copies). They must
    be deduplicated by module_id before any count/volume is summed.
  - unit sparsity has NO native (FP-protected) correction: it measures the
    masked normal quant datapath. Reported as "quant-path unit sparsity".
  - exported BOP proxies are legacy/debug only, NOT physically additive.

Outputs GitHub-flavored markdown to stdout so it can be pasted into results.md.
Also prints the §13 convergence gate (|Δ| < 0.5 pp for H2 vs H1).

Usage:
    python build_results_tables.py --stage h2_30ep
    python build_results_tables.py --stage h2_30ep --compare h1_10ep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP_NAME = "2026-09-13_phaseH_accuracy-preserving-sparsity"
CONFIGS = ["s0_fp8_all", "s1_expert_w4"]
LINEAR_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# S0 quantizes VLM Linear too; S1 leaves VLM in raw FP. For a common-scope
# runtime comparison we drop VLM Linear rows from S0.
COMMON_SCOPE_DROP = lambda r: (  # noqa: E731
    r["component"].lower() == "vlm" and r["operator"] in LINEAR_OPS
)


def _read(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load(stage, cfg, fname, exclude_roles=()):
    root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
    rows = []
    if not os.path.isdir(root):
        return rows
    for d in sorted(os.listdir(root)):
        for r in _read(os.path.join(root, d, "sparsity", fname)):
            if r.get("tensor_role") in exclude_roles:
                continue
            rows.append(r)
    return rows


def _load_one_task(stage, cfg, fname, task="task00"):
    """Static weight is episode-independent → read ONE task, never aggregate."""
    p = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg, task, "sparsity", fname)
    return _read(p)


def _dedupe_modules(rows, component=None):
    """Deduplicate static weight rows by module_id (identical across tasks).

    Static weight CSVs are re-exported per task, so a 10-task aggregate would
    double-count total_bits / parameter volume by 10x. Ratios are unaffected
    but counts and volumes are, so always dedupe.
    """
    uniq = {}
    for r in rows:
        if component and r["component"].lower() != component:
            continue
        uniq[r["module_id"]] = r
    return list(uniq.values())


def _sr(stage, cfg):
    """Episode-weighted SR across tasks (sum successes / sum episodes)."""
    root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
    ok = tot = 0
    n_tasks = 0
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "eval_info.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            o = json.load(f)["overall"]
        n = int(o["n_episodes"])
        ok += round(o["pc_success"] / 100.0 * n)
        tot += n
        n_tasks += 1
    return (ok, tot, n_tasks, 100.0 * ok / tot if tot else 0.0)


def _wilson(ok, n, z=1.959964):
    """Wilson score 95% CI for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = ok / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy else 0.0


def _avg_rank(v):
    """Tie-averaged ranks (needed because SR only takes 3 distinct values)."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(xs, ys):
    return _pearson(_avg_rank(xs), _avg_rank(ys))


def _per_task(stage, cfg):
    """[(task, sr_pct, native_bit_sparsity)] for each task with CSVs."""
    root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
    out = []
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        ms = _read(os.path.join(root, d, "sparsity", "module_sparsity.csv"))
        if not ms:
            continue
        b, _, _ = _ratio(ms, "sparse_bits_native", "total_bits_native")
        sr = None
        p = os.path.join(root, d, "eval_info.json")
        if os.path.isfile(p):
            with open(p) as f:
                sr = json.load(f)["overall"]["pc_success"]
        out.append((d, sr, b))
    return out


def _unit_component_key(r):
    """unit_sparsity.csv has no component/operator columns → derive from module_id."""
    comp, op = _parse_module_id(r["module_id"])
    if op in ("qk", "pv"):
        return (f"{comp}.{op}".upper(), r["phase"], r["tensor_role"])
    return (comp.upper(), r["phase"], r["tensor_role"])


def _ratio(rows, num, den):
    n = sum(float(r[num]) for r in rows)
    d = sum(float(r[den]) for r in rows)
    return (n / d if d else 0.0), n, d


def _pct(x):
    return f"{x * 100:.2f}%"


def _parse_module_id(module_id):
    """module_id → (component, operator).

    Examples:
        vlm.layers.0.self_attn.q_proj   → ("vlm", "q_proj")
        vlm.layers.3.mlp.gate_proj      → ("vlm", "gate_proj")
        expert.layer.0.pv               → ("expert", "pv")
        expert.layers.0.mlp.down_proj   → ("expert", "down_proj")
    """
    parts = module_id.split(".")
    comp = parts[0]
    op = parts[-1]
    return comp, op


def _component_key(r):
    """Map a module_sparsity row to a (component, phase, role) display key."""
    comp = r["component"]
    op = r["operator"]
    role = r["tensor_role"]
    if op in ("qk", "pv"):
        return (f"{comp}.{op}".upper(), r["phase"], role)
    return (comp.upper(), r["phase"], role)


def _section_summary(rows_all):
    """§14.1 summary table."""
    print("### Summary (accuracy + coverage)")
    print()
    print("| Config | Stage | Tasks | Episodes | SR | Wilson 95% CI | Native bit sparsity "
          "(own scope) | Quant-path unit sparsity (2x2, masked) | FP sidepath |")
    print("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for cfg, stage, ms, us in rows_all:
        sr = _sr(stage, cfg)
        bit, _, _ = _ratio(ms, "sparse_bits_native", "total_bits_native")
        unit, _, _ = _ratio(us, "zero_units", "total_units")
        fp, _, _ = _ratio(ms, "protected_elements", "total_elements_reported")
        if sr:
            lo, hi = _wilson(sr[0], sr[1])
            sr_s, ci_s = f"{sr[3]:.1f}% ({sr[0]}/{sr[1]})", f"[{lo:.1%}, {hi:.1%}]"
        else:
            sr_s = ci_s = "n/a"
        n_ep = sr[1] if sr else 0
        n_task = sr[2] if sr else 0
        print(f"| {cfg} | {stage} | {n_task} | {n_ep} | {sr_s} | {ci_s} | {_pct(bit)} | "
              f"{_pct(unit)} | {_pct(fp)} |")
    print()
    print("> `own scope` = 各 config 自己的量化 workload（S0 含 VLM Linear，S1 不含），"
          "**不能直接作差**；同 scope 对比见 *Common-scope* 一节。")
    print("> `Quant-path unit sparsity` 未做 native (FP-protected) 修正，衡量的是 masked "
          "normal quant datapath 中的 2x2 block-zero 机会，**不是** native unit sparsity。")
    print("> SR 的 Wilson 95% CI 在 30ep 下宽度约 22pp，S0/S1 区间大幅重叠 ⇒ 30ep 不足以"
          "确定 3.3pp gap（H3 100ep 用于正式 accuracy claim）。")
    print()


def _section_components(rows_all):
    """§14.2 component table."""
    print("### Component / role breakdown")
    print()
    print("| Config | Component | Phase | Role | Zero native | Bit sparse native | "
          "Quant-path unit sparse (2x2, masked) | FP sidepath |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for cfg, stage, ms, us in rows_all:
        if not ms:
            continue
        agg_ms = defaultdict(list)
        for r in ms:
            agg_ms[_component_key(r)].append(r)
        agg_us = defaultdict(list)
        for r in us:
            agg_us[_unit_component_key(r)].append(r)

        def sort_key(k):
            comp, phase, role = k
            return (0 if comp.startswith("VLM") else 1, comp, 0 if phase == "prefill" else 1, role)

        for k in sorted(agg_ms, key=sort_key):
            comp, phase, role = k
            z, _, _ = _ratio(agg_ms[k], "zero_elements_native", "total_elements_native")
            b, _, _ = _ratio(agg_ms[k], "sparse_bits_native", "total_bits_native")
            fp, _, _ = _ratio(agg_ms[k], "protected_elements", "total_elements_reported")
            urows = agg_us.get(k, [])
            u, _, _ = _ratio(urows, "zero_units", "total_units") if urows else (0.0, 0, 0)
            print(f"| {cfg} | {comp} | {phase} | {role} | {_pct(z)} | {_pct(b)} | {_pct(u)} | {_pct(fp)} |")
    print()


def _section_flow_step(rows_all):
    """§14.3 flow-step table (expert denoise, per step, per MatMul role)."""
    print("### Flow-step table (expert denoise, bit sparse native)")
    print()
    roles = [("qk", "A"), ("qk", "B"), ("qk", "O"), ("pv", "A"), ("pv", "B"), ("pv", "O")]
    header = "| Config | step | " + " | ".join(f"{op.upper()} {r}" for op, r in roles) + " |"
    print(header)
    print("|---|---:|" + "---:|" * len(roles))
    for cfg, stage, ms, _us in rows_all:
        if not ms:
            continue
        per = defaultdict(list)
        for r in ms:
            if r["phase"] != "denoise" or r["operator"] not in ("qk", "pv"):
                continue
            per[(int(float(r["flow_step"])), r["operator"], r["tensor_role"])].append(r)
        for step in range(10):
            cells = []
            for op, role in roles:
                rws = per.get((step, op, role), [])
                v, _, _ = _ratio(rws, "sparse_bits_native", "total_bits_native") if rws else (0.0, 0, 0)
                cells.append(_pct(v))
            print(f"| {cfg} | {step} | " + " | ".join(cells) + " |")
    print()


PREFIX_INVALID_ROLES = ("output", "O")


def _section_convergence(cur, prev, excl=()):
    """§13 convergence gate |Δ| < 0.5 pp between two stages.

    If ``excl`` is empty, rows whose `prev` value comes from the pre-fix
    instrumentation (tensor_role in output/O) are marked N/A instead of FAIL,
    because their `prev` denominator is known-invalid rather than merely
    different.
    """
    print(f"### Convergence gate ({cur} vs {prev})")
    print()
    print(f"Gate: |Δ native bit sparsity| < 0.5 pp; |Δ quant-path unit rate| < 0.5 pp")
    if excl:
        print(f"(excluding tensor_role ∈ {{{', '.join(excl)}}} — pre-fix invalid rows)")
    else:
        print("(output/O rows marked N/A: their `prev` value is pre-fix INVALID, "
              "re-run with --exclude-roles output,O to compare in a clean scope)")
    print()
    print("| Config | Component | Phase | Role | Prev | Cur | Δ (pp) | Gate |")
    print("|---|---|---|---|---:|---:|---:|---|")

    def agg(ms, us, cfg, stage):
        m, u = {}, {}
        for r in ms:
            m.setdefault(_component_key(r), []).append(r)
        for r in us:
            u.setdefault(_unit_component_key(r), []).append(r)
        out = {}
        for k, rows in m.items():
            b, _, _ = _ratio(rows, "sparse_bits_native", "total_bits_native")
            urows = u.get(k, [])
            uz, _, _ = _ratio(urows, "zero_units", "total_units") if urows else (0.0, 0, 0)
            out[k] = (b, uz)
        return out

    for cfg in CONFIGS:
        a = agg(_load(prev, cfg, "module_sparsity.csv", excl),
                _load(prev, cfg, "unit_sparsity.csv", excl), cfg, prev)
        b = agg(_load(cur, cfg, "module_sparsity.csv", excl),
                _load(cur, cfg, "unit_sparsity.csv", excl), cfg, cur)
        for k in sorted(set(a) | set(b), key=lambda k: (0 if k[0].startswith("VLM") else 1, k[0], k[1], k[2])):
            if k not in a or k not in b:
                continue
            dbit = (b[k][0] - a[k][0]) * 100
            dunit = (b[k][1] - a[k][1]) * 100
            if not excl and k[2] in PREFIX_INVALID_ROLES:
                print(f"| {cfg} | {k[0]} | {k[1]} | {k[2]} | {_pct(a[k][0])} *(pre-fix)* | "
                      f"{_pct(b[k][0])} | {dbit:+.2f} | N/A (pre-fix) |")
                continue
            gate = "PASS" if max(abs(dbit), abs(dunit)) < 0.5 else "FAIL"
            print(f"| {cfg} | {k[0]} | {k[1]} | {k[2]} | {_pct(a[k][0])} | {_pct(b[k][0])} | "
                  f"{dbit:+.2f} | {gate} |")
    print("  (Δ shown for bit sparsity; unit rate Δ printed below)")
    for cfg in CONFIGS:
        a = agg(_load(prev, cfg, "module_sparsity.csv", excl),
                _load(prev, cfg, "unit_sparsity.csv", excl), cfg, prev)
        b = agg(_load(cur, cfg, "module_sparsity.csv", excl),
                _load(cur, cfg, "unit_sparsity.csv", excl), cfg, cur)
        for k in sorted(set(a) & set(b)):
            if not excl and k[2] in PREFIX_INVALID_ROLES:
                continue
            dunit = (b[k][1] - a[k][1]) * 100
            if abs(dunit) >= 0.5:
                print(f"  unit-rate gate FAIL: {cfg} {k}: {_pct(a[k][1])} → {_pct(b[k][1])} ({dunit:+.2f} pp)")
    print()


def _section_runtime_scope(stage):
    """Common-scope runtime aggregate: drop S0's VLM Linear rows.

    S0 `linear.include: ['*']` also quantizes VLM Linear; S1 only `expert.*`
    (VLM stays raw FP and never reaches the collector). Comparing the two
    raw aggregates therefore mixes different workloads.
    """
    print("### Common-scope runtime aggregate (apples-to-apples)")
    print()
    print("| Config | Scope | Quantized rows | Native bit sparsity | Δ vs S0 common (pp) |")
    print("|---|---|---:|---:|---:|")
    base = None
    for cfg in CONFIGS:
        own = _load(stage, cfg, "module_sparsity.csv")
        common = [r for r in own if not COMMON_SCOPE_DROP(r)]
        b_own, _, _ = _ratio(own, "sparse_bits_native", "total_bits_native")
        b_cmn, _, _ = _ratio(common, "sparse_bits_native", "total_bits_native")
        print(f"| {cfg} | own (all quantized ops) | {len(own)} | {_pct(b_own)} | — |")
        delta = "— (reference)" if base is None else f"**{(b_cmn - base) * 100:+.3f}**"
        if base is None:
            base = b_cmn
        print(f"| {cfg} | **common** (drop VLM Linear) | {len(common)} | **{_pct(b_cmn)}** | {delta} |")
    print()
    print("> `common` = 双方都量化的算子集合（Expert Linear + VLM/Expert QK·PV）。"
          "只有这一行的 Δ 才能解释为「配置变化对 runtime code sparsity 的影响」。")
    print()


def _section_weight_scope(stage):
    """Static weight sparsity: dedupe + Expert common-scope comparison."""
    print("### Weight static sparsity (episode-independent, deduplicated)")
    print()
    # Raw per-config rows, documenting the scope mismatch.
    print("| Config | Scope | Unique modules | Raw CSV rows | zero_rate | sparse_bit_rate |")
    print("|---|---|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        raw = _load(stage, cfg, "weight_sparsity_static.csv")
        uniq = _dedupe_modules(raw)
        z, _, _ = _ratio(uniq, "zero_elements", "total_elements")
        b, _, _ = _ratio(uniq, "sparse_bits", "total_bits")
        scope = "all Linear (VLM + Expert)" if cfg == CONFIGS[0] else "expert only"
        print(f"| {cfg} | {scope} | {len(uniq)} | {len(raw)} | {_pct(z)} | {_pct(b)} |")
    print()
    # Common scope: Expert only (the only valid W4 comparison).
    expert = {}
    for cfg in CONFIGS:
        rows = _dedupe_modules(_load(stage, cfg, "weight_sparsity_static.csv"), component="expert")
        z, _, _ = _ratio(rows, "zero_elements", "total_elements")
        b, _, _ = _ratio(rows, "sparse_bits", "total_bits")
        expert[cfg] = (len(rows), z, b)
    print("| Metric (Expert common scope) | S0 FP8 | S1 W4 | Δ |")
    print("|---|---:|---:|---:|")
    print(f"| Modules | {expert[CONFIGS[0]][0]} | {expert[CONFIGS[1]][0]} | — |")
    for label, idx in [("zero_rate", 1), ("sparse_bit_rate", 2)]:
        a, b = expert[CONFIGS[0]][idx], expert[CONFIGS[1]][idx]
        print(f"| **{label}** | **{_pct(a)}** | **{_pct(b)}** | **{(b - a) * 100:+.2f} pp** |")
    print()
    print("> **只有 Expert common scope 行可用于论文中的 W4 增益论断。** "
          "原报告中的「41.52% → 73.72%」比较的是 *(VLM FP8 + Expert FP8)* vs *Expert W4*，"
          "scope 不同，不能作为 W4 的 weight sparsity 增益。")
    print("> 每个 task 各导出一份完全相同的 static weight CSV（raw 行数 = 10 × modules）；"
          "本表已按 `module_id` 去重，因此 total_bits / 参数量 / weight BOP 不会被放大 10×。")
    print()


def _section_operators(rows_all):
    """Operator-level (q/k/v/o/gate/up/down) linear breakdown."""
    print("### Operator breakdown (linear modules, aggregated over layers)")
    print()
    print("| Config | Component | Operator | Phase | Role | Bit sparse native | "
          "Quant-path unit sparse (2x2, masked) | FP sidepath |")
    print("|---|---|---|---|---|---:|---:|---:|")
    for cfg, stage, ms, us in rows_all:
        if not ms:
            continue
        ops, uops = defaultdict(list), defaultdict(list)
        for r in ms:
            if r["operator"] not in LINEAR_OPS:
                continue
            ops[(r["component"].upper(), r["operator"], r["phase"], r["tensor_role"])].append(r)
        for r in us:
            comp, op = _parse_module_id(r["module_id"])
            if op not in LINEAR_OPS:
                continue
            uops[(comp.upper(), op, r["phase"], r["tensor_role"])].append(r)
        for k in sorted(ops, key=lambda k: (k[0], LINEAR_OPS.index(k[1]), k[2], k[3])):
            b, _, _ = _ratio(ops[k], "sparse_bits_native", "total_bits_native")
            fp, _, _ = _ratio(ops[k], "protected_elements", "total_elements_reported")
            urows = uops.get(k, [])
            u, _, _ = _ratio(urows, "zero_units", "total_units") if urows else (0.0, 0, 0)
            print(f"| {cfg} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | {_pct(b)} | {_pct(u)} | {_pct(fp)} |")
    print()


def _section_workload(stage):
    """Legacy/debug BOP proxy — explicitly NOT physically additive."""
    print("### Legacy/debug BOP proxy — NOT physically additive")
    print()
    print("| Config | Rows | BOP dense proxy | BOP active proxy | active/dense | Calls |")
    print("|---|---:|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        rows = _load(stage, cfg, "workload.csv")
        if not rows:
            continue
        dense = sum(float(r["BOP_dense_proxy"]) for r in rows)
        active = sum(float(r["BOP_active_proxy"]) for r in rows)
        calls = sum(float(r["calls"]) for r in rows)
        ratio = active / dense if dense else 0.0
        print(f"| {cfg} | {len(rows)} | {dense:.4e} | {active:.4e} | {ratio:.4f} | {calls:.0f} |")
    print()
    print("> ⚠️ **不要用本节数字作论文/硬件 conclusion。** 三处已知缺陷：")
    print("> 1. Linear 应分别缩放 A 与 W（`MAC·b_A(1−S_A)·b_W(1−S_W)`），"
          "但 exporter 把同一个 `S` 同时用于两侧；")
    print("> 2. MatMul 应按 `S_A`、`S_B` 分别缩放，且一个物理算子只应有一行；"
          "当前按 role 逐行生成，A 行用 `S_A` 缩两边、B 行又生成一次完整 operator MAC，"
          "**不是可加的物理量**；")
    print("> 3. 这里取的是 `entry[\"sparse_bit_rate\"]`（reported），**不是** headline 用的 "
          "native corrected sparsity。")
    print("> Phase I 应重写为 *one physical operator, one row*："
          "`(operator, phase, step) → (S_A, S_W/B, S_O, M, K, N, R_FP)`。")
    print()


def _section_taskwise(stage):
    """§3.5 task-wise correlation: SR vs native bit sparsity per task."""
    print("### Task-wise SR vs sparsity")
    print()
    print("| Config | Task | SR | Native bit sparsity | FP sidepath |")
    print("|---|---|---:|---:|---:|")
    for cfg in CONFIGS:
        root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
        if not os.path.isdir(root):
            continue
        for d, sr, bit in _per_task(stage, cfg):
            ms = _read(os.path.join(root, d, "sparsity", "module_sparsity.csv"))
            fp, _, _ = _ratio(ms, "protected_elements", "total_elements_reported")
            sr_s = f"{sr:.1f}%" if sr is not None else "n/a"
            print(f"| {cfg} | {d} | {sr_s} | {_pct(bit)} | {_pct(fp)} |")
    print()
    # Correlation (computed, not asserted).
    print("| Scope | n | Pearson r | Spearman ρ (tie-averaged) |")
    print("|---|---:|---:|---:|")
    pooled_x, pooled_y = [], []
    for cfg in CONFIGS:
        pts = [(sr, b) for _t, sr, b in _per_task(stage, cfg) if sr is not None]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pooled_x += xs
        pooled_y += ys
        if len(xs) > 1:
            print(f"| {cfg} | {len(xs)} | {_pearson(xs, ys):+.4f} | {_spearman(xs, ys):+.4f} |")
    if len(pooled_x) > 1:
        print(f"| **pooled** | {len(pooled_x)} | **{_pearson(pooled_x, pooled_y):+.4f}** | "
              f"**{_spearman(pooled_x, pooled_y):+.4f}** |")
    print()
    print("> **解释限度**：该统计只覆盖 10 tasks × 3 episodes = 30 episodes，且每 task 仅 3 "
          "episodes ⇒ SR 只能取 {0, 33.3, 66.7, 100}% 这 4 个值（大量并列），"
          "相关系数的分辨率极低。只能说 *H2 pilot 中未观察到明显 association*，"
          "**不能据此证明 sparsity 与 SR 统计独立**。")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--compare", default=None, help="earlier stage for the convergence gate")
    ap.add_argument(
        "--exclude-roles",
        default="",
        help="comma-separated tensor_role values to drop before aggregating "
             "(used to strip pre-fix output/O rows for H0/H1)",
    )
    args = ap.parse_args()
    excl = tuple(x for x in args.exclude_roles.split(",") if x)
    load = lambda stage, cfg, f: _load(stage, cfg, f, excl)  # noqa: E731

    rows_all = [
        (cfg, args.stage, load(args.stage, cfg, "module_sparsity.csv"),
         load(args.stage, cfg, "unit_sparsity.csv"))
        for cfg in CONFIGS
    ]

    if excl:
        print(f"> Excluding tensor_role ∈ {{{', '.join(excl)}}} from all aggregates.\n")

    _section_summary(rows_all)
    _section_runtime_scope(args.stage)
    _section_components(rows_all)
    _section_operators(rows_all)
    _section_flow_step(rows_all)
    _section_weight_scope(args.stage)
    _section_workload(args.stage)
    _section_taskwise(args.stage)

    if args.compare:
        _section_convergence(args.stage, args.compare, excl)


if __name__ == "__main__":
    main()
