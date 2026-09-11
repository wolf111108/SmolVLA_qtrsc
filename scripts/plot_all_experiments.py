#!/usr/bin/env python
"""Summary figures for all experiment data in logs 2026-09-01 ~ 2026-09-08.

Figures (all English labels, written to outputs/figures/):
  1. bitwidth阶梯         -> bitwidth_ladder.png           (INT ladder + FP8-PoT)
  2. per-task heatmap     -> bitwidth_ladder_per_task.png  (per-task /10 detail)
  3. granularity ablation -> granularity_ablation_fp8_w4.png
  4. na/ep ablation curve -> ablation_na_curve.png         (inverted-U na curve)
  5. Phase F four-suite   -> phaseF_foursuite.png
  6. Phase E2 sensitivity -> phaseE2_closed_loop.png
  7. sparsity profile     -> sparsity_int8_outlier.png

Data sources: doc/logs/2026-09-0{1..8}_*.md and
outputs/experiments/*/result.json. Values below are verified against the
result.json files on disk where available (2026-09-08).

Usage:
    python scripts/plot_all_experiments.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FP_OBJ = 93.8  # FP baseline, libero_object ep10

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# ============================================================================
# 1. Bit-width ladder (INT ladder, libero_object ep10)
# ============================================================================
def plot_bitwidth_ladder():
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    precisions = ["FP", "INT16", "INT12", "INT8"]
    x = np.arange(len(precisions))

    no_outlier = [93.8, 89.0, 7.0, np.nan]  # INT8 pure: not run
    with_outlier = [93.8, np.nan, 95.0, 92.0]  # INT16+outlier: not tested

    ax.plot(x, no_outlier, "o-", color="red", lw=2, ms=8,
            label="No outlier protection")
    ax.plot(x, with_outlier, "o-", color="tab:blue", lw=2, ms=8,
            label="With outlier (ratio=0.01)")

    # INT8-pure pending marker
    ax.plot([3], [0], marker="*", ms=16, color="red")
    ax.annotate("pending", xy=(3, 0), xytext=(6, 6),
                textcoords="offset points", fontsize=9, color="red")

    for xi, v in enumerate(no_outlier):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, -16),
                        textcoords="offset points", ha="center",
                        fontsize=10, color="red")
    for xi, v in enumerate(with_outlier):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=10, color="tab:blue")

    # FP8-PoT reference band (93.0, both linear-only and linear+matmul)
    ax.axhline(93.0, color="tab:green", ls="--", lw=1.5)
    ax.text(0.02, 93.6, "FP8(e4m3)+PoT scale + outlier: 93.0 (linear / linear+matmul)",
            fontsize=9, color="tab:green")

    ax.set_xticks(x)
    ax.set_xticklabels(precisions)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("LIBERO-object SR vs quantization bit-width (ep10, seed=1000)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="center right")
    save(fig, "bitwidth_ladder.png")


# ============================================================================
# 2. Per-task detail heatmap for the INT ladder (per-task /10)
# ============================================================================
def plot_bitwidth_per_task():
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)

    configs = ["INT16", "INT12 pure", "INT12+outlier", "INT8+outlier",
               "FP8-PoT+outlier\n(linear)", "FP8-PoT+outlier\n(linear+matmul)"]
    rows = [
        [10, 10, 8, 8, 10, 7, 10, 8, 8, 10],   # INT16
        [0, 1, 0, 0, 0, 0, 4, 0, 0, 2],        # INT12 pure
        [10, 10, 10, 10, 10, 8, 10, 8, 9, 10],  # INT12+outlier
        [10, 10, 9, 9, 10, 7, 10, 8, 9, 10],   # INT8+outlier
        [10, 10, 10, 8, 10, 8, 10, 8, 9, 10],  # PoT linear-only
        [10, 10, 8, 10, 10, 8, 10, 9, 9, 9],   # PoT linear+matmul
    ]
    totals = [89, 7, 95, 92, 93, 93]

    data = np.array(rows, dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=10, aspect="auto")

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"t{i}" for i in range(10)])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)

    for i in range(len(configs)):
        for j in range(10):
            ax.text(j, i, int(rows[i][j]), ha="center", va="center",
                    fontsize=9, color="black")
        ax.text(10.0, i, f"Σ {totals[i]}", ha="left", va="center",
                fontsize=10, fontweight="bold")

    ax.set_title("Per-task success (/10) — quantization configs, libero_object ep10")
    fig.colorbar(im, ax=ax, shrink=0.8, label="successes / 10")
    ax.grid(False)
    save(fig, "bitwidth_ladder_per_task.png")


# ============================================================================
# 3. Scale-granularity ablation: FP8 vs w4 (libero_object ep10)
# ============================================================================
def plot_granularity():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)

    gran = ["global", "per_component", "per_layer", "per_site"]
    files = [27, 54, 432, 864]
    fp8 = [89.0, 95.0, 92.0, 91.0]
    w4 = [0.0, 76.0, 0.0, 75.0]

    for ax, vals, title in [
        (axes[0], fp8, "FP8 (pot_fp8_outlier, a/w/o e4m3)"),
        (axes[1], w4, "w4 weights (pot_ao_outlier, w_bit=4)"),
    ]:
        bars = ax.bar(gran, vals, color=["tab:red" if v == 0 else "tab:blue" for v in vals],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10)
        ax.axhline(FP_OBJ, color="gray", ls="--", lw=1)
        ax.text(3.4, FP_OBJ + 1, f"FP {FP_OBJ}", fontsize=9, color="gray", ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(title, fontsize=11)

    # annotate scale-file counts on x labels
    for ax in axes:
        labels = [f"{g}\n({n} files)" for g, n in zip(gran, files)]
        ax.set_xticks(range(len(gran)))
        ax.set_xticklabels(labels, fontsize=9)

    fig.suptitle("Scale granularity ablation — libero_object ep10 "
                 "(per_layer/global collapse to 0% under w4)", fontsize=12)
    save(fig, "granularity_ablation_fp8_w4.png")


# ============================================================================
# 4. n_action_steps / n_episodes ablation (inverted-U na curve)
# ============================================================================
def plot_na_curve():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

    # Left: na curve at ep=50 (four suites + average)
    na = [1, 10, 30, 50]
    suites = {
        "Spatial": [85.2, 86.0, 75.6, 64.0],
        "Object": [78.4, 93.8, 85.8, 72.0],
        "Goal": [78.4, 87.8, 84.6, 79.0],
        "Long": [55.8, 74.2, 62.0, 48.6],
    }
    avg = [74.45, 85.45, 77.0, 65.9]
    colors = ["tab:green", "tab:orange", "tab:purple", "tab:cyan"]

    ax = axes[0]
    for (name, vals), c in zip(suites.items(), colors):
        ax.plot(na, vals, "o--", color=c, lw=1.4, ms=5, label=name)
    ax.plot(na, avg, "o-", color="black", lw=2.5, ms=8, label="Average")
    for xi, v in zip(na, avg):
        ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontweight="bold")
    ax.set_xticks(na)
    ax.set_xlabel("n_action_steps (na)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("na curve @ ep=50 — inverted U, peak at na=10")
    ax.set_ylim(40, 100)
    ax.legend(fontsize=9, ncol=2)

    # Right: attribution decomposition (na vs ep contribution)
    ax = axes[1]
    labels = ["na 1→10\n(ep=10 fixed)", "na 1→10\n(ep=50 fixed)",
              "ep 10→50\n(na=1 fixed)", "ep 10→50\n(na=10 fixed)"]
    vals = [10.5, 11.0, 0.45, 0.95]
    colors_ = ["tab:blue", "tab:blue", "gray", "gray"]
    bars = ax.bar(labels, vals, color=colors_, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"+{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    ax.axhspan(-1, 1.5, color="orange", alpha=0.15)
    ax.text(3.4, 1.7, "noise band", fontsize=8, color="darkorange", ha="right")
    ax.set_ylabel("SR gain (pp)")
    ax.set_title("Attribution: na dominates (+10.5~11 pp), ep is noise-level")
    ax.set_ylim(0, 13)

    fig.suptitle("Ablation matrix P0–P3 + na∈{30,50} (A-class, mj332, 4 suites)", fontsize=12)
    save(fig, "ablation_na_curve.png")


# ============================================================================
# 5. Phase F: outlier protection × w4 across four suites
# ============================================================================
def plot_phaseF():
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)

    suites = ["Spatial", "Object", "Goal", "Libero-10"]
    fp_ep50 = [84.8, 95.2, 87.8, 70.0]  # FP baseline (verify_libero ep50)
    f1 = [84.0, 91.0, 90.0, 67.0]       # FP8-PoT + outlier (done)
    f2 = [80.0, 91.0, 87.0, np.nan]     # FP8-PoT no outlier (3/4 done)
    # f3 (w4): not started -> all NaN
    f3 = [np.nan] * 4

    x = np.arange(len(suites))
    w = 0.2

    series = [
        ("FP baseline (ep50 ref)", fp_ep50, "gray", "//"),
        ("F1: FP8-PoT + outlier", f1, "tab:blue", None),
        ("F2: FP8-PoT no outlier", f2, "tab:red", None),
        ("F3: w4 weights", f3, "tab:orange", None),
    ]
    for i, (name, vals, c, hatch) in enumerate(series):
        offs = x + (i - 1.5) * w
        bars = ax.bar(offs, vals, width=w, label=name, color=c,
                      edgecolor="black", linewidth=0.4, hatch=hatch)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 3, "—", ha="center",
                        fontsize=10, color="dimgray")
            else:
                ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Phase F (per_site, ep10, seed=1000): outlier protection × w4 — 4 suites\n"
                 "(F2 libero-10 running, F3 not started as of 2026-09-08)")
    ax.legend(fontsize=9, loc="upper right", ncols=2)
    save(fig, "phaseF_foursuite.png")


# ============================================================================
# 6. Phase E2: closed-loop noise-injection sensitivity (gpupro6000d)
# ============================================================================
def plot_phaseE2():
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=150)

    labels = [
        "baseline_raw",
        "vlm3_down\nα=0.01", "vlm3_down\nα=0.03", "vlm3_down\nα=0.10",
        "vlm0_down\nα=0.01", "vlm0_down\nα=0.03", "vlm0_down\nα=0.10",
        "exp1_up\nα=0.03", "exp1_up\nα=0.10",
        "exp7qk\nα=0.03", "exp7qk\nα=0.30",
        "exp7qk\nquant_resid",
        "vlm3qk\nα=0.03", "vlm3qk\nα=0.10",
    ]
    sr = [94, 91, 92, 89, 93, 92, 93, 92, 92, 89, 92, 91, 92, 92]
    colors = ["gray"] + ["tab:blue"] * 6 + ["tab:orange"] * 2 + \
             ["tab:green"] * 3 + ["tab:purple"] * 2

    bars = ax.bar(range(len(labels)), sr, color=colors,
                  edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, sr):
        ax.annotate(f"{v}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.axhline(94, color="gray", ls="--", lw=1)
    ax.axhspan(91, 97, color="orange", alpha=0.12)
    ax.text(13.4, 96.8, "±3pp noise band", fontsize=8, color="darkorange", ha="right")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(80, 100)
    ax.set_title("Phase E2 closed-loop noise injection (libero_object 100 ep, per_site scales)\n"
                 "rank-1 site vs rank-156 control: max drop only 5pp — forward sensitivity "
                 "does NOT predict closed-loop impact")
    save(fig, "phaseE2_closed_loop.png")


# ============================================================================
# 7. Sparsity profile (int8+outlier, 8 frames)
#    weight_sparsity.csv is written by collect_model_weight_sparsity();
#    activation stats require enable_sparsity() during rollout and may be
#    absent (empty header-only csv) — plot weights only in that case.
# ============================================================================
def plot_sparsity():
    base = Path(__file__).resolve().parents[1] / "outputs" / "sparsity" / "int8_outlier"

    def read_csv(p):
        with open(p) as f:
            header = f.readline().strip().split(",")
            rows = [line.strip().split(",") for line in f if line.strip()]
        return header, rows

    wpath = base / "weight_sparsity.csv"
    if not wpath.exists():
        print(f"[skip] sparsity csv not found: {wpath}")
        return
    header, rows = read_csv(wpath)
    if not rows:
        print("[skip] weight_sparsity.csv empty")
        return
    col = {c: i for i, c in enumerate(header)}

    names = [r[col["layer_type"]] for r in rows]
    zero_rate = np.array([float(r[col["zero_rate"]]) for r in rows])
    bit_rate = np.array([float(r[col["sparse_bit_rate"]]) for r in rows])
    speedup = np.array([float(r[col["ideal_speed_up"]]) for r in rows])

    # activation csv (may be header-only)
    act_rows = []
    apath = base / "per_layer_sparsity.csv"
    if apath.exists():
        _, act_rows = read_csv(apath)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    # (a) zero-element ratio by op type
    ax = axes[0]
    ops = sorted(set(names))
    data_by_op = [zero_rate[np.array(names) == op] for op in ops]
    bp = ax.boxplot(data_by_op, labels=ops, showfliers=True,
                    flierprops=dict(markersize=3))
    for i, d in enumerate(data_by_op, start=1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.12, 0.12, len(d)),
                   d, s=8, alpha=0.5, zorder=3)
    ax.set_ylabel("zero-element ratio")
    ax.set_title("(a) Weight zero ratio by op", fontsize=11)
    ax.tick_params(axis="x", rotation=45)

    # (b) sparse-bit rate (fraction of bits that are zero, sparse-bit format)
    ax = axes[1]
    data_by_op = [bit_rate[np.array(names) == op] for op in ops]
    ax.boxplot(data_by_op, labels=ops)
    for i, d in enumerate(data_by_op, start=1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.12, 0.12, len(d)),
                   d, s=8, alpha=0.5, zorder=3)
    ax.axhline(0.5, color="red", ls="--", lw=1)
    ax.text(0.98, 0.505, "50%", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=8, color="red")
    ax.set_ylabel("sparse_bit_rate (zero-bit fraction)")
    ax.set_title("(b) Weight sparse-bit rate by op", fontsize=11)
    ax.tick_params(axis="x", rotation=45)

    # (c) ideal speed-up distribution
    ax = axes[2]
    order = np.argsort(speedup)[::-1][:25]
    ax.barh(range(len(order)), speedup[order][::-1], color="tab:blue",
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=6)
    ax.set_xlabel("ideal_speed_up (×)")
    mean_su = speedup.mean()
    ax.axvline(mean_su, color="red", ls="--", lw=1)
    ax.text(mean_su + 0.02, 0.5, f"mean {mean_su:.2f}×", color="red",
            fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_title("(c) Ideal speed-up — top 25 modules", fontsize=11)

    act_note = f"activation rows: {len(act_rows)}" if act_rows else \
               "activation stats not collected in this run"
    fig.suptitle("Sparsity profile — int8+outlier quantized weights "
                 f"(224 Linear modules, 8-frame run; {act_note})", fontsize=12)
    save(fig, "sparsity_int8_outlier.png")


if __name__ == "__main__":
    plot_bitwidth_ladder()
    plot_bitwidth_per_task()
    plot_granularity()
    plot_na_curve()
    plot_phaseF()
    plot_phaseE2()
    plot_sparsity()
    print("done.")
