"""Plot LIBERO and Meta-World benchmark comparisons.

Usage:
    python scripts/plot_benchmarks.py

Outputs:
    outputs/figures/benchmark_libero.png
    outputs/figures/benchmark_metaworld.png

Data sources:
- LIBERO: paper Table 2 + four public checkpoints (suite-level eval_info.json)
- Meta-World: paper 0.45B table + outputs/metaworld/metaworld_mt50_lerobot_smolvla/eval_info.json
All figure text is in English to avoid CJK font issues.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "outputs"

# Colors
C_PAPER = "#c0392b"      # paper (dark red)
C_LEROBOT_A = "#2e86c1"  # lerobot A (blue, quantization baseline)
C_TIANTIANX = "#8e44ad"  # tiantianx C (purple)
C_K1000DAI = "#f39c12"   # k1000dai D (orange)
C_HFVLA = "#16a085"      # HuggingFaceVLA official ref (teal)
C_OURS = "#27ae60"       # our reproduction (green)

# Fallback values for results produced on another machine (hfvla ran on H100;
# its eval_info.json files are not synced to this workspace).
# Source: logs/2026-08-25 (hfvla four-suite completion summary).
FALLBACK = {
    "HuggingFaceVLA (32L/0.5)": {
        "Spatial": 65.0, "Object": 71.0, "Goal": 72.0, "Long": 37.0,
    },
}


def load_eval_pc_success(path: Path) -> float:
    """Read a lerobot eval_info.json and return overall pc_success (%)."""
    info = json.loads(path.read_text())
    per_task = info.get("per_task", [])
    if isinstance(per_task, dict):
        per_task = [{"metrics": v} for v in per_task.values()]
    n_succ = sum(sum(t["metrics"]["successes"]) for t in per_task)
    n_tot = sum(len(t["metrics"]["successes"]) for t in per_task)
    return 100.0 * n_succ / max(n_tot, 1)


def plot_libero():
    suites = ["Spatial", "Object", "Goal", "Long"]
    paper = [90.0, 96.0, 92.0, 71.0]
    paper_avg = 87.3

    base = OUT
    models = {
        "lerobot A (16L/0.75)": {
            "color": C_LEROBOT_A,
            "paths": {
                "Spatial": [base / "libero/baseline_smolvla450m_libero_spatial" / "eval_info.json",
                            base / "libero/baseline_smolvla450m_libero_spatial" / "eval" / "eval_info.json",
                            base / "libero/libero_spatial_rerun_v2" / "eval_info.json"],
                "Object": [base / "libero/lerobot_smolvla_libero_libero_object" / "eval_info.json"],
                "Goal": [base / "libero/lerobot_smolvla_libero_libero_goal" / "eval_info.json"],
                "Long": [base / "libero/lerobot_smolvla_libero_libero_10" / "eval_info.json"],
            },
        },
        "tiantianx C": {
            "color": C_TIANTIANX,
            "paths": {
                "Spatial": [base / "libero/tiantianx_smolvla_libero_libero_spatial" / "eval_info.json"],
                "Object": [base / "libero/tiantianx_smolvla_libero_libero_object" / "eval_info.json"],
                "Goal": [base / "libero/tiantianx_smolvla_libero_libero_goal" / "eval_info.json"],
                "Long": [base / "libero/tiantianx_smolvla_libero_libero_10" / "eval_info.json"],
            },
        },
        "k1000dai D": {
            "color": C_K1000DAI,
            "paths": {
                "Spatial": [base / "libero/k1000dai_ft100k_libero_spatial" / "eval_info.json"],
                "Object": [base / "libero/k1000dai_ft100k_libero_object" / "eval_info.json"],
                "Goal": [base / "libero/k1000dai_ft100k_libero_goal" / "eval_info.json"],
                "Long": [base / "libero/k1000dai_ft100k_libero_10" / "eval_info.json"],
            },
        },
        "HuggingFaceVLA (32L/0.5)": {
            "color": C_HFVLA,
            "paths": {
                "Spatial": [base / "libero/hfvla_libero_libero_spatial" / "eval_info.json"],
                "Object": [base / "libero/hfvla_libero_libero_object" / "eval_info.json"],
                "Goal": [base / "libero/hfvla_libero_libero_goal" / "eval_info.json"],
                "Long": [base / "libero/hfvla_libero_libero_10" / "eval_info.json"],
            },
        },
    }

    def load(name, paths, suite):
        for p in paths[suite]:
            if p.exists():
                return load_eval_pc_success(p)
        fb = FALLBACK.get(name, {}).get(suite)
        if fb is not None:
            print(f"  [{name}] {suite}: fallback {fb:.1f}% (eval_info.json not found locally)")
        return fb

    # Load per-model suite values, compute averages from available suites
    model_vals = {}
    for name, spec in models.items():
        vals = [load(name, spec["paths"], s) for s in suites]
        available = [v for v in vals if v is not None]
        spec["avg"] = sum(available) / len(available) if available else 0.0
        model_vals[name] = vals

    categories = suites + ["Average"]
    n_cats = len(categories)
    n_bars = 1 + len(models)
    width = 0.8 / n_bars
    xs = list(range(n_cats))

    fig, ax = plt.subplots(figsize=(13, 6))

    # Paper bars
    paper_vals = paper + [paper_avg]
    offsets = [x - (n_bars - 1) / 2 * width for x in xs]
    ax.bar(offsets, paper_vals, width, label="Paper 0.45B", color=C_PAPER,
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(paper_vals):
        ax.text(offsets[i], v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                fontsize=9, color=C_PAPER, fontweight="bold")

    # Public checkpoint bars
    for j, (name, spec) in enumerate(models.items(), start=1):
        vals = model_vals[name] + [spec["avg"]]
        offs = [x - (n_bars - 1) / 2 * width + j * width for x in xs]
        ax.bar(offs, [v or 0 for v in vals], width, label=name,
               color=spec["color"], edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(offs[i], v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8, color=spec["color"])

    ax.set_xticks(xs)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_title("LIBERO: Paper vs Public Checkpoints (4 suites + average)", fontsize=14, pad=12)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    out = OUT / "figures" / "benchmark_libero.png"
    fig.savefig(out, dpi=150)
    print(f"Generated {out}")


def plot_metaworld():
    groups = ["Easy", "Medium", "Hard", "Very Hard"]
    group_key = {"Easy": "easy", "Medium": "medium", "Hard": "hard", "Very Hard": "very_hard"}
    paper = [82.5, 41.8, 45.0, 60.0]
    paper_avg = sum(paper) / len(paper)  # 57.3, arithmetic mean as reported in paper

    info = json.loads((OUT / "metaworld" / "metaworld_mt50_lerobot_smolvla" / "eval_info.json").read_text())
    ours_by_group = {k: v["pc_success"] for k, v in info["per_group"].items()}

    # Per-task success rates for min/max annotation
    per_task = info.get("per_task", [])
    if isinstance(per_task, dict):
        per_task = [{"task_group": g, "metrics": v} for g, v in per_task.items()]
    task_sr = {}
    for t in per_task:
        g = t["task_group"]
        sr = 100.0 * sum(t["metrics"]["successes"]) / max(len(t["metrics"]["successes"]), 1)
        task_sr.setdefault(g, []).append(sr)

    ours_vals = [ours_by_group.get(group_key[g], 0) for g in groups]
    ours_avg = sum(ours_vals) / len(ours_vals)

    categories = groups + ["Average"]
    n_cats = len(categories)
    width = 0.32
    xs = list(range(n_cats))

    paper_vals = paper + [paper_avg]
    ours_all = ours_vals + [ours_avg]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar([x - width / 2 for x in xs], paper_vals, width, label="Paper 0.45B",
           color=C_PAPER, edgecolor="black", linewidth=0.5)
    ax.bar([x + width / 2 for x in xs], ours_all, width, label="Reproduction (lerobot/smolvla_metaworld)",
           color=C_OURS, edgecolor="black", linewidth=0.5)

    for i in range(n_cats):
        ax.text(i - width / 2, paper_vals[i] + 1.2, f"{paper_vals[i]:.1f}", ha="center", va="bottom",
                fontsize=10, color=C_PAPER, fontweight="bold")
        ax.text(i + width / 2, ours_all[i] + 1.2, f"{ours_all[i]:.1f}", ha="center", va="bottom",
                fontsize=10, color=C_OURS, fontweight="bold")

    # Per-group task count + min/max spread
    for i, g in enumerate(groups):
        srs = task_sr.get(group_key[g], [])
        if srs:
            ax.text(i, -6.0, f"n={len(srs)} tasks\n[min {min(srs):.0f}, max {max(srs):.0f}]",
                    ha="center", va="top", fontsize=8, color="gray")

    ax.set_xticks(xs)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Meta-World MT50: Paper vs Reproduction (by difficulty + average)", fontsize=14, pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    out = OUT / "figures" / "benchmark_metaworld.png"
    fig.savefig(out, dpi=150)
    print(f"Generated {out}")


if __name__ == "__main__":
    plot_libero()
    plot_metaworld()
