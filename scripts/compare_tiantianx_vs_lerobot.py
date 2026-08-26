"""对比 tiantianx/smolvla_libero 与 lerobot/smolvla_libero 在四个 LIBERO suite 上的结果。

用法:
    python compare_tiantianx_vs_lerobot.py

同时给出论文 Table 2 (0.45B SmolVLA) 作为参照。

数据来源:
- tiantianx (C 类, paper-like): outputs/tiantianx_smolvla_libero_<suite>/eval_info.json
  (由 run_tiantianx_libero_suites.sh产出)
- lerobot (A 类, legacy): outputs/baseline_smolvla450m_libero_spatial/ + outputs/lerobot_smolvla_libero_libero_{object,goal,10}/
"""

import json
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# 论文 Table 2: 0.45B SmolVLA
PAPER = {
    "libero_spatial": 90.0,
    "libero_object": 96.0,
    "libero_goal": 92.0,
    "libero_10": 71.0,
}
PAPER_AVG = 87.3

SUITE_LABEL = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long (LIBERO-10)",
}

MODELS = {
    "tiantianx": {
        "label": "tiantianx/smolvla_libero",
        "tag": "C: paper-like (16/0.75, 100k, expert-only)",
        "suites": {s: f"libero/tiantianx_smolvla_libero_{s}/eval_info.json" for s in SUITES},
    },
    "lerobot": {
        "label": "lerobot/smolvla_libero",
        "tag": "A: legacy (16/0.75, ~25k, full finetune)",
        "suites": {
            "libero_spatial": "libero/baseline_smolvla450m_libero_spatial/eval_info.json",
            "libero_object": "libero/lerobot_smolvla_libero_libero_object/eval_info.json",
            "libero_goal": "libero/lerobot_smolvla_libero_libero_goal/eval_info.json",
            "libero_10": "libero/lerobot_smolvla_libero_libero_10/eval_info.json",
        },
    },
}

OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "tiantianx_vs_lerobot_libero.md"


def load_suite(base: Path, rel: str):
    p = base / rel
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    per_task = info.get("per_task", [])
    if isinstance(per_task, dict):
        per_task = [{"task_id": k, "metrics": v} for k, v in per_task.items()]
    succ = [bool(x) for t in per_task for x in t["metrics"]["successes"]]
    task_sr = {
        f"task_{t.get('task_id', '?')}": 100.0 * sum(t["metrics"]["successes"])
        / max(len(t["metrics"]["successes"]), 1)
        for t in per_task
    }
    return {
        "path": str(p),
        "n_succ": sum(succ),
        "n_tot": len(succ),
        "sr": 100.0 * sum(succ) / max(len(succ), 1),
        "task_sr": task_sr,
    }


def main():
    base = Path(__file__).resolve().parent.parent / "outputs"

    rows = {
        name: {s: load_suite(base, cfg["suites"][s]) for s in SUITES}
        for name, cfg in MODELS.items()
    }

    lines = []
    lines.append("# tiantianx/smolvla_libero vs lerobot/smolvla_libero (LIBERO 四 suite)")
    lines.append("")
    lines.append(f"- tiantianx: {MODELS['tiantianx']['tag']}")
    lines.append(f"- lerobot:   {MODELS['lerobot']['tag']}")
    lines.append(f"- 论文参照:  SmolVLA Table 2 (arXiv:2506.01844)")
    lines.append(f"- 统一协议:  seed=1000, n_action_steps=1, num_steps=10, chunk_size=50, 10 tasks × 10 episodes")
    lines.append("")

    # Suite 级对比
    lines.append("## Suite 级对比")
    lines.append("")
    lines.append("| Suite | 论文 | tiantianx (C) | lerobot (A) | C vs A | C vs 论文 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    tt_srs, lr_srs = [], []
    for s in SUITES:
        tt, lr = rows["tiantianx"][s], rows["lerobot"][s]

        def cell(r):
            return f"{r['sr']:.0f}% ({r['n_succ']}/{r['n_tot']})" if r else "(未测试)"

        def delta(a, b):
            if a is None or b is None:
                return "-"
            return f"{a['sr'] - b['sr']:+.0f}pp"

        if tt:
            tt_srs.append(tt["sr"])
        if lr:
            lr_srs.append(lr["sr"])
        lines.append(
            f"| {SUITE_LABEL[s]} | {PAPER[s]:.0f}% | {cell(tt)} | {cell(lr)} | {delta(tt, lr)} | {delta(tt, {'sr': PAPER[s]})} |"
        )
    n_tt = sum(1 for s in SUITES if rows['tiantianx'][s])
    n_lr = sum(1 for s in SUITES if rows['lerobot'][s])
    avg_tt = sum(tt_srs) / len(tt_srs) if tt_srs else None
    avg_lr = sum(lr_srs) / len(lr_srs) if lr_srs else None
    tt_avg_cell = f"{avg_tt:.1f}%" if avg_tt is not None and n_tt == 4 else f"({n_tt}/4 suite)"
    lr_avg_cell = f"{avg_lr:.1f}%" if avg_lr is not None and n_lr == 4 else f"({n_lr}/4 suite)"
    d_tt_lr = f"{avg_tt - avg_lr:+.1f}pp" if avg_tt is not None and avg_lr is not None else "-"
    d_tt_paper = f"{avg_tt - PAPER_AVG:+.1f}pp" if avg_tt is not None and n_tt == 4 else "-"
    lines.append(f"| **平均** | **{PAPER_AVG}%** | **{tt_avg_cell}** | **{lr_avg_cell}** | **{d_tt_lr}** | **{d_tt_paper}** |")
    lines.append("")

    # 每个 suite 一张 per-task 表
    lines.append("## Per-task 成功率")
    lines.append("")
    for s in SUITES:
        tt, lr = rows["tiantianx"][s], rows["lerobot"][s]
        if not tt and not lr:
            continue
        lines.append(f"### {SUITE_LABEL[s]}")
        lines.append("")
        lines.append("| task | tiantianx (C) | lerobot (A) | diff |")
        lines.append("|---|---:|---:|---:|")
        for i in range(10):
            tt_v = tt["task_sr"].get(f"task_{i}") if tt else None
            lr_v = lr["task_sr"].get(f"task_{i}") if lr else None
            tt_cell = f"{tt_v:.0f}%" if tt_v is not None else "-"
            lr_cell = f"{lr_v:.0f}%" if lr_v is not None else "-"
            d = f"{tt_v - lr_v:+.0f}pp" if (tt_v is not None and lr_v is not None) else "-"
            lines.append(f"| task {i} | {tt_cell} | {lr_cell} | {d} |")
        lines.append("")

    # 数据来源
    lines.append("## 数据来源")
    lines.append("")
    for name, cfg in MODELS.items():
        for s in SUITES:
            r = rows[name][s]
            if r:
                lines.append(f"- {cfg['label']} {SUITE_LABEL[s]}: `{r['path']}`")
            else:
                lines.append(f"- {cfg['label']} {SUITE_LABEL[s]}: 未找到结果文件")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    OUT_PATH.write_text(text)
    print(f"\n[done] 已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
