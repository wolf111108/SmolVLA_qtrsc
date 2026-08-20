"""汇总 lerobot/smolvla_libero 四个 LIBERO suite 结果并与论文对比。

用法:
    python summarize_lerobot_vs_paper.py [--spatial-json PATH]

Spatial 数据来源(按优先级):
1. --spatial-json 指定的 eval_info.json
2. outputs/baseline_smolvla450m_libero_spatial/eval/eval_info.json (历史 81% 运行)
"""

import argparse
import json
from pathlib import Path

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

# suite -> eval_info.json 候选路径(按优先级)
CANDIDATES = {
    "libero_spatial": [
        "baseline_smolvla450m_libero_spatial/eval_info.json",
    ],
    "libero_object": ["lerobot_smolvla_libero_libero_object/eval_info.json"],
    "libero_goal": ["lerobot_smolvla_libero_libero_goal/eval_info.json"],
    "libero_10": ["lerobot_smolvla_libero_libero_10/eval_info.json"],
}

OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "lerobot_smolvla_libero_vs_paper.md"


def load_suite(base: Path, suite: str, override: str | None):
    paths = ([Path(override)] if override else []) + [base / p for p in CANDIDATES[suite]]
    for p in paths:
        if p.exists():
            info = json.loads(p.read_text())
            per_task = info.get("per_task", [])
            if isinstance(per_task, dict):
                per_task = [
                    {"task_id": k, "metrics": v} for k, v in per_task.items()
                ]
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
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial-json", default=None, help="Spatial 的 eval_info.json 路径(可选)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent / "outputs"

    rows = {}
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        r = load_suite(base, suite, args.spatial_json if suite == "libero_spatial" else None)
        rows[suite] = r

    lines = []
    lines.append("# lerobot/smolvla_libero vs 论文 Table 2 (0.45B SmolVLA)")
    lines.append("")
    lines.append(f"- 论文: SmolVLA (arXiv:2506.01844), 每 suite 10 tasks × 10 episodes")
    lines.append(f"- 本地: seed=1000, n_action_steps=1, num_steps=10, chunk_size=50")
    lines.append("")

    # 总表
    lines.append("## Suite 级对比")
    lines.append("")
    lines.append("| Suite | 论文 SR | 本地 SR | 本地 n | 差值 |")
    lines.append("|---|---:|---:|---:|---:|")
    got_any = False
    srs = []
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        r = rows[suite]
        paper = PAPER[suite]
        if r is None:
            lines.append(f"| {SUITE_LABEL[suite]} | {paper:.0f}% | (未测试) | - | - |")
        else:
            got_any = True
            srs.append(r["sr"])
            lines.append(f"| {SUITE_LABEL[suite]} | {paper:.0f}% | {r['sr']:.0f}% ({r['n_succ']}/{r['n_tot']}) | {r['n_tot']} | {r['sr'] - paper:+.0f}pp |")
    n_done = sum(1 for s in rows.values() if s)
    if got_any and n_done == 4:
        avg = sum(srs) / len(srs)
        lines.append(f"| **平均** | **{PAPER_AVG}%** | **{avg:.1f}%** | - | **{avg - PAPER_AVG:+.1f}pp** |")
    else:
        lines.append(f"| 平均 | {PAPER_AVG}% | (完成 {n_done}/4 suite 后给出) | - | - |")
    lines.append("")

    # per-task 明细
    lines.append("## Per-task 成功率 (本地)")
    lines.append("")
    header = ["task"] + [SUITE_LABEL[s].split(" ")[0] for s in rows if rows[s]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for i in range(10):
        cells = [f"task {i}"]
        for suite, r in rows.items():
            if r:
                cells.append(f"{r['task_sr'].get(f'task_{i}', float('nan')):.0f}%")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # 数据来源
    lines.append("## 数据来源")
    lines.append("")
    for suite, r in rows.items():
        if r:
            lines.append(f"- {SUITE_LABEL[suite]}: `{r['path']}`")
        else:
            lines.append(f"- {SUITE_LABEL[suite]}: 未找到结果文件")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    OUT_PATH.write_text(text)
    print(f"\n[done] 已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
