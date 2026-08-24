"""对比 HuggingFaceVLA/smolvla_libero (官方 current reference) 与论文及 A/C/D 类基线。

官方 current reference:
- HuggingFaceVLA/smolvla_libero: 32 层 / 0.5 宽 / SmolVLM2-500M-Instruct / 8D state + image/image2

基线:
- lerobot   (A 类, 16/0.75, full finetune): 69.2%
- k1000dai  (D 类, 16/0.75, 100k, batch64): 65.5%
- tiantianx (C 类, 16/0.75, 100k, expert-only): 59.5%
- 论文 Table 2 (arXiv:2506.01844, 0.45B): 87.3%

用法:
    python compare_hfvla_vs_baselines.py
"""

import json
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABEL = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}
PAPER = {"libero_spatial": 90.0, "libero_object": 96.0, "libero_goal": 92.0, "libero_10": 71.0}
PAPER_AVG = 87.3

MODELS = {
    "hfvla": {
        "label": "HuggingFaceVLA/smolvla_libero (official ref)",
        "tag": "32 层 / 0.5 宽 / 500M-Instruct / 8D state + image/image2",
        "suites": {s: f"hfvla_libero_{s}/eval_info.json" for s in SUITES},
    },
    "lerobot": {
        "label": "lerobot/smolvla_libero (A)",
        "tag": "16 层 / 0.75 宽 / full finetune ~25k",
        "suites": {
            "libero_spatial": "baseline_smolvla450m_libero_spatial/eval_info.json",
            "libero_object": "lerobot_smolvla_libero_libero_object/eval_info.json",
            "libero_goal": "lerobot_smolvla_libero_libero_goal/eval_info.json",
            "libero_10": "lerobot_smolvla_libero_libero_10/eval_info.json",
        },
    },
    "k1000dai": {
        "label": "k1000dai/smolvla_libero_finetune (D)",
        "tag": "16 层 / 0.75 宽 / 100k / batch64",
        "suites": {s: f"k1000dai_ft100k_{s}/eval_info.json" for s in SUITES},
    },
    "tiantianx": {
        "label": "tiantianx/smolvla_libero (C)",
        "tag": "16 层 / 0.75 宽 / 100k / expert-only",
        "suites": {s: f"tiantianx_smolvla_libero_{s}/eval_info.json" for s in SUITES},
    },
}

OUT_MD = Path(__file__).resolve().parent.parent / "outputs" / "hfvla_vs_baselines.md"


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
    rows = {name: {s: load_suite(base, cfg["suites"][s]) for s in SUITES} for name, cfg in MODELS.items()}
    order = ["hfvla", "lerobot", "k1000dai", "tiantianx"]

    lines = ["# HuggingFaceVLA/smolvla_libero vs 基线 (LIBERO 四 suite)", ""]
    for name in order:
        lines.append(f"- {MODELS[name]['label']}: {MODELS[name]['tag']}")
    lines.append(f"- 论文参照: SmolVLA Table 2 (arXiv:2506.01844), 平均 {PAPER_AVG}%")
    lines.append("- 统一协议: seed=1000, n_action_steps=1, num_steps=10, 10 tasks × 10 episodes, EGL(device2)")
    lines.append("")

    lines += ["## Suite 级对比", ""]
    header = ["Suite", "论文"] + [MODELS[n]["label"].split(" (")[0] for n in order]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    avgs = {n: [] for n in order}
    for s in SUITES:
        cells = [SUITE_LABEL[s], f"{PAPER[s]:.0f}%"]
        for n in order:
            r = rows[n][s]
            if r:
                avgs[n].append(r["sr"])
                cells.append(f"{r['sr']:.0f}% ({r['n_succ']}/{r['n_tot']})")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["**平均**", f"**{PAPER_AVG}%**"]
    for n in order:
        v = avgs[n]
        cells.append(f"**{sum(v)/len(v):.1f}%**" if len(v) == 4 else "(%d/4)" % len(v))
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Per-task 成功率", ""]
    for s in SUITES:
        if not any(rows[n][s] for n in order):
            continue
        lines.append(f"### {SUITE_LABEL[s]}")
        lines.append("")
        header = ["task"] + [MODELS[n]["label"].split(" (")[0] for n in order if rows[n][s]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for i in range(10):
            cells = [f"task {i}"]
            for n in order:
                if rows[n][s]:
                    cells.append(f"{rows[n][s]['task_sr'].get(f'task_{i}', float('nan')):.0f}%")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 数据来源", ""]
    for n in order:
        for s in SUITES:
            r = rows[n][s]
            if r:
                lines.append(f"- {MODELS[n]['label']} {SUITE_LABEL[s]}: `{r['path']}`")

    text = "\n".join(lines)
    print(text)
    OUT_MD.write_text(text)
    print(f"\n[done] 已写入 {OUT_MD}")


if __name__ == "__main__":
    main()
