"""对比两个新候选 checkpoint 与 A/C 类基线在四个 LIBERO suite 上的结果, 并写当日日志。

新候选:
- hfvla_ckpts100k: HuggingFaceVLA/smolvla_libero_ckpts 的 100000/pretrained_model (官方 16/0.75)
- k1000dai_ft100k: k1000dai/smolvla_libero_finetune (社区 16/0.75, 100k, batch64)

基线:
- lerobot   (A 类): outputs/baseline_smolvla450m_libero_spatial + lerobot_smolvla_libero_libero_{object,goal,10}
- tiantianx (C 类): outputs/tiantianx_smolvla_libero_<suite>

用法:
    python compare_new_candidates_vs_baselines.py
"""

import json
from datetime import date
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
    "hfvla_ckpts100k": {
        "label": "HuggingFaceVLA/smolvla_libero_ckpts@100k",
        "tag": "官方 16/0.75, pi/libero 数据, DEPRECIATED 但 recipe 贴论文",
        "suites": {s: f"hfvla_ckpts100k_{s}/eval_info.json" for s in SUITES},
    },
    "k1000dai_ft100k": {
        "label": "k1000dai/smolvla_libero_finetune",
        "tag": "社区 16/0.75, 100k steps, batch64, from smolvla_base",
        "suites": {s: f"k1000dai_ft100k_{s}/eval_info.json" for s in SUITES},
    },
    "tiantianx": {
        "label": "tiantianx/smolvla_libero (C)",
        "tag": "paper-like 社区复现, 100k, expert-only",
        "suites": {s: f"tiantianx_smolvla_libero_{s}/eval_info.json" for s in SUITES},
    },
    "lerobot": {
        "label": "lerobot/smolvla_libero (A)",
        "tag": "legacy 社区, ~25k, full finetune",
        "suites": {
            "libero_spatial": "baseline_smolvla450m_libero_spatial/eval_info.json",
            "libero_object": "lerobot_smolvla_libero_libero_object/eval_info.json",
            "libero_goal": "lerobot_smolvla_libero_libero_goal/eval_info.json",
            "libero_10": "lerobot_smolvla_libero_libero_10/eval_info.json",
        },
    },
}

OUT_MD = Path(__file__).resolve().parent.parent / "outputs" / "new_candidates_vs_baselines.md"
OUT_LOG = Path(__file__).resolve().parent.parent / "logs" / f"{date.today().isoformat()}_new_candidates_libero_benchmark.md"


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
    order = ["hfvla_ckpts100k", "k1000dai_ft100k", "tiantianx", "lerobot"]

    lines = ["# 新候选 checkpoint vs 基线 (LIBERO 四 suite)", ""]
    for name in order:
        lines.append(f"- {MODELS[name]['label']}: {MODELS[name]['tag']}")
    lines.append(f"- 论文参照: SmolVLA Table 2 (arXiv:2506.01844), 平均 {PAPER_AVG}%")
    lines.append("- 统一协议: seed=1000, n_action_steps=1, num_steps=10, 10 tasks × 10 episodes, EGL(device2)")
    lines.append("")

    lines += ["## Suite 级对比", ""]
    header = ["Suite", "论文"] + [MODELS[n]["label"].split(" (")[0].split("@")[0] for n in order]
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
        header = ["task"] + [MODELS[n]["label"].split(" (")[0].split("@")[0] for n in order if rows[n][s]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for i in range(10):
            cells = [f"task {i}"]
            for n in order:
                r = rows[n][s]
                if r:
                    v = r["task_sr"].get(f"task_{i}")
                    cells.append(f"{v:.0f}%" if v is not None else "-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 数据来源", ""]
    for n in order:
        for s in SUITES:
            r = rows[n][s]
            lines.append(f"- {MODELS[n]['label']} {SUITE_LABEL[s]}: `{r['path']}`" if r else
                         f"- {MODELS[n]['label']} {SUITE_LABEL[s]}: 未找到结果文件")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    OUT_MD.write_text(text)
    print(f"\n[done] 对比报告 -> {OUT_MD}")

    # 写当日工作日志(存在则跳过: 只在首次生成)
    if not OUT_LOG.exists():
        log_lines = [
            f"# {date.today().isoformat()} 新候选 checkpoint LIBERO benchmark",
            "",
            "## 今日任务",
            "- [ ] HuggingFaceVLA/smolvla_libero_ckpts@100k 四 suite (官方 16/0.75, DEPRECIATED)",
            "- [ ] k1000dai/smolvla_libero_finetune 四 suite (社区 16/0.75, 100k, batch64)",
            "",
            "## 运行进程",
            "```bash",
            "conda activate smolvla_eval",
            "cd ~/VLA_tcs2",
            "nohup bash scripts/run_new_candidates_libero.sh > outputs/new_candidates_run.log 2>&1 &",
            "```",
            "",
            "流程: preflight(自动 rename_map) -> smoke(task0 x1) -> 四 suite x100 episodes(幂等) -> 自动对比。",
            "smoke 失败的模型自动跳过, 不阻塞另一个。",
            "",
            "## 结果",
            "(跑完后由 compare_new_candidates_vs_baselines.py 生成, 见 outputs/new_candidates_vs_baselines.md)",
            "",
            "## 结论与下一步",
            "- 待回填: 哪个候选最接近论文 87.3%; 决定量化 baseline 最终选型。",
            "",
            "## 备查",
            "- 候选发现过程: HF Hub API 搜索 smolvla+libero, 核查 config 架构/train_config recipe;",
            "  hfvla_ckpts100k 需定位子目录 100000/pretrained_model (preflight_policy.py 的 hfsub: 机制);",
            "  两候选相机 key 均为 image/wrist_image, rename_map 由 preflight 自动推导。",
            "",
        ]
        OUT_LOG.write_text("\n".join(log_lines))
        print(f"[done] 当日日志 -> {OUT_LOG}")


if __name__ == "__main__":
    main()
