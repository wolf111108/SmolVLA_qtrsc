"""对比 tiantianx/smolvla_libero normalizer 与 HuggingFaceVLA/libero 数据集 stats。

目的（README Step 2）：
验证 tiantianx checkpoint 的归一化 stats 是否直接来自 HuggingFaceVLA/libero 数据集，
从而确认其训练数据来源与当前官方 LIBERO 输入 contract 的一致性。
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO_ROOT, "outputs", "compare_libero_stats.txt")


def find_state_stats(obj, path=""):
    """递归搜索 json 中 observation.state 相关条目。"""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if "state" in k.lower() and isinstance(v, dict):
                hits.append((p, v))
            else:
                hits.extend(find_state_stats(v, p))
    return hits


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # --- tiantianx normalizer ---
    norm_path = hf_hub_download(
        repo_id="tiantianx/smolvla_libero",
        filename="policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    norm = load_file(norm_path)
    t_mean = norm["observation.state.mean"].cpu().tolist()
    t_std = norm["observation.state.std"].cpu().tolist()
    t_min = norm["observation.state.min"].cpu().tolist()
    t_max = norm["observation.state.max"].cpu().tolist()

    log("=== tiantianx normalizer observation.state ===")
    log(f"mean = {t_mean}")
    log(f"std  = {t_std}")
    log(f"min  = {t_min}")
    log(f"max  = {t_max}")
    log()

    # --- HuggingFaceVLA/libero dataset stats ---
    repo_id = "HuggingFaceVLA/libero"
    stats_path = None
    for filename in [
        "meta/stats.json",
        "stats.json",
        "meta/info.json",
    ]:
        try:
            stats_path = hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type="dataset"
            )
            log(f"downloaded {filename} from dataset {repo_id}")
            break
        except Exception as e:
            log(f"{filename} not available: {type(e).__name__}")

    if stats_path is None:
        log("!! 无法下载 dataset stats，尝试列出 repo 文件")
        from huggingface_hub import list_repo_files
        files = [
            f
            for f in list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".json")
        ]
        log("json files in repo:")
        for f in files:
            log(f"  {f}")
        with open(OUT, "w") as f:
            f.write("\n".join(lines))
        return

    with open(stats_path) as f:
        stats = json.load(f)

    hits = find_state_stats(stats)
    if not hits:
        log("!! stats.json 中未找到 state 条目，顶层 keys:")
        log(f"   {list(stats.keys())[:20]}")
    for p, v in hits:
        log(f"=== dataset {p} ===")
        for stat_name in ["mean", "std", "min", "max"]:
            if stat_name in v:
                vals = v[stat_name]
                # 展平嵌套列表
                flat = json.dumps(vals)
                log(f"{stat_name} = {flat}")

                # 数值对比（仅当维度可对齐时）
                try:
                    tv = {"mean": t_mean, "std": t_std, "min": t_min, "max": t_max}[stat_name]
                    if isinstance(vals, list) and len(flat) < 2000:
                        import numpy as np
                        dv = np.array(vals, dtype=float).flatten()
                        tvv = np.array(tv, dtype=float)
                        if dv.shape == tvv.shape:
                            diff = np.abs(dv - tvv)
                            log(f"  -> |diff| vs tiantianx: max={diff.max():.3e} mean={diff.mean():.3e}")
                except Exception as e:
                    log(f"  (compare failed: {e})")
        log()

    with open(OUT, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[done] 结果已写入 {OUT}")


if __name__ == "__main__":
    main()
