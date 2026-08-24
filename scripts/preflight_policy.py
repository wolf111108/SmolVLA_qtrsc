"""preflight_policy.py — 解析 policy 路径并自动推导 camera rename_map。

用法:
    python preflight_policy.py <spec>

spec 两种形式:
    hub:<repo_id>                 直接使用 hub repo id
    hfsub:<repo_id>:<subfolder>   先 snapshot_download 该子目录, policy.path 用本地绝对路径
                                  (适用于一个 repo 里存多档 checkpoint 的情况)

输出(单行, 用 | 分隔):
    <policy_path>|<rename_map_json>

rename_map 从 LeRobot LIBERO EnvProcessor 输出的相机 key
    observation.images.image   (agentview)
    observation.images.image2  (eye-in-hand)
映射到 checkpoint config 里实际声明的相机 key; 完全一致时输出 {}。
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# LIBERO env 输出的相机 key
ENV_AGENT = "observation.images.image"
ENV_WRIST = "observation.images.image2"

AGENT_CANDS = [ENV_AGENT, "image", "camera1", "agentview_image"]
WRIST_CANDS = ["observation.images.wrist_image", "wrist_image", ENV_WRIST, "image2", "camera2"]


def resolve(spec: str):
    if spec.startswith("dir:"):
        path = spec[4:]
        return path, path  # 本地目录, repo 同 path
    if spec.startswith("hfsub:"):
        _, repo, sub = spec.split(":", 2)
        root = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/**"])
        return f"{root}/{sub}", repo
    if spec.startswith("hub:"):
        repo = spec[4:]
        return repo, repo
    return spec, spec  # 本地路径


def load_config(policy_path: str, repo: str):
    p = Path(policy_path)
    cfg_file = p / "config.json" if p.is_dir() else p
    if not cfg_file.exists():
        cfg_file = Path(hf_hub_download(repo_id=repo, filename="config.json"))
    return json.loads(cfg_file.read_text())


def feature_keys(cfg: dict):
    feats = cfg.get("features") or cfg.get("input_features") or {}
    if isinstance(feats, dict):
        return [k for k in feats if "image" in k or "camera" in k]
    return []


def pick(keys, cands, extra):
    for c in cands:
        if c in keys:
            return c
    for c in cands:  # 前缀匹配, 如 observation.images.wrist_image vs wrist_image
        for k in keys:
            if k.split(".")[-1] == c.split(".")[-1]:
                return k
    for k in keys:
        if extra(k):
            return k
    return None


def main():
    spec = sys.argv[1]
    policy_path, repo = resolve(spec)
    cfg = load_config(policy_path, repo)
    keys = feature_keys(cfg)

    agent = pick(keys, AGENT_CANDS, lambda k: k.endswith(".image") or "agent" in k)
    wrist = pick(keys, WRIST_CANDS, lambda k: "wrist" in k)

    rename = {}
    if agent and agent != ENV_AGENT:
        rename[ENV_AGENT] = agent
    if wrist and wrist != ENV_WRIST:
        rename[ENV_WRIST] = wrist

    print(f"{policy_path}|{json.dumps(rename)}")


if __name__ == "__main__":
    main()
