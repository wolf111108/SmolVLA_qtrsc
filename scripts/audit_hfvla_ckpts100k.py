"""审计 hfvla_ckpts100k (HuggingFaceVLA/smolvla_libero_ckpts@100k) checkpoint。

目的: 判断 0/1 smoke 失败的原因, 以及是否值得修复:
1. safetensors 关键张量形状 -> 推断 VLM hidden dim (960=500M系, 2048=2.2B系),
   expert 层数/宽度, state projection 的输入维度(有无 state 通路);
2. normalizer stats -> 看训练时 state 是几维、是否 LIBERO 8D;
3. 与 tiantianx(可正常评测)的关键差异对照。
"""

import json
from pathlib import Path

from safetensors import safe_open

MIG = Path.home() / "VLA_tcs2/checkpoints/hfvla_smolvla_libero_100k_migrated"
OUT = Path.home() / "VLA_tcs2/outputs/audit_hfvla_ckpts100k.txt"

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=== 1. model.safetensors 关键张量 ===")
with safe_open(MIG / "model.safetensors", framework="pt") as f:
    keys = list(f.keys())
    log(f"total tensors = {len(keys)}")

    # VLM hidden dim: embed_tokens / 关注 language/model embed
    for k in sorted(keys):
        if "embed_tokens" in k or "embeddings" in k:
            log(f"[embed] {k} shape={tuple(f.get_slice(k).get_shape())}")

    # state projection(存在与否决定有没有 state 通路)
    for k in sorted(keys):
        if "state" in k.lower():
            log(f"[state] {k} shape={tuple(f.get_slice(k).get_shape())}")

    # expert / vlm 层数计数
    import re
    vlm_layers = set()
    expert_layers = set()
    for k in keys:
        m = re.search(r"model\.layers\.(\d+)\.", k)
        if m:
            vlm_layers.add(int(m.group(1)))
        m = re.search(r"expert_model\.layers\.(\d+)\.", k) or re.search(
            r"action_expert\.model\.layers\.(\d+)\.", k
        )
        if m:
            expert_layers.add(int(m.group(1)))
    log(f"[layers] vlm layer idx max={max(vlm_layers) if vlm_layers else '?'} count={len(vlm_layers)}")
    log(f"[layers] expert layer idx max={max(expert_layers) if expert_layers else '?'} count={len(expert_layers)}")

log("")
log("=== 2. normalizer stats ===")
with safe_open(MIG / "policy_preprocessor_step_5_normalizer_processor.safetensors", framework="pt") as f:
    for k in sorted(f.keys()):
        shape = tuple(f.get_slice(k).get_shape())
        if "state" in k.lower() or k.endswith(".count"):
            val = ""
            if len(shape) <= 1 and shape[0] <= 40:
                val = f" value={f.get_tensor(k).tolist()}"
            log(f"{k} shape={shape}{val}")
    # image keys
    img_keys = sorted({k.rsplit(".", 1)[0] for k in f.keys() if "image" in k.lower()})
    log(f"image key groups: {img_keys}")

log("")
log("=== 3. preprocessor / postprocessor json ===")
for name in ["policy_preprocessor.json", "policy_postprocessor.json"]:
    p = MIG / name
    if p.exists():
        log(f"--- {name} ---")
        log(p.read_text()[:2000])

log("")
log("=== 4. 原始 vs migrated config diff 要点 ===")
orig = json.loads((MIG.parent / "hfvla_smolvla_libero_100k/config.json").read_text())
mig = json.loads((MIG / "config.json").read_text())
for kk in sorted(set(orig) | set(mig)):
    if orig.get(kk) != mig.get(kk):
        log(f"{kk}: orig={orig.get(kk)!r} -> migrated={mig.get(kk)!r}")

OUT.write_text("\n".join(lines))
print(f"\n[done] 结果已写入 {OUT}")
