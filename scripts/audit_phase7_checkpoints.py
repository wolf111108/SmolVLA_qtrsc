"""PHASE 7.1 — Checkpoint Provenance 自动审计。

读取 5 个 SmolVLA checkpoint 的 config.json / train_config.json 关键字段，
归档到 outputs/table2_repro_audit/07_checkpoint_provenance/configs.txt。

对应手册 PHASE 7.1；与 AUDIT_RECORD.md 中已有的手动审计互补。
"""

from huggingface_hub import hf_hub_download
import json

repos = [
    "lerobot/smolvla_base",
    "lerobot/smolvla_libero",
    "HuggingFaceVLA/smolvla_libero",
    "tiantianx/smolvla_libero",
    "k1000dai/smolvla_libero_finetune",
]

fields = [
    "vlm_model_name",
    "load_vlm_weights",
    "num_vlm_layers",
    "num_expert_layers",
    "expert_width_multiplier",
    "chunk_size",
    "n_action_steps",
    "num_steps",
    "freeze_vision_encoder",
    "train_expert_only",
    "train_state_proj",
    "attention_mode",
    "self_attn_every_n_layers",
    "pad_language_to",
    "prefix_length",
]

lines = []
def log(s=""):
    print(s)
    lines.append(s)

for repo in repos:
    log("\n" + "=" * 80)
    log(repo)

    try:
        p = hf_hub_download(repo, "config.json")
        cfg = json.load(open(p))
        for k in fields:
            log(f"{k:30s} = {cfg.get(k, '<missing>')}")
    except Exception as e:
        log(f"config ERROR: {type(e).__name__}")

    try:
        p = hf_hub_download(repo, "train_config.json")
        t = json.load(open(p))
        for k in ["steps", "batch_size", "seed", "pretrained_path"]:
            log(f"train.{k:24s} = {t.get(k, '<missing>')}")
    except Exception as e:
        log(f"train_config unavailable: {type(e).__name__}")

out = "/home/zyzhao/VLA_tcs2/outputs/table2_repro_audit/07_checkpoint_provenance/configs.txt"
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"\n[done] 已写入 {out}")
