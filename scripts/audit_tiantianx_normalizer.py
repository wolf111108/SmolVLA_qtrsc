"""审计 tiantianx/smolvla_libero 的 normalizer safetensors。

目的（README Step 1）：
确认 observation.state 的归一化 stats 实际是 6D 还是 8D，
以解释 config metadata (6D) 与 LIBERO official processor (8D) 的矛盾。
"""

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(_REPO_ROOT, "outputs"), exist_ok=True)
out_path = os.path.join(_REPO_ROOT, "outputs", "audit_tiantianx_normalizer.txt")
_orig_stdout = sys.stdout
sys.stdout = open(out_path, "w")
try:
    repo = "tiantianx/smolvla_libero"

    path = hf_hub_download(
        repo_id=repo,
        filename="policy_preprocessor_step_5_normalizer_processor.safetensors",
    )

    print("normalizer file =", path)

    d = load_file(path)

    print("\n=== ALL NORMALIZER TENSORS ===")
    for k in sorted(d.keys()):
        v = d[k]
        print(k)
        print("  shape =", tuple(v.shape))
        if v.numel() <= 64:
            print("  value =", v.cpu().tolist())

    # 重点：state 维度判断
    state_keys = [k for k in d.keys() if "state" in k.lower()]
    print("\n=== STATE SUMMARY ===")
    for k in state_keys:
        print(k, "->", tuple(d[k].shape))
finally:
    sys.stdout.close()
    sys.stdout = _orig_stdout

print(f"[done] 结果已写入 {out_path}")
