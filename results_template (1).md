# SmolVLA Sparsity Ratio Quick Scan — Results

| Config | Stage | Runtime element sparsity | Runtime bit sparsity | Weight element sparsity | Weight bit sparsity |
|---|---|---:|---:|---:|---:|
| INT8 | VLM prefill | | | | |
| INT8 | Expert denoise | | | | |
| INT16 | VLM prefill | | | | |
| INT16 | Expert denoise | | | | |
| FP8W4 PoT | VLM prefill | | | | |
| FP8W4 PoT | Expert denoise | | | | |
| FP8 PoT | VLM prefill | | | | |
| FP8 PoT | Expert denoise | | | | |

说明：INT bit sparsity 是 sign-aware sparse-bit；FP8 是 E4M3 4-bit significand zero-bit。二者不是同一编码定义。FP8W4 PoT 当前严格语义为 A/O PoT、W4 weight scale continuous。
