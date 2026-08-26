# 可用 Checkpoint 总结：模型结构、baseline 成绩与论文对比

> 生成时间：2026-08-24
> 统一评测协议：LeRobot 0.6.2 (commit 6adf515) · seed=1000 · n_action_steps=1 · num_steps=10 · chunk_size=50 · 10 tasks × 10 episodes/suite · H100 + EGL(device2)
> 论文参照：SmolVLA (arXiv:2506.01844) Table 2，0.45B 模型，平均 87.3%

---

## 1. 论文 0.45B 模型规格（参照系）

| 项 | 论文值 |
|---|---|
| VLM backbone | SmolVLM2-500M-**Video**-Instruct |
| VLM 层数 | 16（截取） |
| Action Expert | 32 层，hidden = 0.75 × VLM hidden（≈720） |
| Action Expert 参数 | ~100M，总参数 ~450M |
| chunk_size / flow steps | 50 / 10 |
| Fine-tune recipe | 100k steps · batch 64 · VLM frozen · expert-only · BF16 |
| 输入 | 8D state（eef xyz + axis-angle + gripper qpos×2）+ 2 相机（agentview + wrist） |

---

## 2. 三个可用 checkpoint 结构对照

| 结构项 | 论文 0.45B | tiantianx/smolvla_libero (C) | k1000dai/smolvla_libero_finetune (D) | lerobot/smolvla_libero (A) |
|---|---|---|---|---|
| VLM backbone | 500M-Video | ✅ 500M-Video | ✅ 500M-Video（from smolvla_base） | ✅ 500M-Video |
| VLM 层数 | 16 | ✅ 16 | ✅ 16 | ✅ 16 |
| Expert 宽度 | 0.75 | ✅ 0.75 | ✅ 0.75 | ✅ 0.75 |
| Fine-tune steps | 100k | ✅ 100k | ✅ 100k | ❌ ~25k |
| Batch size | 64 | 32（可能 2×GPU=64，未证实） | ✅ 64 | ~32 |
| VLM frozen / expert-only | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ❌ / ❌（full finetune） |
| State | 8D | ✅ 8D（normalizer 已逐位验证 = HuggingFaceVLA/libero） | 8D | 8D |
| 相机 | 2 | ✅ 2（config 的 camera3 为 stale） | 2（wrist_image 命名，需 rename_map） | 2（camera1/2 命名，需 rename_map） |
| config stale 问题 | — | 6D state + camera1/2/3 均 stale（无害） | — | — |
| 与论文 recipe 吻合度 | — | **最高**（6/7 项一致） | 高（6.5/7） | 低（3/7） |

> 结构审计依据：`outputs/audit_tiantianx_normalizer.txt`、`outputs/compare_libero_stats.txt`、
> 各 checkpoint 的 config.json / train_config.json（revision 见 README handoff §22）。

---

## 3. LIBERO 四 suite 成绩对比

| Suite | 论文 | tiantianx (C) | k1000dai (D) | lerobot (A) | 最佳差距 vs 论文 |
|---|---:|---:|---:|---:|---:|
| Spatial | 90% | 76% | 64% | **81%** | -9pp |
| Object | 96% | 59% | **82%** | 60% | -14pp |
| Goal | 92% | 63% | 70% | **77%** | -15pp |
| Long | 71% | 40% | 46% | **59%** | -12pp |
| **平均** | **87.3%** | 59.5% | 65.5% | **69.2%** | **-18.1pp** |

关键观察：

1. **没有任何公开 checkpoint 接近论文 87.3%**——最好的（A 类 69.2%）仍差 18pp；
2. 各 suite 最佳成绩分属不同模型（Object 属 D，其余属 A），说明差异是**系统性训练质量**问题而非单任务偏科；
3. recipe 最贴论文的 C 类（tiantianx）反而最低（59.5%）→ 论文数字依赖未公开的细节（环境版本 / 具体 checkpoint / 训练细节），社区按论文 recipe 复现不出；
4. 曾验证的 HuggingFaceVLA/smolvla_libero（32 层/0.5 宽，已换架构）Task5=9/10，证明 87%+ 的成绩在当前 simulator/pipeline 上**可达**，瓶颈在 checkpoint 而非环境。

---

## 4. 各 checkpoint 角色定位与用途

| Checkpoint | 分类 | 建议用途 |
|---|---|---|
| `tiantianx/smolvla_libero` | paper-like recipe 复现（C） | 论文复现性研究引用；其 8D normalizer 与官方数据集逐位一致，可作为 input contract 参照 |
| `k1000dai/smolvla_libero_finetune` | batch64 全量复现（D） | 说明 batch/数据细节对结果影响大（Object 82% 最高但其余低） |
| `lerobot/smolvla_libero` | legacy 全参数微调（A） | **量化 baseline 首选**：绝对成绩最高（69.2%）、成绩全面、架构仍是论文同款 16/0.75 |

废弃线（已审计关闭）：

- `HuggingFaceVLA/smolvla_libero_ckpts@100k`：实测为 **2.2B-Instruct** backbone（embed 49280×2048）+ pi 风格无 state 数据训练 + DEPRECIATED，非论文结构，判定不值得修（审计：`outputs/audit_hfvla_ckpts100k.txt`）；
- `HuggingFaceVLA/smolvla_libero`：32 层/0.5 宽新架构，仅作 simulator 正确性 reference；
- `jadechoghari/*`、`Alkatt/*`：config 不兼容或优先级低，未测。

---

## 5. 对量化工作的结论

1. **baseline 选型建议 `lerobot/smolvla_libero`（A 类，69.2%）**：
   - 成绩最高且四 suite 齐全，量化前后 SR 对比统计功效最好；
   - 架构与论文同款 16/0.75（量化研究目标针对结构，不针对 recipe）；
   - 注意它是 full-finetune 而非 expert-only，报告时需注明。
2. 论文 87.3% 不可作为量化对照基准——**所有量化退化必须相对自测 baseline 报告**（A 类 69.2%），并以"C 类 recipe 复现也仅 59.5%"作为论文数字不可达的论据。
3. 量化结果表的规范：每行绑定 checkpoint + revision + seed + 量化配置 + 四 suite SR + 平均 SR 退化（pp）。

---

## 数据来源

- `outputs/lerobot_smolvla_libero_vs_paper.md`（A 类四 suite）
- `outputs/tiantianx_vs_lerobot_libero.md`（C 类四 suite）
- `outputs/k1000dai_ft100k_libero_{spatial,object,goal,10}/eval_info.json`（D 类四 suite）
- `outputs/new_candidates_run.log`（D 类汇总 + hfvla smoke 失败记录）
- `outputs/audit_hfvla_ckpts100k.txt`（hfvla 结构审计）
- `outputs/audit_tiantianx_normalizer.txt`、`outputs/compare_libero_stats.txt`（C 类 contract 审计）
