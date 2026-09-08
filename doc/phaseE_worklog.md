# Phase E 工作日志(Sensitivity 实验)

> 仓库:`/home/lfwang/VLA_tcs2`(从原机器 `/home/zyzhao/VLA_tcs2` 迁移)
> 环境:conda `smolvla_eval`,GPU: RTX PRO 6000(与另一任务共享,余量 ~64GB)

---

## 2026-09-07

### 1. 首次 Phase E 运行失败(权限错误)

- 命令:`python main.py --config configs/experiments/phaseE_sensitivity.yaml`(nohup)
- 现象:校准第一个 batch 即崩,`PermissionError: [Errno 13] Permission denied: '/home/zyzhao/VLA_tcs2'`
- 根因:代码从原机器迁移,`DUMP_TENSORS_DIR` 等默认值硬编码 `/home/zyzhao/...` 绝对路径,本机用户为 `lfwang`
- **修复**(7 个文件,全部改为从 `__file__` 推导仓库根):
  - `src/vla_tcs2/quant_linear.py` — `DUMP_TENSORS_DIR`
  - `src/vla_tcs2/quant/utils.py` — `SQNR_LOG_PATH`
  - `scripts/analyze_tensor_dump.py`、`scripts/analyze_tensor_dump_channel.py` — `--dir/--out` 默认值
  - `scripts/compare_libero_stats.py` — `OUT`
  - `scripts/audit_phase7_checkpoints.py` — `out`
  - `scripts/audit_tiantianx_normalizer.py` — `out_path`
- 验证:`DUMP_TENSORS_DIR` 解析为 `/home/lfwang/VLA_tcs2/outputs/tensor_dump`,py_compile 通过

### 2. 重跑成功但结果有 Bug

- 当晚重跑完成:libero_object 100 episodes,**SR = 93.0%**
- 运行配置:target=`expert.layer.7.qk`,方法 `gaussian_rms_output`,α=0.03
- ⚠️ 但该结果后来被判定为**无效的单 site 结论**(见 09-08 的 target 匹配 bug)

---

## 2026-09-08

### 3. 发现并修复 `_matches_target` 逻辑 bug(关键)

- 现象:E1 group sweep 日志中每个 target 都报 `matched 288 modules (expected 1)`
- 根因:`src/vla_tcs2/model_wrapper.py` 的 `_matches_target()` 在 target 只指定
  `module_id` 时,不匹配的模块不返回 `False`,而是落到底部 `return True` →
  **全部 288 个模块都被注噪**
- **影响**:
  - 首次 E1 sweep(18 组)排序无效(已备份为 `sensitivity.csv.invalid_allmodule_bug`)
  - 09-07 的 E2 闭环 93% 实际含义是「**全 288 模块同时注噪 α=3% 仍 93%**」——
    全局鲁棒性结论有效,但不是单 site sensitivity
- **修复**:`module_id`/`module_ids` 改为过滤器(不匹配 → `False`),保持与
  `component`/`layer`/`operator` 的 AND 语义
- 验证:19 个 selector 组合用例全部通过;重跑 sweep 全部 `-> 1 modules in test_forward`

### 4. E1 前向敏感度排序(有效数据,全量 288 site)

- 命令:`python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base --sweep all --out-dir outputs/sensitivity`
- 产物:`outputs/sensitivity/sensitivity.csv`(288 行,按 `max_abs_diff` 降序)、
  `outputs/sensitivity/phaseE_sweep_all.log`
- 方法:`gaussian_rms_output`,α=0.03,seed=0,单 site 注入其余 raw,测 suffix hidden 扰动

**Top 10:**

| rank | module_id | max_abs_diff | relative_norm |
|---:|---|---:|---:|
| 1 | `vlm.layers.3.mlp.down_proj` | 1.133e+00 | 3.85e-01 |
| 2 | `vlm.layers.0.mlp.down_proj` | 1.973e-01 | 4.66e-02 |
| 3 | `vlm.layers.1.self_attn.k_proj` | 1.094e-01 | 2.98e-02 |
| 4 | `vlm.layers.1.mlp.down_proj` | 1.016e-01 | 2.32e-02 |
| 5 | `vlm.layers.0.mlp.gate_proj` | 9.375e-02 | 2.32e-02 |
| 6 | `vlm.layers.2.self_attn.o_proj` | 9.375e-02 | 1.70e-02 |
| 7 | `vlm.layers.3.mlp.up_proj` | 9.375e-02 | 1.98e-02 |
| 8-10 | `vlm.layer.{1,2,3}.qk` | 9.375e-02 | ~2e-02 |

**聚合结论:**

- 组件:vlm(mean 5.40e-02)> expert(mean 4.30e-02)
- 算子:down_proj 最敏感(mean 8.28e-02),gate/qk/k/v 次之,o_proj 最平
- 层:浅层(0–3)≫ 深层;`vlm.layers.3.mlp.down_proj` 异常突出(超第 2 名 5.7 倍)
- **7 个零值 site**(架构性,非 bug):`vlm.layer(s).15` 的 q/o/gate/up/down/qk/pv。
  原因:suffix(expert denoise)只通过 KV cache 感知 VLM,layer 15(最后一层)的
  q/o/mlp 扰动只影响 prefill 最终 hidden(→ 文本 logits),不写 cache。
  已验证:α=0.3 复测仍精确 0;D2 call-count 确认这些 site 都被执行过。
- 旧默认 target `expert.layer.7.qk` 实际排名 **156/288**(中游偏不敏感)

### 5. E2 完整闭环实验矩阵(已启动,进行中)

- 工具:
  - `scripts/gen_phaseE2_configs.py` — 从模板生成 14 个 config
  - `scripts/run_phaseE2_all.sh` — 顺跑 driver(断点续跑:已有 result.json 自动跳过;
    单个失败不阻断)
- 校准:复用 `scales/phaseE_sensitivity/`(864 个 per_site scale,已完整)
- 评测:libero_object,10 episodes,seed=1000,全部 `--skip-calibration`

**矩阵(14 个实验):**

| 组 | 实验 | target | α / 方法 |
|---|---|---|---|
| 基线 | `phaseE2_baseline_raw` | (无注入,全 raw) | — |
| top-1 剂量响应 | `phaseE2_vlm3_down_a001/003/010` | `vlm.layers.3.mlp.down_proj` | 0.01 / 0.03 / 0.10 |
| top-2 剂量响应 | `phaseE2_vlm0_down_a001/003/010` | `vlm.layers.0.mlp.down_proj` | 0.01 / 0.03 / 0.10 |
| expert 侧 top | `phaseE2_exp1_up_a003/010` | `expert.layers.1.mlp.up_proj` | 0.03 / 0.10 |
| 对照 MatMul | `phaseE2_exp7qk_a003/030` | `expert.layer.7.qk` (rank 156) | 0.03 / 0.30 |
| quant_residual | `phaseE2_exp7qk_qr` | `expert.layer.7.qk` | quant_residual_output |
| VLM MatMul | `phaseE2_vlm3qk_a003/010` | `vlm.layer.3.qk` | 0.03 / 0.10 |

- 启动时间:2026-09-08 02:04(driver PID 1767126)
- 预计总时长 ~2–2.5 小时(每个实验约 8–12 分钟)
- 日志:`outputs/phaseE2_driver.log` + 每个 `outputs/phaseE2_<name>.log`
- 结果:每个 `outputs/experiments/<name>/result.json` 的 `overall.pc_success`

**进度记录(2026-09-08 05:50 全部完成,总耗时 ~3h45m):**

- [x] baseline_raw — SR=94%
- [x] vlm3_down_a001=91% / a003=92% / a010=89%
- [x] vlm0_down_a001=93% / a003=92% / a010=93%
- [x] exp1_up_a003=92% / a010=92%
- [x] exp7qk_a003=89% / a030=92% / qr=91%
- [x] vlm3qk_a003=92% / a010=92%

**结果与剂量-响应结论(libero_object,共 100 episodes,1pp ≈ 1 episode):**

| 实验 | α | SR | Δvs基线 |
|---|---|---:|---:|
| baseline_raw | — | 94% | — |
| vlm3_down(前向 rank1) | 0.01/0.03/0.10 | 91/92/89% | −3/−2/**−5** |
| vlm0_down(rank2) | 0.01/0.03/0.10 | 93/92/93% | −1/−2/−1 |
| exp1_up(expert top) | 0.03/0.10 | 92/92% | −2/−2 |
| exp7qk(rank156 对照) | 0.03/0.30 | 89/92% | −5/+2 |
| exp7qk quant_residual | — | 91% | −3 |
| vlm3qk(VLM MatMul) | 0.03/0.10 | 92/92% | −2/−2 |

- **全部 14 组落在 89–94%(最大降幅 5pp)**,与 09-07 发现的全局鲁棒性一致
- **无明确剂量-响应**:vlm3_down 随 α 增大有轻微微降(91→89),但 vlm0_down 全平;
  对照组 exp7qk 在 α=0.30 反而比 α=0.03 高 → 差异基本在 ±3pp 噪声带内
- **前向敏感度排序与闭环 SR 不相关**:最敏感 site(vlm3_down)最差也只 −5pp,
  rank156 的 exp7qk 同样出现 −5pp → `max_abs_diff` 不能预测闭环性能影响
- 结论:该策略对单 site RMS 注噪(α≤0.1)高度鲁棒,闭环 SR 对注噪 site 选择
  和剂量均不敏感;如需拉开差异需更大 α 或多 site 联合注噪
