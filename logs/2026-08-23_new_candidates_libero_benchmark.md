# 2026-08-23 新候选 checkpoint LIBERO benchmark

## 今日任务
- [ ] HuggingFaceVLA/smolvla_libero_ckpts@100k 四 suite (官方 16/0.75, DEPRECIATED)
- [ ] k1000dai/smolvla_libero_finetune 四 suite (社区 16/0.75, 100k, batch64)

## 运行进程
```bash
conda activate smolvla_eval
cd ~/VLA_tcs2
nohup bash scripts/run_new_candidates_libero.sh > outputs/new_candidates_run.log 2>&1 &
```

流程: preflight(自动 rename_map) -> smoke(task0 x1) -> 四 suite x100 episodes(幂等) -> 自动对比。
smoke 失败的模型自动跳过, 不阻塞另一个。

## 结果
(跑完后由 compare_new_candidates_vs_baselines.py 生成, 见 outputs/new_candidates_vs_baselines.md)

## 结论与下一步
- 待回填: 哪个候选最接近论文 87.3%; 决定量化 baseline 最终选型。

## 备查
- 候选发现过程: HF Hub API 搜索 smolvla+libero, 核查 config 架构/train_config recipe;
  hfvla_ckpts100k 需定位子目录 100000/pretrained_model (preflight_policy.py 的 hfsub: 机制);
  两候选相机 key 均为 image/wrist_image, rename_map 由 preflight 自动推导。
