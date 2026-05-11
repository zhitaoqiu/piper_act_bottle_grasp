# ACT 数据与部署排查流程

这组脚本用于排查 Piper 抓瓶任务里最常见的几类问题：静止帧过多、手工裁剪破坏时序对齐、ACT chunk 开环过长、checkpoint 过拟合，以及夹爪尺度不一致。

## 1. 先诊断原始数据

```bash
python3 scripts/analyze_dataset_motion.py \
  --dataset-root data/lerobot_dataset \
  --output-dir reports/motion
```

输出：

- `reports/motion/motion_report.csv`
- `reports/motion/episode_000_motion.png`
- `reports/motion/episode_000_joints.png`
- `reports/motion/episode_000_gripper.png`

重点看 `warnings`：

- `long_static_start`：开头静止太久，模型容易学到“不动”。
- `long_static_end`：结尾静止太久，模型容易学到“抬一下后保持”。
- `frame_index_not_zero` / `timestamp_not_zero` / `metadata_index_mismatch`：很可能是只裁了 parquet 或视频，没有重建 dataset，时序已经不一致。
- `jump`：state/action 存在异常跳变，可能是采集、裁剪或解码错位。

如果要额外检查图像是否低频重复，可以加：

```bash
python3 scripts/analyze_dataset_motion.py \
  --dataset-root data/lerobot_dataset \
  --output-dir reports/motion \
  --check-image-repeats
```

采集端现在默认启用 motion-start 检测：按空格后先缓存 pre-roll，等关节运动超过阈值再正式写入。需要恢复旧行为时：

```bash
python3 teleop/data_collector.py --disable-motion-start-detect
```

## 2. 重建式裁剪，不要原地删数据

```bash
python3 scripts/rebuild_trimmed_dataset.py \
  --input-root data/lerobot_dataset \
  --output-root data/lerobot_dataset_trimmed \
  --motion-threshold 0.005 \
  --preroll-frames 5 \
  --tail-frames 8
```

输出：

- 新数据集：`data/lerobot_dataset_trimmed`
- 裁剪报告：`reports/trim_report.csv`

原则：不要把开头裁得一帧不剩。保留 3-8 帧 pre-roll，让模型在部署时看到固定起点后应该开始运动。

## 3. 做 chunk size 对照训练

原始数据：

```bash
bash training/train_act_chunk10.sh
bash training/train_act_chunk20.sh
bash training/train_act_chunk40.sh
```

重建裁剪后的数据：

```bash
DATASET_ROOT=data/lerobot_dataset_trimmed \
OUTPUT_DIR=outputs/train/piper_bottle_grasp_trimmed_chunk10 \
bash training/train_act_chunk10.sh
```

小数据集优先试 `chunk10`，因为 `chunk_size=20 / n_action_steps=20` 在真机上开环太久时，容易出现只抬一下、后续预测变平的问题。

## 4. 批量评估 checkpoint

```bash
python3 scripts/eval_all_checkpoints.py \
  --train-output outputs/train/piper_bottle_grasp_chunk10 \
  --dataset-root data/lerobot_dataset \
  --episodes 5
```

输出：

- `reports/checkpoint_eval.csv`

不要只看 `last`。小数据集上早期 checkpoint 往往比最后一个更稳。

## 5. 离线 rollout 看预测曲线

```bash
python3 scripts/plot_policy_rollout_on_dataset.py \
  --checkpt outputs/train/piper_bottle_grasp_chunk10/checkpoints/020000/pretrained_model \
  --dataset-root data/lerobot_dataset \
  --episode 0 \
  --output-dir reports/rollout
```

输出：

- `reports/rollout/episode_000_rollout.png`
- `reports/rollout/episode_000_rollout.csv`

图里的竖线是 `n_action_steps` 边界。重点看预测动作是否只在前 10-20 帧有变化、后面变平，或者预测相对真实动作提前/滞后。

也可以模拟每步重新规划：

```bash
python3 scripts/plot_policy_rollout_on_dataset.py \
  --checkpt outputs/train/piper_bottle_grasp_chunk10/checkpoints/020000/pretrained_model \
  --dataset-root data/lerobot_dataset \
  --episode 0 \
  --replan-every-step
```

## 6. 真机部署时打开动作日志

```bash
python3 inference/deploy.py \
  --checkpt outputs/train/piper_bottle_grasp_chunk10/checkpoints/020000/pretrained_model \
  --debug-actions \
  --debug-every 1 \
  --replan-every-step
```

日志解读：

- `target_diff_from_last_target` 很小，说明模型后续预测停住。
- `target_diff_from_last_target` 持续变化但 `state_diff_from_last_state` 很小，优先查机械臂执行、CAN、速度、限位或使能状态。
- 每隔 `n_action_steps` 才明显变化，说明 ACT action queue/open-loop chunk 影响很大。

夹爪范围已统一为 Piper 实际的 `0.0-0.035m`，部署侧会 clamp 到这个范围。
