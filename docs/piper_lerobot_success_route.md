# Piper + LeRobot Bottle Grasp — 成功路线

> 2026-05-13 更新：waypoint replay 已成功。

## 当前分工

| 组件 | 负责 | 状态 |
|------|------|------|
| **waypoint replay** | 全流程抓取（start→pre_grasp→approach→close→lift→home） | ✅ 连续成功 |
| **ACT 模型** | 仅负责 approach（start→pre_grasp→approach），不控制夹爪 | 待验证 |
| **代码** | close gripper + lift + 返回原点 | ✅ 已实现 |
| **Diffusion Policy** | 中期替代 ACT，端到端 approach | 等 baseline 稳定后 |

## 短期路线（现在）

### 步骤 1：保存成功 baseline

```bash
python3 scripts/save_success_waypoints.py \
    --input configs/bottle_grasp_waypoints_today.json
```

### 步骤 2：验证 replay baseline（连续 5 次成功）

```bash
# 先 step-confirm
python3 scripts/quick_bottle_grasp.py \
    --waypoints configs/bottle_grasp_waypoints_success_v1.json \
    --step-confirm --velocity-pct 50 --log-result

# 再自动运行
python3 scripts/quick_bottle_grasp.py \
    --waypoints configs/bottle_grasp_waypoints_success_v1.json \
    --velocity-pct 50 --hold-seconds 2 --log-result
```

目标：同一瓶子位置、同一 start pose，连续 5 次抓取成功。

### 步骤 3：微调抓取点（不重采整条轨迹）

```bash
python3 scripts/quick_bottle_grasp.py \
    --waypoints configs/bottle_grasp_waypoints_success_v1.json \
    --tune-grasp --step-confirm --velocity-pct 50
```

键盘控制：q/a J1±, w/s J2±, e/d J3±, r/f J4±, t/g J5±, y/h J6±, o/c 开关夹爪, A/C/L 保存姿态, x 退出。

### 步骤 4：Hybrid 模式（ACT approach + 代码 close/lift）

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp_delta_v2/checkpoints/030000/pretrained_model \
    --action-mode delta \
    --delta-scale 5.0 --arm-scale 5.0 --wrist-scale 8.0 --gripper-scale 20.0 \
    --hybrid-force-grasp \
    --force-close-after-step 70 --force-close-target 0.0 \
    --lift-after-close \
    --waypoints configs/bottle_grasp_waypoints_success_v1.json \
    --replan-every-step --hz 30 --max-steps 200 \
    --debug-actions --debug-every 10
```

关键点：
- ACT 只负责 arm approach（J1-J6），gripper 由代码在每步设为 open_gripper
- `--force-close-after-step 70` 之后，代码强制 target[6] = 0.0
- `--lift-after-close` 执行代码控制的 lift

## 中期路线（等 baseline 稳定后）

1. **清洗成功 replay 数据** — 记录多次成功抓取的 episode
2. **采集高质量成功 episode** — 用 LeRobot 格式记录
3. **训练 LeRobot Diffusion Policy** — `training/train_diffusion_policy.sh`
4. **部署 Diffusion Policy** — 替代 ACT，端到端 approach

## 当前不建议做的

- ❌ 全流程裸跑 ACT（夹爪输出极小，已在历史中验证）
- ❌ SmolVLA / Pi0.5（社区路线未到这一步）
- ❌ 重新采集整条轨迹（waypoint replay 已可用）
- ❌ 改摄像头配置（当前无摄像头也能 replay）
- ❌ 让 ACT 控制 close gripper 或 lift

## Gripper 关键认知

- **单位**：Piper SDK gripper 使用米（0.00=关, 0.10=最大开度）
- **瓶子宽度** ~0.05m，夹爪闭合时物理上限就是 ~0.05，不会到 0.0
- **warnings 检测**：用夹爪变化率归零（plateau）而非绝对值 → 正确检测到"碰到瓶子"
- **回放时** close_gripper/lift 使用录到的实际夹爪值（~0.05），不覆盖为 0.0

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/record_waypoint_trajectory.py` | 连续录制轨迹，自动提取 waypoints |
| `scripts/quick_bottle_grasp.py` | Waypoint 回放抓取（支持 --tune-grasp, --log-result） |
| `scripts/save_success_waypoints.py` | 版本化保存成功 waypoints |
| `scripts/test_gripper_control.py` | 测试夹爪 0.0~0.10 控制范围 |
| `scripts/check_piper_lerobot_alignment.py` | 检查 robot ↔ dataset gripper 对齐 |
| `inference/deploy.py` | ACT 部署（支持 --hybrid-force-grasp） |
| `inference/deploy_diffusion.py` | Diffusion Policy 部署 |
