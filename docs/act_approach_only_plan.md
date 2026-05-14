# ACT Approach-Only 数据采集与部署计划

> 2026-05-14 制定。ACT 只负责 approach（start → pre_grasp → approach），不负责 close/lift/place/release/retreat/home。

## 为什么是 approach-only

全流程端到端 ACT（start → approach → close → lift → place → release → home）在实践中不可行：

- **夹爪信号弱**：ACT 输出的 gripper 维度变化极小（~0.001-0.002），不足以可靠地驱动夹爪闭合/张开
- **长轨迹坍缩**：10 阶段 pick-and-place 轨迹太长（~15-30s），ACT 的 open-loop action chunk 在后期漂移严重
- **时序耦合**：close/lift 的时机对成功率极其敏感，而 ACT 没有显式时序理解

**分工**：ACT 预测 J1-J6 的 approach 轨迹（视觉引导），代码控制夹爪 + 后续全部阶段（规则驱动）。

## 数据采集规格

### 采集内容

每条 episode 只录 **start → approach**：

| 阶段 | 描述 | 夹爪 |
|------|------|------|
| start | 示教臂在固定起点，夹爪张开，瓶子在下方 | open |
| pre_grasp | 下降至瓶子上方 ~3-5cm | open |
| approach | 到达瓶子抓取位置，夹爪仍在张开 | open |

**不录** close_gripper、lift、place 等后续阶段。

### 硬件要求

- **腕部相机**（RealSense D435i）：必须，拍摄夹爪和瓶子的近景
- **全局相机**（USB SN0002）：必须，拍摄整个工作空间的全局视角
- **示教臂 + 被控臂**：镜像模式，CAN 总线上

### 采集规范

1. **固定起点**：每次从同一个 start_pose 开始（与被控臂回放用的 waypoints 相同）
2. **瓶子位置变化**：每次改变瓶子位置和朝向（左/右/前/后 ~2-5cm，旋转 15-30°）
3. **夹爪保持张开**：整个 episode 夹爪保持 open（~0.10m），不闭合
4. **速度自然**：示教时速度均匀，不要过快或过慢
5. **episode 时长**：控制在 3-5 秒（~90-150 帧 @ 30fps）
6. **数量**：建议 50-100 条，每条覆盖不同的瓶子位置

### 采集流程

```bash
conda activate piper_act
python3 teleop/data_collector.py --global-camera auto
```

操作：
1. 按 E 使能
2. 手动拖示教臂到固定 start_pose，放好瓶子
3. 按空格开始录制
4. 执行 start → pre_grasp → approach（全程夹爪张开）
5. 按空格停止录制
6. 手动拖回 start_pose，换瓶子位置
7. 重复 3-6

### 数据集命名

```
piper_bottle_approach_v1
```

保存路径：`data/lerobot_dataset_approach_v1/`

与已有 grasp 数据集区分：
- `piper_bottle_grasp` — 旧数据，位置已变，不再使用
- `piper_bottle_grasp_delta_v2` — delta 模式实验，不再使用
- `piper_bottle_approach_v1` — **新数据，approach-only**

## 部署架构

### Hybrid 模式流程

```
ACT (视觉引导)                代码 (规则驱动)
─────────────────            ─────────────────
start_pose                   
  ↓ (ACT J1-J6)              
pre_grasp                    
  ↓ (ACT J1-J6)              
approach                     
  ↓                          gripper 保持 open
  │                          
  ├── handoff 检测 ──────────→ 夹爪 close
  │                           dwell 1.0s
  │                           lift (waypoints)
  │                           place_pre
  │                           place
  │                           release (gripper open)
  │                           retreat
  │                           home
```

### Handoff 检测逻辑

ACT 推理过程中，每步检查：
1. 当前 arm 位置与 `approach_pose` 的最大关节差 < `--approach-threshold`（默认 0.05 rad）
2. 夹爪当前值 > `--open-gripper-threshold`（默认 0.06 m，即确认夹爪仍张开）
3. 连续 N 帧满足条件 → 触发 handoff，ACT 停止，代码接管

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_approach_v1/checkpoints/last/pretrained_model \
    --track-approach-distance \
    --approach-threshold 0.05 \
    --open-gripper-threshold 0.06 \
    --stable-frames 5 \
    --max-act-steps 120 \
    --force-close-when-ready \
    --post-close-dwell 1.0 \
    --place-after-lift \
    --return-home-after-place \
    --waypoints configs/bottle_pick_place_waypoints_v1.json \
    --velocity-pct 50
```

## 与现有回放的关系

| 组件 | 用途 | 状态 |
|------|------|------|
| waypoint replay (pick_place) | 全流程规则回放，验证 waypoints 正确 | ✅ 已可用 |
| ACT approach-only | 视觉引导的 approach 阶段，替代固定的 pre_grasp 和 approach waypoints | 待采集数据 |
| 代码 close/lift/place/release/retreat/home | 规则驱动的后续阶段 | ✅ 已实现 |

## 实施步骤

### 步骤 1：确认 waypoints baseline 稳定

```bash
# 全流程 pick_place 回放，验证 10 个 waypoints 都正确
python3 scripts/quick_bottle_grasp.py \
    --waypoints configs/bottle_pick_place_waypoints_v1.json \
    --mode pick_place --step-confirm --velocity-pct 50
```

### 步骤 2：录制 approach-only 数据集

用 `teleop/data_collector.py` 采集 50-100 条 start→approach episode。

### 步骤 3：训练 ACT

```bash
bash training/train_act_approach.sh
```

### 步骤 4：Hybrid 部署验证

先用 `--dry-run` 检查 handoff 检测逻辑，再真机测试。

## 当前不做

- ❌ 让 ACT 输出夹爪动作
- ❌ 端到端训练全流程 pick-and-place
- ❌ 使用旧位置的数据集
- ❌ 改摄像头配置（当前双相机方案不变）
- ❌ 上 Diffusion Policy / SmolVLA（等 ACT approach 稳定后再考虑）
