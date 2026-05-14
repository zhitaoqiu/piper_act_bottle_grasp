# Debug Status — 2026-05-14（最终更新）

## 根因诊断结论

### 现象
1-episode ACT overfit 失败。
5k/10k/15k teacher-forcing: pred_j2 在全轨迹上接近常数（≈0.984–0.989）。
所有 checkpoint 的 pred_j2_std = 0.00000，model_mse ≈ mean_baseline_mse。
improvement_ratio ≈ 1.0 —— 模型不比输出均值好。

### 根因链（Part 1 → Part 3 → Part 5 逐步证实）

**Part 3 — 归一化空间输出 0**
raw_norm_j2_mean: 5k=+0.010, 10k=+0.003, 15k=+0.001
raw_norm_j2_std: 全部 0.00000
模型在归一化动作空间输出 0。反归一化后：unnormalize(0) = action_mean ≈ 0.984。

**Part 5 — 模型忽略所有输入**
- State ablation: 同一 image + 不同 state (t=0 J2=0.006 vs t=150 J2=1.591) → pred 完全不变
- Image ablation: 同一 state + 不同 image (t=0 vs t=150) → pred 完全不变
- 归一化后 state 确实不同 (-1.80 vs +1.15)，但模型完全不使用

**总结：ACT decoder cross-attention collapse**
Decoder 不关注 encoder 输出（state + image features），只依赖 learned query tokens 输出 ≈0。
这是大模型 + 小数据 + dropout=0 下的一种退化。

### Part 4 — 数据没问题
训练 batch 内 state_j2 range ~1.5 rad，action_j2 也正常变化。
数据采集、存储、loader 均正常。

### Part 7 — MLP sanity check PASSED
qpos(7维) → MLP(128 hidden, 3层, 35K参数) → action(7维)
- 200 步即收敛，improvement_ratio = 0.0001
- pred_j2_std = 0.533 ≈ true_action_j2_std = 0.534
- 全轨迹误差 < 0.006 rad
- 文件: outputs/debug/qpos_mlp_1ep.pt, outputs/debug/qpos_mlp_1ep_j2.csv

**结论：数据完全没问题，问题在 ACT / LeRobot 训练管线。**

---

## Part 8 — Tiny ACT 成功打破坍缩

### Tiny ACT (chunk_size=1)
- dim_model=128, n_heads=4, n_encoder_layers=2, n_decoder_layers=2
- chunk_size=1, n_action_steps=1, bs=8, lr=3e-4
- ~11M 参数（vs Big ACT 41M）
- 训练 5K 步，loss 快速下降到 ~0.002

### 真机诊断通过
- Tiny ACT 3k checkpoint 真机 approach 诊断：
  - 成功穿过之前 Big ACT 卡住的 J2=0.45~0.55 区间
  - J2 最终推进到约 1.58，接近预抓取位置
  - 问题从"模型坍缩"变成"部署工程"

### 教训
- 模型太大（41M）+ 数据太少（1 条轨迹）= cross-attention 坍缩
- Tiny 模型（11M）+ 同样数据 = 可以学会
- 小数据场景下，模型容量必须和数据量匹配
- chunk_size=1（逐帧预测）比 chunk_size=10 更容易优化

---

## Part 9 — 部署工程：消抖、平滑、到点停止

### 部署工程迭代

| 迭代 | 参数 | 问题 | 根因 |
|------|------|------|------|
| v0 | α=0.3, MAX_DELTA=0.02, 130步 | J2 只到 0.41，走不动 | 比例限幅：所有关节按最大 delta 统一缩 |
| v1 | α=0.2, 逐关节限幅 0.03, 200步 | J2 只到 0.90，走不动 | α=0.2 平滑拖慢 closed-loop 反馈 |
| v2 | α=0.5, 逐关节限幅 0.03, 200步 | J2 到 1.58，终点抖 | wrist J4-J6 接近终点时模型预测波动 |
| v3 | α=0.5, J1-J3=0.03 / J4-J6=0.012, wrist冻结, ready stop | J2=1.549, 无抖动 | ✅ |

### 部署工程关键发现

#### 1. 比例限幅 bug
```python
# Bug: 所有关节按最大 delta 统一缩放
max_abs = max(|J1_delta|, ..., |J6_delta|)
scale = MAX_DELTA / max_abs  # J3 最大 → 所有关节被拖慢

# Fix: 逐关节独立限幅
for j in range(6):
    delta[j] = np.clip(delta[j], -max_delta[j], max_delta[j])
```

#### 2. α 平滑的闭环反馈效应
- α 不仅是输出平滑器，还参与 closed-loop 反馈循环
- α=0.5：每步走 ~0.02 rad → 模型看到明显推进 → raw 输出爬升到 1.5+ → 能到位
- α=0.2：每步只走 ~0.01 rad → 模型看到变化慢 → raw 输出爬不上去 → 永远到不了位
- 教训：平滑参数不能只看消抖效果，必须考虑对 closed-loop 行为的影响

#### 3. wrist J4-J6 小限幅 + 终点冻结
- 大臂 J1-J3 用 0.03 rad 限幅保证推进速度
- wrist J4-J6 用 0.012 rad 限幅抑制抖动
- J2 > 1.45 时冻结 J4-J6（模型在训练边界外的 wrist 预测不可靠）
- 这是"不同关节不同安全要求"的体现

#### 4. ready stop（自适应到点停止）
- J2 > 1.50 连续 5 步 → 自动停止
- 不需要预先知道精确步数
- 比固定步数更优雅，比纯模型减速检测更可靠

---

## 部署脚本 v0.7.0 功能

### 测试模式

| 模式 | 流程 | 用途 |
|------|------|------|
| Test A | approach → hold → 停止 | 验证夹爪对准 |
| Test B | approach → 闭合 → 抬起 → 回原点 | 验证抓取+抬起 |
| Test C | approach → 下探 → 停止 | 验证下探深度 |
| Test D | approach → 闭合 → 抬起 → 平移放置 → 释放 → 回原点 | 完整抓取流程 |

### 安全机制
- 逐关节独立限幅（J1-J3=0.03, J4-J6=0.012 rad）
- wrist 终点冻结（J2 > 1.45）
- ready stop（J2 > 1.50 连续 5 步）
- 夹爪全程强制打开（approach 阶段）
- stagnation 检测（20 步无进展）
- 关节极限保护
- 键盘中断 = hold 当前位置不失能

### 当前模型
- Tiny ACT 3k: `outputs/train/piper_bottle_approach_tiny_1ep/checkpoints/003000/pretrained_model`
- 1 episode approach-only 数据
- chunk_size=1, absolute action 模式

---

## 保留的 artifact

- 5k/10k/15k checkpoint: outputs/train/piper_bottle_approach_today_1ep_overfit/checkpoints/
- Debug CSV: outputs/debug/1ep_overfit_j2_debug_*_v2.csv
- MLP 模型: outputs/debug/qpos_mlp_1ep.pt
- MLP CSV: outputs/debug/qpos_mlp_1ep_j2.csv
- Tiny ACT 1ep: outputs/train/piper_bottle_approach_tiny_1ep/
- 训练脚本: training/train_act_approach_1ep.sh, training/train_act_approach_1ep_tiny.sh
- 诊断脚本: scripts/debug_1ep_overfit.py, scripts/debug_train_batch.py, scripts/train_qpos_mlp_1ep.py
- 部署脚本: inference/deploy.py
