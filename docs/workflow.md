# 完整操作流程 — Piper ACT Bottle Grasp

> 面向接手同事：从零到抓取的全流程操作指南。每一步都有明确的**前置条件**、**命令**、**预期输出**和**常见问题排查**。

---

## 系统架构

![系统架构图](fig_architecture.png)

---

## 全局前置条件

- 机械臂已上电（电源指示灯亮）
- 示教臂和被控臂在同一 CAN 总线上
- CAN 线已插入工控机
- RealSense D435i 已插入 USB 3.0 口
- SN0002 全局相机已插入 USB 口
- 工控机上已有 conda 环境 `piper_act`

```bash
# 先确认 conda 环境可激活
conda activate piper_act
which python3  # 应输出 ~/miniconda3/envs/piper_act/bin/python3
```

---

## 第一步：环境验证

### 1.1 CAN 总线配置

```bash
# 查看 CAN 设备是否存在
ip link show can0
```

预期输出：
```
3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can
```

如果 `state DOWN`：
```bash
sudo ip link set can0 up type can bitrate 1000000
```

如果 `Device "can0" does not exist`：CAN 线未接或驱动未加载，检查物理连接。

### 1.2 相机检查

```bash
conda activate piper_act
python3 teleop/data_collector.py --list-cameras
```

预期输出（示例）：
```
/dev/video0  Intel RealSense D435I
/dev/video2  USB2.0 Camera: USB Camera
/dev/video4  Intel RealSense D435I
/dev/video6  USB2.0 Camera: USB Camera
```

确认：
- 至少有一个 RealSense 设备（腕部相机）
- 至少有一个 USB Camera（全局相机）

如果缺少某个设备，检查 USB 线是否插紧。

### 1.3 硬件链路自检

```bash
conda activate piper_act
python3 test_hardware.py
```

预期输出：
```
============================================================
  Piper Hardware Test
============================================================
[1/6] Connecting to can0 ...       PASS
[2/6] Enabling motors ...           PASS
[3/6] Reading joint state ...       PASS
  j1=0.123 j2=-0.456 j3=1.789 j4=-2.012 j5=0.567 j6=-1.234 gripper=0.010
[4/6] Test move (j1 +0.05 rad) ... PASS
[5/6] Reading arm status ...        PASS
[6/6] Disabling ...                 PASS
ALL TESTS PASSED
```

### 1.4 相机画面测试

```bash
conda activate piper_act
python3 teleop/data_collector.py --camera-only --global-camera auto
```

预期：弹出两个窗口，分别显示腕部相机和全局相机画面。按 Q 退出。

如果全局相机黑屏：
```bash
# 先列出所有设备
python3 teleop/data_collector.py --list-cameras
# 逐个尝试设备号
python3 teleop/data_collector.py --camera-only --no-wrist --global-camera 0
python3 teleop/data_collector.py --camera-only --no-wrist --global-camera 2
python3 teleop/data_collector.py --camera-only --no-wrist --global-camera 6
# 找到能出画面的设备号后，记住它
```

---

## 第二步：采集示教数据

### 2.1 启动采集程序

```bash
conda activate piper_act
python3 teleop/data_collector.py --global-camera auto
```

如果上一步确定特定设备号（比如 `/dev/video6` 是全局相机）：
```bash
python3 teleop/data_collector.py --global-camera 6
```

### 2.2 键盘操作说明

| 按键 | 功能 | 说明 |
|---|---|---|
| **E** | 使能被控臂 | 启动电机，机械臂有力；每次程序启动后按一次 |
| **D** | 失能被控臂 | 关闭电机，机械臂无力（可手动拖动） |
| **空格** | 开始/停止录制 | 按一次开始，再按一次停止并保存 episode |
| **R** | 丢弃重录 | 当前录制中按 R 丢弃本条，回到待机状态 |
| **Q / ESC** | 退出程序 | 退出前会自动失能机械臂 |

### 2.3 完整采集流程

**第一步：初始化**

1. 按 **E** 使能机械臂（听到电机上电的声音，臂变硬）
2. 左手抓示教臂，右手抓瓶子，确认操作范围无遮挡

**第二步：人工回到固定起点**

1. 手动拖动示教臂到固定起点（比如瓶子正上方 10-15cm，夹爪张开）
2. 被控臂会自动镜像跟随，同时到达相同位置
3. 把瓶子放到本条 episode 的初始位置

**第三步：录制 episode**

1. 确认示教臂、被控臂、瓶子都在本条数据的起点
2. 按**空格**开始录制（终端显示 `>>> Recording EPISODE_N ...`）
3. **完整执行一次抓取**：
   - 从起点下降接近瓶子
   - 合拢夹爪抓住瓶子
   - 提起瓶子
   - 移动到目标位置
   - 松开夹爪放下瓶子
4. 按**空格**停止录制
5. 终端输出：`Episode saved. Total episodes: N, frames: M, duration: X.Xs`

**第四步：重复**

1. 手动把示教臂和被控臂回到同一个固定起点
2. 调整瓶子位置和角度
3. 按**空格**录制下一条
4. 重复 50-100 次

**第五步：变化数据**

每次抓取稍微改变：
- 瓶子的初始位置（左/右/前/后移动 2-5cm）
- 瓶子的朝向（旋转 15-30°）
- 抓取高度（稍高一点/稍低一点）
- 移动速度（稍快/稍慢）

> **关键**：数据多样性直接决定模型泛化能力。如果 50 条都一模一样，模型只能复现这一种抓取。

### 2.4 采集中的异常处理

| 情况 | 处理 |
|---|---|
| 录制中操作失误 | 按 **R** 丢弃当前 episode，回到待机重新录 |
| 机械臂碰撞 | 立刻按 **D** 失能，检查有无损伤后重开程序 |
| 相机画面卡住 | 退出重进，检查 USB 线，必要时重启程序 |
| 被控臂位置和示教臂不对齐 | 手动拖动示教臂到被控臂当前位置，等它们同步 |
| 夹爪合不上/打不开 | 检查示教臂夹爪机械结构是否有异物卡住 |

### 2.5 验证采集数据

```bash
conda activate piper_act
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset('piper/bottle_grasp', root='data/lerobot_dataset')
print(f'Episodes: {d.num_episodes}')
print(f'Total frames: {len(d)}')
print(f'Features: {list(d.meta.features.keys())}')
"
```

预期输出：
```
Episodes: 20 (或你采集的数量)
Total frames: XXXX
Features: ['observation.state', 'action', 'observation.images.wrist_rgb', 'observation.images.global_rgb', ...]
```

---

## 第三步：训练 ACT 模型

### 3.1 训练前检查

```bash
# 确认 GPU 可用
nvidia-smi

# 确认数据集存在
ls data/lerobot_dataset/meta/info.json

# 确认 lerobot patch 已打
grep 'ft.get("names")' ~/third_party/lerobot/src/lerobot/utils/feature_utils.py
```

最后一条命令应该有输出（之前打过的 bug fix），如果没有，需要手动修复：
```bash
sed -i 's/names = ft\["names"\]/names = ft.get("names")/' ~/third_party/lerobot/src/lerobot/utils/feature_utils.py
sed -i 's/if names\[2\] in/if names and names[2] in/' ~/third_party/lerobot/src/lerobot/utils/feature_utils.py
```

### 3.2 启动训练

**前台（终端开着）**：
```bash
conda activate piper_act
bash training/train.sh
```

**后台（关终端也不中断）**：
```bash
conda activate piper_act
nohup bash training/train.sh > /tmp/train_piper_act.log 2>&1 &
echo "PID: $!"  # 记下 PID，用于后续管理
```

### 3.3 监控训练

**查看进度：**
```bash
tail -f /tmp/train_piper_act.log
```

每 200 步输出一行，格式如下：
```
step:10K smpl:80K ep:168 epch:8.42 loss:0.099 grdn:4.019 lr:1.0e-04 updt_s:0.294 data_s:0.103
```

| 字段 | 含义 | 正常范围 |
|---|---|---|
| `step` | 当前训练步数 | 0 → 100,000 |
| `smpl` | 已处理的样本数 | 持续增长 |
| `ep` | 已遍历的 episode 数 | 持续增长 |
| `epch` | epoch 数 | 持续增长 |
| `loss` | L2 重建损失 | 初始 ~5，持续下降，好模型 <0.05 |
| `grdn` | 梯度范数 | 初始 ~120，持续下降，稳定后 <5 |
| `lr` | 学习率 | 当前 1e-4 |
| `updt_s` | 每步更新耗时(秒) | ~0.29s (GPU 计算) |
| `data_s` | 每步数据加载耗时(秒) | ~0.06s |

**判断训练是否正常：**
- ✅ loss 持续下降
- ✅ grad norm 持续下降
- ✅ updt_s 稳定在 ~0.3s
- ❌ loss 突然爆炸（grad 爆炸）→ 降低 lr
- ❌ loss 长时间不降 → 数据可能有问题
- ❌ 进程消失 → 检查 `dmesg | tail` 是否 OOM

### 3.4 Checkpoint 管理

Checkpoint 自动保存在：
```
outputs/train/piper_bottle_grasp/checkpoints/
├── 020000/            # 第 20,000 步
│   ├── pretrained_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── policy_preprocessor*.json/safetensors
│   │   └── policy_postprocessor*.json/safetensors
│   └── training_state/
├── 040000/            # 第 40,000 步
├── 060000/            # 第 60,000 步
├── 080000/            # 第 80,000 步
├── 100000/            # 最终
└── last -> 100000/    # 软链接 → 最新 checkpoint
```

**只用 `last` 路径**：`outputs/.../checkpoints/last/pretrained_model` 始终指向最新。

### 3.5 中断与恢复

**停止训练：**
```bash
kill <PID>  # 正常终止，会保存当前 checkpoint
```

**从中断处恢复：**
```bash
# 修改 training/train.sh，加入 --resume=true
python -m lerobot.scripts.lerobot_train \
    ...
    --resume=true \
    --output_dir=outputs/train/piper_bottle_grasp
```

> 注意：恢复时会自动加载最近 checkpoint 的训练状态（optimizer、scheduler、step），从断点继续。

---

## 第四步：模型评估

### 4.1 离线评估（不需要机械臂）

```bash
conda activate piper_act
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model \
    --episodes 3
```

预期输出：
```
==================================================
  Mean MSE across 3 episodes: 0.013674
  Per-joint MSE:
    j1: 0.004637
    j2: 0.028021
    j3: 0.023866
    j4: 0.013897
    j5: 0.016430
    j6: 0.008783
    gripper: 0.000083
==================================================
```

### 4.2 评估结果解读

| MSE 范围 | 模型质量 | 建议 |
|---|---|---|
| < 0.001 | 优秀 | 可以直接部署 |
| 0.001 - 0.01 | 良好 | 可以试部署，注意观察 |
| 0.01 - 0.05 | 一般 | 可以部署，但动作可能不够精准 |
| > 0.05 | 差 | 检查数据质量或模型是否过拟合 |

**关节维度分析：**
- `j1-j6` 是大关节（旋转），误差通常 0.001-0.03，其中 j2/j3 作为主力关节误差最大是正常的
- `gripper` 是夹爪（直线），误差通常 <0.001——如果这个值也很高，数据可能有问题

**过拟合识别：**
train loss 下降但验证 MSE 上升 = 过拟合。典型表现为早期 checkpoint（如 20K）MSE 反而比最终模型（如 100K）更好。本项目实测数据：

| Checkpoint | Train Loss | 验证 MSE |
|---|---|---|
| 20,000 步 | 0.099 | **0.015** ✅ |
| 100,000 步 | 0.037 | 0.145 ❌ |

> 数据越少、模型越大、训练越久，过拟合风险越高。本项目 20 集数据 + 54M 参数，20K 步之后就开始过拟合。

### 4.3 比较不同 checkpoint

```bash
# 比较 20K vs 40K checkpoint
python3 inference/eval.py --checkpt .../checkpoints/020000/pretrained_model --episodes 3
python3 inference/eval.py --checkpt .../checkpoints/040000/pretrained_model --episodes 3
```

选出 MSE 最低的那个用于部署。

---

## 第五步：真机部署

### 5.1 安全准备

部署前必须确认以下事项，机械臂执行的是模型预测的动作，没有人类实时监督：

1. **机械臂工作空间清空**：轨迹范围内没有障碍物、没有人的手
2. **紧急停止准备**：知道如何快速按 Q 退出或按 D 失能
3. **瓶子放在安全位置**：初始位置在采集数据时瓶子出现的范围内
4. **人站在安全距离**：手不伸入机械臂工作空间

### 5.2 启动部署

```bash
conda activate piper_act
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

如果需要指定全局相机：
```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model \
    --global-camera 6
```

如果不用全局相机：
```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model \
    --no-global
```

调整执行速度（50 = 更快，20 = 更慢更安全）：
```bash
python3 inference/deploy.py \
    --checkpt .../checkpoints/last/pretrained_model \
    --velocity-pct 30
```

### 5.3 部署操作流程

程序启动后，终端输出：
```
============================================================
  Piper ACT Deployment — Bottle Grasp (v0.5.2)
============================================================
  Device: cuda

[1/4] Loading ACT policy from outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model ...
  Policy loaded (chunk_size=20, n_action_steps=20).

[2/4] Building normalizers from dataset stats ...
  Normalizers ready.

[3/4] Connecting Piper (can0) ...
  Robot connected and enabled.

[4/4] Initializing cameras ...
  RealSense ready (640x480 @ 30fps, depth=False)
  USB camera /dev/video6 ready (640x480 @ 30.0fps, backend=V4L2)
  Cameras ready.

----------------------------------------------------------
  SPACE = run grasp    Q/ESC = quit
----------------------------------------------------------
```

弹出两个画面窗口：
- `ACT Deployment - Wrist`：腕部相机
- `ACT Deployment - Global`：全局相机

操作：
1. 把瓶子放到起始位置（和采集数据时一致的区域）
2. 确认机械臂在安全位置
3. 按**空格**执行一次完整抓取
4. 观察机械臂动作是否流畅、是否成功抓取
5. 机械臂停止后，把瓶子放回，调整位置
6. 再次按**空格**测试

每次抓取的轨迹：
```
  >>> Grasp attempt ...
  Trajectory complete.
```

### 5.4 部署中的异常处理

| 情况 | 处理 |
|---|---|
| 机械臂不动 | 检查 CAN 线：`ip link show can0`，确认 state UP |
| 模型加载失败 | 检查 checkpoint 路径是否正确，文件是否完整 |
| 画面不显示 | 检查相机连接，用 `--camera-only` 模式先确认相机可用 |
| 机械臂动作异常 | 立刻按 Q 退出，降低 `--velocity-pct` 重试 |
| 抓取失败 | 检查瓶子位置是否在训练数据覆盖范围内 |
| GPU OOM | 用 `--device cpu` 跑 CPU 推理（会慢但不会 OOM） |

### 5.5 退出

按 **Q** 或 **ESC**。程序会自动失能机械臂并关闭相机。

---

## 第六步：迭代优化

首次部署后通常需要以下迭代：

### 模型效果不好怎么办？

| 问题 | 可能原因 | 解决 |
|---|---|---|
| 抓不到瓶子 | 瓶子位置没在训练数据中 | 补采该位置的数据 |
| 动作太慢/太快 | velocity_pct 不匹配 | 调整 `--velocity-pct` |
| 夹爪时机不对 | 数据中夹爪状态变化不够明确 | 采集时让夹爪动作更干脆 |
| 轨迹抖动 | 模型质量不够 | 继续训练更多步数或增加数据量 |
| 每次都一样 | 模型欠拟合 | 增加训练步数、降低学习率 |

### 持续改进流程

```
数据采集 → 训练 → 评估 → 部署测试 → 发现问题 → 补采数据 → 重新训练 → ...
   ↑                                                              ↓
   └────────────────── 反馈循环 ──────────────────────────────────┘
```

每次迭代：
1. 部署测试，记录哪些情况成功、哪些失败
2. 针对失败情况补采 5-10 条新数据
3. 用新数据重新训练（或 fine-tune 已有 checkpoint）
4. 再次部署测试

---

## 附录 A：快速命令索引

```bash
# === 环境检查 ===
conda activate piper_act
nvidia-smi                              # GPU 状态
ip link show can0                       # CAN 状态

# === CAN 配置 ===
sudo ip link set can0 up type can bitrate 1000000

# === 硬件验证 ===
python3 test_hardware.py                # 机械臂自检
python3 teleop/data_collector.py --list-cameras  # 列出相机
python3 teleop/data_collector.py --camera-only --global-camera auto  # 测试相机

# === 数据采集 ===
python3 teleop/data_collector.py --global-camera auto

# === 训练 ===
nohup bash training/train.sh > /tmp/train_piper_act.log 2>&1 &   # 启动训练
tail -f /tmp/train_piper_act.log          # 监控进度
tr '\r' '\n' < /tmp/train_piper_act.log | grep "INFO.*loss:" | tail -5  # 查看 loss

# === 评估 ===
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model \
    --episodes 5

# === 部署 ===
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

## 附录 B：典型时间预估

| 操作 | 时间 | 备注 |
|---|---|---|
| 环境搭建（首次） | 30-60 min | 取决于网络速度 |
| 硬件验证 | 5 min | 如果全部 PASS |
| 采集 1 条 episode | 10-30 s | 取决于抓取动作时长 |
| 采集 50 条 | 30-60 min | 包含重置和休息 |
| 训练 100K 步（RTX 3060） | 8-10 h | 后台运行 |
| 产生第一个 checkpoint | ~2 h | 20K 步 |
| 离线评估 3 episodes | 2-3 min | GPU |
| 真机部署 1 次抓取 | 3-5 s | 取决于 chunk_size |
