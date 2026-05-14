# Piper ACT Bottle Grasp — 技术文档

## 目录

1. [系统架构](#1-系统架构)
2. [算法原理](#2-算法原理)
3. [数据流与格式](#3-数据流与格式)
4. [操作 SOP](#4-操作-sop)
5. [踩坑记录](#5-踩坑记录)

---

## 1. 系统架构

### 硬件拓扑

```
┌──────────────────────────────────────────────────┐
│                    工控机                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ CAN (can0)│  │ USB 3.0  │  │ USB 2.0         │ │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘ │
│       │             │                │             │
└───────┼─────────────┼────────────────┼─────────────┘
        │             │                │
   ┌────┴────┐   ┌────┴────┐     ┌────┴────┐
   │ 示教臂   │   │ D435i   │     │ SN0002  │
   │ (Leader) │   │ 腕部相机 │     │ 全局相机 │
   └────┬────┘   └─────────┘     └─────────┘
        │
   CAN 总线共享
        │
   ┌────┴────┐
   │ 被控臂   │
   │(Follower)│
   └─────────┘
```

### 镜像模式原理

Piper 机械臂支持**硬件级镜像模式**：示教臂和被控臂连接在同一条 CAN 总线上，示教臂通过 CAN 广播自身关节位置，被控臂在硬件固件层面自动跟随，无需上位机转发。

- **优点**：零延迟跟随，无需软件处理
- **约束**：数据采集时需注意被控臂的状态就是示教臂的相同状态，因此只需读取被控臂的 `get_joint_positions()`
- **起始位姿问题**：如果上一轮采集结束位置与下一轮起始位置不同，示教臂和被控臂会有位置偏差，因为被控臂跟随的是当前示教臂位置。解决方案：人工手动把示教臂拖回起始位置对齐即可

### 软件架构

```
┌─────────────────────────────────────────────────────────┐
│                     应用层                               │
│  teleop/data_collector.py    inference/deploy.py        │
│  采集示教数据                   推理部署                  │
├─────────────────────────────────────────────────────────┤
│                     算法层                               │
│  LeRobot ACT Policy            Normalization Pipeline    │
│  (ResNet18 + Transformer)      (Mean/Std 归一化)        │
├─────────────────────────────────────────────────────────┤
│                     驱动层                               │
│  hardware/piper_wrapper.py     camera/rs_camera.py      │
│  (Piper SDK 封装)              (RealSense + USB)        │
├─────────────────────────────────────────────────────────┤
│                     硬件抽象层                            │
│  piper_sdk_py_driver/          pyrealsense2 / OpenCV    │
│  (CAN 通信)                    (相机采集)                │
└─────────────────────────────────────────────────────────┘
```

### 模块职责

| 模块 | 文件 | 职责 |
|---|---|---|
| 机械臂驱动 | `hardware/piper_wrapper.py` | CAN 通信、使能/失能、关节读写、安全限位 |
| 相机驱动 | `camera/rs_camera.py` | RealSense + USB 相机初始化、帧读取、自动设备扫描 |
| 数据采集 | `teleop/data_collector.py` | 键盘控制、数据记录为 LeRobot v3.0 格式 |
| 训练 | `training/train.sh` | ACT 模型训练、checkpoint 管理 |
| 推理部署 | `inference/deploy.py` | 加载模型 + 归一化 pipeline、实时控制机械臂 |
| 离线评估 | `inference/eval.py` | 在数据集上评估模型精度 |

---

## 2. 算法原理

### 一句话理解

> 给模型看两张图（腕部相机 + 全局相机）+ 当前关节角度 → 让它输出接下来机械臂该怎么动。

### 为什么不一次只预测一步？

假设你在闭眼走路，每走一步才睁眼看一次。踩到坑的概率很大，因为你看不到前方 5 步之外的路况。

ACT 的做法是：**一次预测未来 10 步的动作**，但只执行第 1 步，然后立刻重新观察、重新预测。相当于一边看路一边走，走一步看十步。

### 模型长什么样

![ACT Policy 内部结构](fig_act_policy.png)

分三步走：

| 步骤 | 做什么 | 通俗理解 |
|---|---|---|
| ① ResNet18 看图 | 将 480×640 的 RGB 图像压成固定长度的特征向量 | **眼睛**：把画面转化成「瓶子的位置」「夹爪的距离」这种抽象信息 |
| ② 拼接关节状态 | 把 7 个关节角度数值拼到图像特征后面 | **本体感觉**：告诉模型「我的胳膊现在在哪」 |
| ③ Transformer 推理 | 用注意力和位置编码融合所有信息，输出 10 步动作 | **大脑**：综合看到的东西和身体状态，决定接下来怎么动 |

### 什么是「归一化」

神经网络对数字大小很敏感。举个例子：

```
关节 1 的角度范围：-3.14 ~ 3.14（量级 10⁰）
图像像素值范围：    0 ~ 255    （量级 10²）
```

如果直接喂进去，像素值会把关节角度淹掉。

归一化就是把所有数据缩放到「均值 0、标准差 1」的同一尺度，让网络公平对待每个输入。推理时模型输出的动作也需要反归一化回真实物理值。

### 推理过程（时间线视角）

```
时间 →

第0步   拍照片 + 读关节 → 推理 → 输出10步动作 → 只执行第1步
第1步   拍照片 + 读关节 → 推理 → 输出10步动作 → 只执行第1步
第2步   拍照片 + 读关节 → 推理 → 输出10步动作 → 只执行第1步
 ...     (每次都用最新观察重新推理，形成闭环)
```

每次推理虽然预测了 10 步，但我们只取第 1 步执行，剩下的 9 步只是帮 Transformer 理解轨迹走向。这种「走一步看一步」的策略让机械臂能实时纠正偏差。

### 关键技术决策

| 决策 | 选了 | 为什么 |
|---|---|---|
| 确定性 vs VAE | 确定性（`use_vae=false`） | VAE 在本任务上后验坍缩，直接关掉反而更稳定 |
| 绝对动作 vs Delta | Delta（位移量） | 如果模型摆烂输出零，机械臂只是不动，安全 |
| Phase 信号 | 加入 | 打破时序歧义，让模型知道「抓到哪一步了」 |
| Replan | 每步重新推理 | 开环变闭环，碰到瓶子也能及时纠正 |

---

## 3. 数据流与格式

### 两条数据流

整个系统只有两条数据流，分别对应两个入口程序：

| | 采集 (data_collector.py) | 推理 (deploy.py) |
|---|---|---|
| **方向** | 机械臂 → 硬盘 | 硬盘 → 机械臂 |
| **触发** | 按空格开始录制 | 按空格执行一次抓取 |
| **频率** | 30Hz 固定 | 30Hz 固定 |
| **产物** | Parquet + MP4 | 关节角度指令 |

#### 采集流

![采集数据流](fig_data_collection.png)

每一帧同时做三件事：读关节、读腕部画面、读全局画面。然后 `action[t] = state[t+1]`——下一帧的状态就是当前帧的动作目标。这叫**示教动作**：人怎么动，数据就怎么记。

#### 推理流

![推理数据流](fig_inference_flow.png)

推理比采集多两步关键处理：

- **归一化/反归一化**：网络内部所有数值都在「均值 0、标准差 1」的尺度上运算，输入需要归一化，输出需要反归一化回真实物理值
- **安全裁剪**：反归一化后的动作要硬限制在机械臂物理限位内，防止模型输出越界

### 数据存储格式

#### 目录结构

```
data/lerobot_dataset_50eps_current_cam/
├── meta/
│   ├── info.json              # 特征定义、FPS、shape
│   ├── stats.json             # 归一化统计量 (mean/std/min/max)
│   ├── episodes/              # 每条 episode 的起止帧索引
│   └── tasks.parquet          # 任务描述
├── data/chunk-000/
│   └── file-000.parquet       # 所有 50 episode 的数值数据
└── videos/
    ├── observation.images.wrist_rgb/chunk-000/
    │   └── file-000.mp4       # 腕部视频 (10926帧, AV1)
    └── observation.images.global_rgb/chunk-000/
        └── file-000.mp4       # 全局视频 (10926帧, AV1)
```

> 注意：本项目实测是**一个 Parquet 文件**存所有 episode，不是每个 episode 一个文件。LeRobot 的 chunk 大小决定了何时切分新文件。

#### 特征一览

| 特征名 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `observation.state` | float32 | (7,) | 当前关节角度 `[j1~j6, gripper]` |
| `action` | float32 | (7,) | 下一帧的关节角度（绝对动作）/ 位移量（Delta） |
| `observation.images.wrist_rgb` | video | (3, 480, 640) | 腕部 RealSense D435i, 30fps |
| `observation.images.global_rgb` | video | (3, 480, 640) | 全局 USB 相机, 30fps |
| `timestamp` | float32 | 标量 | 帧时间戳（秒） |
| `episode_index` | int64 | 标量 | 所属 episode 编号 |
| `frame_index` | int64 | 标量 | episode 内帧序号 |

#### 数值范围

| 维度 | 单位 | 范围 | 说明 |
|---|---|---|---|
| j1 ~ j6 | rad | [-3.14, 3.14] | 硬件限位裁剪 |
| gripper | m | [0, 0.10] | 夹爪开度，0=合拢 |

#### 当前数据集概况

| 数据集 | Episodes | 总帧数 | 说明 |
|---|---|---|---|
| `lerobot_dataset_50eps_current_cam` | 50 | 10,926 | 当前主力数据集，绝对动作格式 |
| `lerobot_dataset_delta_phase` | 50 | 10,926 | Delta 动作 + Phase 信号（训练用） |
| `lerobot_dataset_rebuilt_v1` | 50 | 10,926 | 重建后的 v1 版本 |

---

## 4. 操作 SOP

### 日常操作流程

#### 开机

```bash
# 1. 确认机械臂上电
# 2. 配置 CAN
sudo bash scripts/setup_can.sh
# 3. 验证硬件
conda activate piper_act
python3 test_hardware.py
```

#### 采集数据

```bash
conda activate piper_act
python3 teleop/data_collector.py
```

1. 按 **E** 使能
2. 手动把示教臂和被控臂回到固定起点
3. 把瓶子放到本条 episode 的初始位置
4. 按**空格**开始录制
5. 执行一次完整抓取
6. 按**空格**停止并保存
7. 保存后再手动回到固定起点，重复步骤 2-6，目标 50-100 条

**关键**：不同 bottle 位置、不同角度、不同抓取策略都要覆盖

#### 训练

```bash
conda activate piper_act
bash training/train.sh
# 后台执行:
nohup bash training/train.sh > /tmp/train_piper_act.log 2>&1 &
```

监控：
```bash
tail -f /tmp/train_piper_act.log
```

#### 评估

```bash
conda activate piper_act
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

评估报告示例解读：
```
Mean MSE across 3 episodes: 0.001234
Per-joint MSE:
  j1: 0.000856    <- 大关节误差通常较大
  j2: 0.001234
  j3: 0.000923
  j4: 0.000567    <- 小关节误差通常较小
  j5: 0.000432
  j6: 0.000398
  gripper: 0.000012  <- 夹爪开合比较确定
```

#### 部署

```bash
conda activate piper_act
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

### 故障处理速查

| 症状 | 检查项 | 解决 |
|---|---|---|
| 机械臂不动 | `ip link show can0` | `sudo bash scripts/setup_can.sh` |
| cv2 报错 _ARRAY_API | NumPy 版本 | `pip install "numpy<2" --force-reinstall opencv-python` |
| 全局相机黑屏 | 设备探测 | `python3 teleop/data_collector.py --list-cameras`，然后用 `--global-camera N` 指定 |
| RealSense 无深度 | D435/D435i 兼容性 | 会自动回退到 RGB-only |
| 训练崩溃 KeyError:'names' | lerobot patch | 确认 `feature_utils.py` 的 patch 已打 |
| PYTHONPATH 污染 | ROS2 环境 | `unset PYTHONPATH` 后重试 |

---

## 5. 踩坑记录

### 5.1 双机械臂架构的演进

**最初设想**：示教臂和被控臂分别接 can0/can1，上位机读示教臂状态并转发指令给被控臂（软件转发模式）。

**实际发现**：用户插电后发现两只臂已经在同一 CAN 总线上，被控臂自动镜像示教臂动作。这是 Piper 的硬件级镜像模式，完全不需要软件转发。

**启发**：先确认硬件能力再设计软件方案，避免不必要的工作。

### 5.2 LeRobot 版本选择

- **v0.4.4**：最初选型，API 较简单但已过时
- **v0.5.2**：最终使用版本，API 更成熟但 CLI 和配置系统有较大变化
- 升级要点：`train` → `lerobot_train`、`parse_args` → `parse_arg`、归一化引入 processor pipeline

### 5.3 归一化与 feature_utils.py Bug

LeRobot v0.5.2 的 `dataset_to_policy_features()` 假设所有图像/视频特征都有 `names` 字段来标注通道顺序。但 v3.0 格式的 video 特征 shape 已经是 (C, H, W) 格式，不需要 `names` 字段。

**修复**：`ft["names"]` → `ft.get("names")`，并加空值检查。

这是 LeRobot 的一个已知兼容性 bug，可能在后续版本修复。

### 5.4 PYTHONPATH 与 ROS2 冲突

ROS2 安装后会在 `~/.bashrc` 中设置 PYTHONPATH，指向系统级 Python 包路径（如 `/opt/ros/humble/lib/python3.10/site-packages`）。即使激活 conda 环境，PYTHONPATH 优先级高于 conda 环境，导致 import 时加载系统旧版包。

**现象**：`import lerobot` 导入的是 `~/.local/lib/python3.10/site-packages/` 的旧版而非 conda 环境的最新版。

**解决方案**：在 conda 环境的 `activate.d/` 和 `deactivate.d/` 中添加 hooks。

### 5.5 CAN 初始化的教训

`sudo ip link set can0 up type can bitrate 1000000` 只需要在系统启动后执行一次（或者 CAN 线重新插拔后）。不需要每次运行程序都重新配置。

**注意**：如果 `setup_can.sh` 先 `down` 再 `up`，而机械臂正在运行中，会导致失能。生产环境中应避免不必要的 CAN 重启。

### 5.6 AV1 编码与视频性能

LeRobot v3.0 默认使用 AV1 编码存储视频。AV1 压缩率高但解码计算量大，在训练时 CPU 视频解码可能成为瓶颈。`torchcodec` 支持 GPU 加速解码但当前 Linux 环境下不可用（用 `pyav` 回退）。

如果训练速度明显受限于视频解码，可考虑：
- 减小图像分辨率
- 使用预处理脚本将视频转为图像序列

### 5.7 起始位姿管理

镜像模式下，软件只控制被控臂，示教臂不会被程序自动拖回起点。如果只让被控臂回到某个保存位姿，而示教臂还在另一个位置，CAN 镜像会立即覆盖被控臂位置：

```
follower 移动到固定起点 → CAN 镜像检测到 leader 位置不同 → follower 立刻跟随 leader
```

**结论**：当前流程不再提供 P/H 自动起点管理。每条数据开始前，人工把示教臂和被控臂一起摆到固定起点；部署真机抓取前也用同一个固定起点。

### 5.8 图像增广对训练速度的影响

启用 `dataset.image_transforms.enable=true` 后，每个 batch 需要 CPU 执行 ColorJitter / SharpnessJitter / RandomAffine 等操作。这会导致训练速度下降约 10-20%。如果 GPU 利用率未满，可以降低 `num_workers` 减少 CPU 竞争；如果 CPU 是瓶颈，考虑关掉增广。

---

## 附录

### A. 依赖版本锁定

```
python=3.10
lerobot==0.5.2  (editable install from ~/third_party/lerobot)
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pyrealsense2>=2.55.0
numpy>=1.24.0,<2.0
datasets>=3.0.0
pyarrow>=14.0.0
av>=11.0.0
```

### B. 关键路径速查

| 路径 | 说明 |
|---|---|
| `~/miniconda3/envs/piper_act/` | Conda 环境 |
| `~/third_party/lerobot/` | LeRobot 源码（editable 安装） |
| `data/lerobot_dataset/` | 采集的数据集 |
| `outputs/train/piper_bottle_grasp/` | 训练输出 |
| `/tmp/train_piper_act.log` | 训练日志 |
