# Piper ACT Bottle Grasp

基于 Piper 双臂（镜像模式）+ ACT 算法的瓶子抓取项目。

## 硬件配置

| 组件 | 型号 | 连接方式 | 用途 |
|---|---|---|---|
| 被控臂 | Piper | CAN (can0) | 执行抓取动作 |
| 示教臂 | Piper | 同一条 CAN 总线 | 人手拖动，被控臂自动镜像跟随 |
| 腕部相机 | RealSense D435i | USB 3.0 | 末端 RGB |
| 全局相机 | USB SN0002 | USB | 俯视 RGB |
| GPU | NVIDIA RTX 3060 12GB | — | 训练 + 推理 |

> **镜像模式**：示教臂和被控臂共享同一条 CAN 总线（can0），由硬件层面自动完成镜像跟随，无需软件转发。采集时只需读取被控臂状态即可。

## 环境搭建

### 1. Conda 环境

```bash
conda create -n piper_act python=3.10 -y
conda activate piper_act
pip install -r requirements.txt
```

关键依赖：
- `lerobot >= 0.5.2`（本项目使用 `/home/huatec/third_party/lerobot` 的 editable 安装）
- `torch >= 2.0`（CUDA 版）
- `pyrealsense2`（RealSense 相机）
- `opencv-python`（图像处理）
- `numpy < 2.0`（避免与 cv2 的 NumPy ABI 冲突）

### 2. Conda 环境隔离（重要）

如果系统安装了 ROS2，PYTHONPATH 会污染 conda 环境。已在 `~/miniconda3/envs/piper_act/etc/conda/` 下配置了自动 hooks：
- `activate.d/unset_pythonpath.sh` — 激活环境时自动清空 PYTHONPATH
- `deactivate.d/restore_pythonpath.sh` — 退出环境时恢复

### 3. CAN 总线

```bash
# 只需要 can0（示教臂和被控臂共享同一 CAN 总线）
sudo ip link set can0 down 2>/dev/null; sudo ip link set can0 up type can bitrate 1000000

# 或者用脚本
sudo bash scripts/setup_can.sh
```

## 项目结构

```
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml              # 参考配置
│
├── test_hardware.py              # 硬件链路验证（6 步自检）
├── hardware/
│   └── piper_wrapper.py          # Piper SDK 高层封装（安全限位、指令下发）
├── camera/
│   └── rs_camera.py              # RealSense D435i + USB 相机驱动（自动扫描）
│
├── teleop/
│   └── data_collector.py         # 示教数据采集（LeRobot v3.0 格式）
│
├── training/
│   └── train.sh                  # ACT 训练启动脚本
│
├── inference/
│   ├── deploy.py                 # 真机推理部署
│   └── eval.py                   # 离线评估（计算 MSE，不需要机械臂）
│
├── scripts/
│   ├── setup_can.sh              # CAN 双通道配置
│   └── setup_env.sh              # venv 环境安装脚本
│
├── data/
│   └── lerobot_dataset/          # 采集的 LeRobot v3.0 数据集
├── outputs/                      # 训练输出
│   └── train/
│       └── piper_bottle_grasp/
│           └── checkpoints/
│               ├── step_020000/  # 各步数 checkpoint
│               └── last/         # → 指向最新 checkpoint 的软链接
│
└── piper_sdk_py_driver/          # Piper SDK 驱动（外部提供）
```

## 完整工作流

> 详细的逐步操作指南请阅读 **[docs/workflow.md](docs/workflow.md)**，包含每一步的预期输出、故障排查和操作细节。以下为快速概览。
>
> ACT 静止帧、裁剪对齐、chunk size 和 checkpoint 排查请阅读 **[docs/act_debugging.md](docs/act_debugging.md)**。

### Step 1 — 验证硬件

```bash
conda activate piper_act
python3 test_hardware.py                # 机械臂 6 步自检
python3 teleop/data_collector.py --list-cameras  # 列出相机
python3 teleop/data_collector.py --camera-only --global-camera auto  # 测试画面
```

### Step 2 — 采集示教数据

```bash
conda activate piper_act
python3 teleop/data_collector.py --global-camera auto
```

| 按键 | 功能 |
|---|---|
| E | 使能被控臂 |
| 空格 | 开始/停止录制 |
| R | 丢弃当前条重录 |
| Q | 退出 |

> 每条数据开始前，手动把示教臂和被控臂一起回到固定起点，再按空格录制。建议采集 **50-100 条**，变化瓶子位置和角度。详见 [docs/workflow.md](docs/workflow.md) 第二步。

### Step 3 — 训练 ACT 模型

```bash
conda activate piper_act
nohup bash training/train.sh > /tmp/train_piper_act.log 2>&1 &   # 后台训练
tail -f /tmp/train_piper_act.log   # 监控
```

训练超参见 [training/train.sh](training/train.sh)，checkpoint 保存在 `outputs/train/piper_bottle_grasp/checkpoints/`。

### Step 4 — 离线评估

```bash
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

逐帧对比预测 vs 真值，输出 MSE。不需要机械臂。

### Step 5 — 真机部署

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

按**空格**执行一次抓取。可选 `--velocity-pct 30` 调低速度更安全。

调试“抬不起来 / 抬一下停住”时建议先用：

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/020000/pretrained_model \
    --debug-actions \
    --debug-every 1 \
    --replan-every-step
```

完整数据诊断、重建式裁剪、chunk size 对照训练和 checkpoint 批量评估命令见 [docs/act_debugging.md](docs/act_debugging.md)。

## 已知问题与修复

### 1. lerobot 源码 patch

本项目使用的 lerobot v0.5.2 有一处 bug 已在本机修复：

**文件**：`~/third_party/lerobot/src/lerobot/utils/feature_utils.py:153`

**问题**：LeRobotDataset v3.0 中 video 类型的数据没有 `names` 字段，但 `dataset_to_policy_features` 假设它存在，导致训练启动时崩溃。

**修复**：将 `ft["names"]` 改为 `ft.get("names")`，并在后续检查中加入空值保护：
```python
names = ft.get("names")
if names and names[2] in ["channel", "channels"]:
    shape = (shape[2], shape[0], shape[1])
```

如果重装 lerobot 或换机器，需要重新打这个 patch。

### 2. NumPy 版本

`numpy<2.0` 是必须的。OpenCV 的 Python 包编译时链接了 NumPy 1.x ABI，用 NumPy 2.x 会报 `_ARRAY_API` 错误。

如果出现此错误：
```bash
pip install "numpy<2" --force-reinstall opencv-python
```

### 3. PYTHONPATH 污染

如果系统有 ROS2，激活 conda 环境后系统 Python 包可能被加载。已在 conda hooks 中处理，如果换了环境需要重新配置。

### 4. USB 摄像头无法打开

SN0002 摄像头在某些系统上 V4L2 打开失败。已实现自动扫描 `/dev/video*` 设备并逐个尝试。可以用 `--list-cameras` 先列出所有设备，再用 `--global-camera N` 指定具体设备号。

## SDK API 速查

`PiperRobot`（`hardware/piper_wrapper.py`）：

| 方法 | 返回值 | 说明 |
|---|---|---|
| `connect()` | — | 连接 CAN |
| `enable(blocking=True)` | — | 使能电机（阻塞直到完成） |
| `disable()` | — | 失能电机 |
| `get_joint_positions()` | `list[float]` × 7 | 读取 [j1..j6, gripper]，单位 rad / m |
| `get_joint_state()` | `JointState` | 位置 + 速度 + 力矩 |
| `get_end_pose()` | `EndPose` | 末端位姿 (x, y, z, rx, ry, rz) |
| `set_joint_positions(pos, velocity_pct)` | — | 下发关节指令 |
| `disconnect()` | — | 断开 CAN 连接 |

底层基于 `piper_sdk.C_PiperInterface`，关节角单位 rad，夹爪单位 m。安全限位 ±3.14 rad，夹爪范围 0~0.035 m。

## 眼镜哥接手注意事项

1. **环境位置**：conda 环境在 `~/miniconda3/envs/piper_act/`，可编辑安装的 lerobot 在 `~/third_party/lerobot/`
2. **CAN 线**：示教臂和被控臂共享同一 CAN 总线（can0），只需要一根 CAN 线同时连接两个臂
3. **数据采集**：拖动示教臂 → 被控臂自动跟随 → 程序读取被控臂状态 + 摄像头 → 保存为 LeRobot 格式
4. **起始位姿**：不要让程序自动回位；每条开始前手动把示教臂和被控臂一起回到固定起点
5. **训练监控**：`tail -f /tmp/train_piper_act.log`
6. **checkpoint 路径**：用 `checkpoints/last/pretrained_model` 软链接，始终指向最新 checkpoint
7. **换机器需要做的事**：
   - 装 conda 环境 + 依赖
   - 装 lerobot（`pip install -e ~/third_party/lerobot`）
   - 打上面提到的 `feature_utils.py` patch
   - 配置 conda hooks（PYTHONPATH 隔离）
   - 装 pyrealsense2 + OpenCV
   - 配置 CAN 总线
