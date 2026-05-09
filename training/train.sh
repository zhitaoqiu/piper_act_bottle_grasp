#!/bin/bash
# ============================================================
# ACT Training for Piper Bottle Grasp (via LeRobot v0.5.2)
# ============================================================
# Prerequisites:
#   1. Data collected in data/lerobot_dataset (via teleop/data_collector.py)
#   2. conda activate piper_act
#
# Usage:  bash training/train.sh
# ============================================================

set -e

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_grasp \
    --dataset.root=data/lerobot_dataset \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.chunk_size=20 \
    --policy.n_action_steps=20 \
    --policy.dim_model=512 \
    --policy.dim_feedforward=2048 \
    --policy.n_heads=8 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=4 \
    --policy.dropout=0.1 \
    --policy.use_vae=false \
    --policy.kl_weight=1.0 \
    --policy.optimizer_lr=5e-4 \
    --policy.optimizer_lr_backbone=1e-4 \
    --policy.repo_id=piper/bottle_grasp_act \
    --policy.push_to_hub=false \
    --batch_size=4 \
    --steps=50000 \
    --save_freq=10000 \
    --eval_freq=10000 \
    --output_dir=outputs/train/piper_bottle_grasp \
    --job_name=piper_act_training
