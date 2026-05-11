#!/bin/bash
# ACT Training for Piper Bottle Grasp, chunk_size=10 / n_action_steps=10

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_grasp_chunk10}"

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_grasp \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
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
    --policy.repo_id=piper/bottle_grasp_act_chunk10 \
    --policy.push_to_hub=false \
    --batch_size=4 \
    --steps=50000 \
    --save_freq=10000 \
    --eval_freq=10000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_act_chunk10
