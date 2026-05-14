#!/bin/bash
# ACT Overfit — current-environment approach-only, clean 1 episode.
# Build DATASET_ROOT first with training/build_today_approach_1ep_dataset.sh.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset_today_approach_1ep}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_approach_today_1ep_overfit}"
STEPS="${STEPS:-5000}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/tmp/piper_act_hf_cache}"

export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" || ! -d "${DATASET_ROOT}/data" ]]; then
    echo "[ERROR] Clean one-episode dataset is missing: ${DATASET_ROOT}" >&2
    echo "        Run: bash training/build_today_approach_1ep_dataset.sh" >&2
    exit 1
fi

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_approach_today_1ep \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes='[0]' \
    --dataset.image_transforms.enable=false \
    --policy.type=act \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.dim_model=512 \
    --policy.dim_feedforward=2048 \
    --policy.n_heads=8 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=4 \
    --policy.dropout=0.0 \
    --policy.use_vae=false \
    --policy.kl_weight=1.0 \
    --policy.optimizer_lr=3e-4 \
    --policy.optimizer_lr_backbone=1e-4 \
    --policy.repo_id=piper/bottle_approach_today_1ep_overfit \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --num_workers=0 \
    --persistent_workers=false \
    --steps="${STEPS}" \
    --save_freq=5000 \
    --eval_freq=5000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_act_approach_today_1ep_overfit
