#!/usr/bin/env python3
"""Check training batch state/action variation for 1-episode dataset.

Objective: verify that training batches contain varying state_j2 and action_j2,
not constants or identical values across samples.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

J2_IDX = 1


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from itertools import cycle

    dataset_root = PROJECT_ROOT / "data/lerobot_dataset_today_approach_1ep"
    repo_id = "piper/bottle_approach_today_1ep"
    batch_size = 8

    print(f"Loading dataset: {dataset_root}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[0],
        image_transforms=None,
    )
    n = len(dataset)
    print(f"  {n} frames, {dataset.num_episodes} episode(s)")

    # Feature names
    feature_keys = list(dataset[0].keys())
    print(f"  Frame keys: {feature_keys}")

    # Create dataloader matching training config (shuffle=True, no sampler for simple case)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    dl_iter = cycle(dataloader)

    print(f"\n{'='*80}")
    print(f"Batch inspection — batch_size={batch_size}")
    print(f"Checking state_j2 and action_j2 variation across samples in batch")
    print(f"{'='*80}")

    for batch_idx in range(5):
        batch = next(dl_iter)
        obs_state = batch["observation.state"]  # [batch, 7]
        action = batch["action"]  # [batch, chunk_size, 7] or [batch, 7]

        batch_size_actual = obs_state.shape[0]
        state_j2 = obs_state[:, J2_IDX].numpy()

        if action.ndim == 3:
            # [batch, chunk_size, 7]
            action_j2_first = action[:, 0, J2_IDX].numpy()  # first action step J2
            action_j2_all = action[0, :, J2_IDX].numpy()  # first sample's full chunk J2
            action_shape_str = f"[B,{action.shape[1]},7]"
        else:
            action_j2_first = action[:, J2_IDX].numpy()
            action_j2_all = None
            action_shape_str = f"[B,7]"

        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"  observation.state shape: {obs_state.shape}")
        print(f"  action shape:            {action.shape} {action_shape_str}")
        print(f"  {'sample':>6s}  {'state_j2':>10s}  {'action_j2':>10s}  "
              f"{'delta_j2':>10s}  {'act_j2_range':>14s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

        for s in range(batch_size_actual):
            sj2 = float(state_j2[s])
            aj2 = float(action_j2_first[s])
            dj2 = aj2 - sj2
            print(f"  {s:6d}  {sj2:10.5f}  {aj2:10.5f}  {dj2:10.5f}")

        # Sample 0 full action chunk
        if action_j2_all is not None:
            print(f"\n  sample 0 full action chunk J2: {[f'{x:.5f}' for x in action_j2_all]}")
            print(f"  sample 0 state J2:           {float(state_j2[0]):.5f}")
            print(f"  sample 0 delta per step:     {[f'{x-float(state_j2[0]):.5f}' for x in action_j2_all]}")

        # Variation check
        state_j2_range = float(state_j2.max() - state_j2.min())
        action_j2_range = float(action_j2_first.max() - action_j2_first.min())
        print(f"\n  state_j2 range in batch:  {state_j2_range:.5f}  "
              f"({'VARIED' if state_j2_range > 0.01 else 'CONSTANT!'})")
        print(f"  action_j2 range in batch: {action_j2_range:.5f}  "
              f"({'VARIED' if action_j2_range > 0.01 else 'CONSTANT!'})")

    # ── Check if action labels are qpos[t] or qpos[t+1] ──
    print(f"\n{'='*80}")
    print(f"Action alignment check: is action[t] ≈ qpos[t] or qpos[t+1]?")
    print(f"{'='*80}")
    for batch_idx in range(2):
        batch = next(dl_iter)
        obs_state = batch["observation.state"]  # [batch, 7]
        action = batch["action"]

        if action.ndim == 3:
            action_first = action[:, 0, :]  # [batch, 7]
        else:
            action_first = action

        # For each sample, check if action ≈ state (identity) or different
        state_arr = obs_state.numpy()
        action_arr = action_first.numpy()

        diffs = np.abs(action_arr - state_arr).mean(axis=1)
        print(f"  Batch {batch_idx+1}: mean|action-state| per sample: "
              f"{[f'{d:.5f}' for d in diffs]}")
        max_diff = diffs.max()
        print(f"    max diff = {max_diff:.5f} — "
              f"{'action != state (good, not identity)' if max_diff > 0.01 else 'action ≈ state (SUSPICIOUS)'}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
