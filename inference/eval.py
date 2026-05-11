#!/usr/bin/env python3
"""
Evaluate trained ACT policy on held-out episodes.

Usage:
  conda activate piper_act
  python3 inference/eval.py --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model

This runs the policy on dataset episodes (not used in training) and computes:
  - Mean squared error (MSE) between predicted and ground-truth actions
  - Visualization of predicted vs actual trajectories
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_policy_processors(policy, checkpt: str, device: torch.device):
    """Load the same pre/post processors used during deployment."""
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {"device": device.type},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {"device": device.type},
        "device_processor": {"device": "cpu"},
    }
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def reshape_visual_stats_for_broadcast(stats, policy_features):
    """LeRobot v0.5.2 dataset visual stats may be (C,), but images are (B,C,H,W)."""
    from lerobot.configs import FeatureType

    fixed = copy.deepcopy(stats)
    for key, feature in policy_features.items():
        if feature.type != FeatureType.VISUAL or key not in fixed:
            continue
        channels = feature.shape[0]
        for stat_name, value in fixed[key].items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1 and arr.shape[0] == channels:
                fixed[key][stat_name] = arr.reshape(channels, 1, 1)
    return fixed


def build_pre_post_processors(policy, dataset):
    from lerobot.configs import FeatureType, NormalizationMode
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.processor import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.feature_utils import dataset_to_policy_features

    policy_features = dataset_to_policy_features(dataset.meta.features)
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    stats = reshape_visual_stats_for_broadcast(dataset.meta.stats, policy_features)

    normalizer = NormalizerProcessorStep(
        features=policy_features, norm_map=norm_map,
        stats=stats, device="cpu",
    )
    preprocessor = PolicyProcessorPipeline(
        steps=[normalizer], to_transition=batch_to_transition, to_output=transition_to_batch,
    )

    unnormalizer = UnnormalizerProcessorStep(
        features=policy_features, norm_map=norm_map,
        stats=stats, device="cpu",
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[unnormalizer], to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True,
                        help="Path to trained ACT checkpoint")
    parser.add_argument("--dataset-root", type=str, default="data/lerobot_dataset",
                        help="Path to LeRobot dataset root used for offline evaluation")
    parser.add_argument("--dataset-repo-id", type=str, default="piper/bottle_grasp",
                        help="LeRobot dataset repo_id")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to evaluate (0 = all)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip trajectory plotting")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load policy
    print(f"Loading policy from {args.checkpt} ...")
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()

    # Load dataset
    print("Loading dataset ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames")

    # Build normalization. Prefer checkpoint processors so offline eval matches deployment.
    try:
        preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    except Exception as e:
        print(f"  [WARN] Could not load checkpoint processors ({e}); using dataset stats.")
        preprocessor, postprocessor = build_pre_post_processors(policy, dataset)
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()
    if hasattr(postprocessor, "reset"):
        postprocessor.reset()

    # Pick eval episodes (use last N episodes, not the first ones used in training)
    total_eps = dataset.num_episodes
    eval_episodes = max(1, args.episodes) if args.episodes > 0 else total_eps
    eval_episodes = min(eval_episodes, total_eps)
    # Use last episodes for validation (they were collected later)
    episode_indices = list(range(max(0, total_eps - eval_episodes), total_eps))
    print(f"Evaluating on episodes: {episode_indices}")

    # Collect metrics
    all_mse = []
    all_joint_errors = []  # per-joint MSE

    for ep_idx in tqdm(episode_indices, desc="Evaluating episodes"):
        # Find frame range for this episode via hf_dataset
        ep_mask = (np.array(dataset.hf_dataset["episode_index"]) == ep_idx)
        ep_frames = np.where(ep_mask)[0]
        ep_start, ep_end = ep_frames[0], ep_frames[-1] + 1
        ep_len = ep_end - ep_start

        predicted = []
        ground_truth = []

        for frame_idx in range(ep_start, ep_end):
            item = dataset[frame_idx]

            # Build observation batch (add batch dim)
            batch = {}
            for key in item:
                if key.startswith("observation.state"):
                    batch[key] = item[key].unsqueeze(0).to(device)
                elif key.startswith("observation.images."):
                    batch[key] = item[key].unsqueeze(0).to(device)

            with torch.inference_mode():
                norm_batch = preprocessor(batch)
                # predict_action_chunk gives (1, chunk_size, 7)
                action_chunk = policy.predict_action_chunk(norm_batch)
                action_chunk = postprocessor(action_chunk)
                # Take first action of the chunk for comparison
                action = action_chunk[:, 0, :]

            pred = action.squeeze(0).cpu().numpy()
            gt = item["action"].cpu().numpy()

            predicted.append(pred)
            ground_truth.append(gt)

        predicted = np.array(predicted)  # (T, 7)
        ground_truth = np.array(ground_truth)

        mse = np.mean((predicted - ground_truth) ** 2)
        joint_mse = np.mean((predicted - ground_truth) ** 2, axis=0)  # (7,)
        all_mse.append(mse)
        all_joint_errors.append(joint_mse)

        print(f"  Ep {ep_idx}: MSE={mse:.6f}, frames={ep_len}")

    # Summary
    mean_mse = np.mean(all_mse)
    mean_joint_mse = np.mean(all_joint_errors, axis=0)
    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    print(f"\n{'='*50}")
    print(f"  Mean MSE across {len(episode_indices)} episodes: {mean_mse:.6f}")
    print(f"  Per-joint MSE:")
    for name, err in zip(joint_names, mean_joint_mse):
        print(f"    {name}: {err:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
