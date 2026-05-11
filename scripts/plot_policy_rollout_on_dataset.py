#!/usr/bin/env python3
"""Plot ACT deployment-style rollout predictions against dataset actions."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Install it with: pip install matplotlib"
        ) from exc


def set_hf_cache_defaults(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))


def load_policy_processors(policy, checkpt: str, device: torch.device):
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


def reset_if_supported(obj) -> None:
    if hasattr(obj, "reset"):
        obj.reset()


def build_observation_batch(item: dict, device: torch.device) -> dict:
    batch = {}
    for key, value in item.items():
        if key == "observation.state" or key.startswith("observation.images."):
            if hasattr(value, "unsqueeze"):
                batch[key] = value.unsqueeze(0).to(device)
            else:
                batch[key] = torch.as_tensor(value).unsqueeze(0).to(device)
    return batch


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_action(policy, normalized_obs, replan_every_step: bool):
    if replan_every_step:
        policy.reset()
    return policy.select_action(normalized_obs)


def plot_rollout(
    plt,
    out_path: Path,
    episode: int,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    chunk_interval: int,
):
    x = np.arange(len(predicted))
    fig, axes = plt.subplots(7, 1, figsize=(13, 13), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(x, ground_truth[:, dim], label="ground truth", linewidth=1.3)
        ax.plot(x, predicted[:, dim], label="predicted", linewidth=1.2, alpha=0.85)
        if chunk_interval > 1:
            for boundary in range(chunk_interval, len(predicted), chunk_interval):
                ax.axvline(boundary, color="0.75", linestyle=":", linewidth=0.8)
        ax.set_ylabel(JOINT_NAMES[dim])
        ax.grid(True, alpha=0.22)
    axes[0].set_title(f"Episode {episode:03d} ACT rollout")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("local frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_rollout_csv(out_path: Path, predicted: np.ndarray, ground_truth: np.ndarray) -> None:
    fieldnames = ["frame"]
    fieldnames += [f"pred_{name}" for name in JOINT_NAMES]
    fieldnames += [f"gt_{name}" for name in JOINT_NAMES]
    fieldnames += [f"err_{name}" for name in JOINT_NAMES]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (pred, gt) in enumerate(zip(predicted, ground_truth, strict=True)):
            row = {"frame": idx}
            row.update({f"pred_{name}": float(pred[i]) for i, name in enumerate(JOINT_NAMES)})
            row.update({f"gt_{name}": float(gt[i]) for i, name in enumerate(JOINT_NAMES)})
            row.update({f"err_{name}": float(pred[i] - gt[i]) for i, name in enumerate(JOINT_NAMES)})
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot_dataset"))
    parser.add_argument("--repo-id", default="piper/bottle_grasp")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Maximum frames to roll out. 0 means the whole episode.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rollout"))
    parser.add_argument("--replan-every-step", action="store_true",
                        help="Reset the ACT action queue before every select_action call.")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/piper_act_hf_cache"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_hf_cache_defaults(args.cache_dir)
    sys.path.insert(0, str(PROJECT_ROOT))

    plt = require_matplotlib()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    print(f"Loading dataset: {args.dataset_root}")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    episode_positions = np.flatnonzero(episode_index == args.episode)
    if len(episode_positions) == 0:
        raise SystemExit(f"Episode {args.episode} not found in dataset")
    if args.max_steps > 0:
        episode_positions = episode_positions[: args.max_steps]

    print(f"Loading policy: {args.checkpt}")
    policy = ACTPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)

    reset_if_supported(policy)
    reset_if_supported(preprocessor)
    reset_if_supported(postprocessor)

    predicted = []
    ground_truth = []

    for step, dataset_pos in enumerate(episode_positions):
        item = dataset[int(dataset_pos)]
        batch = build_observation_batch(item, device)
        with torch.inference_mode():
            normalized_obs = preprocessor(batch)
            action = select_action(policy, normalized_obs, args.replan_every_step)
            action = postprocessor(action)

        if action.dim() == 2:
            action = action.squeeze(0)
        predicted.append(tensor_to_numpy(action).astype(np.float32))
        ground_truth.append(tensor_to_numpy(item["action"]).astype(np.float32))

        if (step + 1) % 50 == 0:
            print(f"  rolled out {step + 1}/{len(episode_positions)} frames")

    predicted = np.stack(predicted)
    ground_truth = np.stack(ground_truth)
    mse = float(np.mean((predicted - ground_truth) ** 2))
    per_joint_mse = np.mean((predicted - ground_truth) ** 2, axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_replan" if args.replan_every_step else ""
    png_path = args.output_dir / f"episode_{args.episode:03d}_rollout{suffix}.png"
    csv_path = args.output_dir / f"episode_{args.episode:03d}_rollout{suffix}.csv"
    chunk_interval = int(getattr(policy.config, "n_action_steps", getattr(policy.config, "chunk_size", 1)))
    plot_rollout(plt, png_path, args.episode, predicted, ground_truth, chunk_interval)
    write_rollout_csv(csv_path, predicted, ground_truth)

    print(f"Mean MSE: {mse:.6f}")
    for name, value in zip(JOINT_NAMES, per_joint_mse, strict=True):
        print(f"  {name}: {float(value):.6f}")
    print(f"Saved rollout plot to {png_path}")
    print(f"Saved rollout CSV to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
