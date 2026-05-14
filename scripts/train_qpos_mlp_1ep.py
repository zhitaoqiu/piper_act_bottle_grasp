#!/usr/bin/env python3
"""qpos-only MLP sanity check: train a small MLP to map observation.state → action.

If this succeeds but ACT fails, the problem is in the ACT/LeRobot training pipeline.
If this also fails, the data/labels/normalization has a fundamental problem.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

J2_IDX = 1


class QPosMLP(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=7, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = PROJECT_ROOT / "data/lerobot_dataset_today_approach_1ep"
    print(f"Loading dataset: {dataset_root}")
    dataset = LeRobotDataset(
        repo_id="piper/bottle_approach_today_1ep",
        root=dataset_root,
        episodes=[0],
    )
    n = len(dataset)
    print(f"  {n} frames")

    # Extract all qpos → action pairs
    all_qpos = np.zeros((n, 7), dtype=np.float32)
    all_action = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        frame = dataset[i]
        all_qpos[i] = frame["observation.state"].numpy()
        all_action[i] = frame["action"].numpy()

    # Convert to tensors
    qpos_t = torch.from_numpy(all_qpos).to(device)
    action_t = torch.from_numpy(all_action).to(device)

    print(f"\nData stats:")
    print(f"  qpos J2 range:    [{all_qpos[:, J2_IDX].min():.4f}, {all_qpos[:, J2_IDX].max():.4f}]")
    print(f"  action J2 range:  [{all_action[:, J2_IDX].min():.4f}, {all_action[:, J2_IDX].max():.4f}]")
    print(f"  action J2 mean:   {all_action[:, J2_IDX].mean():.4f}")
    print(f"  action J2 std:    {all_action[:, J2_IDX].std():.4f}")

    # Pre-compute mean baseline for J2
    mean_baseline_j2 = float(all_action[:, J2_IDX].mean())
    mean_baseline_mse_j2 = float(((all_action[:, J2_IDX] - mean_baseline_j2) ** 2).mean())
    true_action_j2_std = float(all_action[:, J2_IDX].std())

    # Model
    model = QPosMLP(input_dim=7, hidden_dim=128, output_dim=7, num_layers=3)
    model.to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 32
    steps = 5000
    print_freq = 200

    print(f"\nTraining: {steps} steps, batch_size={batch_size}, lr=1e-3")
    print(f"{'='*80}")

    losses = []
    for step in range(steps):
        # Random batch
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = qpos_t[idx]  # [B, 7]
        y = action_t[idx]  # [B, 7]

        pred = model(x)  # [B, 7]
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step == 0 or (step + 1) % print_freq == 0:
            # Evaluate on full dataset
            model.eval()
            with torch.inference_mode():
                all_pred = model(qpos_t).cpu().numpy()
            model.train()

            all_pred_j2 = all_pred[:, J2_IDX]
            true_j2 = all_action[:, J2_IDX]

            model_mse_j2 = float(((true_j2 - all_pred_j2) ** 2).mean())
            pred_j2_std = float(all_pred_j2.std())
            improvement = model_mse_j2 / mean_baseline_mse_j2 if mean_baseline_mse_j2 > 1e-10 else float("inf")

            print(f"  step {step+1:5d}: loss={losses[-1]:.6f}  "
                  f"model_mse_j2={model_mse_j2:.6f}  "
                  f"baseline_mse_j2={mean_baseline_mse_j2:.6f}  "
                  f"improv_ratio={improvement:.4f}  "
                  f"pred_std={pred_j2_std:.4f}  true_std={true_action_j2_std:.4f}  "
                  f"{'PASS' if improvement < 0.5 else 'FAIL'}")

    # ── Final evaluation ──
    model.eval()
    with torch.inference_mode():
        all_pred = model(qpos_t).cpu().numpy()
    all_pred_j2 = all_pred[:, J2_IDX]
    true_j2 = all_action[:, J2_IDX]
    qpos_j2 = all_qpos[:, J2_IDX]

    model_mse_j2 = float(((true_j2 - all_pred_j2) ** 2).mean())
    pred_j2_std = float(all_pred_j2.std())
    improvement = model_mse_j2 / mean_baseline_mse_j2 if mean_baseline_mse_j2 > 1e-10 else float("inf")

    print(f"\n{'='*80}")
    print(f"Final results")
    print(f"{'='*80}")
    print(f"  true_action_j2_mean:  {mean_baseline_j2:.5f}")
    print(f"  pred_j2_mean:         {float(all_pred_j2.mean()):.5f}")
    print(f"  true_action_j2_std:   {true_action_j2_std:.5f}")
    print(f"  pred_j2_std:          {pred_j2_std:.5f}")
    print(f"  mean_baseline_mse_j2: {mean_baseline_mse_j2:.6f}")
    print(f"  model_mse_j2:         {model_mse_j2:.6f}")
    print(f"  improvement_ratio_j2: {improvement:.4f}")

    if improvement < 0.5:
        print(f"\n  VERDICT: MLP PASSED — can learn qpos→action mapping")
    else:
        print(f"\n  VERDICT: MLP FAILED — data/label alignment has fundamental problem")

    # ── Per-interval stats ──
    J2_INTERVALS = [
        (0.0, 0.2, "0.0-0.2"),
        (0.2, 0.5, "0.2-0.5"),
        (0.45, 0.55, "0.45-0.55"),
        (0.8, 1.2, "0.8-1.2"),
        (1.2, 1.6, "1.2-1.6"),
    ]
    print(f"\n  Per-interval (J2 range vs pred):")
    for lo, hi, label in J2_INTERVALS:
        mask = (qpos_j2 >= lo) & (qpos_j2 < hi)
        if not mask.any():
            continue
        print(f"    {label}:  n={mask.sum():3d}  "
              f"true_mean={true_j2[mask].mean():.5f}  "
              f"pred_mean={all_pred_j2[mask].mean():.5f}  "
              f"err={np.abs(all_pred_j2[mask]-true_j2[mask]).mean():.5f}")

    # ── Trajectory summary ──
    print(f"\n  Trajectory:")
    for pct in [0, 5, 10, 20, 40, 60, 80, 95]:
        t = int(n * pct / 100)
        if t >= n: t = n - 1
        err = abs(all_pred_j2[t] - true_j2[t])
        print(f"    t={t:3d} ({pct:2d}%): qpos={qpos_j2[t]:.4f}  "
              f"true={true_j2[t]:.4f}  pred={all_pred_j2[t]:.4f}  "
              f"err={err:+.4f}  {'OK' if err<0.05 else 'BAD'}")

    # ── Save model ──
    save_dir = PROJECT_ROOT / "outputs/debug"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "qpos_mlp_1ep.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved to {model_path}")

    # ── Save CSV ──
    csv_path = save_dir / "qpos_mlp_1ep_j2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "qpos_j2", "true_action_j2", "pred_j2", "pred_error_j2"])
        for t in range(n):
            writer.writerow([
                t, qpos_j2[t], true_j2[t], all_pred_j2[t],
                all_pred_j2[t] - true_j2[t]
            ])
    print(f"  CSV saved to {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
