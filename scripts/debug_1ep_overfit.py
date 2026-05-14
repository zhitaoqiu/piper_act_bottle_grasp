#!/usr/bin/env python3
"""Offline teacher-forcing evaluation of 1-episode ACT overfit.

Feeds every frame's real observation to the model and compares pred vs true action.
Outputs CSV and per-interval statistics, focused on J2.

Part 1: mean-baseline comparison to diagnose mean action collapse.
Part 2: uses predict_action_chunk (stateless) + policy.reset() per frame — no action queue.
Part 3: prints raw normalized output and action processor stats.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

J2_IDX = 1
NEAR_ZERO_THRESH = 0.003
J2_INTERVALS = [
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.5, "0.2-0.5"),
    (0.45, 0.55, "0.45-0.55"),
    (0.8, 1.2, "0.8-1.2"),
    (1.2, 1.6, "1.2-1.6"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str,
                        default="outputs/train/piper_bottle_approach_today_1ep_overfit/checkpoints/last/pretrained_model")
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "data/lerobot_dataset_today_approach_1ep")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-csv", type=Path,
                        default=PROJECT_ROOT / "outputs/debug/1ep_overfit_j2_debug_with_mean_baseline.csv")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from inference.deploy import load_policy_processors, policy_state_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset_root}")
    dataset = LeRobotDataset(
        repo_id="piper/bottle_approach_today_1ep",
        root=args.dataset_root,
        episodes=[args.episode],
    )
    n = len(dataset)
    print(f"  {n} frames")

    # Load policy
    print(f"\nLoading policy: {args.checkpt}")
    policy = ACTPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    print(f"  state_dim={policy_state_dim(policy)}, "
          f"chunk_size={policy.config.chunk_size}, "
          f"n_action_steps={policy.config.n_action_steps}")

    # Load processors
    print("\nLoading pre/post processors ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    print("  Done.")

    # ── Part 3: action normalization stats ──
    print("\n" + "=" * 80)
    print("Part 3: Action normalization stats (J2)")
    print("=" * 80)
    action_stats = None
    for mod in postprocessor:
        mod_name = type(mod).__name__
        if hasattr(mod, "stats") and mod.stats is not None:
            action_stats = mod.stats
            break
    if action_stats is not None and "action" in action_stats:
        act_j2 = action_stats["action"]
        if len(act_j2) == 7:
            print(f"  action_mean_j2:   {act_j2[J2_IDX]:.5f}")
        elif len(act_j2) == 2:
            print(f"  action_stats keys: {list(action_stats.keys())}")
            print(f"  action mean:  {act_j2[0][J2_IDX]:.5f}")
            print(f"  action std:   {act_j2[1][J2_IDX]:.5f}")
            print(f"  action min:   {(act_j2[0] - 3*act_j2[1])[J2_IDX]:.5f}")
            print(f"  action max:   {(act_j2[0] + 3*act_j2[1])[J2_IDX]:.5f}")
    else:
        print("  WARNING: could not find action stats in postprocessor")
        # Try to find stats in preprocessor
        for mod in preprocessor:
            mod_name = type(mod).__name__
            if hasattr(mod, "stats") and mod.stats is not None:
                stats = mod.stats
                if "action" in stats:
                    act_j2 = stats["action"]
                    if hasattr(act_j2, 'shape') and len(act_j2) == 2:
                        print(f"  Found in PREPROCESSOR: action_mean_j2={act_j2[0][J2_IDX]:.5f}, "
                              f"action_std_j2={act_j2[1][J2_IDX]:.5f}")
                    else:
                        print(f"  Found in PREPROCESSOR: action_stats={act_j2}")

    # ── Teacher-forcing: feed each real obs, compare pred vs true action ──
    print(f"\nRunning teacher-forcing over {n} frames ...")
    print("Part 2: using predict_action_chunk (stateless) + policy.reset() per frame")
    rows = []
    for t in range(n):
        # Part 2: force reset to clear any action queue
        policy.reset()

        frame = dataset[t]
        qpos = frame["observation.state"].numpy()
        true_action = frame["action"].numpy()

        # Build obs dict for policy
        obs = {}
        for k, v in frame.items():
            if k == "action":
                continue
            if isinstance(v, torch.Tensor):
                obs[k] = v.unsqueeze(0).to(device)

        with torch.inference_mode():
            # Part 3: get raw normalized output BEFORE postprocessor
            normalized_obs = preprocessor(obs)
            raw_normalized = policy.predict_action_chunk(normalized_obs)  # before unnormalize
            action_chunk = postprocessor(raw_normalized.clone())

        # First predicted action (action_chunk shape: [1, chunk_size, 7])
        pred_first = action_chunk[:, 0, :].cpu().numpy().squeeze()  # (7,)
        pred_chunk_j2 = action_chunk[0, :, J2_IDX].cpu().numpy()  # (chunk_size,)
        raw_norm_j2 = raw_normalized[0, 0, J2_IDX].cpu().item()  # normalized J2 before unnormalize
        raw_chunk_norm_j2 = raw_normalized[0, :, J2_IDX].cpu().numpy()  # normalized chunk

        qj2 = float(qpos[J2_IDX])
        qnj2 = float(true_action[J2_IDX])  # true action J2 = next qpos J2
        pj2 = float(pred_first[J2_IDX])
        true_delta_j2 = qnj2 - qj2
        pred_delta_j2 = pj2 - qj2
        pred_error_j2 = pj2 - qnj2

        rows.append({
            "t": t,
            "qpos_j2": qj2,
            "qpos_next_j2": qnj2,
            "true_action_j2": qnj2,
            "pred_first_j2": pj2,
            "pred_delta_j2": pred_delta_j2,
            "true_delta_j2": true_delta_j2,
            "pred_error_j2": pred_error_j2,
            "abs_pred_error_j2": abs(pred_error_j2),
            "pred_chunk_j2": ",".join(f"{x:.5f}" for x in pred_chunk_j2),
            "raw_norm_j2": raw_norm_j2,
            "raw_chunk_norm_j2": ",".join(f"{x:.5f}" for x in raw_chunk_norm_j2),
            "input_state_j2": qj2,  # Part 2: sanity check — input state varies
        })

        # Part 2: sanity print at key frames
        if t in (0, 30, 60, 100, 150) or (0.45 <= qj2 <= 0.55):
            print(f"  [sanity t={t:3d}] input_state_j2={qj2:.5f}  "
                  f"raw_norm_j2={raw_norm_j2:.5f}  "
                  f"pred_robot_j2={pj2:.5f}  "
                  f"true_act_j2={qnj2:.5f}  "
                  f"raw_chunk_j2={[f'{x:.5f}' for x in raw_chunk_norm_j2[:5]]}")

    # ── Write CSV ──
    fieldnames = ["t", "qpos_j2", "qpos_next_j2", "true_action_j2", "pred_first_j2",
                  "pred_delta_j2", "true_delta_j2", "pred_error_j2", "abs_pred_error_j2",
                  "pred_chunk_j2", "raw_norm_j2", "raw_chunk_norm_j2", "input_state_j2"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved to {args.output_csv}")

    # ── Gather arrays ──
    all_qj2 = np.array([r["qpos_j2"] for r in rows])
    all_aj2 = np.array([r["true_action_j2"] for r in rows])
    all_pj2 = np.array([r["pred_first_j2"] for r in rows])
    all_raw_norm = np.array([r["raw_norm_j2"] for r in rows])
    all_tdelta = all_aj2 - all_qj2
    all_pdelta = all_pj2 - all_qj2

    # ── Part 1: mean baseline statistics ──
    true_action_j2_mean = float(all_aj2.mean())
    pred_j2_mean = float(all_pj2.mean())
    true_action_j2_std = float(all_aj2.std())
    pred_j2_std = float(all_pj2.std())
    raw_norm_mean = float(all_raw_norm.mean())
    raw_norm_std = float(all_raw_norm.std())

    mean_baseline_j2 = true_action_j2_mean
    all_mean_baseline_error = np.abs(all_aj2 - mean_baseline_j2)
    mean_baseline_mse_j2 = float(((all_aj2 - mean_baseline_j2) ** 2).mean())
    model_mse_j2 = float(((all_aj2 - all_pj2) ** 2).mean())
    model_mae_j2 = float(np.abs(all_aj2 - all_pj2).mean())
    improvement_ratio_j2 = model_mse_j2 / mean_baseline_mse_j2 if mean_baseline_mse_j2 > 1e-10 else float("inf")

    # ── Part 1: extended per-frame fields ──
    # Rewrite CSV with mean-baseline columns
    fieldnames_v2 = ["t", "qpos_j2", "true_action_j2", "pred_first_j2",
                     "mean_baseline_j2",
                     "pred_minus_mean_j2", "true_minus_mean_j2",
                     "model_abs_error_j2", "mean_baseline_abs_error_j2",
                     "pred_chunk_j2", "raw_norm_j2", "input_state_j2"]
    # Write the extended CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_v2)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({
                "t": r["t"],
                "qpos_j2": r["qpos_j2"],
                "true_action_j2": r["true_action_j2"],
                "pred_first_j2": r["pred_first_j2"],
                "mean_baseline_j2": mean_baseline_j2,
                "pred_minus_mean_j2": r["pred_first_j2"] - mean_baseline_j2,
                "true_minus_mean_j2": r["true_action_j2"] - mean_baseline_j2,
                "model_abs_error_j2": abs(r["pred_first_j2"] - r["true_action_j2"]),
                "mean_baseline_abs_error_j2": abs(r["true_action_j2"] - mean_baseline_j2),
                "pred_chunk_j2": r["pred_chunk_j2"],
                "raw_norm_j2": r["raw_norm_j2"],
                "input_state_j2": r["input_state_j2"],
            })
    print(f"  Extended CSV (with mean baseline) saved to {args.output_csv}")

    # ── Part 1: per-interval stats ──
    print("\n" + "=" * 80)
    print("Part 1: Per-interval teacher-forcing stats (with mean baseline)")
    print("=" * 80)
    header = (f"{'Interval':>12s}  {'cnt':>5s}  {'mean_qj2':>10s}  {'mean_act':>10s}  "
              f"{'mean_pred':>10s}  {'mean_tdelt':>10s}  {'mean_pdelt':>10s}  "
              f"{'abs_err':>10s}  {'nz_pred':>8s}  {'act_std':>10s}  {'pred_std':>10s}")
    print(header)
    print("-" * len(header))

    overall_abs_err = np.abs(all_pj2 - all_aj2).mean()
    print(f"{'OVERALL':>12s}  {len(rows):5d}  {all_qj2.mean():10.5f}  {all_aj2.mean():10.5f}  "
          f"{all_pj2.mean():10.5f}  {all_tdelta.mean():10.5f}  {all_pdelta.mean():10.5f}  "
          f"{overall_abs_err:10.5f}  {(np.abs(all_pdelta)<NEAR_ZERO_THRESH).sum():8d}  "
          f"{true_action_j2_std:10.5f}  {pred_j2_std:10.5f}")

    for lo, hi, label in J2_INTERVALS:
        mask = (all_qj2 >= lo) & (all_qj2 < hi)
        if not mask.any():
            print(f"{label:>12s}  {'(empty)':>5s}")
            continue
        cnt = mask.sum()
        m_qj2 = all_qj2[mask].mean()
        m_act = all_aj2[mask].mean()
        m_pred = all_pj2[mask].mean()
        m_tdelta = all_tdelta[mask].mean()
        m_pdelta = all_pdelta[mask].mean()
        abs_err = np.abs(all_pj2[mask] - all_aj2[mask]).mean()
        nz_pred = (np.abs(all_pdelta[mask]) < NEAR_ZERO_THRESH).sum()
        a_std = all_aj2[mask].std()
        p_std = all_pj2[mask].std()
        print(f"{label:>12s}  {cnt:5d}  {m_qj2:10.5f}  {m_act:10.5f}  "
              f"{m_pred:10.5f}  {m_tdelta:10.5f}  {m_pdelta:10.5f}  "
              f"{abs_err:10.5f}  {nz_pred:8d}  {a_std:10.5f}  {p_std:10.5f}")

    # ── Part 1: keynote summary ──
    print("\n" + "=" * 80)
    print("Part 1: Mean baseline collapse diagnosis")
    print("=" * 80)
    print(f"  true_action_j2_mean:         {true_action_j2_mean:.5f}")
    print(f"  pred_j2_mean:                {pred_j2_mean:.5f}")
    print(f"  true_action_j2_std:          {true_action_j2_std:.5f}")
    print(f"  pred_j2_std:                 {pred_j2_std:.5f}")
    print(f"  mean_baseline_mse_j2:        {mean_baseline_mse_j2:.6f}")
    print(f"  model_mse_j2:                {model_mse_j2:.6f}")
    print(f"  improvement_ratio_j2:        {improvement_ratio_j2:.4f}  (<0.5 = using input, ~1 = mean collapse)")
    print(f"  raw_norm_j2_mean:            {raw_norm_mean:.5f}")
    print(f"  raw_norm_j2_std:             {raw_norm_std:.5f}")

    # Part 3 verdict
    print("\n" + "=" * 80)
    print("Part 3: Normalized-space verdict")
    print("=" * 80)
    if abs(raw_norm_mean) < 0.05:
        print(f"  raw_norm_j2_mean ≈ 0 ({raw_norm_mean:.5f}) → model outputs near-zero in normalized space")
        print(f"  unnormalize(0) = action_mean = {true_action_j2_mean:.5f}")
        print(f"  pred_robot_j2_mean = {pred_j2_mean:.5f}")
        print("  VERDICT: Model outputs normalized 0, unnormalize restores mean → MEAN COLLAPSE CONFIRMED")
    else:
        print(f"  raw_norm_j2_mean = {raw_norm_mean:.5f} (not near zero)")
        print("  Model might be doing something else — check raw_chunk values")

    # Part 1 verdict
    print("\n" + "=" * 80)
    print("Part 1: Final verdict")
    print("=" * 80)
    if pred_j2_std < 0.01:
        print(f"  pred_j2_std = {pred_j2_std:.5f} < 0.01 → MODEL OUTPUTS CONSTANT")
    else:
        print(f"  pred_j2_std = {pred_j2_std:.5f} >= 0.01 → model has some variation")

    if improvement_ratio_j2 > 0.8:
        print(f"  improvement_ratio_j2 = {improvement_ratio_j2:.4f} > 0.8 → model NOT better than mean baseline")
    else:
        print(f"  improvement_ratio_j2 = {improvement_ratio_j2:.4f} <= 0.8 → model beats mean baseline")

    # Part 2 verdict
    print("\n" + "=" * 80)
    print("Part 2: Action queue / input state sanity check")
    print("=" * 80)
    # Check input states at sampled times are different
    t0_state = rows[0]["input_state_j2"]
    t30_state = rows[30]["input_state_j2"] if len(rows) > 30 else t0_state
    t60_state = rows[60]["input_state_j2"] if len(rows) > 60 else t0_state
    t100_state = rows[100]["input_state_j2"] if len(rows) > 100 else t0_state
    t150_state = rows[150]["input_state_j2"] if len(rows) > 150 else t0_state
    input_values = [t0_state, t30_state, t60_state, t100_state, t150_state]
    unique_inputs = len(set(f"{v:.5f}" for v in input_values))
    print(f"  input_state_j2 at t=0,30,60,100,150: {[f'{v:.5f}' for v in input_values]}")
    if unique_inputs > 1:
        print(f"  Input varies ({unique_inputs} unique) → debug data pipeline OK, not a read-repeat bug")
    else:
        print("  INPUT IS CONSTANT → dataset read or observation construction is broken!")
    print(f"  Method: predict_action_chunk (stateless) + policy.reset() each frame → no queue contamination")

    # ── Critical zone detail ──
    print("\n" + "=" * 80)
    print("CRITICAL: J2 0.45-0.55 zone — teacher-forcing detail")
    print("=" * 80)
    crit_mask = (all_qj2 >= 0.45) & (all_qj2 <= 0.55)
    crit_idxs = np.where(crit_mask)[0]
    if len(crit_idxs) == 0:
        print("  No frames in this range.")
    else:
        print(f"  {'t':>4s}  {'qpos':>10s}  {'true_act':>10s}  "
              f"{'pred':>10s}  {'tdelta':>10s}  {'pdelta':>10s}  {'err':>10s}  raw_norm  chunk_J2")
        for t in crit_idxs:
            r = rows[t]
            print(f"  {t:4d}  {r['qpos_j2']:10.5f}  {r['true_action_j2']:10.5f}  "
                  f"{r['pred_first_j2']:10.5f}  {r['true_delta_j2']:10.5f}  "
                  f"{r['pred_delta_j2']:10.5f}  {r['pred_error_j2']:10.5f}  "
                  f"{r['raw_norm_j2']:8.5f}  [{r['pred_chunk_j2']}]")

        crit_pdelta = all_pdelta[crit_idxs]
        crit_nz = (np.abs(crit_pdelta) < NEAR_ZERO_THRESH).sum()
        print(f"\n  Mean pred_delta: {crit_pdelta.mean():.5f}")
        print(f"  Near-zero pred_deltas: {crit_nz}/{len(crit_idxs)}")
        if crit_pdelta.mean() < 0.002:
            print("  VERDICT: Model predicts near-zero delta here — stuck point.")
        else:
            print("  VERDICT: Model predicts positive delta — should move through this zone.")

    # ── Trajectory summary ──
    print("\n" + "=" * 80)
    print("Trajectory summary: can pred_first_j2 follow true_action_j2?")
    print("=" * 80)
    for pct in [0, 5, 10, 20, 40, 60, 80, 95]:
        t = int(n * pct / 100)
        if t >= n:
            t = n - 1
        r = rows[t]
        print(f"  t={t:3d} ({pct:2d}%): qpos={r['qpos_j2']:.4f}  "
              f"true_act={r['true_action_j2']:.4f}  "
              f"pred={r['pred_first_j2']:.4f}  "
              f"err={r['pred_error_j2']:+.4f}  "
              f"({'OK' if abs(r['pred_error_j2'])<0.05 else 'BAD'})")

    print("\nDone.")


if __name__ == "__main__":
    main()
