#!/usr/bin/env python3
"""
Deploy trained ACT policy on Piper arm for bottle grasping.

Usage:
  conda activate piper_act
  python3 inference/deploy.py --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model

Controls:
  SPACE  = run one grasp attempt
  Q/ESC  = quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.piper_wrapper import PiperRobot
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

PIPER_GRIPPER_MAX_M = 0.035


def load_policy_processors(policy, checkpt: str, device: torch.device):
    """Load normalization pipelines saved with the trained policy."""
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


def prepare_observation(state, wrist_img, global_img, device):
    """Convert raw numpy data to inference-ready tensors (batched, on device)."""
    obs = {}
    obs["observation.state"] = torch.from_numpy(
        np.asarray(state, dtype=np.float32)
    ).unsqueeze(0).to(device)

    if wrist_img is not None:
        t = torch.from_numpy(wrist_img).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)
        obs["observation.images.wrist_rgb"] = t

    if global_img is not None:
        t = torch.from_numpy(global_img).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)
        obs["observation.images.global_rgb"] = t

    return obs


def build_preview(wrist_frame, global_frame, text: str, color=(0, 255, 0)):
    preview = wrist_frame.rgb.copy()
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if global_frame is not None:
        g_preview = cv2.resize(global_frame.rgb, (preview.shape[1], preview.shape[0]))
        preview = np.hstack([preview, g_preview])
    return preview


def should_quit(key: int) -> bool:
    return key in (27, ord('q'), ord('Q'))


def fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def max_abs_diff(cur, prev) -> float:
    if prev is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(cur, dtype=np.float32) - np.asarray(prev, dtype=np.float32))))


def select_policy_action(policy, postprocessor, normalized_obs, replan_every_step: bool):
    """Return one unnormalized action, optionally bypassing ACT's open-loop queue."""
    if replan_every_step:
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(normalized_obs)
            action_chunk = postprocessor(action_chunk)
            return action_chunk[:, 0, :]

        policy.reset()

    action = policy.select_action(normalized_obs)
    return postprocessor(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True,
                        help="Path to trained ACT checkpoint")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control loop frequency. Keep this equal to dataset fps.")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum action steps for one grasp attempt.")
    parser.add_argument("--no-global", action="store_true",
                        help="Disable global camera")
    parser.add_argument("--global-camera", type=str, default="auto",
                        help="Global camera device ID or 'auto'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run policy and preview targets without sending robot commands.")
    parser.add_argument("--debug-actions", action="store_true",
                        help="Print current state, predicted target, and delta during rollout.")
    parser.add_argument("--debug-every", type=int, default=10,
                        help="Print one debug line every N action steps.")
    parser.add_argument("--replan-every-step", action="store_true",
                        help="Recompute a fresh ACT action chunk every control step instead of consuming the action queue.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Piper ACT Deployment — Bottle Grasp (v0.5.2)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Load ACT policy ---
    print(f"\n[1/4] Loading ACT policy from {args.checkpt} ...")
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    chunk_size = policy.config.chunk_size
    n_action_steps = policy.config.n_action_steps
    print(f"  Policy loaded (chunk_size={chunk_size}, n_action_steps={n_action_steps}).")

    # --- Load pre/post processors ---
    print("\n[2/4] Loading pre/post processors from checkpoint ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    print("  Processors ready.")

    # --- Connect robot ---
    print(f"\n[3/4] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port)
    robot.connect()
    robot.enable(blocking=True)
    print("  Robot connected and enabled.")

    # --- Init cameras ---
    print("\n[4/4] Initializing cameras ...")
    rs_serials = find_realsense_devices()
    wrist_serial = rs_serials[0] if rs_serials else ""
    wrist_cam = RealSenseCamera(serial=wrist_serial, width=640, height=480, fps=30,
                                enable_depth=False)

    global_cam = None
    requires_global = "observation.images.global_rgb" in policy.config.input_features
    if args.no_global and requires_global:
        raise ValueError("This policy was trained with global_rgb, so --no-global cannot be used.")
    if not args.no_global:
        try:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
        except IOError as e:
            if requires_global:
                raise
            print(f"  Global camera skipped: {e}")
    print("  Cameras ready.")

    print("\n" + "-" * 60)
    print("  SPACE = run grasp    Q/ESC = quit")
    print("  Manually place the arm at the same start pose used for collection before SPACE.")
    if args.dry_run:
        print("  DRY RUN: robot commands will not be sent.")
    if args.replan_every_step:
        print("  REPLAN: policy will predict a fresh first action at every step.")
    print("-" * 60 + "\n")

    try:
        while True:
            # --- Live preview ---
            wrist_frame = wrist_cam.read()
            global_frame = global_cam.read() if global_cam else None

            preview = build_preview(wrist_frame, global_frame, "READY - SPACE run")
            cv2.imshow("ACT Deployment", preview)

            key = cv2.waitKey(1) & 0xFF
            if should_quit(key):
                break
            if key != ord(' '):
                continue

            # --- Execute grasp ---
            print("  >>> Grasp attempt ...")

            # Reset action queue before new trajectory
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            action_total = max(1, args.max_steps)
            last_target = None
            last_state = None
            for step in range(action_total):
                loop_start = time.time()

                # Capture fresh observation
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                robot_state = robot.get_joint_positions()

                # Build observation
                wrist_img = wrist_frame.rgb  # (H, W, 3) uint8
                global_img = global_frame.rgb if global_frame else None
                obs = prepare_observation(robot_state, wrist_img, global_img, device)

                # Run inference
                with torch.inference_mode():
                    normalized_obs = preprocessor(obs)
                    action = select_policy_action(
                        policy, postprocessor, normalized_obs, args.replan_every_step
                    )

                # action shape: (1, 7) -> (7,)
                if action.dim() == 2:
                    action = action.squeeze(0)
                target = action.cpu().numpy()

                # Safety clamp
                target[:6] = np.clip(target[:6], -3.14, 3.14)
                target[6] = np.clip(abs(target[6]), 0.0, PIPER_GRIPPER_MAX_M)

                delta = target - np.asarray(robot_state, dtype=np.float32)
                max_arm_delta = float(np.max(np.abs(delta[:6])))
                gripper_delta = float(abs(delta[6]))
                target_diff = max_abs_diff(target, last_target)
                state_diff = max_abs_diff(robot_state, last_state)
                if args.debug_actions and (
                    step == 0 or step == action_total - 1 or step % max(1, args.debug_every) == 0
                ):
                    print(
                        f"  step {step+1:03d}: "
                        f"max_arm_delta={max_arm_delta:.4f} rad, "
                        f"gripper_delta={gripper_delta:.4f} m, "
                        f"target_diff_from_last_target={target_diff:.4f}, "
                        f"state_diff_from_last_state={state_diff:.4f}"
                    )
                    print(f"    state : {fmt_vec(robot_state)}")
                    print(f"    target: {fmt_vec(target)}")
                    print(f"    delta : {fmt_vec(delta)}")

                if not args.dry_run:
                    sent = robot.set_joint_positions(target.tolist(), velocity_pct=args.velocity_pct)
                    if args.debug_actions and not sent:
                        print("    command: failed")

                # Update preview
                preview = build_preview(
                    wrist_frame, global_frame, f"EXEC {step+1}/{action_total}", color=(0, 0, 255)
                )
                cv2.imshow("ACT Deployment", preview)
                if should_quit(cv2.waitKey(1) & 0xFF):
                    break

                elapsed = time.time() - loop_start
                step_time = 1.0 / args.hz
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

                last_target = target.copy()
                last_state = np.asarray(robot_state, dtype=np.float32).copy()

            print("  Trajectory complete.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("  Shutting down ...")
        robot.disable()
        robot.disconnect()
        wrist_cam.close()
        if global_cam:
            global_cam.close()
        cv2.destroyAllWindows()
        print("  Done.")


if __name__ == "__main__":
    main()
