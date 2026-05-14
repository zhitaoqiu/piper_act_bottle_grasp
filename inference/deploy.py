#!/usr/bin/env python3
"""
Deploy trained ACT policy on Piper arm for bottle grasping.

Usage:
  conda activate piper_act
  # Test mode A (approach only):
  python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_approach_tiny_1ep/checkpoints/003000/pretrained_model \
    --test-mode A --debug-actions --replan-every-step
  # Test mode B (approach + close + lift):
  python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_approach_tiny_1ep/checkpoints/003000/pretrained_model \
    --test-mode B --debug-actions --replan-every-step

Controls:
  SPACE  = run one approach attempt
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
from hardware.config_piper import PiperRobotConfig
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

PIPER_GRIPPER_MAX_M = 0.101

# ── Approach-phase constants ──
GRIPPER_OPEN = 0.08          # gripper fully open (m)
GRIPPER_CLOSE = 0.0          # gripper fully closed (m)
# Per-joint max delta: J1-J3 arm joints get 0.03, J4-J6 wrist get 0.012
MAX_DELTA_PER_JOINT = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012], dtype=np.float32)
ACTION_SMOOTH_ALPHA = 0.5    # EMA smoothing factor
APPROACH_STEPS_DEFAULT = 200
WRIST_FREEZE_J2 = 1.45       # freeze J4-J6 when J2 exceeds this
READY_J2 = 1.50              # J2 threshold for ready_count
READY_COUNT_MIN = 5          # consecutive steps above READY_J2 to trigger stop
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008  # rad — below this for N consecutive steps = stuck


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


def prepare_observation(state, wrist_img, global_img, device, expected_state_dim=7, phase=0.0,
                        gripper_unit_scale=1.0):
    """Convert raw numpy data to inference-ready tensors (batched, on device).

    state: raw robot joint positions [j1..j6, gripper] in robot units.
    gripper_unit_scale: multiply state[6] by this before feeding to policy,
      so the policy sees values in its training distribution.
    """
    obs = {}
    state_arr = np.asarray(state, dtype=np.float32).copy()
    state_arr[6] *= gripper_unit_scale
    if expected_state_dim == len(state_arr) + 1:
        state_arr = np.concatenate([state_arr, np.asarray([phase], dtype=np.float32)])
    elif expected_state_dim != len(state_arr):
        raise ValueError(
            f"Policy expects observation.state dim {expected_state_dim}, "
            f"but robot provides {len(state_arr)} joints. Only dim 7 or 8-with-phase is supported."
        )
    obs["observation.state"] = torch.from_numpy(
        state_arr
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
    preview = cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if global_frame is not None:
        g_preview = cv2.cvtColor(global_frame.rgb, cv2.COLOR_RGB2BGR)
        g_preview = cv2.resize(g_preview, (preview.shape[1], preview.shape[0]))
        preview = np.hstack([preview, g_preview])
    return preview


def should_quit(key: int) -> bool:
    return key in (27, ord('q'), ord('Q'))


def fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def policy_state_dim(policy) -> int:
    feature = policy.config.input_features.get("observation.state")
    if feature is None:
        return 0
    return int(feature.shape[0])


def max_abs_diff(cur, prev) -> float:
    if prev is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(cur, dtype=np.float32) - np.asarray(prev, dtype=np.float32))))


def interpolate_joint_path(start: np.ndarray, target: np.ndarray,
                           max_step_rad: float, max_step_gripper: float):
    """Generate intermediate joint targets from start to target (excluding start, including target)."""
    diff = np.asarray(target, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)
    waypoints = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp = np.asarray(start, dtype=np.float32) + diff * alpha
        waypoints.append(interp)
    return waypoints


def select_policy_action(policy, postprocessor, normalized_obs, replan_every_step: bool):
    """Return (first_action, full_chunk). full_chunk is None when queue-based."""
    if replan_every_step:
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(normalized_obs)
            action_chunk = postprocessor(action_chunk)
            return action_chunk[:, 0, :], action_chunk.squeeze(0).cpu().numpy()  # (chunk, 7)

        policy.reset()

    action = policy.select_action(normalized_obs)
    return postprocessor(action), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True,
                        help="Path to trained ACT checkpoint")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control loop frequency. Keep this equal to dataset fps.")
    parser.add_argument("--max-steps", type=int, default=APPROACH_STEPS_DEFAULT,
                        help="Maximum action steps for one grasp attempt.")
    parser.add_argument("--test-mode", choices=("A", "B", "C", "D"), default="A",
                        help="A: approach only. B: approach + close + lift. C: approach + descend. D: full grasp + place + release.")
    parser.add_argument("--descend-j2-delta", type=float, default=0.04,
                        help="J2 increment (rad) for descent phase in test mode C.")
    parser.add_argument("--place-j1-offset", type=float, default=0.30,
                        help="J1 offset (rad) to move bottle to side before release (test mode D).")
    parser.add_argument("--approach-steps", type=int, default=APPROACH_STEPS_DEFAULT,
                        help="Stop ACT after this many steps and begin handover to code control.")
    parser.add_argument("--action-smooth", type=float, default=ACTION_SMOOTH_ALPHA,
                        help="EMA smoothing factor for consecutive action predictions (0=disabled, 0.3=default).")
    parser.add_argument("--no-global", action="store_true",
                        help="Disable global camera")
    parser.add_argument("--global-camera", type=str, default="auto",
                        help="Global camera device ID or 'auto'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run policy and preview targets without sending robot commands.")
    parser.add_argument("--debug-actions", action="store_true",
                        help="Print raw/clipped/smoothed/sent action at every debug-every step.")
    parser.add_argument("--debug-every", type=int, default=10,
                        help="Print one debug line every N action steps.")
    parser.add_argument("--replan-every-step", action="store_true",
                        help="Recompute a fresh ACT action chunk every control step instead of consuming the action queue.")
    parser.add_argument("--action-mode", choices=("absolute", "delta"), default="absolute",
                        help="Interpret policy action as absolute joint target or state-relative delta waypoint.")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="Multiplier for model-predicted delta before adding to current state.")
    parser.add_argument("--arm-scale", type=float, default=None,
                        help="Scale for J1/J2/J3 (arm joints). Defaults to --delta-scale.")
    parser.add_argument("--wrist-scale", type=float, default=None,
                        help="Scale for J4/J5/J6 (wrist joints). Defaults to --delta-scale.")
    parser.add_argument("--gripper-scale", type=float, default=None,
                        help="Scale for gripper (dim 7). Defaults to --delta-scale.")
    parser.add_argument("--gripper-deadband", type=float, default=0.0,
                        help="Ignore raw gripper action below this absolute value.")
    parser.add_argument("--min-gripper-delta", type=float, default=0.0,
                        help="If abs(raw_gripper) > deadband but abs(scaled) < min, override to min_gripper_delta.")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable cv2 GUI windows; use terminal input (Enter=grasp, q=quit).")
    parser.add_argument("--gripper-unit-scale", type=float, default=1.0,
                        help="Scale robot gripper state before feeding to policy, "
                             "and inverse-scale the target back before sending to robot.")
    parser.add_argument("--training-gripper-min", type=float, default=None,
                        help="Min gripper value in training data, for OOD warning.")
    parser.add_argument("--training-gripper-max", type=float, default=None,
                        help="Max gripper value in training data, for OOD warning.")
    parser.add_argument("--no-return-to-start", action="store_true",
                        help="Disable automatic return to start_pose after trajectory.")
    parser.add_argument("--max-joint-delta", type=float, default=None,
                        help="Override per-joint max delta for all arm joints. 0=disabled. Default uses per-joint limits.")
    args = parser.parse_args()

    # Build per-dimension scale array: [J1, J2, J3, J4, J5, J6, Grip]
    arm_s = args.arm_scale if args.arm_scale is not None else args.delta_scale
    wrist_s = args.wrist_scale if args.wrist_scale is not None else args.delta_scale
    grip_s = args.gripper_scale if args.gripper_scale is not None else args.delta_scale
    dim_scale = np.array([arm_s, arm_s, arm_s, wrist_s, wrist_s, wrist_s, grip_s], dtype=np.float32)

    # Build per-joint max delta: use override if provided, else per-joint defaults
    if args.max_joint_delta is not None and args.max_joint_delta > 0:
        max_delta = np.full(6, args.max_joint_delta, dtype=np.float32)
    else:
        max_delta = MAX_DELTA_PER_JOINT.copy()

    print("=" * 60)
    print("  Piper ACT Deployment — Bottle Approach (v0.7.0)")
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
    expected_state_dim = policy_state_dim(policy)
    uses_phase = expected_state_dim == 8
    print(
        f"  Policy loaded (chunk_size={chunk_size}, n_action_steps={n_action_steps}, "
        f"state_dim={expected_state_dim})."
    )

    # --- Load pre/post processors ---
    print("\n[2/4] Loading pre/post processors from checkpoint ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    print("  Processors ready.")

    # --- Connect robot ---
    print(f"\n[3/4] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port, disable_torque_on_disconnect=False)
    robot.connect()  # connect + enable in one call
    print("  Robot connected and enabled (disable_torque_on_disconnect=False).")

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

    # --- Gripper distribution check ---
    robot_state = robot.get_joint_positions()
    grip_raw = robot_state[6]
    grip_policy = grip_raw * args.gripper_unit_scale
    print(f"\n  Gripper state (robot units): {grip_raw:.6f}")
    if args.gripper_unit_scale != 1.0:
        print(f"  Gripper state (policy units): {grip_policy:.6f}  [×{args.gripper_unit_scale}]")
    if args.training_gripper_min is not None and args.training_gripper_max is not None:
        if grip_policy < args.training_gripper_min or grip_policy > args.training_gripper_max:
            print(f"  [WARN] Deployment gripper state ({grip_policy:.4f} in policy units)"
                  f" is outside training distribution"
                  f" [{args.training_gripper_min:.3f}, {args.training_gripper_max:.3f}].")

    print("\n" + "-" * 60)
    print("  SPACE = run approach    Q/ESC = quit")
    print(f"  TEST MODE: {args.test_mode}  |  APPROACH STEPS: {args.approach_steps}"
          f"  |  SMOOTH: α={args.action_smooth}")
    print(f"  Per-joint max_delta: J1-J3={max_delta[0]:.3f}  J4-J6={max_delta[3]:.3f} rad")
    print(f"  Wrist freeze @ J2 > {WRIST_FREEZE_J2:.2f}  |  Ready stop @ J2 > {READY_J2:.2f} ×{READY_COUNT_MIN}")
    print(f"  Gripper forced OPEN ({GRIPPER_OPEN:.3f} m) during ACT approach.")
    if args.test_mode == "A":
        print("  → Approach only — no close, no lift.")
    elif args.test_mode == "C":
        print(f"  → Approach + descend (J2 += {args.descend_j2_delta:.3f} rad) — no close.")
    elif args.test_mode == "D":
        print(f"  → Full grasp: approach + close + lift + place(J1+={args.place_j1_offset:.2f}) + release + return.")
    else:
        print(f"  → Approach + close ({GRIPPER_CLOSE:.3f} m) + lift (J3 -= 0.06).")
    if args.dry_run:
        print("  DRY RUN: robot commands will not be sent.")
    if args.replan_every_step:
        print("  REPLAN: policy will predict a fresh first action at every step.")
    print("-" * 60 + "\n")

    try:
        while True:
            # --- Live preview ---
            if args.no_gui:
                cmd = input("  Press ENTER to run approach, Q then ENTER to quit: ").strip().lower()
                if cmd == "q":
                    break
            else:
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None

                preview = build_preview(wrist_frame, global_frame, "READY - SPACE run")
                cv2.imshow("ACT Deployment", preview)

                key = cv2.waitKey(1) & 0xFF
                if should_quit(key):
                    break
                if key != ord(' '):
                    continue

            # ================================================================
            #  ACT APPROACH PHASE
            # ================================================================
            print(f"  >>> Approach attempt (test-mode={args.test_mode}, {args.approach_steps} steps) ...")

            # Save start position for auto return
            start_robot_state = np.asarray(robot.get_joint_positions(), dtype=np.float32)

            # Reset action queue before new trajectory
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            approach_steps = args.approach_steps
            last_smoothed = None
            last_state = None
            raw_actions = []
            paused = False
            user_quit = False
            stagnation_count = 0
            ready_count = 0
            stop_reason = "completed"

            for step in range(approach_steps):
                loop_start = time.time()

                # Capture fresh observation
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                robot_state = robot.get_joint_positions()
                phase = 0.0 if approach_steps <= 1 else min(1.0, step / float(approach_steps - 1))

                # Build observation
                wrist_img = wrist_frame.rgb
                global_img = global_frame.rgb if global_frame else None
                obs = prepare_observation(
                    robot_state, wrist_img, global_img, device, expected_state_dim, phase,
                    gripper_unit_scale=args.gripper_unit_scale,
                )

                # Run inference
                with torch.inference_mode():
                    normalized_obs = preprocessor(obs)
                    action, _ = select_policy_action(
                        policy, postprocessor, normalized_obs, args.replan_every_step
                    )

                # action shape: (1, 7) -> (7,)
                if action.dim() == 2:
                    action = action.squeeze(0)
                model_action = action.cpu().numpy()
                raw_actions.append(model_action.copy())

                robot_state_arr = np.asarray(robot_state, dtype=np.float32)

                # ── Compute raw_target from model output ──
                if args.action_mode == "delta":
                    scaled_action = model_action * dim_scale
                    policy_state_arr = robot_state_arr.copy()
                    policy_state_arr[6] *= args.gripper_unit_scale
                    policy_target = policy_state_arr + scaled_action
                    raw_target = policy_target.copy()
                    raw_target[6] /= args.gripper_unit_scale
                else:
                    raw_target = model_action.copy()

                # ── Step 1: per-joint independent delta clamp ──
                raw_delta = raw_target - robot_state_arr
                for j in range(6):
                    raw_delta[j] = np.clip(raw_delta[j], -max_delta[j], max_delta[j])
                clipped = robot_state_arr + raw_delta

                # Force gripper open throughout approach
                clipped[6] = GRIPPER_OPEN

                # ── Step 2: wrist freeze when J2 > WRIST_FREEZE_J2 ──
                wrist_frozen = False
                if robot_state_arr[1] > WRIST_FREEZE_J2:
                    clipped[3:6] = robot_state_arr[3:6]
                    wrist_frozen = True

                # ── Step 3: EMA smoothing ──
                alpha = args.action_smooth
                if last_smoothed is not None and alpha > 0:
                    smoothed_arm = alpha * clipped[:6] + (1.0 - alpha) * last_smoothed[:6]
                else:
                    smoothed_arm = clipped[:6].copy()
                sent_target = np.concatenate([smoothed_arm, [GRIPPER_OPEN]])

                # Safety clamp to joint limits
                sent_target[:6] = np.clip(sent_target[:6], -3.14, 3.14)
                sent_target[6] = np.clip(sent_target[6], 0.0, PIPER_GRIPPER_MAX_M)

                # ── Ready stop: J2 > READY_J2 for READY_COUNT_MIN consecutive steps ──
                if robot_state_arr[1] > READY_J2 and step > 150:
                    ready_count += 1
                else:
                    ready_count = 0
                stop_act = (ready_count >= READY_COUNT_MIN) or (step + 1 >= approach_steps)

                # ── Logging ──
                if args.debug_actions and (
                    step == 0 or step == approach_steps - 1 or (step + 1) % args.debug_every == 0
                    or wrist_frozen or stop_act
                ):
                    print(f"  --- step {step+1:03d}/{approach_steps} ---")
                    print(f"    robot_state  : {fmt_vec(robot_state_arr)}")
                    print(f"    raw_action   : {fmt_vec(raw_target)}")
                    print(f"    clipped      : {fmt_vec(clipped)}")
                    print(f"    smoothed     : {fmt_vec(sent_target)}")
                    print(f"    sent_target  : {fmt_vec(sent_target)}")
                    sent_delta = sent_target - robot_state_arr
                    print(f"    delta (sent) : {fmt_vec(sent_delta)}")
                    print(f"    J2={robot_state_arr[1]:.4f}  wrist_frozen={wrist_frozen}"
                          f"  ready={ready_count}/{READY_COUNT_MIN}  stop_act={stop_act}")

                # ── Safety stop: joint limit violation ──
                if np.any(np.abs(sent_target[:6]) > 3.0):
                    print(f"\n  [STOP] Joint limit violation: target={fmt_vec(sent_target)}")
                    stop_reason = "joint_limit"
                    break

                # ── Safety stop: stagnation (state barely moves, not near end) ──
                state_diff = max_abs_diff(robot_state, last_state)
                near_end = step > approach_steps * 0.7
                if not near_end and last_state is not None and state_diff < STAGNATION_THRESHOLD:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count >= STAGNATION_STEPS:
                    print(f"\n  [STOP] Stagnation: {STAGNATION_STEPS} consecutive steps"
                          f" with state_diff < {STAGNATION_THRESHOLD} before 70% progress")
                    print(f"    step={step+1}/{approach_steps}  state_diff={state_diff:.6f}")
                    stop_reason = "stagnation"
                    break

                # ── Ready stop: break after sending ──
                if stop_act:
                    if ready_count >= READY_COUNT_MIN:
                        stop_reason = "ready"
                    else:
                        stop_reason = "max_steps"
                    # Send the final target, then break
                    if not args.dry_run:
                        robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)
                    # Update last_smoothed before breaking
                    last_smoothed = smoothed_arm.copy()
                    last_state = np.asarray(robot_state, dtype=np.float32).copy()
                    print(f"\n  [STOP] Approach complete ({stop_reason})"
                          f"  J2={robot_state_arr[1]:.4f}  step={step+1}")
                    break

                # ── Send to robot ──
                if not args.dry_run:
                    robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)

                # Update preview + handle pause/quit
                if not args.no_gui:
                    label = f"PAUSED {step+1}/{approach_steps}" if paused else f"APPROACH {step+1}/{approach_steps}"
                    color = (0, 165, 255) if paused else (0, 0, 255)
                    preview = build_preview(wrist_frame, global_frame, label, color=color)
                    cv2.imshow("ACT Deployment", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if should_quit(key):
                        user_quit = True
                        stop_reason = "user_quit"
                        break
                    if key == ord(' '):
                        paused = not paused
                        if paused:
                            print("  ⏸  PAUSED — SPACE to resume, Q to quit")
                        else:
                            print("  ▶  RESUMED")

                elapsed = time.time() - loop_start
                step_time = 1.0 / args.hz
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

                # ── Pause loop ──
                while paused:
                    if not args.no_gui:
                        preview = build_preview(wrist_frame, global_frame,
                                                f"PAUSED {step+1}/{approach_steps}", color=(0, 165, 255))
                        cv2.imshow("ACT Deployment", preview)
                    if last_smoothed is not None and not args.dry_run:
                        hold_pos = np.concatenate([last_smoothed, [GRIPPER_OPEN]])
                        robot.set_joint_positions(hold_pos.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                    if not args.no_gui:
                        key = cv2.waitKey(1) & 0xFF
                        if should_quit(key):
                            paused = False
                            user_quit = True
                            stop_reason = "user_quit"
                            break
                        if key == ord(' '):
                            paused = False
                            print("  ▶  RESUMED")
                    else:
                        import select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            line = sys.stdin.readline().strip().lower()
                            if line == 'q':
                                paused = False
                                user_quit = True
                                stop_reason = "user_quit"
                                break
                            if line == '':
                                paused = False
                                print("  ▶  RESUMED")
                if user_quit:
                    break

                last_smoothed = smoothed_arm.copy()
                last_state = np.asarray(robot_state, dtype=np.float32).copy()

            # ── Print raw action stats ──
            if raw_actions:
                ra = np.array(raw_actions)
                jnames = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
                print(f"\n  Raw action stats over {len(raw_actions)} steps:")
                print(f"  {'Dim':>6}  {'mean':>12}  {'abs_mean':>12}  {'min':>12}  {'max':>12}")
                for d in range(ra.shape[1]):
                    print(f"  {jnames[d]:>6}  {ra[:, d].mean():12.6f}  "
                          f"{np.abs(ra[:, d]).mean():12.6f}  "
                          f"{ra[:, d].min():12.6f}  {ra[:, d].max():12.6f}")

            print(f"\n  ACT approach finished ({stop_reason}, {len(raw_actions)} steps).")

            # ================================================================
            #  HANDOVER: hold position
            # ================================================================
            if not user_quit and not args.dry_run:
                print("  >>> Handover: hold position (0.3s) ...")
                cur = robot.get_joint_positions()
                hold_start = time.time()
                while time.time() - hold_start < 0.3:
                    robot.set_joint_positions(cur, velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Hold complete.")

            # ================================================================
            #  TEST MODE A: approach only — done
            # ================================================================
            if args.test_mode == "A":
                if not user_quit:
                    print("  [TEST-A] Approach only — gripper stays open, no close/lift.")
                    print("  [TEST-A] Check: is gripper aligned with bottle at pre-grasp position?")
                    final_state = robot.get_joint_positions()
                    print(f"  [TEST-A] Final J2 = {final_state[1]:.5f} rad")

            # ================================================================
            #  TEST MODE C: approach → descend 2-3cm → stop (no close)
            # ================================================================
            if args.test_mode == "C" and not user_quit:
                print(f"\n  >>> [TEST-C] Descending (J2 += {args.descend_j2_delta:.3f} rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                descend_pose = cur.copy()
                descend_pose[1] += args.descend_j2_delta
                descend_pose[1] = np.clip(descend_pose[1], -3.14, 3.14)
                descend_pose[6] = GRIPPER_OPEN
                descend_path = interpolate_joint_path(cur, descend_pose,
                                                      max_step_rad=0.015, max_step_gripper=0.002)
                for di, dt in enumerate(descend_path):
                    t_start = time.time()
                    dt[6] = GRIPPER_OPEN
                    if not args.dry_run:
                        robot.set_joint_positions(dt.tolist(), velocity_pct=args.velocity_pct)
                    if di == 0 or di == len(descend_path) - 1:
                        print(f"    descend {di+1:3d}/{len(descend_path):3d}  {fmt_vec(dt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                # Hold at descended position
                hold_start = time.time()
                print("  Holding descended position for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(descend_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                final = robot.get_joint_positions()
                print(f"  Descent complete. Final J2 = {final[1]:.5f} rad")
                print("  [TEST-C] Check: is gripper at correct grasp depth?")

            # ================================================================
            #  TEST MODE B: close gripper + lift
            # ================================================================
            if args.test_mode == "B" and not user_quit:
                # --- Close gripper ---
                print(f"\n  >>> [TEST-B] Closing gripper to {GRIPPER_CLOSE:.3f} m ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close_pose = cur.copy()
                close_pose[6] = GRIPPER_CLOSE
                close_path = interpolate_joint_path(cur, close_pose,
                                                    max_step_rad=0.02, max_step_gripper=0.002)
                for ci, ct in enumerate(close_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                    if ci == 0 or ci == len(close_path) - 1:
                        print(f"    close {ci+1:3d}/{len(close_path):3d}  grip={ct[6]:.4f}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                # Hold closed
                hold_start = time.time()
                print("  Holding close for 0.6s ...")
                while time.time() - hold_start < 0.6:
                    if not args.dry_run:
                        robot.set_joint_positions(close_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Gripper closed.")

                # --- Lift: J3 -= 0.06 rad ---
                print("\n  >>> [TEST-B] Lifting (J3 -= 0.06 rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                lift_pose = cur.copy()
                lift_pose[2] -= 0.06
                lift_pose[2] = np.clip(lift_pose[2], -3.14, 3.14)
                lift_pose[6] = GRIPPER_CLOSE  # keep gripper closed
                lift_path = interpolate_joint_path(cur, lift_pose,
                                                   max_step_rad=0.02, max_step_gripper=0.002)
                for li, lt in enumerate(lift_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                    if li == 0 or li == len(lift_path) - 1:
                        print(f"    lift {li+1:3d}/{len(lift_path):3d}  {fmt_vec(lt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                # Hold lift
                hold_start = time.time()
                print("  Holding lift for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(lift_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Lift complete.")
                print("  [TEST-B] Verify: is bottle grasped and lifted?")

            # ================================================================
            #  TEST MODE D: full grasp → place → release → return
            # ================================================================
            if args.test_mode == "D" and not user_quit:
                # --- Close gripper (same as Test B) ---
                print(f"\n  >>> [TEST-D] Closing gripper to {GRIPPER_CLOSE:.3f} m ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close_pose = cur.copy()
                close_pose[6] = GRIPPER_CLOSE
                close_path = interpolate_joint_path(cur, close_pose,
                                                    max_step_rad=0.02, max_step_gripper=0.002)
                for ci, ct in enumerate(close_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                    if ci == 0 or ci == len(close_path) - 1:
                        print(f"    close {ci+1:3d}/{len(close_path):3d}  grip={ct[6]:.4f}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                hold_start = time.time()
                print("  Holding close for 0.6s ...")
                while time.time() - hold_start < 0.6:
                    if not args.dry_run:
                        robot.set_joint_positions(close_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Gripper closed.")

                # --- Lift: J3 -= 0.06 ---
                print("\n  >>> [TEST-D] Lifting (J3 -= 0.06 rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                lift_pose = cur.copy()
                lift_pose[2] -= 0.06
                lift_pose[2] = np.clip(lift_pose[2], -3.14, 3.14)
                lift_pose[6] = GRIPPER_CLOSE
                lift_path = interpolate_joint_path(cur, lift_pose,
                                                   max_step_rad=0.02, max_step_gripper=0.002)
                for li, lt in enumerate(lift_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                    if li == 0 or li == len(lift_path) - 1:
                        print(f"    lift {li+1:3d}/{len(lift_path):3d}  {fmt_vec(lt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                hold_start = time.time()
                print("  Holding lift for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(lift_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Lift complete.")

                # --- Place: J1 offset to move bottle to side ---
                print(f"\n  >>> [TEST-D] Moving to side (J1 += {args.place_j1_offset:.2f} rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                place_pose = cur.copy()
                place_pose[0] += args.place_j1_offset
                place_pose[0] = np.clip(place_pose[0], -3.14, 3.14)
                place_pose[6] = GRIPPER_CLOSE
                place_path = interpolate_joint_path(cur, place_pose,
                                                    max_step_rad=0.03, max_step_gripper=0.002)
                for pi, pt in enumerate(place_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(pt.tolist(), velocity_pct=args.velocity_pct)
                    if pi == 0 or pi == len(place_path) - 1 or (pi + 1) % 5 == 0:
                        print(f"    place {pi+1:3d}/{len(place_path):3d}  {fmt_vec(pt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Moved to side.")

                # --- Release: open gripper ---
                print(f"\n  >>> [TEST-D] Releasing gripper (open to {GRIPPER_OPEN:.3f} m) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                release_pose = cur.copy()
                release_pose[6] = GRIPPER_OPEN
                release_path = interpolate_joint_path(cur, release_pose,
                                                      max_step_rad=0.02, max_step_gripper=0.004)
                for ri, rt in enumerate(release_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                    if ri == 0 or ri == len(release_path) - 1:
                        print(f"    release {ri+1:3d}/{len(release_path):3d}  grip={rt[6]:.4f}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Gripper released.")

                # Dwell after release
                hold_start = time.time()
                print("  Holding release for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(release_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  [TEST-D] Full grasp + place + release complete.")

            # ================================================================
            #  RETURN TO START
            # ================================================================
            auto_return = (not args.no_return_to_start and not user_quit and not args.dry_run)
            if auto_return:
                start_pose = start_robot_state.copy()
                print("\n  >>> Returning to start position ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                start_pose[6] = cur[6]  # preserve current gripper
                path = interpolate_joint_path(cur, start_pose, max_step_rad=0.03, max_step_gripper=0.004)
                for ri, rt in enumerate(path):
                    t_start = time.time()
                    rt_clamped = rt.copy()
                    rt_clamped[:6] = np.clip(rt_clamped[:6], -3.14, 3.14)
                    rt_clamped[6] = np.clip(rt_clamped[6], 0.0, PIPER_GRIPPER_MAX_M)
                    if not args.dry_run:
                        robot.set_joint_positions(rt_clamped.tolist(), velocity_pct=args.velocity_pct)
                    if ri == 0 or ri == len(path) - 1 or (ri + 1) % 10 == 0:
                        print(f"    return {ri+1:3d}/{len(path):3d}  {fmt_vec(rt_clamped, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Returned to start position.")

            print("  Trajectory complete.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        # Emergency stop: hold current position, do NOT disable
        try:
            cur = robot.get_joint_positions()
            robot.set_joint_positions(cur, velocity_pct=50)
        except Exception:
            pass
        print("  Stopped. Arm stays ENABLED at current position.")
        wrist_cam.close()
        if global_cam:
            global_cam.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
