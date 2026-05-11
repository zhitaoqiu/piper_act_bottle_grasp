#!/usr/bin/env python3
"""Run offline eval over every checkpoint in a training output directory."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_MSE_RE = re.compile(r"Mean MSE across\s+\d+\s+episodes:\s+([0-9.eE+-]+)")


def checkpoint_sort_key(path: Path):
    if path.name == "last":
        return (1, float("inf"), path.name)
    match = re.search(r"(\d+)$", path.name)
    step = int(match.group(1)) if match else float("inf")
    return (0, step, path.name)


def find_checkpoints(train_output: Path) -> list[tuple[str, Path]]:
    checkpoints_dir = train_output / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")

    found = []
    for child in sorted(checkpoints_dir.iterdir(), key=checkpoint_sort_key):
        if not child.is_dir() and not child.is_symlink():
            continue
        pretrained = child / "pretrained_model"
        if pretrained.exists():
            found.append((child.name, pretrained))
        elif (child / "config.json").exists():
            found.append((child.name, child))
    return found


def parse_mean_mse(output: str) -> float | None:
    match = MEAN_MSE_RE.search(output)
    return float(match.group(1)) if match else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot_dataset"))
    parser.add_argument("--dataset-repo-id", default="piper/bottle_grasp")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/checkpoint_eval.csv"))
    parser.add_argument("--eval-script", type=Path, default=Path("inference/eval.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extra-eval-arg", action="append", default=[],
                        help="Additional argument passed to inference/eval.py. May be used multiple times.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = find_checkpoints(args.train_output)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {args.train_output / 'checkpoints'}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", "/tmp/piper_act_hf_cache/hf_home")
    env.setdefault("HF_DATASETS_CACHE", "/tmp/piper_act_hf_cache/datasets")

    rows = []
    for name, checkpoint_path in checkpoints:
        cmd = [
            args.python,
            str(args.eval_script),
            "--checkpt",
            str(checkpoint_path),
            "--dataset-root",
            str(args.dataset_root),
            "--dataset-repo-id",
            args.dataset_repo_id,
            "--episodes",
            str(args.episodes),
            "--no-plot",
            *args.extra_eval_arg,
        ]
        print(f"\n=== Evaluating {name}: {checkpoint_path} ===")
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        output = proc.stdout + "\n" + proc.stderr
        mse = parse_mean_mse(output)
        status = "ok" if proc.returncode == 0 and mse is not None else "failed"
        rows.append(
            {
                "checkpoint": name,
                "checkpoint_path": str(checkpoint_path),
                "mean_mse": "" if mse is None else mse,
                "returncode": proc.returncode,
                "status": status,
            }
        )
        if mse is not None:
            print(f"{name}: Mean MSE = {mse:.6f}")
        else:
            print(f"{name}: failed to parse Mean MSE (returncode={proc.returncode})")
            tail = "\n".join(output.strip().splitlines()[-20:])
            if tail:
                print(tail)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "checkpoint_path", "mean_mse", "returncode", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote checkpoint eval summary to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
