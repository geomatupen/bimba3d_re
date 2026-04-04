#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(args: list[str]) -> int:
    proc = subprocess.run(args, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset build + train + replay eval for adaptive AI model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", default="bimba3d_backend/data/_adaptive_ai_global/dataset_v1.json")
    parser.add_argument("--model-root", default="bimba3d_backend/data/_adaptive_ai_global")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    model_root = Path(args.model_root)

    build_cmd = [
        args.python,
        "-m",
        "bimba3d_backend.scripts.build_ai_adaptive_dataset",
        "--out",
        str(dataset),
    ]
    train_cmd = [
        args.python,
        "-m",
        "bimba3d_backend.scripts.train_ai_adaptive_model",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(model_root),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
    ]
    eval_cmd = [
        args.python,
        "-m",
        "bimba3d_backend.scripts.evaluate_ai_adaptive_replay",
        "--dataset",
        str(dataset),
        "--model-root",
        str(model_root),
    ]

    if run_cmd(build_cmd) != 0:
        return 1
    if run_cmd(train_cmd) != 0:
        return 1
    if run_cmd(eval_cmd) != 0:
        return 1

    print("continual_update_status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
