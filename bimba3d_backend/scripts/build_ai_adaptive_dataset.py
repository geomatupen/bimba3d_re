#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_project_run_logs(projects_root: Path):
    for project_dir in sorted(projects_root.glob("*")):
        if not project_dir.is_dir():
            continue
        runs_dir = project_dir / "adaptive_ai" / "runs"
        if not runs_dir.exists():
            continue
        for log_path in sorted(runs_dir.glob("*.jsonl")):
            yield project_dir.name, log_path


def build_dataset(projects_root: Path) -> list[dict]:
    dataset: list[dict] = []
    for project_id, log_path in iter_project_run_logs(projects_root):
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                transition = event.get("trained_transition")
                if not isinstance(transition, dict):
                    continue
                features = transition.get("features")
                action = transition.get("action")
                reward = transition.get("reward")
                if not isinstance(features, list) or action is None or reward is None:
                    continue
                dataset.append(
                    {
                        "project_id": project_id,
                        "run_log": str(log_path),
                        "step": int(transition.get("step") or 0),
                        "features": [float(v) for v in features],
                        "action": str(action),
                        "reward": float(reward),
                    }
                )
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline dataset from adaptive AI run logs")
    parser.add_argument("--projects-root", type=Path, default=Path("bimba3d_backend/data/projects"))
    parser.add_argument("--out", type=Path, default=Path("bimba3d_backend/data/_adaptive_ai_global/dataset_v1.json"))
    args = parser.parse_args()

    dataset = build_dataset(args.projects_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(dataset),
        "items": dataset,
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"dataset_samples={len(dataset)}")
    print(f"dataset_path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
