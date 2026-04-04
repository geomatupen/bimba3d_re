#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bimba3d_backend.worker.ai_adaptive_light import ACTIONS, TinyMLP


def load_dataset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Train lightweight adaptive MLP from offline transitions")
    parser.add_argument("--dataset", type=Path, default=Path("bimba3d_backend/data/_adaptive_ai_global/dataset_v1.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out-dir", type=Path, default=Path("bimba3d_backend/data/_adaptive_ai_global"))
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"dataset_not_found={args.dataset}")
        return 1

    items = load_dataset(args.dataset)
    if not items:
        print("dataset_empty=1")
        return 1

    feature_dim = len(items[0].get("features", []))
    if feature_dim <= 0:
        print("invalid_feature_dim=1")
        return 1

    model = TinyMLP(input_dim=feature_dim)

    for _ in range(max(1, int(args.epochs))):
        for row in items:
            features = np.asarray(row["features"], dtype=np.float64)
            action = str(row["action"])
            reward = float(row["reward"])
            if action not in ACTIONS:
                continue
            action_idx = ACTIONS.index(action)
            model.train_selected_action(features, action_idx, reward, learning_rate=float(args.lr))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = args.out_dir / "model_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "version": 1,
        "saved_at": time.time(),
        "trained_on_samples": len(items),
        "model": model.to_dict(),
    }
    version_name = f"model_{int(time.time())}.json"
    version_path = versions_dir / version_name
    version_path.write_text(json.dumps(model_payload), encoding="utf-8")

    previous_active = None
    registry_path = args.out_dir / "model_registry.json"
    if registry_path.exists():
        try:
            previous_active = json.loads(registry_path.read_text(encoding="utf-8")).get("active")
        except Exception:
            previous_active = None

    registry_payload = {
        "active": version_name,
        "previous": previous_active,
        "updated_at": time.time(),
    }
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")

    (args.out_dir / "model_v1.json").write_text(json.dumps(model_payload), encoding="utf-8")

    print(f"trained_samples={len(items)}")
    print(f"active_model={version_name}")
    print(f"model_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
