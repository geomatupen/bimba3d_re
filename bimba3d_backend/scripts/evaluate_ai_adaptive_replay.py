#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bimba3d_backend.worker.ai_adaptive_light import ACTIONS, TinyMLP


def load_model(model_root: Path) -> TinyMLP | None:
    registry_path = model_root / "model_registry.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            active = str(registry.get("active") or "").strip()
            if active:
                payload = json.loads((model_root / "model_versions" / active).read_text(encoding="utf-8"))
                model_data = payload.get("model") if isinstance(payload, dict) else None
                if isinstance(model_data, dict):
                    return TinyMLP.from_dict(model_data)
        except Exception:
            pass

    fallback = model_root / "model_v1.json"
    if fallback.exists():
        payload = json.loads(fallback.read_text(encoding="utf-8"))
        model_data = payload.get("model") if isinstance(payload, dict) else None
        if isinstance(model_data, dict):
            return TinyMLP.from_dict(model_data)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay evaluation for adaptive AI model")
    parser.add_argument("--dataset", type=Path, default=Path("bimba3d_backend/data/_adaptive_ai_global/dataset_v1.json"))
    parser.add_argument("--model-root", type=Path, default=Path("bimba3d_backend/data/_adaptive_ai_global"))
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"dataset_not_found={args.dataset}")
        return 1

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not items:
        print("dataset_empty=1")
        return 1

    model = load_model(args.model_root)
    if model is None:
        print(f"model_not_found={args.model_root}")
        return 1

    rewards: list[float] = []
    policy_rewards: list[float] = []
    action_matches = 0

    for row in items:
        x = np.asarray(row["features"], dtype=np.float64)
        _, _, logits = model.forward(x)
        chosen_action = ACTIONS[int(np.argmax(logits))]
        logged_action = str(row["action"])
        reward = float(row["reward"])
        rewards.append(reward)
        if chosen_action == logged_action:
            action_matches += 1
            policy_rewards.append(reward)

    match_rate = float(action_matches) / float(max(len(items), 1))
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_reward_on_matches = float(np.mean(policy_rewards)) if policy_rewards else 0.0

    print(f"samples={len(items)}")
    print(f"match_rate={match_rate:.4f}")
    print(f"avg_reward={avg_reward:.6f}")
    print(f"avg_reward_on_matches={avg_reward_on_matches:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
