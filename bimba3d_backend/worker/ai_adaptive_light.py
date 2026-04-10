from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Action space used by the lightweight controller.
# Keep actions are intentionally explicit to simplify event logs and analysis.
ACTION_KEEP = "keep"
ACTION_LR_UP = "lr_up_3pct"
ACTION_LR_DOWN = "lr_down_3pct"
ACTION_DENSIFY_UP = "densify_thresh_up_small"
ACTION_DENSIFY_DOWN = "densify_thresh_down_small"
ACTION_PRUNE_UP = "prune_aggr_up_small"

ACTIONS = [
    ACTION_KEEP,
    ACTION_LR_UP,
    ACTION_LR_DOWN,
    ACTION_DENSIFY_UP,
    ACTION_DENSIFY_DOWN,
    ACTION_PRUNE_UP,
]


@dataclass
class AdaptiveDecision:
    """Single controller decision returned to the training loop."""

    action: str
    reason: str
    gate_threshold: float
    relative_improvement: float | None
    reward_from_previous: float | None
    features: list[float]
    action_scores: list[float]


class TinyMLP:
    """Small fully-connected policy/value approximator used online during training.

    The network predicts one score per action. The selected action's score is then
    nudged toward the observed reward from the next decision step.
    """

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 64, output_dim: int = len(ACTIONS), seed: int = 7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.08, size=(input_dim, hidden1)).astype(np.float64)
        self.b1 = np.zeros((hidden1,), dtype=np.float64)
        self.W2 = rng.normal(0.0, 0.08, size=(hidden1, hidden2)).astype(np.float64)
        self.b2 = np.zeros((hidden2,), dtype=np.float64)
        self.W3 = rng.normal(0.0, 0.08, size=(hidden2, output_dim)).astype(np.float64)
        self.b3 = np.zeros((output_dim,), dtype=np.float64)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Return hidden activations as well so backprop can reuse them.
        h1_pre = x @ self.W1 + self.b1
        h1 = np.tanh(h1_pre)
        h2_pre = h1 @ self.W2 + self.b2
        h2 = np.tanh(h2_pre)
        logits = h2 @ self.W3 + self.b3
        return h1, h2, logits

    def train_selected_action(self, x: np.ndarray, action_idx: int, reward: float, learning_rate: float = 0.001) -> None:
        # One-step supervised target: selected action score should match observed reward.
        h1, h2, logits = self.forward(x)
        pred = float(logits[action_idx])
        grad_out = np.zeros_like(logits)
        grad_out[action_idx] = pred - float(reward)

        grad_W3 = np.outer(h2, grad_out)
        grad_b3 = grad_out

        grad_h2 = self.W3 @ grad_out
        grad_h2_pre = grad_h2 * (1.0 - np.square(h2))

        grad_W2 = np.outer(h1, grad_h2_pre)
        grad_b2 = grad_h2_pre

        grad_h1 = self.W2 @ grad_h2_pre
        grad_h1_pre = grad_h1 * (1.0 - np.square(h1))

        grad_W1 = np.outer(x, grad_h1_pre)
        grad_b1 = grad_h1_pre

        self.W3 -= learning_rate * grad_W3
        self.b3 -= learning_rate * grad_b3
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1

    def to_dict(self) -> dict[str, Any]:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TinyMLP":
        model = cls(input_dim=len(payload.get("W1", [])))
        model.W1 = np.asarray(payload["W1"], dtype=np.float64)
        model.b1 = np.asarray(payload["b1"], dtype=np.float64)
        model.W2 = np.asarray(payload["W2"], dtype=np.float64)
        model.b2 = np.asarray(payload["b2"], dtype=np.float64)
        model.W3 = np.asarray(payload["W3"], dtype=np.float64)
        model.b3 = np.asarray(payload["b3"], dtype=np.float64)
        return model


class CoreAIAdaptiveController:
    """Lightweight adaptive controller that tunes LR/strategy during gsplat runs.

    Main loop:
    1) Build features from recent loss/gaussian/optimizer state.
    2) Score actions with TinyMLP.
    3) Apply safety gates/cooldowns and (optionally) apply action.
    4) Train previous transition using newly observed reward.
    """

    def __init__(
        self,
        *,
        project_dir: Path,
        run_id: str,
        max_steps: int,
        tune_start_step: int,
        tune_end_step: int,
        strategy_start_step: int,
        strategy_end_step: int,
        base_min_improvement: float,
        decision_interval: int,
        reward_step_weight: float = 0.65,
        reward_trend_weight: float = 0.35,
        trend_scope: str = "run",
    ):
        self.project_dir = Path(project_dir)
        self.run_id = str(run_id)
        self.max_steps = max(1, int(max_steps))
        self.tune_start_step = max(1, int(tune_start_step))
        self.tune_end_step = max(self.tune_start_step, int(tune_end_step))
        self.strategy_start_step = max(1, int(strategy_start_step))
        self.strategy_end_step = max(self.strategy_start_step, int(strategy_end_step))
        self.base_min_improvement = float(max(0.0, min(1.0, base_min_improvement)))
        self.decision_interval = max(1, int(decision_interval))
        self.reward_step_weight = max(0.0, float(reward_step_weight))
        self.reward_trend_weight = max(0.0, float(reward_trend_weight))
        self.trend_scope = str(trend_scope).strip().lower() or "run"
        if self.trend_scope not in {"run", "phase"}:
            self.trend_scope = "run"

        # Policy safety and stability knobs.
        self.cooldown_intervals = 2
        self.cooldown_left = 0
        self.gate_alpha = 0.6
        self.learning_rate = 0.001
        self.small_change_band = 0.015
        self.quality_priority_start_step = max(self.tune_start_step, int(self.max_steps * 0.7))
        self.quality_priority_gate_multiplier = 2.0
        self.quality_priority_risky_actions = {
            ACTION_LR_UP,
            ACTION_DENSIFY_UP,
            ACTION_DENSIFY_DOWN,
        }

        self.loss_history: deque[float] = deque(maxlen=16)
        self.gaussian_history: deque[float] = deque(maxlen=16)
        self.last_loss: float | None = None
        self.last_step: int | None = None
        self.last_action: str | None = None
        self.pending_transition: dict[str, Any] | None = None
        self.updates_since_save = 0

        # Online full-run trend stats (O(1) updates) for reward shaping.
        self._trend_count = 0
        self._trend_sum_x = 0.0
        self._trend_sum_y = 0.0
        self._trend_sum_xx = 0.0
        self._trend_sum_xy = 0.0
        self._phase_trend_stats: dict[int, dict[str, float]] = {
            0: {"count": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_xx": 0.0, "sum_xy": 0.0},
            1: {"count": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_xx": 0.0, "sum_xy": 0.0},
            2: {"count": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_xx": 0.0, "sum_xy": 0.0},
        }

        self._run_root_dir = self.project_dir / "runs" / self.run_id
        # Per-run artifacts (events/summaries) and reusable model snapshots.
        self._storage_dir = self._run_root_dir / "adaptive_ai"
        self._runs_dir = self._storage_dir / "runs"
        self._project_model_dir = self.project_dir / "models" / "adaptive_ai"
        self._versions_dir = self._project_model_dir / "model_versions"
        self._model_path = self._project_model_dir / "model_v1.json"
        self._registry_path = self._project_model_dir / "model_registry.json"
        self._global_storage_dir = self.project_dir.parent / "_adaptive_ai_global"
        self._global_versions_dir = self._global_storage_dir / "model_versions"
        self._global_model_path = self._global_storage_dir / "model_v1.json"
        self._global_registry_path = self._global_storage_dir / "model_registry.json"
        self._events_path = self._runs_dir / f"{self.run_id}.jsonl"
        self._summary_path = self._runs_dir / f"{self.run_id}.summary.json"
        self._run_root_dir.mkdir(parents=True, exist_ok=True)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._project_model_dir.mkdir(parents=True, exist_ok=True)
        self._versions_dir.mkdir(parents=True, exist_ok=True)

        self.feature_dim = 20
        self.model = self._load_model()

    def _load_model(self) -> TinyMLP:
        # Load precedence: project active version -> project fallback -> global active version -> global fallback.
        # This keeps project-level specialization while still allowing global warm starts.
        if self._registry_path.exists():
            try:
                registry = json.loads(self._registry_path.read_text(encoding="utf-8"))
                active_name = str(registry.get("active") or "").strip()
                if active_name:
                    active_path = self._versions_dir / active_name
                    if active_path.exists():
                        payload = json.loads(active_path.read_text(encoding="utf-8"))
                        model_data = payload.get("model") if isinstance(payload, dict) else None
                        if isinstance(model_data, dict):
                            return TinyMLP.from_dict(model_data)
            except Exception:
                pass
        if self._model_path.exists():
            try:
                payload = json.loads(self._model_path.read_text(encoding="utf-8"))
                model_data = payload.get("model") if isinstance(payload, dict) else None
                if isinstance(model_data, dict):
                    return TinyMLP.from_dict(model_data)
            except Exception:
                pass

        if self._global_registry_path.exists():
            try:
                registry = json.loads(self._global_registry_path.read_text(encoding="utf-8"))
                active_name = str(registry.get("active") or "").strip()
                if active_name:
                    active_path = self._global_versions_dir / active_name
                    if active_path.exists():
                        payload = json.loads(active_path.read_text(encoding="utf-8"))
                        model_data = payload.get("model") if isinstance(payload, dict) else None
                        if isinstance(model_data, dict):
                            return TinyMLP.from_dict(model_data)
            except Exception:
                pass
        if self._global_model_path.exists():
            try:
                payload = json.loads(self._global_model_path.read_text(encoding="utf-8"))
                model_data = payload.get("model") if isinstance(payload, dict) else None
                if isinstance(model_data, dict):
                    return TinyMLP.from_dict(model_data)
            except Exception:
                pass
        return TinyMLP(input_dim=self.feature_dim)

    def _save_model(self) -> None:
        # Save immutable version + update active pointer + keep legacy single-file fallback.
        version_name = f"model_{int(time.time())}.json"
        version_path = self._versions_dir / version_name
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "model": self.model.to_dict(),
        }
        version_path.write_text(json.dumps(payload), encoding="utf-8")

        previous_active = None
        if self._registry_path.exists():
            try:
                previous_active = json.loads(self._registry_path.read_text(encoding="utf-8")).get("active")
            except Exception:
                previous_active = None
        registry_payload = {
            "active": version_name,
            "previous": previous_active,
            "updated_at": time.time(),
        }
        self._registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")

        tmp = self._model_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._model_path)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _phase_info(self, step: int) -> tuple[int, list[float], float]:
        # 3-phase schedule: early (lr focus), middle (full adapt), late (quality-priority).
        if step < self.strategy_start_step:
            phase = 0
        elif step <= self.strategy_end_step:
            phase = 1
        else:
            phase = 2
        one_hot = [1.0 if i == phase else 0.0 for i in range(3)]
        progress = float(step) / float(self.max_steps)
        return phase, one_hot, self._clamp(progress, 0.0, 1.0)

    def _window_stats(self) -> tuple[float, float, float]:
        # Local trend/volatility metrics used by gates and features.
        if len(self.loss_history) < 3:
            return 0.0, 0.0, 0.0
        arr = np.asarray(self.loss_history, dtype=np.float64)
        x = np.arange(arr.shape[0], dtype=np.float64)
        x = x - x.mean()
        y = arr - arr.mean()
        denom = float(np.dot(x, x)) if float(np.dot(x, x)) > 1e-12 else 1.0
        slope = float(np.dot(x, y) / denom)
        variance = float(np.var(arr))
        recent_delta = float(arr[-1] - arr[-2])
        return slope, variance, recent_delta

    def _gaussian_growth(self) -> float:
        # Relative gaussian count change between the latest two observations.
        if len(self.gaussian_history) < 2:
            return 0.0
        prev = float(self.gaussian_history[-2])
        cur = float(self.gaussian_history[-1])
        denom = max(abs(prev), 1.0)
        return (cur - prev) / denom

    def _normalized_reward_weights(self) -> tuple[float, float]:
        # Normalize to convex combination so reward mix is stable regardless of raw input scale.
        total = self.reward_step_weight + self.reward_trend_weight
        if total <= 1e-12:
            return 1.0, 0.0
        return self.reward_step_weight / total, self.reward_trend_weight / total

    def _update_run_trend_reward(self, loss: float) -> float:
        # Online linear-regression slope over full run (O(1) update per step).
        self._trend_count += 1
        x = float(self._trend_count)
        y = float(loss)
        self._trend_sum_x += x
        self._trend_sum_y += y
        self._trend_sum_xx += x * x
        self._trend_sum_xy += x * y

        if self._trend_count < 3:
            return 0.0

        n = float(self._trend_count)
        denom = (n * self._trend_sum_xx) - (self._trend_sum_x * self._trend_sum_x)
        if abs(denom) < 1e-12:
            return 0.0

        slope = ((n * self._trend_sum_xy) - (self._trend_sum_x * self._trend_sum_y)) / denom
        mean_loss = self._trend_sum_y / n
        # Lower loss trend (negative slope) should increase reward.
        relative_slope = -slope / max(abs(mean_loss), 1e-8)
        return self._clamp(relative_slope, -0.05, 0.05)

    def _update_phase_trend_reward(self, *, phase: int, loss: float) -> float:
        # Same slope reward as run scope, but reset per phase to avoid cross-phase dilution.
        stats = self._phase_trend_stats.get(int(phase))
        if stats is None:
            return 0.0

        count = stats["count"] + 1.0
        x = count
        y = float(loss)
        sum_x = stats["sum_x"] + x
        sum_y = stats["sum_y"] + y
        sum_xx = stats["sum_xx"] + (x * x)
        sum_xy = stats["sum_xy"] + (x * y)

        stats["count"] = count
        stats["sum_x"] = sum_x
        stats["sum_y"] = sum_y
        stats["sum_xx"] = sum_xx
        stats["sum_xy"] = sum_xy

        if count < 3.0:
            return 0.0

        denom = (count * sum_xx) - (sum_x * sum_x)
        if abs(denom) < 1e-12:
            return 0.0

        slope = ((count * sum_xy) - (sum_x * sum_y)) / denom
        mean_loss = sum_y / count
        relative_slope = -slope / max(abs(mean_loss), 1e-8)
        return self._clamp(relative_slope, -0.05, 0.05)

    def _extract_runner_stats(self, runner_obj: Any) -> tuple[dict[str, float], dict[str, float], float]:
        # Read a compact state snapshot from runner optimizers/strategy/splats.
        lrs: dict[str, float] = {}
        for key in ("means", "opacities", "scales", "quats", "sh0", "shN"):
            optimizer = getattr(runner_obj, "optimizers", {}).get(key)
            if optimizer is None or not getattr(optimizer, "param_groups", None):
                lrs[key] = 0.0
                continue
            lrs[key] = float(optimizer.param_groups[0].get("lr", 0.0))

        strategy = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
        strategy_vals = {
            "grow_grad2d": float(getattr(strategy, "grow_grad2d", 0.0)) if strategy is not None else 0.0,
            "prune_opa": float(getattr(strategy, "prune_opa", 0.0)) if strategy is not None else 0.0,
            "refine_every": float(getattr(strategy, "refine_every", 0.0)) if strategy is not None else 0.0,
            "reset_every": float(getattr(strategy, "reset_every", 0.0)) if strategy is not None else 0.0,
        }

        gaussians = 0.0
        means_tensor = getattr(runner_obj, "splats", {}).get("means")
        if means_tensor is not None and hasattr(means_tensor, "shape") and len(means_tensor.shape) > 0:
            try:
                gaussians = float(means_tensor.shape[0])
            except Exception:
                gaussians = 0.0
        return lrs, strategy_vals, gaussians

    def _build_features(
        self,
        *,
        step: int,
        loss: float,
        relative_improvement: float | None,
        runner_obj: Any,
    ) -> list[float]:
        # Fixed feature vector contract consumed by TinyMLP.
        phase, phase_one_hot, progress = self._phase_info(step)
        slope, variance, recent_delta = self._window_stats()
        lrs, strategy_vals, gaussians = self._extract_runner_stats(runner_obj)
        self.gaussian_history.append(gaussians)
        gaussian_growth = self._gaussian_growth()

        features = [
            float(loss),
            float(relative_improvement or 0.0),
            float(slope),
            float(variance),
            float(recent_delta),
            float(gaussians),
            float(gaussian_growth),
            float(lrs.get("means", 0.0)),
            float(lrs.get("opacities", 0.0)),
            float(lrs.get("sh0", 0.0)),
            float(lrs.get("shN", 0.0)),
            float(strategy_vals.get("grow_grad2d", 0.0)),
            float(strategy_vals.get("prune_opa", 0.0)),
            float(strategy_vals.get("refine_every", 0.0)),
            float(strategy_vals.get("reset_every", 0.0)),
            float(progress),
            float(phase_one_hot[0]),
            float(phase_one_hot[1]),
            float(phase_one_hot[2]),
            float(phase),
        ]
        return features

    def _phase_base_threshold(self, phase: int, progress: float) -> float:
        # Gate target becomes stricter in volatile/early phase and softer later.
        if phase == 0:
            base = self.base_min_improvement + 0.004
        elif phase == 1:
            base = self.base_min_improvement
        else:
            base = max(0.001, self.base_min_improvement - 0.002)
        return self._clamp(base + 0.002 * progress, 0.0005, 0.05)

    def _adaptive_gate_threshold(self, phase: int, progress: float) -> float:
        # Volatility-aware gate: higher recent sigma means we require stronger evidence to intervene.
        _, variance, _ = self._window_stats()
        sigma = math.sqrt(max(variance, 0.0))
        return self._clamp(self._phase_base_threshold(phase, progress) + self.gate_alpha * sigma, 0.0005, 0.08)

    def _action_allowed(self, action: str, *, apply_lr: bool, apply_strategy: bool) -> bool:
        if action == ACTION_KEEP:
            return True
        if action in {ACTION_LR_UP, ACTION_LR_DOWN}:
            return apply_lr
        if action in {ACTION_DENSIFY_UP, ACTION_DENSIFY_DOWN, ACTION_PRUNE_UP}:
            return apply_strategy
        return False

    def _apply_action(self, runner_obj: Any, action: str, *, apply_lr: bool, apply_strategy: bool) -> tuple[dict[str, float], dict[str, float]]:
        # Apply bounded multiplicative edits to avoid destructive jumps.
        lrs_before: dict[str, float] = {}
        strategy_before: dict[str, float] = {}
        if action == ACTION_KEEP:
            return lrs_before, strategy_before

        optimizers = getattr(runner_obj, "optimizers", {})
        strategy = getattr(getattr(runner_obj, "cfg", None), "strategy", None)

        if action in {ACTION_LR_UP, ACTION_LR_DOWN} and apply_lr:
            mult = 1.03 if action == ACTION_LR_UP else 0.97
            for key in ("means", "opacities", "scales", "quats", "sh0", "shN"):
                optimizer = optimizers.get(key)
                if optimizer is None or not getattr(optimizer, "param_groups", None):
                    continue
                current = float(optimizer.param_groups[0].get("lr", 0.0))
                lrs_before[key] = current
                updated = self._clamp(current * mult, 1e-7, 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = updated

        if strategy is not None and apply_strategy:
            if action in {ACTION_DENSIFY_UP, ACTION_DENSIFY_DOWN}:
                grow_before = float(getattr(strategy, "grow_grad2d", 0.0))
                strategy_before["grow_grad2d"] = grow_before
                mult = 1.02 if action == ACTION_DENSIFY_UP else 0.98
                strategy.grow_grad2d = self._clamp(grow_before * mult, 5e-5, 5e-3)
            elif action == ACTION_PRUNE_UP:
                prune_before = float(getattr(strategy, "prune_opa", 0.0))
                strategy_before["prune_opa"] = prune_before
                strategy.prune_opa = self._clamp(prune_before * 1.01, 1e-4, 5e-2)

        return lrs_before, strategy_before

    def _log_event(self, payload: dict[str, Any]) -> None:
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _update_summary(self) -> None:
        # Small summary file used by UI/monitoring to display last controller state quickly.
        summary = {
            "run_id": self.run_id,
            "updated_at": time.time(),
            "decisions": self.updates_since_save,
            "last_action": self.last_action,
            "last_loss": self.last_loss,
            "last_step": self.last_step,
        }
        tmp = self._summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary), encoding="utf-8")
        tmp.replace(self._summary_path)

    def decide_and_apply(
        self,
        *,
        step: int,
        loss: float,
        runner_obj: Any,
        apply_lr: bool,
        apply_strategy: bool,
    ) -> AdaptiveDecision:
        # 1) Observe new loss and compute trend reward for this step.
        loss_value = float(loss)
        self.loss_history.append(loss_value)
        phase, _, _ = self._phase_info(step)
        run_trend_reward = (
            self._update_phase_trend_reward(phase=phase, loss=loss_value)
            if self.trend_scope == "phase"
            else self._update_run_trend_reward(loss_value)
        )

        previous_loss = self.last_loss
        relative_improvement = None
        if previous_loss is not None:
            denom = max(abs(previous_loss), 1e-8)
            relative_improvement = (previous_loss - loss_value) / denom

        # 2) If a previous transition exists, train it now using the newly observed reward.
        # Reward = weighted blend of one-step improvement + trend signal.
        reward_from_previous = None
        step_reward_from_previous = None
        trend_reward_from_previous = None
        trained_transition: dict[str, Any] | None = None
        if self.pending_transition is not None and previous_loss is not None:
            denom = max(abs(previous_loss), 1e-8)
            step_reward = (previous_loss - loss_value) / denom
            step_reward = self._clamp(step_reward, -0.05, 0.05)
            trend_reward = float(run_trend_reward)
            w_step, w_trend = self._normalized_reward_weights()
            reward = self._clamp((w_step * step_reward) + (w_trend * trend_reward), -0.05, 0.05)
            prev_feat = np.asarray(self.pending_transition.get("features", [0.0] * self.feature_dim), dtype=np.float64)
            prev_action_idx = int(self.pending_transition.get("action_idx", 0))
            self.model.train_selected_action(prev_feat, prev_action_idx, float(reward), learning_rate=self.learning_rate)
            reward_from_previous = float(reward)
            step_reward_from_previous = float(step_reward)
            trend_reward_from_previous = float(trend_reward)
            trained_transition = {
                "step": int(self.pending_transition.get("step", 0) or 0),
                "features": [float(v) for v in self.pending_transition.get("features", [])],
                "action": str(self.pending_transition.get("action") or ACTION_KEEP),
                "reward": float(reward),
                "step_reward": float(step_reward),
                "run_trend_reward": float(trend_reward),
                "reward_step_weight": float(w_step),
                "reward_trend_weight": float(w_trend),
                "trend_scope": self.trend_scope,
            }

        # 3) Score current state and run policy gates.
        #    Cooldown and quality-priority gates intentionally bias toward safe "keep" in uncertain regions.
        _, _, progress = self._phase_info(step)
        gate_threshold = self._adaptive_gate_threshold(phase, progress)
        strict_gate_threshold = self._clamp(
            gate_threshold * self.quality_priority_gate_multiplier,
            gate_threshold,
            0.12,
        )
        quality_priority_active = int(step) >= self.quality_priority_start_step
        features = self._build_features(
            step=step,
            loss=loss_value,
            relative_improvement=relative_improvement,
            runner_obj=runner_obj,
        )
        x = np.asarray(features, dtype=np.float64)
        _, _, scores = self.model.forward(x)
        action_idx = int(np.argmax(scores))
        action = ACTIONS[action_idx]
        reason = "mlp_policy"

        if self.cooldown_left > 0:
            action = ACTION_KEEP
            reason = "cooldown"
            self.cooldown_left = max(0, self.cooldown_left - 1)
        elif relative_improvement is None:
            action = ACTION_KEEP
            reason = "warmup_no_prev_loss"
        else:
            rel = float(relative_improvement)
            if quality_priority_active and action in self.quality_priority_risky_actions:
                if rel >= float(strict_gate_threshold):
                    reason = "late_phase_gate_allow"
                else:
                    action = ACTION_KEEP
                    reason = "late_phase_quality_priority"
            elif rel >= float(gate_threshold):
                action = ACTION_KEEP
                reason = "adaptive_gate_keep"
            elif rel >= -float(self.small_change_band):
                action = ACTION_KEEP
                reason = "stable_small_change"

        # 4) Apply action if allowed by current window and feature schedule.
        if not self._action_allowed(action, apply_lr=apply_lr, apply_strategy=apply_strategy):
            action = ACTION_KEEP
            reason = "outside_window"

        self._apply_action(runner_obj, action, apply_lr=apply_lr, apply_strategy=apply_strategy)

        if action != ACTION_KEEP:
            self.cooldown_left = self.cooldown_intervals

        # 5) Stash transition for next-step credit assignment.
        self.pending_transition = {
            "step": int(step),
            "features": features,
            "action": action,
            "action_idx": ACTIONS.index(action),
        }

        self.last_action = action
        self.last_loss = loss_value
        self.last_step = int(step)
        self.updates_since_save += 1

        # 6) Persist event stream and periodic model checkpoints.
        event = {
            "time": time.time(),
            "step": int(step),
            "loss": float(loss_value),
            "relative_improvement": float(relative_improvement) if relative_improvement is not None else None,
            "gate_threshold": float(gate_threshold),
            "quality_priority_active": bool(quality_priority_active),
            "quality_priority_start_step": int(self.quality_priority_start_step),
            "quality_priority_gate_threshold": float(strict_gate_threshold),
            "action": action,
            "reason": reason,
            "reward_from_previous": float(reward_from_previous) if reward_from_previous is not None else None,
            "step_reward_from_previous": float(step_reward_from_previous) if step_reward_from_previous is not None else None,
            "run_trend_reward": float(run_trend_reward),
            "trend_reward_from_previous": float(trend_reward_from_previous) if trend_reward_from_previous is not None else None,
            "reward_step_weight": float(self._normalized_reward_weights()[0]),
            "reward_trend_weight": float(self._normalized_reward_weights()[1]),
            "trend_scope": self.trend_scope,
            "trained_transition": trained_transition,
            "apply_lr": bool(apply_lr),
            "apply_strategy": bool(apply_strategy),
            "features": [float(v) for v in features],
            "scores": [float(v) for v in scores.tolist()],
        }
        self._log_event(event)

        if self.updates_since_save % 5 == 0:
            self._save_model()
        self._update_summary()

        return AdaptiveDecision(
            action=action,
            reason=reason,
            gate_threshold=float(gate_threshold),
            relative_improvement=float(relative_improvement) if relative_improvement is not None else None,
            reward_from_previous=float(reward_from_previous) if reward_from_previous is not None else None,
            features=[float(v) for v in features],
            action_scores=[float(v) for v in scores.tolist()],
        )
