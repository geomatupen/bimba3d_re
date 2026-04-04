from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_TUNE_SCOPE = "core_individual_plus_strategy"
VALID_TUNE_SCOPES = {
    "core_individual",
    "core_only",
    "core_ai_optimization",
    "core_individual_plus_strategy",
    "with_strategy",
}


@dataclass(frozen=True)
class RuleProfile:
    name: str
    lr_multipliers: dict[str, float]
    strategy_multipliers: dict[str, float]


def normalize_tune_scope(value: object, default: str = DEFAULT_TUNE_SCOPE) -> str:
    candidate = str(value or default).strip().lower()
    if candidate == "with_strategy":
        return "core_individual_plus_strategy"
    return candidate if candidate in VALID_TUNE_SCOPES else default


def select_rule_profile(loss_value: float) -> RuleProfile:
    if loss_value >= 0.20:
        return RuleProfile(
            name="high_loss",
            lr_multipliers={
                "means": 0.92,
                "opacities": 0.86,
                "scales": 1.00,
                "quats": 1.00,
                "sh0": 0.96,
                "shN": 0.96,
            },
            strategy_multipliers={
                "grow_grad2d": 1.02,
                "prune_opa": 1.01,
                "refine_every": 1.04,
                "reset_every": 1.02,
            },
        )

    if loss_value >= 0.08:
        return RuleProfile(
            name="mid_loss",
            lr_multipliers={
                "means": 0.96,
                "opacities": 0.92,
                "scales": 1.00,
                "quats": 1.00,
                "sh0": 1.00,
                "shN": 1.00,
            },
            strategy_multipliers={
                "grow_grad2d": 1.01,
                "prune_opa": 1.005,
                "refine_every": 1.02,
                "reset_every": 1.01,
            },
        )

    return RuleProfile(
        name="low_loss",
        lr_multipliers={
            "means": 1.02,
            "opacities": 0.99,
            "scales": 1.00,
            "quats": 1.00,
            "sh0": 1.05,
            "shN": 1.05,
        },
        strategy_multipliers={
            "grow_grad2d": 1.00,
            "prune_opa": 1.00,
            "refine_every": 1.01,
            "reset_every": 1.00,
        },
    )


def build_rule_multiplier_summary(scope: str, profile: RuleProfile) -> dict[str, float]:
    summary = {
        "lr": float(profile.lr_multipliers["means"]),
        "opacity_lr_mult": float(profile.lr_multipliers["opacities"]),
        "sh_lr_mult": float(
            (float(profile.lr_multipliers["sh0"]) + float(profile.lr_multipliers["shN"])) / 2.0
        ),
        "position_lr_mult": float(profile.lr_multipliers["means"]),
    }
    if scope in {"core_only", "core_ai_optimization", "core_individual_plus_strategy", "with_strategy"}:
        summary["densify_threshold_mult"] = float(profile.strategy_multipliers["grow_grad2d"])
    return summary


def _apply_lr_scaling(runner_obj: Any, lr_multipliers: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    before_lrs: dict[str, float] = {}
    after_lrs: dict[str, float] = {}

    for name, mult in lr_multipliers.items():
        optimizer = getattr(runner_obj, "optimizers", {}).get(name)
        if optimizer is None or not getattr(optimizer, "param_groups", None):
            continue

        current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
        before_lrs[name] = current_lr
        new_lr = max(1e-7, min(1.0, current_lr * float(mult)))
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        after_lrs[name] = new_lr

    return before_lrs, after_lrs


def _capture_strategy_values(strategy: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("grow_grad2d", "prune_opa", "refine_every", "reset_every"):
        if hasattr(strategy, key):
            values[key] = float(getattr(strategy, key))
    return values


def _apply_core_strategy_only(strategy: Any, strategy_multipliers: dict[str, float]) -> None:
    strategy.grow_grad2d = max(
        5e-5,
        min(5e-3, float(strategy.grow_grad2d) * float(strategy_multipliers["grow_grad2d"])),
    )


def _apply_extended_strategy(strategy: Any, strategy_multipliers: dict[str, float]) -> None:
    strategy.prune_opa = max(
        1e-4,
        min(5e-2, float(strategy.prune_opa) * float(strategy_multipliers["prune_opa"])),
    )
    strategy.refine_every = max(
        25,
        min(300, int(float(strategy.refine_every) * float(strategy_multipliers["refine_every"]))),
    )
    strategy.reset_every = max(
        max(strategy.refine_every, 1000),
        min(6000, int(float(strategy.reset_every) * float(strategy_multipliers["reset_every"]))),
    )


def apply_scope_core_individual(
    runner_obj: Any,
    profile: RuleProfile,
    *,
    apply_lr: bool = True,
) -> dict[str, Any]:
    before_lrs: dict[str, float] = {}
    after_lrs: dict[str, float] = {}
    adjustments: list[str] = []
    if apply_lr:
        before_lrs, after_lrs = _apply_lr_scaling(runner_obj, profile.lr_multipliers)
        adjustments.append("rule_based_lr_scaling")
    return {
        "before_lrs": before_lrs,
        "after_lrs": after_lrs,
        "strategy_before": {},
        "strategy_after": {},
        "adjustments": adjustments,
    }


def apply_scope_core_only(
    runner_obj: Any,
    profile: RuleProfile,
    *,
    apply_lr: bool = True,
    apply_strategy: bool = True,
) -> dict[str, Any]:
    before_lrs: dict[str, float] = {}
    after_lrs: dict[str, float] = {}
    adjustments: list[str] = []
    if apply_lr:
        before_lrs, after_lrs = _apply_lr_scaling(runner_obj, profile.lr_multipliers)
        adjustments.append("rule_based_lr_scaling")

    strategy_before: dict[str, float] = {}
    strategy_after: dict[str, float] = {}
    strategy = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
    if apply_strategy and strategy is not None:
        strategy_before = _capture_strategy_values(strategy)
        _apply_core_strategy_only(strategy, profile.strategy_multipliers)
        strategy_after = _capture_strategy_values(strategy)
        adjustments.append("rule_based_strategy_scaling_core")

    return {
        "before_lrs": before_lrs,
        "after_lrs": after_lrs,
        "strategy_before": strategy_before,
        "strategy_after": strategy_after,
        "adjustments": adjustments,
    }


def apply_scope_core_plus_strategy(
    runner_obj: Any,
    profile: RuleProfile,
    *,
    apply_lr: bool = True,
    apply_strategy: bool = True,
) -> dict[str, Any]:
    before_lrs: dict[str, float] = {}
    after_lrs: dict[str, float] = {}
    adjustments: list[str] = []
    if apply_lr:
        before_lrs, after_lrs = _apply_lr_scaling(runner_obj, profile.lr_multipliers)
        adjustments.append("rule_based_lr_scaling")

    strategy_before: dict[str, float] = {}
    strategy_after: dict[str, float] = {}
    strategy = getattr(getattr(runner_obj, "cfg", None), "strategy", None)
    if apply_strategy and strategy is not None:
        strategy_before = _capture_strategy_values(strategy)
        _apply_core_strategy_only(strategy, profile.strategy_multipliers)
        _apply_extended_strategy(strategy, profile.strategy_multipliers)
        strategy_after = _capture_strategy_values(strategy)
        adjustments.append("rule_based_strategy_scaling")

    return {
        "before_lrs": before_lrs,
        "after_lrs": after_lrs,
        "strategy_before": strategy_before,
        "strategy_after": strategy_after,
        "adjustments": adjustments,
    }


def apply_tune_scope(
    scope: str,
    runner_obj: Any,
    profile: RuleProfile,
    *,
    apply_lr: bool = True,
    apply_strategy: bool = True,
) -> dict[str, Any]:
    normalized_scope = normalize_tune_scope(scope)
    if normalized_scope == "core_individual":
        return apply_scope_core_individual(runner_obj, profile, apply_lr=apply_lr)
    if normalized_scope in {"core_only", "core_ai_optimization"}:
        return apply_scope_core_only(
            runner_obj,
            profile,
            apply_lr=apply_lr,
            apply_strategy=apply_strategy,
        )
    return apply_scope_core_plus_strategy(
        runner_obj,
        profile,
        apply_lr=apply_lr,
        apply_strategy=apply_strategy,
    )
