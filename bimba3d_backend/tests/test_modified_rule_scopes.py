import unittest
from types import SimpleNamespace

from bimba3d_backend.worker.modified_rule_scopes import (
    RuleProfile,
    apply_tune_scope,
    normalize_tune_scope,
    select_rule_profile,
)


class DummyOptimizer:
    def __init__(self, lr: float):
        self.param_groups = [{"lr": lr}]


class DummyStrategy:
    def __init__(self):
        self.grow_grad2d = 1e-3
        self.prune_opa = 5e-3
        self.refine_every = 100
        self.reset_every = 2000


class ModifiedRuleScopesTests(unittest.TestCase):
    def _make_runner(self):
        optimizers = {
            "means": DummyOptimizer(0.01),
            "opacities": DummyOptimizer(0.02),
            "scales": DummyOptimizer(0.03),
            "quats": DummyOptimizer(0.04),
            "sh0": DummyOptimizer(0.05),
            "shN": DummyOptimizer(0.06),
        }
        cfg = SimpleNamespace(strategy=DummyStrategy())
        return SimpleNamespace(optimizers=optimizers, cfg=cfg)

    def _mid_profile(self) -> RuleProfile:
        return select_rule_profile(0.10)

    def test_normalize_tune_scope_accepts_known_and_falls_back(self):
        self.assertEqual(normalize_tune_scope("core_individual"), "core_individual")
        self.assertEqual(normalize_tune_scope("core_only"), "core_only")
        self.assertEqual(normalize_tune_scope("core_ai_optimization"), "core_ai_optimization")
        self.assertEqual(normalize_tune_scope("core_individual_plus_strategy"), "core_individual_plus_strategy")
        self.assertEqual(normalize_tune_scope("with_strategy"), "core_individual_plus_strategy")
        self.assertEqual(normalize_tune_scope("unknown"), "core_individual_plus_strategy")

    def test_core_individual_changes_only_learning_rates(self):
        runner = self._make_runner()
        strategy_before = (
            runner.cfg.strategy.grow_grad2d,
            runner.cfg.strategy.prune_opa,
            runner.cfg.strategy.refine_every,
            runner.cfg.strategy.reset_every,
        )

        result = apply_tune_scope("core_individual", runner, self._mid_profile())

        self.assertIn("rule_based_lr_scaling", result["adjustments"])
        self.assertNotIn("rule_based_strategy_scaling", result["adjustments"])
        self.assertEqual(result["strategy_before"], {})
        self.assertEqual(result["strategy_after"], {})

        strategy_after = (
            runner.cfg.strategy.grow_grad2d,
            runner.cfg.strategy.prune_opa,
            runner.cfg.strategy.refine_every,
            runner.cfg.strategy.reset_every,
        )
        self.assertEqual(strategy_before, strategy_after)

        self.assertLess(result["after_lrs"]["means"], result["before_lrs"]["means"])
        self.assertLess(result["after_lrs"]["opacities"], result["before_lrs"]["opacities"])

    def test_core_only_changes_core_strategy_but_not_extended_strategy(self):
        runner = self._make_runner()
        strategy_before = {
            "grow_grad2d": runner.cfg.strategy.grow_grad2d,
            "prune_opa": runner.cfg.strategy.prune_opa,
            "refine_every": runner.cfg.strategy.refine_every,
            "reset_every": runner.cfg.strategy.reset_every,
        }

        result = apply_tune_scope("core_only", runner, self._mid_profile())

        self.assertIn("rule_based_strategy_scaling_core", result["adjustments"])
        self.assertNotIn("rule_based_strategy_scaling", result["adjustments"])

        self.assertNotEqual(runner.cfg.strategy.grow_grad2d, strategy_before["grow_grad2d"])
        self.assertEqual(runner.cfg.strategy.prune_opa, strategy_before["prune_opa"])
        self.assertEqual(runner.cfg.strategy.refine_every, strategy_before["refine_every"])
        self.assertEqual(runner.cfg.strategy.reset_every, strategy_before["reset_every"])

    def test_core_individual_plus_strategy_changes_all_strategy_fields(self):
        runner = self._make_runner()
        strategy_before = {
            "grow_grad2d": runner.cfg.strategy.grow_grad2d,
            "prune_opa": runner.cfg.strategy.prune_opa,
            "refine_every": runner.cfg.strategy.refine_every,
            "reset_every": runner.cfg.strategy.reset_every,
        }

        result = apply_tune_scope("core_individual_plus_strategy", runner, self._mid_profile())

        self.assertIn("rule_based_strategy_scaling", result["adjustments"])
        self.assertEqual(set(result["strategy_before"].keys()), {"grow_grad2d", "prune_opa", "refine_every", "reset_every"})
        self.assertEqual(set(result["strategy_after"].keys()), {"grow_grad2d", "prune_opa", "refine_every", "reset_every"})

        # Mid-loss profile includes no-op multipliers for some fields (1.0),
        # so we only require that fields expected to change in this profile changed.
        self.assertNotEqual(runner.cfg.strategy.grow_grad2d, strategy_before["grow_grad2d"])
        self.assertNotEqual(runner.cfg.strategy.refine_every, strategy_before["refine_every"])

    def test_core_ai_optimization_uses_core_strategy_only(self):
        runner = self._make_runner()
        strategy_before = {
            "grow_grad2d": runner.cfg.strategy.grow_grad2d,
            "prune_opa": runner.cfg.strategy.prune_opa,
            "refine_every": runner.cfg.strategy.refine_every,
            "reset_every": runner.cfg.strategy.reset_every,
        }

        result = apply_tune_scope("core_ai_optimization", runner, self._mid_profile())

        self.assertIn("rule_based_strategy_scaling_core", result["adjustments"])
        self.assertNotIn("rule_based_strategy_scaling", result["adjustments"])
        self.assertNotEqual(runner.cfg.strategy.grow_grad2d, strategy_before["grow_grad2d"])
        self.assertEqual(runner.cfg.strategy.prune_opa, strategy_before["prune_opa"])
        self.assertEqual(runner.cfg.strategy.refine_every, strategy_before["refine_every"])
        self.assertEqual(runner.cfg.strategy.reset_every, strategy_before["reset_every"])

    def test_with_strategy_alias_still_maps_to_plus_strategy(self):
        runner = self._make_runner()
        result = apply_tune_scope("with_strategy", runner, self._mid_profile())
        self.assertIn("rule_based_strategy_scaling", result["adjustments"])


if __name__ == "__main__":
    unittest.main()
