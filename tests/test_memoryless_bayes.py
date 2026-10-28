"""CPU tests for memoryless Bayes (C1 caps 12/40)."""

import unittest

from vlm_minigrid_rl.memoryless_bayes import (
    ACTION_FORWARD,
    ACTION_TURN,
    C1_EMPTY_16_MAX_STEPS,
    C1_EMPTY_8_MAX_STEPS,
    POLICY_SEARCH,
    POLICY_SPIN,
    VERDICT_FLIP,
    VERDICT_MIXED,
    bayes_empty_action,
    build_memoryless_report,
    c1_empty_max_steps,
    evaluate_policy,
    maps_bayes_verdict,
    p_far_given_empty,
)


def _cell(env_size: int, policy: str, seed: int, rate: float) -> dict:
    return {
        "env_size": env_size,
        "policy": policy,
        "episodes": 250,
        "seed": seed,
        "max_steps": c1_empty_max_steps(env_size),
        "successes": 0,
        "success_rate": rate,
        "by_stratum": {},
    }


class CapsAndBayesRuleTest(unittest.TestCase):
    def test_c1_caps_not_slim_profile(self):
        self.assertEqual(C1_EMPTY_8_MAX_STEPS, 12)
        self.assertEqual(C1_EMPTY_16_MAX_STEPS, 40)
        self.assertEqual(c1_empty_max_steps(8), 12)
        self.assertEqual(c1_empty_max_steps(16), 40)

    def test_p_far_given_empty(self):
        self.assertAlmostEqual(p_far_given_empty(477, 28), 28 / 505)
        self.assertIsNone(p_far_given_empty(0, 0))

    def test_bayes_prefers_turn_when_spin_wins(self):
        self.assertEqual(
            bayes_empty_action(spin_success=90.0, search_success=40.0),
            ACTION_TURN,
        )

    def test_bayes_prefers_forward_when_search_wins(self):
        self.assertEqual(
            bayes_empty_action(spin_success=40.0, search_success=70.0),
            ACTION_FORWARD,
        )

    def test_maps_flip_is_8_turn_16_forward(self):
        self.assertEqual(maps_bayes_verdict(ACTION_TURN, ACTION_FORWARD), VERDICT_FLIP)
        self.assertEqual(maps_bayes_verdict(ACTION_TURN, ACTION_TURN), VERDICT_MIXED)
        self.assertEqual(maps_bayes_verdict(ACTION_FORWARD, ACTION_FORWARD), VERDICT_MIXED)

    def test_report_ascii_reason_and_flip(self):
        seeds = [42, 123, 456]
        evals = {
            "8": {
                POLICY_SPIN: [_cell(8, POLICY_SPIN, s, 95.0) for s in seeds],
                POLICY_SEARCH: [_cell(8, POLICY_SEARCH, s, 50.0) for s in seeds],
            },
            "16": {
                POLICY_SPIN: [_cell(16, POLICY_SPIN, s, 44.0) for s in seeds],
                POLICY_SEARCH: [_cell(16, POLICY_SEARCH, s, 70.0) for s in seeds],
            },
        }
        prior = {
            "8": {"p_far_given_empty": 0.055},
            "16": {"p_far_given_empty": 0.68},
        }
        report = build_memoryless_report(
            priors=prior,
            evals=evals,
            prior_n=2000,
            prior_seed=42,
            eval_episodes=250,
            eval_seeds=seeds,
        )
        self.assertEqual(report["verdict"], VERDICT_FLIP)
        self.assertEqual(report["by_env"]["8"]["bayes_empty_action"], ACTION_TURN)
        self.assertEqual(report["by_env"]["16"]["bayes_empty_action"], ACTION_FORWARD)
        report["reason"].encode("ascii")


class GymSmokeTest(unittest.TestCase):
    def test_spin_episode_on_8x8_runs(self):
        cell = evaluate_policy(8, POLICY_SPIN, episodes=3, seed=42)
        self.assertEqual(cell["max_steps"], 12)
        self.assertEqual(cell["episodes"], 3)
        self.assertGreaterEqual(cell["success_rate"], 0.0)
        self.assertLessEqual(cell["success_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
