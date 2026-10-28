"""CPU tests for GRPO p_all_fail / p_std_zero by start stratum."""

from __future__ import annotations

import random
import unittest

from vlm_minigrid_rl.memoryless_bayes import POLICY_SPIN, start_stratum
from vlm_minigrid_rl.minigrid_utils import create_minigrid_env, reset_env_with_goal
from vlm_minigrid_rl.p_std_zero_stratum import (
    C1_E9_MAX_STEPS,
    GROUP_ALL_FAIL,
    GROUP_ALL_SUCCESS,
    GROUP_MIXED,
    VERDICT_FAR_ALLFAIL_DOMINATES,
    VERDICT_MIXED,
    VERDICT_NOT_SHOWN,
    classify_layout,
    cpu_group_all_fail,
    decode_log_bytes,
    grpo_layout_seeds,
    group_outcome,
    join_rows,
    parse_grpo_episode_rows,
    raw_std_zero,
    select_c1_e9_run,
    split_log_runs,
    tally_by_stratum,
    verdict_from_rates,
    wilson_ci,
    _pool_tallies,
)


def _find_layout(stratum: str, *, limit: int = 400) -> int:
    env = create_minigrid_env(8, max_steps=C1_E9_MAX_STEPS)
    try:
        for seed in range(limit):
            reset_env_with_goal(env, seed, goal_color="green")
            if start_stratum(env) == stratum:
                return seed
    finally:
        env.close()
    raise AssertionError(f"no {stratum} layout in seeds 0..{limit - 1}")


class SeedAndStdTest(unittest.TestCase):
    def test_grpo_layout_seeds_match_dedicated_rng(self):
        rng = random.Random(42)
        expected = [rng.randint(0, 100000) for _ in range(3)]
        self.assertEqual(grpo_layout_seeds(42, 3), expected)

    def test_grpo_layout_seeds_ignore_global_draws(self):
        canonical = grpo_layout_seeds(42, 3)
        random.seed(42)
        for _ in range(10):
            random.randint(0, 100000)
        self.assertEqual(grpo_layout_seeds(42, 3), canonical)

    def test_raw_std_zero_all_fail_and_mixed(self):
        self.assertTrue(raw_std_zero([0.0] * 16))
        self.assertFalse(raw_std_zero([0.0, 0.9] + [0.0] * 14))


class ParserTest(unittest.TestCase):
    def test_decode_utf16_le_log(self):
        payload = (
            "Config: lr=5e-06, seed=123, rollout=generate, update_batch_size=32\n"
            "Ep 1/100 | Mean Return: 0.000 | Success: 0.0%\n"
        )
        text = decode_log_bytes(payload.encode("utf-16"))
        rows = parse_grpo_episode_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["success_rate"], 0.0)

    def test_parse_three_episode_rows(self):
        text = (
            "Ep 1/100 | Mean Return: 0.180 | Success: 18.8%\n"
            "Ep 2/100 | Mean Return: 0.979 | Success: 100.0%\n"
            "Ep 11/100 | Mean Return: 0.000 | Success: 0.0%\n"
        )
        rows = parse_grpo_episode_rows(text)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["episode"], 1)
        self.assertAlmostEqual(rows[0]["success_rate"], 18.8)
        self.assertEqual(rows[2]["episode"], 11)
        self.assertAlmostEqual(rows[2]["mean_return"], 0.0)
        self.assertAlmostEqual(rows[2]["success_rate"], 0.0)

    def test_split_two_config_blocks(self):
        text = (
            "Config: lr=5e-06, epsilon=0.1, beta=0.1, lora_dropout=0.0, "
            "seed=123, rollout=generate, update_batch_size=32\n"
            "Ep 1/100 | Mean Return: 0.976 | Success: 100.0%\n"
            "Сохранено в I:\\VS Code\\vlm-minigrid-rl\\checkpoints\\"
            "grpo_adapter_8x8_handoff_from_e10_e9_seed123/episode-100\n"
            "Config: lr=5e-06, epsilon=0.1, beta=0.1, lora_dropout=0.0, "
            "seed=456, rollout=generate, update_batch_size=32\n"
            "Ep 16/100 | Mean Return: 0.000 | Success: 0.0%\n"
            "Сохранено в I:\\VS Code\\vlm-minigrid-rl\\checkpoints\\"
            "grpo_adapter_8x8_handoff_from_e10_e9_seed456/episode-100\n"
        )
        runs = split_log_runs(text)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["seed"], 123)
        self.assertTrue(runs[0]["adapter"].endswith("handoff_from_e10_e9_seed123"))
        self.assertEqual(runs[1]["seed"], 456)
        picked = select_c1_e9_run(runs, 456)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["seed"], 456)
        self.assertEqual(picked["episodes"][0]["episode"], 16)

    def test_select_unsuffixed_e9_not_earlier_epoch(self):
        text = (
            "Config: lr=5e-06, seed=42, rollout=generate, update_batch_size=32\n"
            "Ep 1/100 | Mean Return: 0.060 | Success: 6.2%\n"
            "Сохранено в checkpoints\\grpo_adapter_8x8_handoff_from_e10_e1/episode-100\n"
            "Config: lr=5e-06, seed=42, rollout=generate, update_batch_size=32\n"
            "Ep 1/100 | Mean Return: 0.180 | Success: 18.8%\n"
            "Сохранено в checkpoints\\grpo_adapter_8x8_handoff_from_e10_e9/episode-100\n"
        )
        runs = split_log_runs(text)
        picked = select_c1_e9_run(runs, 42)
        self.assertIsNotNone(picked)
        self.assertAlmostEqual(picked["episodes"][0]["success_rate"], 18.8)
        self.assertTrue(picked["adapter"].endswith("handoff_from_e10_e9"))


class TallyJoinTest(unittest.TestCase):
    def test_group_outcome(self):
        self.assertEqual(group_outcome(0.0), GROUP_ALL_FAIL)
        self.assertEqual(group_outcome(100.0), GROUP_ALL_SUCCESS)
        self.assertEqual(group_outcome(18.8), GROUP_MIXED)

    def test_join_far_all_fail_does_not_count_as_miss(self):
        rows = join_rows(
            ["far", "facing_miss"],
            [
                {"episode": 1, "mean_return": 0.0, "success_rate": 0.0},
                {"episode": 2, "mean_return": 0.97, "success_rate": 100.0},
            ],
        )
        tally = tally_by_stratum(rows)
        self.assertEqual(tally["far"]["n"], 1)
        self.assertEqual(tally["far"]["n_all_fail"], 1)
        self.assertEqual(tally["facing_miss"]["n_all_fail"], 0)
        self.assertAlmostEqual(tally["far"]["p_all_fail"], 1.0)
        self.assertAlmostEqual(tally["facing_miss"]["p_all_fail"], 0.0)

    def test_pool_weighted_mean_success(self):
        a = tally_by_stratum(
            [{"stratum": "far", "outcome": GROUP_MIXED, "success_rate": 80.0}]
        )
        b = tally_by_stratum(
            [{"stratum": "far", "outcome": GROUP_ALL_FAIL, "success_rate": 0.0}]
        )
        pooled = _pool_tallies([a, b])
        self.assertEqual(pooled["far"]["n_all_fail"], 1)
        self.assertAlmostEqual(pooled["far"]["mean_success"], 40.0)

    def test_wilson_ci_all_fail_small_n(self):
        lo, hi = wilson_ci(1, 3)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, hi)

    def test_verdict_rules(self):
        self.assertEqual(
            verdict_from_rates(p_far=1.0, p_miss=0.0, cpu_spin_far=0.99),
            VERDICT_FAR_ALLFAIL_DOMINATES,
        )
        self.assertEqual(
            verdict_from_rates(p_far=0.0, p_miss=0.0, cpu_spin_far=1.0),
            VERDICT_MIXED,
        )
        self.assertEqual(
            verdict_from_rates(p_far=None, p_miss=None, cpu_spin_far=1.0),
            VERDICT_NOT_SHOWN,
        )


class GymSpinStratumTest(unittest.TestCase):
    def test_spin_all_fail_on_far_not_on_facing_miss(self):
        far_seed = _find_layout("far")
        miss_seed = _find_layout("facing_miss")
        self.assertEqual(classify_layout(8, far_seed), "far")
        self.assertEqual(classify_layout(8, miss_seed), "facing_miss")
        self.assertTrue(cpu_group_all_fail(8, far_seed, POLICY_SPIN))
        self.assertFalse(cpu_group_all_fail(8, miss_seed, POLICY_SPIN))


if __name__ == "__main__":
    unittest.main()
