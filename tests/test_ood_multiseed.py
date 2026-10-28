"""Unit tests for C1 multi-seed OOD helpers and e9 verdict."""

import unittest
from pathlib import Path

from vlm_minigrid_rl.handoff_sweep import build_grpo_eval_command
from vlm_minigrid_rl.ood_multiseed import (
    PRIMARY_EPOCH,
    SUITE_GOAL_RED,
    SUITE_TRANSFER,
    SUITES,
    VERDICT_HURTS,
    VERDICT_INCONCLUSIVE,
    VERDICT_MIXED,
    VERDICT_NOT_HURTS,
    build_suite_epoch_row,
    build_summary_payload,
    ood_cell_path,
    ood_verdict,
    payload_env_size,
    rates_from_ood_payload,
    reuse_payload_matches_suite,
    suite_max_steps,
)


def _payload(
    *,
    env_size: int,
    goal_color: str,
    prompt_goal_color: str,
    sft_epoch: int,
    sft_rate: float,
    grpo_rate: float,
    episodes: int = 250,
    grpo_episode: int = 100,
) -> dict:
    return {
        "config": {
            "env_size": env_size,
            "sft_adapter_path": (
                f"I:/VS Code/vlm-minigrid-rl/checkpoints/"
                f"sft_adapter_8x8_bs32_e10/epoch-{sft_epoch}"
            ),
            "grpo_adapter_path": (
                f"I:/VS Code/vlm-minigrid-rl/checkpoints/"
                f"grpo_adapter_8x8_handoff_from_e10_e{sft_epoch}/episode-{grpo_episode}"
            ),
            "episodes": episodes,
            "goal_color": goal_color,
            "prompt_goal_color": prompt_goal_color,
            "seed": 42,
        },
        "policies": [
            {"name": "SFT", "metrics": {"success_rate": sft_rate}},
            {"name": "GRPO", "metrics": {"success_rate": grpo_rate}},
        ],
    }


class OodMultiseedHelpersTest(unittest.TestCase):
    def test_cell_path_matches_a1_layout(self):
        run = Path("runs/2026-08-ood-multiseed-8x8")
        self.assertEqual(
            ood_cell_path(run, "transfer8to16", 9, 123),
            run / "8x8" / "ood_transfer8to16_e9_seed123.json",
        )
        self.assertEqual(
            ood_cell_path(run, "goalred", 1, 42),
            run / "8x8" / "ood_goalred_e1_seed42.json",
        )

    def test_suite_max_steps(self):
        self.assertEqual(suite_max_steps(SUITES[SUITE_TRANSFER]), 38)
        self.assertEqual(suite_max_steps(SUITES[SUITE_GOAL_RED]), 16)

    def test_payload_env_size_from_config(self):
        payload = _payload(
            env_size=16,
            goal_color="green",
            prompt_goal_color="green",
            sft_epoch=1,
            sft_rate=48.4,
            grpo_rate=40.4,
        )
        self.assertEqual(payload_env_size(payload), 16)

    def test_reuse_accepts_a1_transfer_e1(self):
        payload = _payload(
            env_size=16,
            goal_color="green",
            prompt_goal_color="green",
            sft_epoch=1,
            sft_rate=48.4,
            grpo_rate=40.4,
        )
        self.assertTrue(
            reuse_payload_matches_suite(
                payload, suite=SUITES[SUITE_TRANSFER], sft_epoch=1
            )
        )

    def test_reuse_rejects_wrong_env_or_goal(self):
        transfer = SUITES[SUITE_TRANSFER]
        wrong_env = _payload(
            env_size=8,
            goal_color="green",
            prompt_goal_color="green",
            sft_epoch=1,
            sft_rate=48.4,
            grpo_rate=40.4,
        )
        self.assertFalse(
            reuse_payload_matches_suite(wrong_env, suite=transfer, sft_epoch=1)
        )
        wrong_goal = _payload(
            env_size=16,
            goal_color="red",
            prompt_goal_color="red",
            sft_epoch=1,
            sft_rate=48.4,
            grpo_rate=40.4,
        )
        self.assertFalse(
            reuse_payload_matches_suite(wrong_goal, suite=transfer, sft_epoch=1)
        )

    def test_reuse_rejects_epoch_10_as_epoch_1(self):
        payload = _payload(
            env_size=16,
            goal_color="green",
            prompt_goal_color="green",
            sft_epoch=10,
            sft_rate=48.4,
            grpo_rate=40.4,
        )
        self.assertFalse(
            reuse_payload_matches_suite(
                payload, suite=SUITES[SUITE_TRANSFER], sft_epoch=1
            )
        )
        transfer = SUITES[SUITE_TRANSFER]
        wrong_ep = _payload(
            env_size=16,
            goal_color="green",
            prompt_goal_color="green",
            sft_epoch=1,
            sft_rate=48.4,
            grpo_rate=40.4,
            grpo_episode=75,
        )
        self.assertFalse(
            reuse_payload_matches_suite(wrong_ep, suite=transfer, sft_epoch=1)
        )
        self.assertFalse(
            reuse_payload_matches_suite(
                _payload(
                    env_size=16,
                    goal_color="green",
                    prompt_goal_color="green",
                    sft_epoch=5,
                    sft_rate=60.8,
                    grpo_rate=45.2,
                ),
                suite=transfer,
                sft_epoch=1,
            )
        )

    def test_rates_from_ood_payload(self):
        payload = _payload(
            env_size=8,
            goal_color="red",
            prompt_goal_color="red",
            sft_epoch=9,
            sft_rate=14.8,
            grpo_rate=1.6,
        )
        sft, grpo = rates_from_ood_payload(payload)
        self.assertAlmostEqual(sft, 14.8)
        self.assertAlmostEqual(grpo, 1.6)

    def test_row_counts_grpo_worse(self):
        row = build_suite_epoch_row(
            suite_id=SUITE_TRANSFER,
            sft_epoch=9,
            seeds=[42, 123, 456],
            sft_rates=[57.6, 55.0, 56.0],
            grpo_rates=[45.2, 44.0, 46.0],
        )
        self.assertEqual(row["n_grpo_worse"], 3)
        self.assertLess(row["delta_mean"], 0.0)
        self.assertAlmostEqual(row["by_seed"]["42"]["sft"], 57.6)
        self.assertAlmostEqual(row["by_seed"]["42"]["grpo"], 45.2)

    def test_verdict_hurts_when_e9_both_suites_negative(self):
        rows = [
            build_suite_epoch_row(
                suite_id=SUITE_TRANSFER,
                sft_epoch=1,
                seeds=[42, 123, 456],
                sft_rates=[50.0, 50.0, 50.0],
                grpo_rates=[40.0, 40.0, 40.0],
            ),
            build_suite_epoch_row(
                suite_id=SUITE_TRANSFER,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[57.6, 55.0, 56.0],
                grpo_rates=[45.2, 44.0, 46.0],
            ),
            build_suite_epoch_row(
                suite_id=SUITE_GOAL_RED,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[14.8, 12.0, 13.0],
                grpo_rates=[1.6, 2.0, 1.2],
            ),
        ]
        reading = ood_verdict(rows)
        self.assertEqual(reading["verdict"], VERDICT_HURTS)

    def test_verdict_mixed_when_e9_suites_disagree(self):
        rows = [
            build_suite_epoch_row(
                suite_id=SUITE_TRANSFER,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[50.0, 50.0, 50.0],
                grpo_rates=[40.0, 40.0, 40.0],
            ),
            build_suite_epoch_row(
                suite_id=SUITE_GOAL_RED,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[10.0, 10.0, 10.0],
                grpo_rates=[12.0, 11.0, 13.0],
            ),
        ]
        self.assertEqual(ood_verdict(rows)["verdict"], VERDICT_MIXED)

    def test_verdict_not_hurts_when_e9_both_nonnegative(self):
        rows = [
            build_suite_epoch_row(
                suite_id=SUITE_TRANSFER,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[50.0, 50.0, 50.0],
                grpo_rates=[50.0, 51.0, 52.0],
            ),
            build_suite_epoch_row(
                suite_id=SUITE_GOAL_RED,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[10.0, 10.0, 10.0],
                grpo_rates=[10.0, 10.0, 10.0],
            ),
        ]
        self.assertEqual(ood_verdict(rows)["verdict"], VERDICT_NOT_HURTS)

    def test_verdict_inconclusive_without_both_e9_suites(self):
        rows = [
            build_suite_epoch_row(
                suite_id=SUITE_TRANSFER,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[57.6, 55.0, 56.0],
                grpo_rates=[45.2, 44.0, 46.0],
            ),
        ]
        self.assertEqual(ood_verdict(rows)["verdict"], VERDICT_INCONCLUSIVE)

    def test_summary_payload_nests_suites_and_verdict(self):
        rows = [
            build_suite_epoch_row(
                suite_id=SUITE_TRANSFER,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[57.6, 55.0, 56.0],
                grpo_rates=[45.2, 44.0, 46.0],
            ),
            build_suite_epoch_row(
                suite_id=SUITE_GOAL_RED,
                sft_epoch=PRIMARY_EPOCH,
                seeds=[42, 123, 456],
                sft_rates=[14.8, 12.0, 13.0],
                grpo_rates=[1.6, 2.0, 1.2],
            ),
        ]
        payload = build_summary_payload(
            rows,
            config={"eval_seeds": [42, 123, 456], "primary_epoch": 9},
        )
        self.assertEqual(payload["verdict"], VERDICT_HURTS)
        self.assertIn("9", payload["suites"][SUITE_TRANSFER])
        self.assertAlmostEqual(payload["suites"][SUITE_TRANSFER]["9"]["by_seed"]["42"]["sft"], 57.6)


class GrpoEvalCommandGoalColorTest(unittest.TestCase):
    def test_goal_color_flags_appended_when_set(self):
        cmd = build_grpo_eval_command(
            python="python",
            test_models=Path("scripts/test_models.py"),
            env_size=8,
            dataset=Path("datasets/dataset_8x8"),
            sft_adapter=Path("checkpoints/sft_e9"),
            grpo_adapter=Path("checkpoints/grpo_e9/episode-100"),
            episodes=250,
            seed=123,
            max_steps=12,
            output_json=Path("out.json"),
            goal_color="red",
            prompt_goal_color="red",
        )
        self.assertIn("--goal-color", cmd)
        self.assertEqual(cmd[cmd.index("--goal-color") + 1], "red")
        self.assertEqual(cmd[cmd.index("--prompt-goal-color") + 1], "red")
        self.assertNotIn("--skip-sft", cmd)

    def test_goal_color_omitted_by_default(self):
        cmd = build_grpo_eval_command(
            python="python",
            test_models=Path("scripts/test_models.py"),
            env_size=8,
            dataset=Path("datasets/dataset_8x8"),
            sft_adapter=Path("checkpoints/sft_e9"),
            grpo_adapter=Path("checkpoints/grpo_e9/episode-100"),
            episodes=250,
            seed=42,
            max_steps=12,
            output_json=Path("out.json"),
        )
        self.assertNotIn("--goal-color", cmd)
        self.assertNotIn("--prompt-goal-color", cmd)


if __name__ == "__main__":
    unittest.main()
