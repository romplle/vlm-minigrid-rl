"""CPU tests for e9 OOD GRPO train-seed replication (C1 caps 12/40)."""

import unittest
from pathlib import Path

from vlm_minigrid_rl.ood_multiseed import (
    SUITE_GOAL_RED,
    SUITE_TRANSFER,
    SUITES,
    VERDICT_HURTS,
    VERDICT_MIXED,
    VERDICT_NOT_HURTS,
    build_suite_epoch_row,
)
from vlm_minigrid_rl.ood_trainseed import (
    C1_GOAL_RED_MAX_STEPS,
    C1_TRANSFER_MAX_STEPS,
    FROZEN_C1_RUN_NAME,
    FROZEN_E9_RUN_NAME,
    PRIMARY_EPOCH,
    adapter_bundle_complete,
    assert_run_dir_allowed,
    build_trainseed_summary_payload,
    c1_reuse_matches,
    experiment_id_for_epoch,
    grpo_path_matches_train_seed,
    suite_c1_max_steps,
    trainseed_cell_path,
    trainseed_ood_verdict,
)


def _c1_payload(
    *,
    suite_id: str,
    eval_seed: int,
    sft_rate: float,
    grpo_rate: float,
    train_seed: int = 42,
    max_steps: int | None = None,
    episodes: int = 250,
    sft_epoch: int = PRIMARY_EPOCH,
) -> dict:
    suite = SUITES[suite_id]
    steps = suite_c1_max_steps(suite) if max_steps is None else max_steps
    if train_seed == 42:
        grpo = (
            "I:/VS Code/vlm-minigrid-rl/checkpoints/"
            f"grpo_adapter_8x8_handoff_from_e10_e{sft_epoch}/episode-100"
        )
    else:
        grpo = (
            "I:/VS Code/vlm-minigrid-rl/checkpoints/"
            f"grpo_adapter_8x8_handoff_from_e10_e{sft_epoch}_seed{train_seed}/"
            "episode-100"
        )
    return {
        "config": {
            "env_size": int(suite["eval_env"]),
            "sft_adapter_path": (
                "I:/VS Code/vlm-minigrid-rl/checkpoints/"
                f"sft_adapter_8x8_bs32_e10/epoch-{sft_epoch}"
            ),
            "grpo_adapter_path": grpo,
            "episodes": episodes,
            "max_steps": steps,
            "goal_color": suite["goal_color"],
            "prompt_goal_color": suite["prompt_goal_color"],
            "seed": eval_seed,
        },
        "policies": [
            {"name": "SFT", "metrics": {"success_rate": sft_rate}},
            {"name": "GRPO", "metrics": {"success_rate": grpo_rate}},
        ],
    }


class C1CapsAndPathsTest(unittest.TestCase):
    def test_c1_caps_not_slim_profile(self):
        self.assertEqual(C1_TRANSFER_MAX_STEPS, 40)
        self.assertEqual(C1_GOAL_RED_MAX_STEPS, 12)
        self.assertEqual(suite_c1_max_steps(SUITES[SUITE_TRANSFER]), 40)
        self.assertEqual(suite_c1_max_steps(SUITES[SUITE_GOAL_RED]), 12)

    def test_cell_path_nests_train_seed(self):
        run = Path("runs/2026-09-ood-trainseed-e9-8x8")
        self.assertEqual(
            trainseed_cell_path(run, 123, "transfer8to16", 42),
            run / "train123" / "8x8" / "ood_transfer8to16_e9_seed42.json",
        )

    def test_grpo_path_seed42_is_unsuffixed(self):
        path = "checkpoints/grpo_adapter_8x8_handoff_from_e10_e9/episode-100"
        self.assertTrue(grpo_path_matches_train_seed(path, 42))
        self.assertFalse(grpo_path_matches_train_seed(path, 123))

    def test_grpo_path_seed123_does_not_match_42(self):
        path = "checkpoints/grpo_adapter_8x8_handoff_from_e10_e9_seed123/episode-100"
        self.assertTrue(grpo_path_matches_train_seed(path, 123))
        self.assertFalse(grpo_path_matches_train_seed(path, 42))
        self.assertFalse(grpo_path_matches_train_seed(path, 456))

    def test_c1_reuse_requires_max_steps_12_or_40(self):
        ok = _c1_payload(
            suite_id=SUITE_TRANSFER, eval_seed=42, sft_rate=57.6, grpo_rate=45.2
        )
        self.assertTrue(
            c1_reuse_matches(ok, suite=SUITES[SUITE_TRANSFER], eval_seed=42)
        )
        slim = _c1_payload(
            suite_id=SUITE_TRANSFER,
            eval_seed=42,
            sft_rate=57.6,
            grpo_rate=45.2,
            max_steps=38,
        )
        self.assertFalse(
            c1_reuse_matches(slim, suite=SUITES[SUITE_TRANSFER], eval_seed=42)
        )

    def test_grpo_path_e1_and_e5_do_not_match_e9(self):
        for epoch in (1, 5):
            unsuffixed = (
                f"checkpoints/grpo_adapter_8x8_handoff_from_e10_e{epoch}/episode-100"
            )
            seed123 = (
                f"checkpoints/grpo_adapter_8x8_handoff_from_e10_e{epoch}"
                "_seed123/episode-100"
            )
            self.assertTrue(grpo_path_matches_train_seed(unsuffixed, 42, sft_epoch=epoch))
            self.assertFalse(grpo_path_matches_train_seed(unsuffixed, 42))
            self.assertTrue(grpo_path_matches_train_seed(seed123, 123, sft_epoch=epoch))
            self.assertFalse(grpo_path_matches_train_seed(seed123, 123))
            self.assertFalse(grpo_path_matches_train_seed(unsuffixed, 123, sft_epoch=epoch))

    def test_cell_path_uses_sft_epoch(self):
        run = Path("runs/2026-09-ood-trainseed-e1-8x8")
        self.assertEqual(
            trainseed_cell_path(run, 123, "transfer8to16", 42, sft_epoch=1),
            run / "train123" / "8x8" / "ood_transfer8to16_e1_seed42.json",
        )

    def test_c1_reuse_matches_e1_and_e5(self):
        for epoch in (1, 5):
            ok = _c1_payload(
                suite_id=SUITE_TRANSFER,
                eval_seed=42,
                sft_rate=57.6,
                grpo_rate=45.2,
                sft_epoch=epoch,
            )
            self.assertTrue(
                c1_reuse_matches(
                    ok,
                    suite=SUITES[SUITE_TRANSFER],
                    eval_seed=42,
                    sft_epoch=epoch,
                )
            )
            self.assertFalse(
                c1_reuse_matches(ok, suite=SUITES[SUITE_TRANSFER], eval_seed=42)
            )

    def test_run_dir_guards(self):
        assert_run_dir_allowed(Path("runs") / FROZEN_E9_RUN_NAME, 9)
        with self.assertRaises(ValueError):
            assert_run_dir_allowed(Path("runs") / FROZEN_E9_RUN_NAME, 1)
        with self.assertRaises(ValueError):
            assert_run_dir_allowed(Path("runs") / FROZEN_C1_RUN_NAME, 1)
        self.assertEqual(experiment_id_for_epoch(1), "2026-09-ood-trainseed-e1-8x8")
        self.assertEqual(experiment_id_for_epoch(5), "2026-09-ood-trainseed-e5-8x8")
        self.assertEqual(experiment_id_for_epoch(9), FROZEN_E9_RUN_NAME)


class TrainseedVerdictTest(unittest.TestCase):
    def test_hurts_when_all_three_train_seeds_negative_on_both_suites(self):
        reading = trainseed_ood_verdict(
            transfer_deltas={42: -12.4, 123: -10.0, 456: -8.0},
            goal_deltas={42: -15.5, 123: -11.0, 456: -9.0},
        )
        self.assertEqual(reading["verdict"], VERDICT_HURTS)
        self.assertEqual(reading["transfer_n_train_negative"], 3)
        self.assertEqual(reading["goal_n_train_negative"], 3)
        reading["reason"].encode("ascii")

    def test_mixed_if_one_suite_has_a_positive_train_seed(self):
        reading = trainseed_ood_verdict(
            transfer_deltas={42: -12.4, 123: 1.0, 456: -8.0},
            goal_deltas={42: -15.5, 123: -11.0, 456: -9.0},
        )
        self.assertEqual(reading["verdict"], VERDICT_MIXED)

    def test_not_hurts_if_both_suites_nonnegative_means(self):
        reading = trainseed_ood_verdict(
            transfer_deltas={42: 1.0, 123: 2.0, 456: 0.5},
            goal_deltas={42: 0.1, 123: 0.2, 456: 0.0},
        )
        self.assertEqual(reading["verdict"], VERDICT_NOT_HURTS)

    def test_summary_aggregates_across_train_seeds(self):
        by_train = {}
        for train_seed, t_delta, g_delta in (
            (42, -12.0, -15.0),
            (123, -9.0, -12.0),
            (456, -6.0, -9.0),
        ):
            by_train[str(train_seed)] = {
                SUITE_TRANSFER: build_suite_epoch_row(
                    suite_id=SUITE_TRANSFER,
                    sft_epoch=9,
                    seeds=[42, 123, 456],
                    sft_rates=[50.0, 50.0, 50.0],
                    grpo_rates=[50.0 + t_delta, 50.0 + t_delta, 50.0 + t_delta],
                ),
                SUITE_GOAL_RED: build_suite_epoch_row(
                    suite_id=SUITE_GOAL_RED,
                    sft_epoch=9,
                    seeds=[42, 123, 456],
                    sft_rates=[20.0, 20.0, 20.0],
                    grpo_rates=[20.0 + g_delta, 20.0 + g_delta, 20.0 + g_delta],
                ),
            }
        payload = build_trainseed_summary_payload(
            by_train,
            config={"train_seeds": [42, 123, 456], "eval_seeds": [42, 123, 456]},
        )
        self.assertEqual(payload["verdict"], VERDICT_HURTS)
        self.assertAlmostEqual(payload["across_train_seed"]["transfer8to16"]["delta_mean"], -9.0)
        self.assertEqual(
            payload["across_train_seed"]["transfer8to16"]["n_negative"], 3
        )

    def test_adapter_bundle_requires_config_and_weights(self):
        missing = Path("checkpoints/does-not-exist-adapter")
        self.assertFalse(adapter_bundle_complete(missing))


if __name__ == "__main__":
    unittest.main()
