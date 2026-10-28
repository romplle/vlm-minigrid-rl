"""Unit tests for greedy-vs-sampled F3 summary helpers."""

import unittest

from vlm_minigrid_rl.greedy_sampled_diag import (
    build_epoch_row,
    build_f3_reading,
    build_summary_payload,
    cell_json_name,
)


class GreedySampledSummaryTest(unittest.TestCase):
    def test_cell_json_name(self):
        self.assertEqual(
            cell_json_name("sft", 3, "generate", 42),
            "sft_e3_modegenerate_seed42.json",
        )
        self.assertEqual(
            cell_json_name("grpo", 8, "argmax3", 123),
            "grpo_e8_ep100_modeargmax3_seed123.json",
        )

    def test_f3_temperature_artifact_when_greedy_flat_sampled_drops(self):
        rows = [
            build_epoch_row(
                1,
                sft_sampled=[70.0, 70.0, 70.0],
                sft_greedy=[80.0, 80.0, 80.0],
                grpo_sampled=[75.0, 75.0, 75.0],
                grpo_greedy=[81.0, 81.0, 81.0],
            ),
            build_epoch_row(
                3,
                sft_sampled=[78.0, 78.0, 78.0],
                sft_greedy=[81.0, 81.0, 81.0],
                grpo_sampled=[81.0, 81.0, 81.0],
                grpo_greedy=[82.0, 82.0, 82.0],
            ),
            build_epoch_row(
                8,
                sft_sampled=[82.0, 82.0, 82.0],
                sft_greedy=[82.0, 82.0, 82.0],
                grpo_sampled=[82.5, 82.5, 82.5],
                grpo_greedy=[82.5, 82.5, 82.5],
            ),
        ]
        f3 = build_f3_reading(rows)
        self.assertEqual(f3["early_epoch"], 1)
        self.assertEqual(f3["late_epoch"], 8)
        self.assertAlmostEqual(f3["slope_sampled"], -4.5)
        self.assertLess(abs(f3["slope_greedy"]), 1.0)
        self.assertTrue(f3["temperature_artifact_suspected"])
        self.assertEqual(f3["verdict"], "temperature_artifact")

    def test_f3_policy_improvement_when_deltas_track(self):
        rows = [
            build_epoch_row(
                1,
                sft_sampled=[70.0],
                sft_greedy=[71.0],
                grpo_sampled=[75.0],
                grpo_greedy=[76.0],
            ),
            build_epoch_row(
                8,
                sft_sampled=[80.0],
                sft_greedy=[81.0],
                grpo_sampled=[82.0],
                grpo_greedy=[83.0],
            ),
        ]
        f3 = build_f3_reading(rows)
        self.assertAlmostEqual(f3["slope_sampled"], f3["slope_greedy"])
        self.assertTrue(f3["policy_improvement_suspected"])
        self.assertEqual(f3["verdict"], "policy_improvement")

    def test_f3_policy_improvement_when_greedy_amplifies(self):
        """Same-direction large slopes with different magnitudes still count as policy."""
        rows = [
            build_epoch_row(
                1,
                sft_sampled=[63.0],
                sft_greedy=[80.0],
                grpo_sampled=[69.0],
                grpo_greedy=[91.0],
            ),
            build_epoch_row(
                8,
                sft_sampled=[83.0],
                sft_greedy=[84.0],
                grpo_sampled=[82.0],
                grpo_greedy=[84.0],
            ),
        ]
        f3 = build_f3_reading(rows)
        self.assertLess(f3["slope_sampled"], -2.0)
        self.assertLess(f3["slope_greedy"], -2.0)
        self.assertFalse(f3["temperature_artifact_suspected"])
        self.assertTrue(f3["policy_improvement_suspected"])
        self.assertEqual(f3["verdict"], "policy_improvement")

    def test_build_summary_payload_includes_f3(self):
        rows = [
            build_epoch_row(
                1,
                sft_sampled=[1.0],
                sft_greedy=[1.0],
                grpo_sampled=[1.0],
                grpo_greedy=[1.0],
            ),
            build_epoch_row(
                8,
                sft_sampled=[1.0],
                sft_greedy=[1.0],
                grpo_sampled=[1.0],
                grpo_greedy=[1.0],
            ),
        ]
        payload = build_summary_payload(rows, config={"env_size": 16})
        self.assertEqual(payload["config"]["env_size"], 16)
        self.assertEqual(payload["f3"]["verdict"], "both_flat")
        self.assertEqual(len(payload["epochs"]), 2)


if __name__ == "__main__":
    unittest.main()
