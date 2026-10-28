"""CPU tests for native 16x16 SFT eval-seed helper. No VLM."""

import unittest
from pathlib import Path

from vlm_minigrid_rl.native16_sft_evalseed import (
    FROZEN_FAR_RUN,
    REUSE_EVAL_SEED,
    STABILITY_GATE_PP,
    aggregate_far,
    assert_not_frozen_far_run_dir,
    cell_out_path,
    reuse_seed42_row,
)


class FrozenDirTest(unittest.TestCase):
    def test_refuse_frozen_far_run_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, FROZEN_FAR_RUN):
            assert_not_frozen_far_run_dir(Path("runs") / FROZEN_FAR_RUN)
        with self.assertRaisesRegex(ValueError, FROZEN_FAR_RUN):
            assert_not_frozen_far_run_dir(
                Path("I:/VS Code/vlm-minigrid-rl/runs") / FROZEN_FAR_RUN / "nested"
            )
        assert_not_frozen_far_run_dir(Path("runs/2026-09-native16-sft-evalseed"))


class ReuseRowTest(unittest.TestCase):
    def test_reuse_seed42_row_reads_success_far(self) -> None:
        payload = {
            "config": {
                "success_far": 81.69014084507042,
                "n_far": 142,
                "n_visible": 41,
                "n_facing_miss": 67,
                "success_visible": 100.0,
                "success_facing_miss": 73.13432835820896,
                "eval_max_steps": 40,
                "sft_adapter_path": "epoch-8",
            },
            "policies": [{"name": "SFT", "metrics": {"success_rate": 82.4}}],
        }
        row = reuse_seed42_row(payload)
        self.assertEqual(row["seed"], REUSE_EVAL_SEED)
        self.assertEqual(row["source"], "reuse")
        self.assertEqual(row["n_far"], 142)
        self.assertAlmostEqual(row["success_far"], 81.69014084507042)


class AggregateTest(unittest.TestCase):
    def test_aggregate_far_gate(self) -> None:
        reuse = {"seed": 42, "success_far": 81.69, "n_far": 142, "source": "reuse"}
        gpu_ok = [
            {"seed": 123, "success_far": 80.0, "n_far": 140, "source": "gpu"},
            {"seed": 456, "success_far": 83.0, "n_far": 138, "source": "gpu"},
        ]
        gpu = aggregate_far(gpu_ok)
        self.assertEqual(gpu["n"], 2)
        self.assertIsNone(gpu["within_3pp_of_seed42"])
        three = aggregate_far([reuse, *gpu_ok])
        self.assertTrue(three["within_3pp_of_seed42"])
        self.assertLessEqual(abs(three["delta_vs_seed42_pp"]), STABILITY_GATE_PP)
        far_off = aggregate_far(
            [reuse, {"seed": 123, "success_far": 70.0, "n_far": 140, "source": "gpu"}]
        )
        self.assertFalse(far_off["within_3pp_of_seed42"])

    def test_cell_out_path(self) -> None:
        self.assertEqual(cell_out_path(Path("runs/x"), 123).name, "sft_native16_s123.json")


if __name__ == "__main__":
    unittest.main()
