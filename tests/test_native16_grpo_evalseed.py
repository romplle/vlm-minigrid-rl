"""CPU tests for native 16x16 GRPO eval-seed verdict. No VLM."""

import unittest
from pathlib import Path

from vlm_minigrid_rl.c1_geometry import C1_EVAL_16X16
from vlm_minigrid_rl.native16_grpo_evalseed import (
    FAR_CELL_FLOOR,
    FROZEN_FAR_RUN,
    GPU_EVAL_SEEDS,
    REUSE_EVAL_SEED,
    TRAIN_SEEDS,
    VERDICT_FLOOR,
    VERDICT_INVALID,
    VERDICT_SIGN,
    VERDICT_SOFT,
    VERDICT_TRANSFER,
    assert_not_frozen_far_run_dir,
    cell_out_path,
    decide_verdict,
    planned_gpu_cells,
    reuse_row_from_payload,
)

SFT_FAR = {
    42: 81.69014084507042,
    123: 79.86111111111111,
    456: 69.46564885496183,
}
REUSE_FAR = {
    42: 71.12676056338029,
    123: 78.16901408450704,
    456: 71.83098591549296,
}


def _sft_rows() -> dict[int, dict]:
    return {
        seed: {"seed": seed, "success_far": far, "success_facing_miss": 80.0}
        for seed, far in SFT_FAR.items()
    }


def _row(train: int, eval_seed: int, far: float, *, facing: float = 80.0) -> dict:
    return {
        "train_seed": train,
        "eval_seed": eval_seed,
        "source": "reuse" if eval_seed == REUSE_EVAL_SEED else "gpu",
        "success_rate": far,
        "success_far": far,
        "success_facing_miss": facing,
        "success_visible": 100.0,
        "n_far": 140,
        "n_facing_miss": 60,
        "n_visible": 50,
        "eval_max_steps": C1_EVAL_16X16,
    }


def _grid(*, gpu_far: dict[int, float] | None = None, reuse_far: dict[int, float] | None = None) -> list[dict]:
    reuse = REUSE_FAR if reuse_far is None else reuse_far
    gpu = {123: 70.0, 456: 60.0} if gpu_far is None else gpu_far
    rows = [_row(train, REUSE_EVAL_SEED, reuse[train]) for train in TRAIN_SEEDS]
    for eval_seed in GPU_EVAL_SEEDS:
        for train in TRAIN_SEEDS:
            rows.append(_row(train, eval_seed, gpu[eval_seed]))
    return rows


class FrozenDirTest(unittest.TestCase):
    def test_refuse_frozen_far_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, FROZEN_FAR_RUN):
            assert_not_frozen_far_run_dir(Path("runs") / FROZEN_FAR_RUN)
        with self.assertRaisesRegex(ValueError, FROZEN_FAR_RUN):
            assert_not_frozen_far_run_dir(
                Path("I:/VS Code/vlm-minigrid-rl/runs") / FROZEN_FAR_RUN / "nested"
            )
        assert_not_frozen_far_run_dir(Path("runs/2026-09-native16-grpo-evalseed"))

    def test_six_gpu_cells(self) -> None:
        cells = planned_gpu_cells()
        self.assertEqual(len(cells), 6)
        self.assertNotIn(REUSE_EVAL_SEED, {eval_seed for _, eval_seed in cells})
        self.assertEqual(
            cell_out_path(Path("runs/x"), 42, 123).name,
            "grpo_t42_s123.json",
        )


class ReuseRowTest(unittest.TestCase):
    def test_reuse_reads_config_far(self) -> None:
        payload = {
            "config": {
                "seed": 42,
                "eval_max_steps": 40,
                "method": "G1",
                "success_far": REUSE_FAR[42],
                "n_far": 142,
                "success_facing_miss": 88.06,
            },
            "policies": [{"name": "GRPO42", "metrics": {"success_rate": 80.4}}],
        }
        row = reuse_row_from_payload(payload, 42)
        self.assertEqual(row["eval_seed"], 42)
        self.assertAlmostEqual(row["success_far"], REUSE_FAR[42])
        payload["config"]["eval_max_steps"] = 38
        with self.assertRaisesRegex(ValueError, "40"):
            reuse_row_from_payload(payload, 42)


class VerdictTest(unittest.TestCase):
    def test_seed42_reuse_is_registered_mean_and_soft_delta(self) -> None:
        decision = decide_verdict(grpo_rows=_grid(), sft_rows=_sft_rows())
        self.assertTrue(decision["reuse_ok"])
        self.assertAlmostEqual(decision["reuse_grpo_far_mean"], 73.70892018779343, places=4)
        self.assertLessEqual(abs(decision["reuse_grpo_far_mean"] - 73.71), 0.05)
        delta42 = decision["paired_deltas"]["42"]["delta"]
        self.assertAlmostEqual(delta42, -7.981220657277, places=3)
        self.assertLess(delta42, 0.0)
        self.assertGreater(delta42, -21.0)
        self.assertEqual(decision["verdict"], VERDICT_SOFT)

    def test_sign_differs_when_a_paired_delta_is_nonnegative(self) -> None:
        decision = decide_verdict(
            grpo_rows=_grid(gpu_far={123: 85.0, 456: 60.0}),
            sft_rows=_sft_rows(),
        )
        self.assertGreaterEqual(decision["paired_deltas"]["123"]["delta"], 0.0)
        self.assertEqual(decision["verdict"], VERDICT_SIGN)

    def test_transfer_scale_when_a_negative_delta_reaches_the_hole(self) -> None:
        decision = decide_verdict(
            grpo_rows=_grid(gpu_far={123: 70.0, 456: 40.0}),
            sft_rows=_sft_rows(),
        )
        self.assertLessEqual(decision["paired_deltas"]["456"]["delta"], -21.0)
        self.assertEqual(decision["verdict"], VERDICT_TRANSFER)

    def test_floor_breach_beats_sign(self) -> None:
        rows = [
            row
            for row in _grid(gpu_far={123: 90.0, 456: 60.0})
            if not (row["train_seed"] == 42 and row["eval_seed"] == 123)
        ]
        rows.append(_row(42, 123, FAR_CELL_FLOOR))
        decision = decide_verdict(grpo_rows=rows, sft_rows=_sft_rows())
        self.assertEqual(decision["verdict"], VERDICT_FLOOR)
        self.assertTrue(decision["floor_cells"])

    def test_invalid_reuse_blocks_the_new_seeds(self) -> None:
        decision = decide_verdict(
            grpo_rows=_grid(reuse_far={42: 90.0, 123: 90.0, 456: 90.0}),
            sft_rows=_sft_rows(),
        )
        self.assertFalse(decision["reuse_ok"])
        self.assertEqual(decision["verdict"], VERDICT_INVALID)

    def test_facing_miss_does_not_change_the_far_verdict(self) -> None:
        low = _grid()
        high = _grid()
        for row in high:
            row["success_facing_miss"] = 99.0
        self.assertEqual(
            decide_verdict(grpo_rows=low, sft_rows=_sft_rows())["verdict"],
            decide_verdict(grpo_rows=high, sft_rows=_sft_rows())["verdict"],
        )


if __name__ == "__main__":
    unittest.main()
