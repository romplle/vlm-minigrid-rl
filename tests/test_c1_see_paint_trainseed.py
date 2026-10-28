"""CPU tests for SEE/PAINT train-seed paths and across-train verdicts."""

import unittest
from pathlib import Path

from vlm_minigrid_rl.c1_see_paint_trainseed import (
    G1_TRANSFER_GRPO,
    across_axis_verdict,
    assert_not_frozen_run_dir,
    g1_sanity_ok,
    joint_mechanism_verdict,
    planned_cells,
    train_run_dir,
)
from vlm_minigrid_rl.handoff_sweep import grpo_episode_dir, grpo_handoff_root
from vlm_minigrid_rl.ood_trainseed import grpo_path_matches_train_seed


class PathsTest(unittest.TestCase):
    def test_seed_suffix(self):
        root = grpo_handoff_root(8, 9, Path("checkpoints"), train_seed=123)
        ep = grpo_episode_dir(root, 100)
        self.assertTrue(grpo_path_matches_train_seed(ep, 123))
        self.assertFalse(grpo_path_matches_train_seed(ep, 42))

    def test_nested_run_dir(self):
        self.assertEqual(
            train_run_dir(Path("runs/2026-09-c1-see-paint-trainseed"), 456).as_posix(),
            "runs/2026-09-c1-see-paint-trainseed/train456",
        )

    def test_refuse_frozen(self):
        with self.assertRaises(ValueError):
            assert_not_frozen_run_dir(Path("runs/2026-08-c1-search-8x8"))


class SanityTest(unittest.TestCase):
    def test_canonical_123_transfer(self):
        self.assertTrue(g1_sanity_ok(suite="see", train_seed=123, grpo_mean=46.93))
        self.assertFalse(g1_sanity_ok(suite="see", train_seed=123, grpo_mean=40.0))
        self.assertEqual(G1_TRANSFER_GRPO[42], 45.07)


class AcrossTest(unittest.TestCase):
    def test_replicated(self):
        see = across_axis_verdict(
            {42: "SEARCH_PARTIAL", 123: "SEARCH_PARTIAL", 456: "SEARCH_RECOVERS"},
            good={"SEARCH_PARTIAL", "SEARCH_RECOVERS"},
            prefix="SEE",
        )
        color = across_axis_verdict(
            {42: "COLOR_RECOVERS", 123: "COLOR_RECOVERS", 456: "COLOR_RECOVERS"},
            good={"COLOR_RECOVERS"},
            prefix="COLOR",
        )
        self.assertEqual(see, "SEE_REPLICATED")
        self.assertEqual(color, "COLOR_REPLICATED")
        self.assertEqual(joint_mechanism_verdict(see, color), "MECHANISM_REPLICATED")

    def test_seed42_only(self):
        see = across_axis_verdict(
            {42: "SEARCH_PARTIAL", 123: "SEARCH_NULL", 456: "SEARCH_NULL"},
            good={"SEARCH_PARTIAL", "SEARCH_RECOVERS"},
            prefix="SEE",
        )
        color = across_axis_verdict(
            {42: "COLOR_RECOVERS", 123: "COLOR_NULL", 456: "COLOR_PARTIAL"},
            good={"COLOR_RECOVERS"},
            prefix="COLOR",
        )
        self.assertEqual(see, "SEE_SEED42_ONLY")
        self.assertEqual(color, "COLOR_SEED42_ONLY")
        self.assertEqual(joint_mechanism_verdict(see, color), "MECHANISM_SEED42")

    def test_color_partial_is_not_replicated(self):
        color = across_axis_verdict(
            {42: "COLOR_RECOVERS", 123: "COLOR_PARTIAL", 456: "COLOR_RECOVERS"},
            good={"COLOR_RECOVERS"},
            prefix="COLOR",
        )
        self.assertEqual(color, "COLOR_MIXED")


class PlannedCellsTest(unittest.TestCase):
    def test_gpu_order_and_count(self):
        rows = planned_cells()
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0], (123, "PAINT", "G1", 42))
        self.assertEqual(rows[-1], (456, "SEE", "SEE", 456))


if __name__ == "__main__":
    unittest.main()

