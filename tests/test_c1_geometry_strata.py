"""CPU tests for C1 start strata. No VLM."""

import unittest

from vlm_minigrid_rl.c1_geometry import (
    C1_EVAL_16X16,
    STRATUM_FAR,
    STRATUM_FACING_MISS,
    STRATUM_VISIBLE,
    episode_stratum,
    stratify_geometry,
)
from vlm_minigrid_rl.env_profiles import EMPTY_16X16


class HorizonConstantTest(unittest.TestCase):
    def test_c1_transfer_is_not_slim_eval(self):
        self.assertEqual(C1_EVAL_16X16, 40)
        self.assertEqual(EMPTY_16X16.eval_max_steps, 38)
        self.assertNotEqual(C1_EVAL_16X16, EMPTY_16X16.eval_max_steps)


class EpisodeStratumTest(unittest.TestCase):
    def test_visible_wins(self):
        self.assertEqual(
            episode_stratum({"start_goal_visible": True, "start_visible_any_dir": True}),
            STRATUM_VISIBLE,
        )

    def test_facing_miss(self):
        self.assertEqual(
            episode_stratum({"start_goal_visible": False, "start_visible_any_dir": True}),
            STRATUM_FACING_MISS,
        )

    def test_far(self):
        self.assertEqual(
            episode_stratum({"start_goal_visible": False, "start_visible_any_dir": False}),
            STRATUM_FAR,
        )


class StratifyTest(unittest.TestCase):
    def test_success_far(self):
        rows = [
            {"start_goal_visible": False, "start_visible_any_dir": False, "success": True},
            {"start_goal_visible": False, "start_visible_any_dir": False, "success": False},
            {"start_goal_visible": True, "start_visible_any_dir": True, "success": True},
        ]
        stats = stratify_geometry(rows)
        self.assertEqual(stats["n_far"], 2)
        self.assertEqual(stats["success_far"], 50.0)
        self.assertEqual(stats["n_visible"], 1)


if __name__ == "__main__":
    unittest.main()
