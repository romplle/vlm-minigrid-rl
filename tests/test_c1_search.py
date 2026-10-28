"""C1 search-then-generate: BFS-to-see, restore pose, verdicts. No VLM."""

from __future__ import annotations

import unittest

from minigrid.core.world_object import Goal

from vlm_minigrid_rl.c1_geometry import STRATUM_FAR
from vlm_minigrid_rl.c1_search import (
    C1_EVAL_16X16,
    LOW_N_MIN,
    METHOD_G1,
    METHOD_SEE,
    CellResult,
    c1_search_verdict,
    c1_search_verdict_grpo,
    far_see_truncated_frac,
)
from vlm_minigrid_rl.env_profiles import EMPTY_16X16
from vlm_minigrid_rl.minigrid_utils import (
    create_minigrid_env,
    dirs_that_see_goal,
    goal_in_agent_view,
    goal_in_agent_view_at,
    goal_visible_any_dir,
    reset_env_with_goal,
    shortest_path_to_see_goal,
)
from vlm_minigrid_rl.training_utils import ACTION_TO_ID, ID_TO_ACTION


def _find_goal(env) -> tuple[int, int]:
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    raise AssertionError("no goal on grid")


def _pose_facing_goal(goal: tuple[int, int], env_size: int) -> tuple[tuple[int, int], int]:
    gx, gy = goal
    limit = env_size - 2
    candidates = [
        ((gx - 1, gy), 0),
        ((gx + 1, gy), 2),
        ((gx, gy - 1), 1),
        ((gx, gy + 1), 3),
    ]
    for pos, direction in candidates:
        if 1 <= pos[0] <= limit and 1 <= pos[1] <= limit:
            return pos, direction
    raise AssertionError(f"no interior neighbor for goal {goal}")


def _cell(
    method: str,
    policy: str,
    seed: int,
    success_rate: float,
    **kwargs,
) -> CellResult:
    return CellResult(method=method, policy=policy, seed=seed, success_rate=success_rate, **kwargs)


def _triple(method: str, policy: str, success_rate: float, **kwargs) -> list[CellResult]:
    return [_cell(method, policy, seed, success_rate, **kwargs) for seed in (42, 123, 456)]


class HorizonConstantTest(unittest.TestCase):
    def test_c1_transfer_is_not_slim_eval(self):
        self.assertEqual(C1_EVAL_16X16, 40)
        self.assertEqual(EMPTY_16X16.eval_max_steps, 38)
        self.assertNotEqual(C1_EVAL_16X16, EMPTY_16X16.eval_max_steps)


class FarTruncatedFracTest(unittest.TestCase):
    def test_far_only(self):
        rows = [
            {
                "start_goal_visible": True,
                "start_visible_any_dir": True,
                "see_truncated": True,
            },
            {
                "start_goal_visible": False,
                "start_visible_any_dir": False,
                "see_truncated": True,
            },
            {
                "start_goal_visible": False,
                "start_visible_any_dir": False,
                "see_truncated": False,
            },
        ]
        self.assertEqual(far_see_truncated_frac(rows), 0.5)


class LiveFovSearchTest(unittest.TestCase):
    def test_view_at_restores_pos_and_dir(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            reset_env_with_goal(env, seed=0, goal_color="green")
            pos = tuple(int(v) for v in env.unwrapped.agent_pos)
            direction = int(env.unwrapped.agent_dir)
            goal_in_agent_view_at(env, (1, 1), (direction + 1) % 4)
            self.assertEqual(tuple(int(v) for v in env.unwrapped.agent_pos), pos)
            self.assertEqual(int(env.unwrapped.agent_dir), direction)
        finally:
            env.close()

    def test_already_visible_path_empty(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            reset_env_with_goal(env, seed=0, goal_color="green")
            goal = _find_goal(env)
            pos, facing = _pose_facing_goal(goal, env_size=8)
            env.unwrapped.agent_pos = pos
            env.unwrapped.agent_dir = facing
            self.assertTrue(goal_in_agent_view(env))
            before = (tuple(int(v) for v in env.unwrapped.agent_pos), int(env.unwrapped.agent_dir))
            self.assertEqual(shortest_path_to_see_goal(env), [])
            after = (tuple(int(v) for v in env.unwrapped.agent_pos), int(env.unwrapped.agent_dir))
            self.assertEqual(before, after)
        finally:
            env.close()

    def test_facing_miss_path_is_turns_only(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            reset_env_with_goal(env, seed=0, goal_color="green")
            goal = _find_goal(env)
            pos, facing = _pose_facing_goal(goal, env_size=8)
            env.unwrapped.agent_pos = pos
            env.unwrapped.agent_dir = (facing + 2) % 4
            self.assertFalse(goal_in_agent_view(env))
            self.assertTrue(goal_visible_any_dir(env))
            path = shortest_path_to_see_goal(env)
            self.assertTrue(path)
            self.assertTrue(all(action in (ACTION_TO_ID["left"], ACTION_TO_ID["right"]) for action in path))
            self.assertFalse(goal_in_agent_view(env))
            self.assertEqual(int(env.unwrapped.agent_dir), (facing + 2) % 4)
        finally:
            env.close()

    def test_far_corner_path_sees_goal_without_planning_step(self):
        env = create_minigrid_env(16, max_steps=40)
        try:
            reset_env_with_goal(env, seed=1, goal_color="green")
            env.unwrapped.agent_pos = (1, 1)
            env.unwrapped.agent_dir = 0
            goal = _find_goal(env)
            env.unwrapped.grid.set(goal[0], goal[1], None)
            env.unwrapped.grid.set(14, 14, Goal("green"))
            self.assertFalse(goal_in_agent_view(env))
            self.assertFalse(goal_visible_any_dir(env))
            before_pos = tuple(int(v) for v in env.unwrapped.agent_pos)
            before_dir = int(env.unwrapped.agent_dir)
            path = shortest_path_to_see_goal(env)
            self.assertTrue(path, "planner bug: empty path on far 16x16 corner")
            self.assertEqual(tuple(int(v) for v in env.unwrapped.agent_pos), before_pos)
            self.assertEqual(int(env.unwrapped.agent_dir), before_dir)
            self.assertEqual(STRATUM_FAR, "far")
            for action in path:
                obs, _reward, terminated, truncated, _ = env.step(action)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
            self.assertTrue(goal_in_agent_view(env))
            self.assertTrue(dirs_that_see_goal(env))
            self.assertTrue(any(ID_TO_ACTION[action] == "forward" for action in path))
        finally:
            env.close()


class VerdictTest(unittest.TestCase):
    _n = dict(
        n_visible=40,
        n_facing_miss=40,
        n_far=170,
        n_empty=210,
        far_see_truncated_frac=0.0,
    )

    def _pack(
        self,
        *,
        g1_sft: float,
        g1_grpo: float,
        see_sft: float,
        see_grpo: float,
        vis_sft: float = 99.0,
        vis_grpo: float = 96.0,
        vis_see_sft: float = 99.0,
        vis_see_grpo: float = 96.0,
        face_g1_sft: float = 95.0,
        face_g1_grpo: float = 96.0,
        face_see_sft: float = 95.0,
        face_see_grpo: float = 96.0,
        far_g1_sft: float = 26.0,
        far_g1_grpo: float = 5.0,
        far_see_sft: float = 26.0,
        far_see_grpo: float = 5.0,
        n_facing_miss: int = 40,
        far_see_truncated_frac: float = 0.0,
    ) -> list[CellResult]:
        nkw = dict(self._n)
        nkw["n_facing_miss"] = n_facing_miss
        nkw["far_see_truncated_frac"] = far_see_truncated_frac
        cells = []
        cells.extend(
            _triple(
                METHOD_G1,
                "SFT",
                g1_sft,
                success_visible=vis_sft,
                success_facing_miss=face_g1_sft,
                success_far=far_g1_sft,
                success_empty=40.0,
                **nkw,
            )
        )
        cells.extend(
            _triple(
                METHOD_G1,
                "GRPO",
                g1_grpo,
                success_visible=vis_grpo,
                success_facing_miss=face_g1_grpo,
                success_far=far_g1_grpo,
                success_empty=28.0,
                **nkw,
            )
        )
        cells.extend(
            _triple(
                METHOD_SEE,
                "SFT",
                see_sft,
                success_visible=vis_see_sft,
                success_facing_miss=face_see_sft,
                success_far=far_see_sft,
                success_empty=42.0,
                **nkw,
            )
        )
        cells.extend(
            _triple(
                METHOD_SEE,
                "GRPO",
                see_grpo,
                success_visible=vis_see_grpo,
                success_facing_miss=face_see_grpo,
                success_far=far_see_grpo,
                success_empty=50.0,
                **nkw,
            )
        )
        return cells

    def test_search_null(self):
        cells = self._pack(
            g1_sft=57.47,
            g1_grpo=45.07,
            see_sft=57.5,
            see_grpo=45.2,
            far_see_grpo=6.0,
        )
        self.assertIn("SEARCH_NULL", c1_search_verdict(cells))
        self.assertNotIn("INCONCLUSIVE", c1_search_verdict(cells))

    def test_search_recovers(self):
        cells = self._pack(
            g1_sft=57.47,
            g1_grpo=45.07,
            see_sft=80.0,
            see_grpo=90.0,
            far_see_grpo=92.0,
        )
        verdict = c1_search_verdict(cells)
        self.assertIn("SEARCH_RECOVERS", verdict)
        self.assertNotIn("SEE_TRUNCATED", verdict)

    def test_search_partial(self):
        cells = self._pack(
            g1_sft=57.47,
            g1_grpo=45.07,
            see_sft=60.0,
            see_grpo=55.0,
            far_see_grpo=30.0,
        )
        verdict = c1_search_verdict(cells)
        self.assertIn("SEARCH_PARTIAL", verdict)
        self.assertNotIn("SEARCH_RECOVERS", verdict)

    def test_visible_leak_inconclusive(self):
        cells = self._pack(
            g1_sft=57.47,
            g1_grpo=45.07,
            see_sft=57.5,
            see_grpo=45.2,
            vis_see_grpo=90.0,
            far_see_grpo=92.0,
        )
        self.assertIn("INCONCLUSIVE", c1_search_verdict(cells))
        self.assertNotIn("SEARCH_RECOVERS", c1_search_verdict(cells))

    def test_truncated_blocks_recovers(self):
        cells = self._pack(
            g1_sft=57.47,
            g1_grpo=45.07,
            see_sft=80.0,
            see_grpo=90.0,
            far_see_grpo=92.0,
            far_see_truncated_frac=0.10,
        )
        verdict = c1_search_verdict(cells)
        self.assertIn("SEE_TRUNCATED", verdict)
        self.assertNotIn("SEARCH_RECOVERS", verdict)
        self.assertIn("SEARCH_PARTIAL", verdict)

    def test_low_n(self):
        self.assertLess(5, LOW_N_MIN)
        cells = self._pack(
            g1_sft=57.47,
            g1_grpo=45.07,
            see_sft=80.0,
            see_grpo=90.0,
            far_see_grpo=92.0,
            n_facing_miss=5,
        )
        verdict = c1_search_verdict(cells)
        self.assertIn("LOW_N", verdict)
        self.assertIn("INCONCLUSIVE", verdict)
        self.assertNotIn("SEARCH_RECOVERS", verdict)

    def test_g1_off_c1_is_inconclusive(self):
        cells = self._pack(
            g1_sft=20.0,
            g1_grpo=10.0,
            see_sft=20.0,
            see_grpo=10.0,
        )
        self.assertEqual(c1_search_verdict(cells), "INCONCLUSIVE")

    def test_empty_cells(self):
        self.assertEqual(c1_search_verdict([]), "INCONCLUSIVE")


class GrpoOnlyVerdictTest(unittest.TestCase):
    def test_partial_like_seed42(self):
        cells = []
        for seed in (42, 123, 456):
            cells.append(
                _cell(
                    METHOD_G1,
                    "GRPO",
                    seed,
                    45.0,
                    n_visible=50,
                    n_facing_miss=65,
                    n_far=135,
                    success_visible=95.88,
                    success_facing_miss=95.82,
                    success_far=4.61,
                    far_see_truncated_frac=0.0,
                )
            )
            cells.append(
                _cell(
                    METHOD_SEE,
                    "GRPO",
                    seed,
                    90.0,
                    n_visible=50,
                    n_facing_miss=65,
                    n_far=135,
                    success_visible=100.0,
                    success_facing_miss=97.0,
                    success_far=87.54,
                    far_see_truncated_frac=0.0,
                )
            )
        out = c1_search_verdict_grpo(cells)
        self.assertEqual(out["scientific"], "SEARCH_PARTIAL")
        self.assertEqual(out["machine"], "INCONCLUSIVE")

    def test_null_if_far_stuck(self):
        cells = []
        for seed in (42, 123, 456):
            cells.append(
                _cell(
                    METHOD_G1,
                    "GRPO",
                    seed,
                    45.0,
                    n_visible=50,
                    n_facing_miss=65,
                    n_far=135,
                    success_visible=96.0,
                    success_facing_miss=96.0,
                    success_far=5.0,
                    far_see_truncated_frac=0.0,
                )
            )
            cells.append(
                _cell(
                    METHOD_SEE,
                    "GRPO",
                    seed,
                    45.0,
                    n_visible=50,
                    n_facing_miss=65,
                    n_far=135,
                    success_visible=96.5,
                    success_facing_miss=96.2,
                    success_far=6.0,
                    far_see_truncated_frac=0.0,
                )
            )
        out = c1_search_verdict_grpo(cells)
        self.assertEqual(out["scientific"], "SEARCH_NULL")
        self.assertEqual(out["machine"], "OK")


if __name__ == "__main__":
    unittest.main()
