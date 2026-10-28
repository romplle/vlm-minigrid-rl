"""C1 goal-red PAINT: recolor, prepare_obs, verdicts. No VLM."""

from __future__ import annotations

import unittest

from vlm_minigrid_rl.c1_goalred import (
    C1_EVAL_8X8,
    COLOR_RECOVERS_MIN,
    METHOD_G1,
    METHOD_PAINT,
    CellResult,
    c1_goalred_verdict,
    c1_goalred_verdict_grpo,
    color_recovers_min,
    prepare_goalred_obs,
)
from vlm_minigrid_rl.env_profiles import EMPTY_8X8
from vlm_minigrid_rl.minigrid_utils import (
    create_minigrid_env,
    goal_in_agent_view,
    recolor_goal,
    reset_env_with_goal,
)


def _find_goal(env) -> tuple[int, int]:
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    raise AssertionError("no goal on grid")


def _goal_color(env) -> str:
    gx, gy = _find_goal(env)
    cell = env.unwrapped.grid.get(gx, gy)
    return str(cell.color)


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


def _cell(method: str, policy: str, seed: int, success_rate: float) -> CellResult:
    return CellResult(method=method, policy=policy, seed=seed, success_rate=success_rate)


def _triple(method: str, policy: str, success_rate: float) -> list[CellResult]:
    return [_cell(method, policy, seed, success_rate) for seed in (42, 123, 456)]


class HorizonConstantTest(unittest.TestCase):
    def test_c1_8x8_is_not_slim_eval(self):
        self.assertEqual(C1_EVAL_8X8, 12)
        self.assertEqual(EMPTY_8X8.eval_max_steps, 16)
        self.assertNotEqual(C1_EVAL_8X8, EMPTY_8X8.eval_max_steps)
        self.assertGreater(COLOR_RECOVERS_MIN, 90.0)

    def test_c1_goalred_is_not_slim_eval(self):
        self.assertEqual(C1_EVAL_8X8, 12)
        self.assertEqual(EMPTY_8X8.eval_max_steps, 16)


class RecolorTest(unittest.TestCase):
    def test_recolor_green_keeps_pose_and_one_goal(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            reset_env_with_goal(env, seed=0, goal_color="red")
            self.assertEqual(_goal_color(env), "red")
            pos = tuple(int(v) for v in env.unwrapped.agent_pos)
            direction = int(env.unwrapped.agent_dir)
            goal_before = _find_goal(env)
            saw = goal_in_agent_view(env)
            recolor_goal(env, "green")
            self.assertEqual(_find_goal(env), goal_before)
            self.assertEqual(_goal_color(env), "green")
            self.assertEqual(tuple(int(v) for v in env.unwrapped.agent_pos), pos)
            self.assertEqual(int(env.unwrapped.agent_dir), direction)
            self.assertEqual(goal_in_agent_view(env), saw)
        finally:
            env.close()

    def test_recolor_visible_stays_visible(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            reset_env_with_goal(env, seed=0, goal_color="red")
            goal = _find_goal(env)
            pos, facing = _pose_facing_goal(goal, env_size=8)
            env.unwrapped.agent_pos = pos
            env.unwrapped.agent_dir = facing
            self.assertTrue(goal_in_agent_view(env))
            recolor_goal(env, "green")
            self.assertTrue(goal_in_agent_view(env))
            self.assertEqual(_goal_color(env), "green")
        finally:
            env.close()

    def test_prepare_g1_does_not_recolor(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            prepare_goalred_obs(env, METHOD_G1, seed=3)
            self.assertEqual(_goal_color(env), "red")
        finally:
            env.close()

    def test_prepare_paint_recolors_green(self):
        env = create_minigrid_env(8, max_steps=12)
        try:
            _, painted = prepare_goalred_obs(env, METHOD_PAINT, seed=3)
            self.assertTrue(painted)
            self.assertEqual(_goal_color(env), "green")
        finally:
            env.close()


class VerdictTest(unittest.TestCase):
    def _pack(self, g1_sft: float, g1_grpo: float, paint_sft: float, paint_grpo: float):
        cells = []
        cells.extend(_triple(METHOD_G1, "SFT", g1_sft))
        cells.extend(_triple(METHOD_G1, "GRPO", g1_grpo))
        cells.extend(_triple(METHOD_PAINT, "SFT", paint_sft))
        cells.extend(_triple(METHOD_PAINT, "GRPO", paint_grpo))
        return cells

    def test_color_null(self):
        cells = self._pack(17.87, 2.40, 18.0, 3.0)
        self.assertEqual(c1_goalred_verdict(cells), "COLOR_NULL")

    def test_color_recovers(self):
        cells = self._pack(17.87, 2.40, 98.0, 97.0)
        self.assertEqual(c1_goalred_verdict(cells), "COLOR_RECOVERS")

    def test_color_partial(self):
        cells = self._pack(17.87, 2.40, 40.0, 25.0)
        self.assertEqual(c1_goalred_verdict(cells), "COLOR_PARTIAL")

    def test_g1_off_c1_is_inconclusive(self):
        cells = self._pack(50.0, 40.0, 90.0, 90.0)
        self.assertEqual(c1_goalred_verdict(cells), "INCONCLUSIVE")

    def test_empty_cells(self):
        self.assertEqual(c1_goalred_verdict([]), "INCONCLUSIVE")


class PerSeedFloorTest(unittest.TestCase):
    def test_floors(self):
        self.assertAlmostEqual(color_recovers_min(42), 93.53)
        self.assertAlmostEqual(color_recovers_min(123), 93.13)
        self.assertAlmostEqual(color_recovers_min(456), 93.13)


class GrpoVerdictTest(unittest.TestCase):
    def test_recovers_like_seed42(self):
        cells = []
        for seed in (42, 123, 456):
            cells.append(CellResult(METHOD_G1, "GRPO", seed, 2.40))
            cells.append(CellResult(METHOD_PAINT, "GRPO", seed, 97.73))
        self.assertEqual(c1_goalred_verdict_grpo(cells, train_seed=123), "COLOR_RECOVERS")

    def test_null_if_paint_does_nothing(self):
        cells = []
        for seed in (42, 123, 456):
            cells.append(CellResult(METHOD_G1, "GRPO", seed, 4.27))
            cells.append(CellResult(METHOD_PAINT, "GRPO", seed, 5.00))
        self.assertEqual(c1_goalred_verdict_grpo(cells, train_seed=123), "COLOR_NULL")


if __name__ == "__main__":
    unittest.main()
