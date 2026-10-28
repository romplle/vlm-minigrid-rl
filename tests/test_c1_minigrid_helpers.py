"""CPU tests for vis_mask / BFS-to-see / recolor_goal. No VLM."""

from __future__ import annotations

import unittest

from vlm_minigrid_rl.minigrid_utils import (
    create_minigrid_env,
    dirs_that_see_goal,
    goal_in_agent_view,
    goal_in_agent_view_at,
    goal_visible_any_dir,
    recolor_goal,
    reset_env_with_goal,
    shortest_path_to_see_goal,
)
from vlm_minigrid_rl.training_utils import ACTION_TO_ID


def _find_goal(env) -> tuple[int, int]:
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    raise AssertionError("no goal")


class RecolorGoalTest(unittest.TestCase):
    def test_paint_changes_only_goal_color(self):
        env = create_minigrid_env(8, max_steps=12)
        reset_env_with_goal(env, 0, goal_color="red")
        gx, gy = _find_goal(env)
        recolor_goal(env, "green")
        cell = env.unwrapped.grid.get(gx, gy)
        self.assertEqual(cell.type, "goal")
        self.assertEqual(cell.color, "green")
        env.close()


class ShortestPathToSeeTest(unittest.TestCase):
    def test_empty_when_already_visible(self):
        env = create_minigrid_env(8, max_steps=12)
        reset_env_with_goal(env, 0, goal_color="green")
        if goal_in_agent_view(env):
            self.assertEqual(shortest_path_to_see_goal(env), [])
        env.close()

    def test_path_makes_goal_visible_and_is_not_touch_bfs(self):
        env = create_minigrid_env(16, max_steps=40)
        found = False
        for offset in range(400):
            reset_env_with_goal(env, 42 + offset, goal_color="green")
            if goal_in_agent_view(env) or goal_visible_any_dir(env):
                continue
            path = shortest_path_to_see_goal(env)
            self.assertTrue(path)
            unwrapped = env.unwrapped
            pos = list(unwrapped.agent_pos)
            direction = int(unwrapped.agent_dir)
            dir_delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for action in path:
                if action == ACTION_TO_ID["left"]:
                    direction = (direction - 1) % 4
                elif action == ACTION_TO_ID["right"]:
                    direction = (direction + 1) % 4
                else:
                    dx, dy = dir_delta[direction]
                    pos[0] += dx
                    pos[1] += dy
            self.assertTrue(goal_in_agent_view_at(env, pos, direction))
            found = True
            break
        env.close()
        self.assertTrue(found)

    def test_dirs_restore_agent_dir(self):
        env = create_minigrid_env(8, max_steps=12)
        reset_env_with_goal(env, 1, goal_color="green")
        before = int(env.unwrapped.agent_dir)
        dirs_that_see_goal(env)
        self.assertEqual(int(env.unwrapped.agent_dir), before)
        env.close()


if __name__ == "__main__":
    unittest.main()
