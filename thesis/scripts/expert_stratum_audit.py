"""CPU audit: Empty expert starts and empty-FOV actions by vis_mask stratum.

Does not load a VLM. Replays the same reset+BFS recipe as dataset_generation.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.core.world_object import Goal
from minigrid.wrappers import RGBImgPartialObsWrapper

ROOT = Path(__file__).resolve().parents[2]

LEFT, RIGHT, FORWARD = 0, 1, 2
DIR_DELTA = ((1, 0), (0, 1), (-1, 0), (0, -1))


def create_env(env_size: int):
    env = gym.make(f"MiniGrid-Empty-{env_size}x{env_size}-v0", render_mode="rgb_array")
    return RGBImgPartialObsWrapper(env, tile_size=32)


def reset_with_goal(env, seed: int):
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped.place_agent()
    for x in range(unwrapped.grid.width):
        for y in range(unwrapped.grid.height):
            cell = unwrapped.grid.get(x, y)
            if cell and cell.type == "goal":
                unwrapped.grid.set(x, y, None)
    unwrapped.place_obj(Goal("green"))
    return env.observation(unwrapped.gen_obs())


def goal_in_view(env) -> bool:
    unwrapped = env.unwrapped
    grid, vis_mask = unwrapped.gen_obs_grid()
    view_size = unwrapped.agent_view_size
    agent_cell = (view_size // 2, view_size - 1)
    for i in range(grid.width):
        for j in range(grid.height):
            if not vis_mask[i, j] or (i, j) == agent_cell:
                continue
            cell = grid.get(i, j)
            if cell is not None and cell.type == "goal":
                return True
    return False


def dirs_that_see_goal(env) -> list[int]:
    unwrapped = env.unwrapped
    original = int(unwrapped.agent_dir)
    seeing: list[int] = []
    try:
        for direction in range(4):
            unwrapped.agent_dir = direction
            if goal_in_view(env):
                seeing.append(direction)
    finally:
        unwrapped.agent_dir = original
    return seeing


def start_stratum(env) -> str:
    if goal_in_view(env):
        return "visible"
    if dirs_that_see_goal(env):
        return "facing_miss"
    return "far"


def pose_stratum(env) -> str:
    return start_stratum(env)


def shortest_path(env, action_order=(LEFT, RIGHT, FORWARD)):
    grid = env.unwrapped.grid
    agent_pos = tuple(int(v) for v in env.unwrapped.agent_pos)
    agent_dir = int(env.unwrapped.agent_dir)
    goal_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
                break
        if goal_pos:
            break
    if not goal_pos:
        return []
    start = (agent_pos[0], agent_pos[1], agent_dir)
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (x, y, direction), path = queue.popleft()
        if (x, y) == goal_pos:
            return path
        for action in action_order:
            nx, ny, nd = x, y, direction
            if action == LEFT:
                nd = (direction - 1) % 4
            elif action == RIGHT:
                nd = (direction + 1) % 4
            else:
                dx, dy = DIR_DELTA[direction]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue
                cell = grid.get(nx, ny)
                if cell is not None and cell.type == "wall":
                    continue
            state = (nx, ny, nd)
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [action]))
    return []


def turn_balance(actions):
    return actions.count(LEFT) - actions.count(RIGHT)


def choose_path(env, balance: int):
    candidates = [
        shortest_path(env, (LEFT, RIGHT, FORWARD)),
        shortest_path(env, (RIGHT, LEFT, FORWARD)),
    ]
    candidates = [c for c in candidates if c]
    if not candidates:
        return []
    return min(candidates, key=lambda c: (abs(balance + turn_balance(c)), turn_balance(c)))


def audit(env_size: int, n_episodes: int, seed_base: int) -> dict:
    env = create_env(env_size)
    start_counts: Counter[str] = Counter()
    empty_start_actions: dict[str, Counter[str]] = defaultdict(Counter)
    empty_step_actions: dict[str, Counter[str]] = defaultdict(Counter)
    n_steps = 0
    n_empty_steps = 0
    skipped = 0
    balance = 0
    action_name = {LEFT: "left", RIGHT: "right", FORWARD: "forward"}

    for episode in range(n_episodes):
        seed = seed_base + episode
        reset_with_goal(env, seed)
        path = choose_path(env, balance)
        if not path:
            skipped += 1
            continue
        balance += turn_balance(path)
        stratum0 = start_stratum(env)
        start_counts[stratum0] += 1
        if stratum0 != "visible":
            empty_start_actions[stratum0][action_name[path[0]]] += 1

        for action in path:
            n_steps += 1
            pose = pose_stratum(env)
            if pose != "visible":
                n_empty_steps += 1
                empty_step_actions[pose][action_name[action]] += 1
            env.step(action)

    env.close()
    n_ok = sum(start_counts.values())
    n_empty_starts = start_counts["facing_miss"] + start_counts["far"]

    def share(counter: Counter) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {k: round(counter[k] / total, 4) for k in ("left", "right", "forward")}

    return {
        "env_size": env_size,
        "n_episodes": n_episodes,
        "seed_base": seed_base,
        "skipped": skipped,
        "n_ok": n_ok,
        "n_steps": n_steps,
        "n_empty_steps": n_empty_steps,
        "start_counts": dict(start_counts),
        "start_rates": {k: round(v / n_ok, 4) for k, v in start_counts.items()},
        "p_far_given_empty_start": round(start_counts["far"] / n_empty_starts, 4)
        if n_empty_starts
        else None,
        "expert_first_action_on_empty_start": {
            k: {"n": sum(empty_start_actions[k].values()), "share": share(empty_start_actions[k])}
            for k in ("facing_miss", "far")
        },
        "expert_actions_on_empty_fov_steps": {
            k: {"n": sum(empty_step_actions[k].values()), "share": share(empty_step_actions[k])}
            for k in ("facing_miss", "far")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=42)
    args = parser.parse_args()
    out_dir = ROOT / "thesis" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "8": audit(8, args.episodes, args.seed_base),
        "16": audit(16, args.episodes, args.seed_base),
    }
    path = out_dir / "expert_stratum_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
