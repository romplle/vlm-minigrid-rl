"""CPU: are facing_miss and far ego-views visually separable on Empty?"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.core.world_object import Goal
from minigrid.wrappers import RGBImgPartialObsWrapper

ROOT = Path(__file__).resolve().parents[2]


def create_env(env_size: int):
    env = gym.make(f"MiniGrid-Empty-{env_size}x{env_size}-v0", render_mode="rgb_array")
    return RGBImgPartialObsWrapper(env, tile_size=32)


def reset_with_goal(env, seed: int):
    env.reset(seed=seed)
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


def dirs_that_see(env) -> list[int]:
    unwrapped = env.unwrapped
    original = int(unwrapped.agent_dir)
    seeing = []
    try:
        for d in range(4):
            unwrapped.agent_dir = d
            if goal_in_view(env):
                seeing.append(d)
    finally:
        unwrapped.agent_dir = original
    return seeing


def stratum(env) -> str:
    if goal_in_view(env):
        return "visible"
    if dirs_that_see(env):
        return "facing_miss"
    return "far"


def stats(img: np.ndarray) -> dict:
    arr = np.asarray(img, dtype=np.float32)
    # MiniGrid ego RGB; walls are grey-brown, floor light, goal saturated green
    green = (arr[..., 1] > 150) & (arr[..., 1] > arr[..., 0] + 40) & (arr[..., 1] > arr[..., 2] + 40)
    dark = arr.mean(axis=2) < 80
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "green_frac": float(green.mean()),
        "dark_frac": float(dark.mean()),
    }


def collect(env_size: int, n: int, seed0: int) -> dict:
    env = create_env(env_size)
    buckets = {"visible": [], "facing_miss": [], "far": []}
    for i in range(n):
        obs = reset_with_goal(env, seed0 + i)
        s = stratum(env)
        buckets[s].append(stats(obs["image"]))
    env.close()
    out = {}
    for k, rows in buckets.items():
        if not rows:
            out[k] = {"n": 0}
            continue
        out[k] = {
            "n": len(rows),
            "mean": float(np.mean([r["mean"] for r in rows])),
            "std": float(np.mean([r["std"] for r in rows])),
            "green_frac": float(np.mean([r["green_frac"] for r in rows])),
            "dark_frac": float(np.mean([r["dark_frac"] for r in rows])),
        }
    return out


def main():
    report = {
        "n_resets": 2000,
        "8": collect(8, 2000, 42),
        "16": collect(16, 2000, 42),
        "note": (
            "Simple RGB summaries of the 7x7 ego view. Not a trained classifier. "
            "If facing_miss and far differ on dark_frac/mean, observations are not identical."
        ),
    }
    path = ROOT / "thesis" / "data" / "ego_rgb_stratum_stats.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
