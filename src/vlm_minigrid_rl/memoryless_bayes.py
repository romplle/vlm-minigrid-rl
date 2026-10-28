"""CPU memoryless Bayes: spin vs search on empty FOV. C1 caps 12/40. No VLM."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from vlm_minigrid_rl.minigrid_utils import (
    apply_episode_horizon,
    create_minigrid_env,
    get_shortest_path_actions,
    reset_env_with_goal,
)
from vlm_minigrid_rl.ood_multiseed import mean_sd_sample
from vlm_minigrid_rl.training_utils import ACTION_TO_ID

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = ROOT / "thesis" / "data" / "memoryless_bayes.json"

C1_EMPTY_8_MAX_STEPS = 12
C1_EMPTY_16_MAX_STEPS = 40
C1_EMPTY_MAX_STEPS = {
    8: C1_EMPTY_8_MAX_STEPS,
    16: C1_EMPTY_16_MAX_STEPS,
}

POLICY_SPIN = "spin"
POLICY_SEARCH = "search"
ACTION_TURN = "turn"
ACTION_FORWARD = "forward"
VERDICT_FLIP = "BAYES_FLIP"
VERDICT_MIXED = "MIXED"

DIR_DELTA = ((1, 0), (0, 1), (-1, 0), (0, -1))
STRATA = ("visible", "facing_miss", "far")

NOTE = (
    "Visible-goal steps use omniscient BFS once vis_mask contains the goal "
    "(on Empty: turn and walk to the seen cell). Walls use the occupancy grid, "
    "which is present in the ego view. This is not a VLM and is not identified "
    "with GRPO weights. Caps are C1 12/40, not slim 16/38."
)


def c1_empty_max_steps(env_size: int) -> int:
    size = int(env_size)
    if size not in C1_EMPTY_MAX_STEPS:
        raise KeyError(f"No C1 empty max_steps for env_size={size}")
    return int(C1_EMPTY_MAX_STEPS[size])


def p_far_given_empty(n_facing_miss: int, n_far: int) -> float | None:
    empty = int(n_facing_miss) + int(n_far)
    if empty <= 0:
        return None
    return float(n_far) / float(empty)


def bayes_empty_action(*, spin_success: float, search_success: float) -> str:
    if float(spin_success) > float(search_success):
        return ACTION_TURN
    return ACTION_FORWARD


def maps_bayes_verdict(action_8: str, action_16: str) -> str:
    if action_8 == ACTION_TURN and action_16 == ACTION_FORWARD:
        return VERDICT_FLIP
    return VERDICT_MIXED


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


def front_is_blocked(env) -> bool:
    unwrapped = env.unwrapped
    x, y = (int(v) for v in unwrapped.agent_pos)
    direction = int(unwrapped.agent_dir)
    dx, dy = DIR_DELTA[direction]
    nx, ny = x + dx, y + dy
    width, height = unwrapped.grid.width, unwrapped.grid.height
    if not (0 <= nx < width and 0 <= ny < height):
        return True
    cell = unwrapped.grid.get(nx, ny)
    return cell is not None and cell.type == "wall"


def select_memoryless_action(env, policy: str) -> int:
    if goal_in_view(env):
        path = get_shortest_path_actions(env)
        if path:
            return int(path[0])
        return ACTION_TO_ID["left"]
    if policy == POLICY_SPIN:
        return ACTION_TO_ID["left"]
    if policy != POLICY_SEARCH:
        raise ValueError(f"Unknown policy {policy!r}")
    if front_is_blocked(env):
        return ACTION_TO_ID["left"]
    return ACTION_TO_ID["forward"]


def collect_prior(env_size: int, n: int, seed: int) -> dict[str, Any]:
    env = create_minigrid_env(env_size, max_steps=c1_empty_max_steps(env_size))
    counts: Counter[str] = Counter()
    for i in range(int(n)):
        reset_env_with_goal(env, int(seed) + i, goal_color="green")
        counts[start_stratum(env)] += 1
    env.close()
    total = sum(counts.values()) or 1
    n_miss = int(counts["facing_miss"])
    n_far = int(counts["far"])
    n_visible = int(counts["visible"])
    n_empty = n_miss + n_far
    p_far_empty = p_far_given_empty(n_miss, n_far)
    return {
        "env_size": int(env_size),
        "n": int(n),
        "seed": int(seed),
        "counts": {k: int(counts[k]) for k in STRATA},
        "rates": {k: float(counts[k]) / float(total) for k in STRATA},
        "n_empty": n_empty,
        "p_far_given_empty": p_far_empty,
        "not_far_rate": float(n_visible + n_miss) / float(total),
    }


def run_episode(env, policy: str, max_steps: int) -> tuple[bool, str]:
    stratum = start_stratum(env)
    for _ in range(int(max_steps)):
        action = select_memoryless_action(env, policy)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated:
            return True, stratum
        if truncated:
            return False, stratum
    return False, stratum


def evaluate_policy(
    env_size: int,
    policy: str,
    *,
    episodes: int,
    seed: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    steps = c1_empty_max_steps(env_size) if max_steps is None else int(max_steps)
    env = create_minigrid_env(env_size, max_steps=steps)
    apply_episode_horizon(env, steps)
    wins = 0
    by_stratum = {k: {"n": 0, "successes": 0} for k in STRATA}
    for i in range(int(episodes)):
        reset_env_with_goal(env, int(seed) + i, goal_color="green")
        success, stratum = run_episode(env, policy, steps)
        by_stratum[stratum]["n"] += 1
        if success:
            wins += 1
            by_stratum[stratum]["successes"] += 1
    env.close()
    n = int(episodes)
    stratum_success = {}
    for key, row in by_stratum.items():
        denom = row["n"] or 1
        stratum_success[key] = {
            "n": row["n"],
            "success_rate": 100.0 * row["successes"] / float(denom) if row["n"] else None,
        }
    return {
        "env_size": int(env_size),
        "policy": policy,
        "episodes": n,
        "seed": int(seed),
        "max_steps": steps,
        "successes": wins,
        "success_rate": 100.0 * wins / float(n),
        "by_stratum": stratum_success,
    }


def _mean_rate(cells: list[dict[str, Any]]) -> dict[str, float]:
    rates = [float(c["success_rate"]) for c in cells]
    stats = mean_sd_sample(rates)
    return {"mean": stats["mean"], "sd": stats["sd"], "n": stats["n"], "rates": rates}


def build_memoryless_report(
    *,
    priors: dict[str, dict[str, Any]],
    evals: dict[str, dict[str, list[dict[str, Any]]]],
    prior_n: int,
    prior_seed: int,
    eval_episodes: int,
    eval_seeds: list[int],
) -> dict[str, Any]:
    by_env: dict[str, Any] = {}
    actions: dict[str, str] = {}
    for env_key in ("8", "16"):
        spin_stats = _mean_rate(evals[env_key][POLICY_SPIN])
        search_stats = _mean_rate(evals[env_key][POLICY_SEARCH])
        action = bayes_empty_action(
            spin_success=spin_stats["mean"],
            search_success=search_stats["mean"],
        )
        actions[env_key] = action
        by_env[env_key] = {
            "max_steps": c1_empty_max_steps(int(env_key)),
            "prior": priors[env_key],
            "spin": spin_stats,
            "search": search_stats,
            "bayes_empty_action": action,
            "evals": {
                POLICY_SPIN: evals[env_key][POLICY_SPIN],
                POLICY_SEARCH: evals[env_key][POLICY_SEARCH],
            },
        }
    verdict = maps_bayes_verdict(actions["8"], actions["16"])
    return {
        "note": NOTE,
        "config": {
            "prior_n": int(prior_n),
            "prior_seed": int(prior_seed),
            "eval_episodes": int(eval_episodes),
            "eval_seeds": [int(s) for s in eval_seeds],
            "max_steps": dict(C1_EMPTY_MAX_STEPS),
            "goal_color": "green",
        },
        "by_env": by_env,
        "verdict": verdict,
        "reason": (
            f"empty-FOV Bayes: 8x8 {actions['8']}, 16x16 {actions['16']}"
        ),
    }


ProgressCallback = Callable[[str], None]


def run_audit(
    *,
    prior_n: int = 2000,
    prior_seed: int = 42,
    eval_episodes: int = 250,
    eval_seeds: tuple[int, ...] = (42, 123, 456),
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    priors: dict[str, dict[str, Any]] = {}
    for env_size in (8, 16):
        log(f"prior Empty {env_size}x{env_size} n={prior_n} seed={prior_seed}")
        priors[str(env_size)] = collect_prior(env_size, prior_n, prior_seed)

    evals: dict[str, dict[str, list[dict[str, Any]]]] = {
        "8": {POLICY_SPIN: [], POLICY_SEARCH: []},
        "16": {POLICY_SPIN: [], POLICY_SEARCH: []},
    }
    for env_size in (8, 16):
        for policy in (POLICY_SPIN, POLICY_SEARCH):
            for seed in eval_seeds:
                log(
                    f"eval Empty {env_size}x{env_size} policy={policy} "
                    f"seed={seed} episodes={eval_episodes} "
                    f"max_steps={c1_empty_max_steps(env_size)}"
                )
                cell = evaluate_policy(
                    env_size,
                    policy,
                    episodes=eval_episodes,
                    seed=seed,
                )
                evals[str(env_size)][policy].append(cell)
    return build_memoryless_report(
        priors=priors,
        evals=evals,
        prior_n=prior_n,
        prior_seed=prior_seed,
        eval_episodes=eval_episodes,
        eval_seeds=list(eval_seeds),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU memoryless Bayes audit (spin vs search, C1 caps 12/40)."
    )
    parser.add_argument("--prior-n", type=int, default=2000)
    parser.add_argument("--prior-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    report = run_audit(
        prior_n=args.prior_n,
        prior_seed=args.prior_seed,
        eval_episodes=args.episodes,
        eval_seeds=seeds,
        progress=print,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env8 = report["by_env"]["8"]
    env16 = report["by_env"]["16"]
    print(f"wrote {args.output}")
    print(f"caps 8={C1_EMPTY_MAX_STEPS[8]} 16={C1_EMPTY_MAX_STEPS[16]}")
    print(
        f"8x8 p_far_empty={env8['prior']['p_far_given_empty']} "
        f"spin={env8['spin']['mean']:.2f} search={env8['search']['mean']:.2f} "
        f"bayes={env8['bayes_empty_action']}"
    )
    print(
        f"16x16 p_far_empty={env16['prior']['p_far_given_empty']} "
        f"spin={env16['spin']['mean']:.2f} search={env16['search']['mean']:.2f} "
        f"bayes={env16['bayes_empty_action']}"
    )
    print(f"verdict={report['verdict']} {report['reason']}")


if __name__ == "__main__":
    cli_main()
