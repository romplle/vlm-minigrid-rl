"""C1 search-then-generate: G1 plus oracle walk-to-see on far starts.

Eval-only. Transfer 16x16 uses the C1 cap (40), not slim eval_max_steps (38).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from transformers import GenerationConfig

from .c1_geometry import (
    C1_EVAL_16X16,
    C1_GRPO_EPISODE,
    C1_GRPO_TRANSFER_MEAN,
    C1_SANITY_TOL_PP,
    C1_SFT_EPOCH,
    C1_SFT_TRANSFER_MEAN,
    C1_TRAIN_8X8,
    DEFAULT_GRPO_ADAPTER,
    DEFAULT_SFT_ADAPTER,
    LOW_N_MIN,
    STRATUM_FAR,
    episode_stratum,
)
from .minigrid_utils import (
    create_minigrid_env,
    empty_metrics,
    finalize_metrics,
    goal_in_agent_view,
    goal_visible_any_dir,
    print_metrics,
    reset_env_with_goal,
    shortest_path_to_see_goal,
)
from .model_utils import generate_action
from .training_utils import ID_TO_ACTION, set_global_seed

EXPERIMENT_ID = "2026-08-c1-search-8x8"

METHOD_G1 = "G1"
METHOD_SEE = "SEE"
METHODS = (METHOD_G1, METHOD_SEE)

SEE_TRUNCATED_FRAC = 0.05


@dataclass(frozen=True)
class CellResult:
    method: str
    policy: str
    seed: int
    success_rate: float
    n_visible: int | None = None
    n_facing_miss: int | None = None
    n_far: int | None = None
    n_empty: int | None = None
    success_visible: float | None = None
    success_facing_miss: float | None = None
    success_far: float | None = None
    success_empty: float | None = None
    far_see_truncated_frac: float | None = None
    inconclusive: bool = False


def c1s_cell_path(run_dir: Path, method: str, policy: str, seed: int) -> Path:
    return run_dir / f"c1s_{method}_{policy}_seed{seed}.json"


def far_see_truncated_frac(episode_rows: list[dict]) -> float | None:
    far = [row for row in episode_rows if episode_stratum(row) == STRATUM_FAR]
    if not far:
        return None
    return sum(1 for row in far if row.get("see_truncated")) / len(far)


def see_oracle_stats(episode_rows: list[dict]) -> dict:
    far = [row for row in episode_rows if episode_stratum(row) == STRATUM_FAR]
    n_far = len(far)
    if not n_far:
        return {
            "n_far": 0,
            "mean_n_see_steps": None,
            "see_completed_frac": None,
            "see_failed_frac": None,
            "see_truncated_frac": None,
        }
    steps = [int(row.get("n_see_steps") or 0) for row in far]
    remaining = [
        row.get("remaining_after_see")
        for row in far
        if row.get("see_completed") and row.get("remaining_after_see") is not None
    ]
    return {
        "n_far": n_far,
        "mean_n_see_steps": sum(steps) / n_far,
        "see_completed_frac": sum(1 for row in far if row.get("see_completed")) / n_far,
        "see_failed_frac": sum(1 for row in far if row.get("see_failed")) / n_far,
        "see_truncated_frac": sum(1 for row in far if row.get("see_truncated")) / n_far,
        "mean_remaining_after_see": (
            None if not remaining else sum(float(v) for v in remaining) / len(remaining)
        ),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _cell_map(cells: list[CellResult]) -> dict[tuple[str, str, int], CellResult]:
    return {(cell.method, cell.policy, cell.seed): cell for cell in cells if not cell.inconclusive}


def _seeds(cells: list[CellResult]) -> list[int]:
    return sorted({cell.seed for cell in cells})


def _complete(cells: list[CellResult], method: str) -> bool:
    present = {
        (cell.policy, cell.seed)
        for cell in cells
        if cell.method == method and not cell.inconclusive
    }
    seeds = _seeds(cells)
    if not seeds:
        return False
    return all((policy, seed) in present for policy in ("SFT", "GRPO") for seed in seeds)


def _mean_attr(cells: list[CellResult], method: str, policy: str, attr: str) -> float | None:
    values = [
        getattr(cell, attr)
        for cell in cells
        if cell.method == method and cell.policy == policy and not cell.inconclusive
    ]
    numeric = [float(value) for value in values if value is not None]
    return _mean(numeric)


def _paired_see_minus_g1(cells: list[CellResult], policy: str, attr: str) -> float | None:
    by_key = _cell_map(cells)
    deltas: list[float] = []
    for seed in _seeds(cells):
        g1 = by_key.get((METHOD_G1, policy, seed))
        see = by_key.get((METHOD_SEE, policy, seed))
        if g1 is None or see is None:
            return None
        left = getattr(g1, attr)
        right = getattr(see, attr)
        if left is None or right is None:
            return None
        deltas.append(float(right) - float(left))
    return _mean(deltas)


def _low_n(cells: list[CellResult]) -> bool:
    for cell in cells:
        for attr in ("n_visible", "n_facing_miss", "n_far"):
            value = getattr(cell, attr)
            if value is None:
                return True
            if int(value) < LOW_N_MIN:
                return True
    return False


def _g1_sanity_ok(cells: list[CellResult]) -> bool:
    sft = _mean_attr(cells, METHOD_G1, "SFT", "success_rate")
    grpo = _mean_attr(cells, METHOD_G1, "GRPO", "success_rate")
    if sft is None or grpo is None:
        return False
    return (
        abs(sft - C1_SFT_TRANSFER_MEAN) <= C1_SANITY_TOL_PP
        and abs(grpo - C1_GRPO_TRANSFER_MEAN) <= C1_SANITY_TOL_PP
    )


def _leak_or_missing(delta: float | None) -> bool:
    return delta is None or abs(delta) >= 3.0


def _see_truncated(cells: list[CellResult]) -> bool:
    fracs = [
        float(cell.far_see_truncated_frac)
        for cell in cells
        if cell.method == METHOD_SEE
        and not cell.inconclusive
        and cell.far_see_truncated_frac is not None
    ]
    if not fracs:
        return False
    return _mean(fracs) is not None and (_mean(fracs) or 0.0) >= SEE_TRUNCATED_FRAC


def c1_search_verdict(cells: list[CellResult]) -> str:
    if not cells:
        return "INCONCLUSIVE"
    if not _complete(cells, METHOD_G1) or not _complete(cells, METHOD_SEE):
        return "INCONCLUSIVE"
    if not _g1_sanity_ok(cells):
        return "INCONCLUSIVE"

    tags: list[str] = []
    if _low_n(cells):
        tags.append("LOW_N")
        tags.append("INCONCLUSIVE")
        return " | ".join(tags)

    vis_sft = _paired_see_minus_g1(cells, "SFT", "success_visible")
    vis_grpo = _paired_see_minus_g1(cells, "GRPO", "success_visible")
    face_sft = _paired_see_minus_g1(cells, "SFT", "success_facing_miss")
    face_grpo = _paired_see_minus_g1(cells, "GRPO", "success_facing_miss")
    if (
        _leak_or_missing(vis_sft)
        or _leak_or_missing(vis_grpo)
        or _leak_or_missing(face_sft)
        or _leak_or_missing(face_grpo)
    ):
        tags.append("INCONCLUSIVE")
        return " | ".join(tags)

    truncated = _see_truncated(cells)
    if truncated:
        tags.append("SEE_TRUNCATED")

    delta_far = _paired_see_minus_g1(cells, "GRPO", "success_far")
    see_far_grpo = _mean_attr(cells, METHOD_SEE, "GRPO", "success_far")
    g1_vis_grpo = _mean_attr(cells, METHOD_G1, "GRPO", "success_visible")

    search = "SEARCH_MIXED"
    if delta_far is None or see_far_grpo is None or g1_vis_grpo is None:
        search = "SEARCH_MIXED"
    elif (
        not truncated
        and delta_far >= 10.0
        and see_far_grpo >= g1_vis_grpo - 5.0
    ):
        search = "SEARCH_RECOVERS"
    elif delta_far >= 5.0:
        search = "SEARCH_PARTIAL"
    elif abs(delta_far) < 3.0:
        search = "SEARCH_NULL"
    tags.append(search)
    return " | ".join(tags)


def _complete_grpo(cells: list[CellResult], method: str) -> bool:
    seeds = _seeds(cells)
    if not seeds:
        return False
    present = {
        cell.seed
        for cell in cells
        if cell.method == method and cell.policy == "GRPO" and not cell.inconclusive
    }
    return all(seed in present for seed in seeds)


def c1_search_verdict_grpo(cells: list[CellResult]) -> dict[str, str]:
    """GRPO-only. Returns machine_tag, scientific_tag (SEARCH_*)."""
    if not _complete_grpo(cells, METHOD_G1) or not _complete_grpo(cells, METHOD_SEE):
        return {"machine": "INCONCLUSIVE", "scientific": "INCONCLUSIVE"}
    vis = _paired_see_minus_g1(cells, "GRPO", "success_visible")
    face = _paired_see_minus_g1(cells, "GRPO", "success_facing_miss")
    machine = "INCONCLUSIVE" if (_leak_or_missing(vis) or _leak_or_missing(face)) else "OK"
    truncated = _see_truncated([cell for cell in cells if cell.policy == "GRPO"])
    delta_far = _paired_see_minus_g1(cells, "GRPO", "success_far")
    see_far = _mean_attr(cells, METHOD_SEE, "GRPO", "success_far")
    g1_vis = _mean_attr(cells, METHOD_G1, "GRPO", "success_visible")
    if delta_far is None or see_far is None or g1_vis is None:
        scientific = "SEARCH_MIXED"
    elif (not truncated) and delta_far >= 10.0 and see_far >= g1_vis - 5.0:
        scientific = "SEARCH_RECOVERS"
    elif delta_far >= 5.0:
        scientific = "SEARCH_PARTIAL"
    elif abs(delta_far) < 3.0:
        scientific = "SEARCH_NULL"
    else:
        scientific = "SEARCH_MIXED"
    return {"machine": machine, "scientific": scientific}


def _pose_key(env) -> tuple[tuple[int, int], int]:
    unwrapped = getattr(env, "unwrapped", env)
    pos = tuple(int(v) for v in unwrapped.agent_pos)
    return pos, int(unwrapped.agent_dir)


def evaluate_search_in_env(
    model,
    tokenizer,
    image_processor,
    prompt,
    *,
    method: str,
    env_size: int,
    tile_size: int,
    max_steps: int,
    seed: int,
    device: str,
    model_name: str,
    episodes: int,
    goal_color: str,
) -> tuple[dict, list[dict]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")

    set_global_seed(seed)
    env = create_minigrid_env(env_size, tile_size=tile_size, max_steps=max_steps)
    model.eval()
    model.generation_config = GenerationConfig()
    metrics = empty_metrics()
    episode_rows: list[dict] = []

    print(f"\n[{model_name}] Запуск симуляции ({episodes} эпизодов)...")

    for episode in tqdm(range(episodes), desc=f"Testing {model_name}"):
        obs = reset_env_with_goal(env, seed + episode, goal_color=goal_color)
        start_visible = goal_in_agent_view(env)
        start_any = goal_visible_any_dir(env)
        stratum = episode_stratum(
            {
                "start_goal_visible": start_visible,
                "start_visible_any_dir": start_any,
            }
        )
        start_pos = tuple(int(v) for v in env.unwrapped.agent_pos)
        start_dir = int(env.unwrapped.agent_dir)
        episode_reward = 0.0
        success = False
        timeout = False
        invalid = False
        n_see_steps = 0
        see_failed = False
        see_truncated = False
        see_path: list[int] = []
        need_oracle = method == METHOD_SEE and stratum == STRATUM_FAR
        if need_oracle:
            see_path = shortest_path_to_see_goal(env)
            if not see_path:
                see_failed = True
        see_completed = not need_oracle
        path_idx = 0
        actions: list[str] = []
        seen_poses = {_pose_key(env)}
        pose_loop = False
        first_visible_step = 0 if start_visible else None
        still_empty_after_4 = not start_visible
        pos_unchanged_first8 = True

        for step in range(max_steps):
            currently_visible = goal_in_agent_view(env)
            in_oracle = (
                need_oracle
                and not see_failed
                and not see_completed
                and path_idx < len(see_path)
                and not currently_visible
            )
            if in_oracle:
                action_idx = see_path[path_idx]
                action_name = ID_TO_ACTION[action_idx]
                path_idx += 1
                n_see_steps += 1
            else:
                action_name, action_idx, _ = generate_action(
                    model,
                    tokenizer,
                    image_processor,
                    obs["image"],
                    prompt,
                    device,
                )
                if action_name is None or action_idx is None:
                    metrics["invalid_action_episodes"] += 1
                    invalid = True
                    break

            obs, reward, terminated, truncated, _ = env.step(action_idx)
            episode_reward += reward
            metrics["actions"][action_name] += 1
            actions.append(action_name)
            step_index = step + 1

            if tuple(int(v) for v in env.unwrapped.agent_pos) != start_pos:
                if len(actions) <= 8:
                    pos_unchanged_first8 = False

            pose = _pose_key(env)
            if pose in seen_poses:
                pose_loop = True
            seen_poses.add(pose)

            now_visible = goal_in_agent_view(env)
            if first_visible_step is None and now_visible:
                first_visible_step = step_index
            if step_index == 4:
                still_empty_after_4 = not now_visible
            if need_oracle and now_visible:
                see_completed = True

            if terminated:
                metrics["successes"] += 1
                metrics["total_reward"] += episode_reward
                metrics["total_steps_in_success"] += step_index
                success = True
                break
            if truncated or step == max_steps - 1:
                metrics["timeouts"] += 1
                timeout = True
                still_need_oracle = (
                    need_oracle
                    and not see_failed
                    and not see_completed
                    and path_idx < len(see_path)
                    and not now_visible
                )
                if in_oracle or still_need_oracle:
                    see_truncated = True
                break

        remaining_after_see = None
        if see_completed and need_oracle:
            remaining_after_see = max_steps - n_see_steps
        elif see_completed and not need_oracle:
            remaining_after_see = max_steps

        first4 = actions[:4]
        no_search_turn = (
            len(first4) == 4
            and all(name == "forward" for name in first4)
            and still_empty_after_4
        )
        first8 = actions[:8]
        spin_in_place = (
            bool(first8)
            and all(name in ("left", "right") for name in first8)
            and pos_unchanged_first8
        )

        episode_rows.append(
            {
                "success": success,
                "timeout": timeout,
                "invalid": invalid,
                "start_goal_visible": start_visible,
                "start_visible_any_dir": start_any,
                "stratum": stratum,
                "n_see_steps": n_see_steps,
                "see_completed": see_completed,
                "see_failed": see_failed,
                "see_truncated": see_truncated,
                "remaining_after_see": remaining_after_see,
                "first_visible_step": first_visible_step,
                "pose_loop": pose_loop,
                "actions": actions,
                "start_pos": list(start_pos),
                "start_dir": start_dir,
                "pos_unchanged_first8": spin_in_place,
                "no_search_turn": no_search_turn,
            }
        )

    env.close()
    result = finalize_metrics(metrics, episodes)
    print_metrics(model_name, result)
    return result, episode_rows
