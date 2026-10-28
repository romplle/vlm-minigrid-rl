"""C1 goal-red PAINT: same red layouts, recolor goal to green, prompt stays red.

Eval-only. Empty 8x8 uses the C1 cap (12), not slim eval_max_steps (16).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from transformers import GenerationConfig

from .minigrid_utils import (
    create_minigrid_env,
    empty_metrics,
    finalize_metrics,
    goal_in_agent_view,
    goal_visible_any_dir,
    print_metrics,
    recolor_goal,
    reset_env_with_goal,
)
from .model_utils import generate_action
from .training_utils import set_global_seed

C1_EVAL_8X8 = 12
C1_TRAIN_8X8 = 12
EXPERIMENT_ID = "2026-08-c1-goalred-8x8"
C1_SFT_EPOCH = 9
C1_GRPO_EPISODE = 100
DEFAULT_SFT_ADAPTER = "checkpoints/sft_adapter_8x8_bs32_e10/epoch-9"
DEFAULT_GRPO_ADAPTER = "checkpoints/grpo_adapter_8x8_handoff_from_e10_e9/episode-100"

METHOD_G1 = "G1"
METHOD_PAINT = "PAINT"
METHODS = (METHOD_G1, METHOD_PAINT)

C1_SFT_GOALRED_MEAN = 17.87
C1_GRPO_GOALRED_MEAN = 2.40
C1_SANITY_TOL_PP = 3.0
IN_DOMAIN_GRPO_AT_12 = 98.53
COLOR_RECOVERS_MIN = IN_DOMAIN_GRPO_AT_12 - 5.0
IN_DOMAIN_GRPO_BY_TRAIN_SEED = {42: 98.53, 123: 98.13, 456: 98.13}


@dataclass(frozen=True)
class CellResult:
    method: str
    policy: str
    seed: int
    success_rate: float
    n_visible: int | None = None
    inconclusive: bool = False


def c1r_cell_path(run_dir: Path, method: str, policy: str, seed: int) -> Path:
    return run_dir / f"c1r_{method}_{policy}_seed{seed}.json"


def prepare_goalred_obs(env, method: str, seed: int):
    """Reset as C1 red/red. PAINT recolors the goal to green; G1 does not."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    obs = reset_env_with_goal(env, seed, goal_color="red")
    painted = False
    if method == METHOD_PAINT:
        obs = recolor_goal(env, "green")
        painted = True
    return obs, painted


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


def _paired_paint_minus_g1(cells: list[CellResult], policy: str) -> float | None:
    by_key = _cell_map(cells)
    deltas: list[float] = []
    for seed in _seeds(cells):
        g1 = by_key.get((METHOD_G1, policy, seed))
        paint = by_key.get((METHOD_PAINT, policy, seed))
        if g1 is None or paint is None:
            return None
        deltas.append(float(paint.success_rate) - float(g1.success_rate))
    return _mean(deltas)


def _g1_sanity_ok(cells: list[CellResult]) -> bool:
    sft = _mean_attr(cells, METHOD_G1, "SFT", "success_rate")
    grpo = _mean_attr(cells, METHOD_G1, "GRPO", "success_rate")
    if sft is None or grpo is None:
        return False
    return (
        abs(sft - C1_SFT_GOALRED_MEAN) <= C1_SANITY_TOL_PP
        and abs(grpo - C1_GRPO_GOALRED_MEAN) <= C1_SANITY_TOL_PP
    )


def c1_goalred_verdict(cells: list[CellResult]) -> str:
    if not cells:
        return "INCONCLUSIVE"
    if not _complete(cells, METHOD_G1) or not _complete(cells, METHOD_PAINT):
        return "INCONCLUSIVE"
    if not _g1_sanity_ok(cells):
        return "INCONCLUSIVE"

    delta = _paired_paint_minus_g1(cells, "GRPO")
    paint_grpo = _mean_attr(cells, METHOD_PAINT, "GRPO", "success_rate")
    if delta is None or paint_grpo is None:
        return "INCONCLUSIVE"
    if delta >= 10.0 and paint_grpo >= COLOR_RECOVERS_MIN:
        return "COLOR_RECOVERS"
    if delta >= 5.0:
        return "COLOR_PARTIAL"
    if abs(delta) < 3.0:
        return "COLOR_NULL"
    return "COLOR_MIXED"


def color_recovers_min(train_seed: int) -> float:
    return IN_DOMAIN_GRPO_BY_TRAIN_SEED[int(train_seed)] - 5.0


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


def c1_goalred_verdict_grpo(cells: list[CellResult], *, train_seed: int) -> str:
    if not cells:
        return "INCONCLUSIVE"
    if not _complete_grpo(cells, METHOD_G1) or not _complete_grpo(cells, METHOD_PAINT):
        return "INCONCLUSIVE"
    delta = _paired_paint_minus_g1(cells, "GRPO")
    paint_grpo = _mean_attr(cells, METHOD_PAINT, "GRPO", "success_rate")
    if delta is None or paint_grpo is None:
        return "INCONCLUSIVE"
    if delta >= 10.0 and paint_grpo >= color_recovers_min(train_seed):
        return "COLOR_RECOVERS"
    if delta >= 5.0:
        return "COLOR_PARTIAL"
    if abs(delta) < 3.0:
        return "COLOR_NULL"
    return "COLOR_MIXED"


def evaluate_goalred_in_env(
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
        obs, painted = prepare_goalred_obs(env, method, seed + episode)
        start_goal_visible = goal_in_agent_view(env)
        start_visible_any_dir = goal_visible_any_dir(env)
        episode_reward = 0.0
        success = False
        timeout = False
        invalid = False

        for step in range(max_steps):
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

            if terminated:
                metrics["successes"] += 1
                metrics["total_reward"] += episode_reward
                metrics["total_steps_in_success"] += step + 1
                success = True
                break
            if truncated or step == max_steps - 1:
                metrics["timeouts"] += 1
                timeout = True
                break

        episode_rows.append(
            {
                "success": success,
                "timeout": timeout,
                "invalid": invalid,
                "painted": painted,
                "start_goal_visible": start_goal_visible,
                "start_visible_any_dir": start_visible_any_dir,
            }
        )

    env.close()
    result = finalize_metrics(metrics, episodes)
    print_metrics(model_name, result)
    return result, episode_rows
