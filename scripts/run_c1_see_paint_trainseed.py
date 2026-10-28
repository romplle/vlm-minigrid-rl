"""Eval-only C1 SEE/PAINT on GRPO train seeds 123/456 (caps 12/40).

Does not train. Seed 42 is read from frozen C1 JSON. SFT is not re-evaled.
Adapters load from the primary clone; `.worktrees` paths are refused.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.c1_geometry import (
    C1_GRPO_EPISODE,
    C1_SFT_EPOCH,
    C1_TRAIN_8X8,
    behavior_stats,
    stratify_geometry,
)
from vlm_minigrid_rl.c1_goalred import (
    C1_EVAL_8X8,
    METHOD_G1,
    METHOD_PAINT,
    CellResult as PaintCell,
    c1_goalred_verdict_grpo,
    c1r_cell_path,
    evaluate_goalred_in_env,
)
from vlm_minigrid_rl.c1_search import (
    C1_EVAL_16X16,
    METHOD_SEE,
    CellResult as SeeCell,
    c1_search_verdict_grpo,
    c1s_cell_path,
    evaluate_search_in_env,
    far_see_truncated_frac,
    see_oracle_stats,
)
from vlm_minigrid_rl.c1_see_paint_trainseed import (
    COLOR_GOOD,
    EVAL_SEEDS,
    EXPERIMENT_ID,
    FROZEN_PAINT_RUN,
    FROZEN_SEE_RUN,
    REUSE_TRAIN_SEED,
    SEE_GOOD,
    TRAIN_SEEDS_GPU,
    across_axis_verdict,
    assert_not_frozen_run_dir,
    g1_sanity_ok,
    joint_mechanism_verdict,
    planned_cells,
    train_run_dir,
)
from vlm_minigrid_rl.experiment_config import BASE_MODEL_ID, DEFAULT_EVAL_EPISODES, DEFAULT_TILE_SIZE
from vlm_minigrid_rl.handoff_sweep import grpo_episode_dir, grpo_handoff_root
from vlm_minigrid_rl.model_utils import load_vlm_model_with_adapters
from vlm_minigrid_rl.ood_trainseed import adapter_bundle_complete, grpo_path_matches_train_seed
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import build_navigation_prompt

DEFAULT_SFT_ADAPTER = "checkpoints/sft_adapter_8x8_bs32_e10/epoch-9"
ARM_PAINT = "PAINT"
ARM_SEE = "SEE"
ARMS = (ARM_PAINT, ARM_SEE)


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only SEE/PAINT on GRPO train seeds 123/456 (C1 caps 12/40)."
    )
    parser.add_argument("--run-dir", default=f"runs/{EXPERIMENT_ID}")
    parser.add_argument("--sft-adapter-path", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument(
        "--train-seeds",
        default=",".join(str(seed) for seed in TRAIN_SEEDS_GPU),
    )
    parser.add_argument(
        "--eval-seeds",
        default=",".join(str(seed) for seed in EVAL_SEEDS),
    )
    parser.add_argument("--arms", default="PAINT,SEE")
    parser.add_argument("--policies", default="GRPO")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--frozen-see-dir", default=FROZEN_SEE_RUN)
    parser.add_argument("--frozen-paint-dir", default=FROZEN_PAINT_RUN)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="4 episodes, train 123, eval seed 42, both arms; runs/_archive/.",
    )
    return parser.parse_args()


def resolve_run_dir(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    here = project_path(".").resolve()
    for parent in [here, *here.parents]:
        if ".worktrees" in parent.parts:
            continue
        if (parent / "checkpoints").is_dir() or (parent / "runs").is_dir():
            return parent / path_str
    return project_path(path_str)


def resolve_existing(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    here = project_path(".").resolve()
    for parent in [here, *here.parents]:
        if ".worktrees" in parent.parts:
            continue
        probe = parent / path_str
        if probe.exists():
            return probe
    return project_path(path_str)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _opt_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _opt_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def cell_out_path(nested: Path, arm: str, method: str, eval_seed: int) -> Path:
    if arm == ARM_SEE:
        return c1s_cell_path(nested, method, "GRPO", eval_seed)
    return c1r_cell_path(nested, method, "GRPO", eval_seed)


def see_cell_payload(
    *,
    method: str,
    seed: int,
    episodes: int,
    sft_path: Path,
    grpo_path: Path,
    train_seed: int,
    metrics: dict,
    episode_rows: list[dict],
) -> dict:
    strata = stratify_geometry(episode_rows)
    n_total = int(strata["n_visible"] or 0) + int(strata["n_empty"] or 0)
    truncated = far_see_truncated_frac(episode_rows)
    oracle = see_oracle_stats(episode_rows)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "c1s_method": method,
        "env_size": 16,
        "sft_adapter_path": str(sft_path),
        "grpo_adapter_path": str(grpo_path),
        "sft_epoch": C1_SFT_EPOCH,
        "grpo_episode": C1_GRPO_EPISODE,
        "train_seed": int(train_seed),
        "episodes": episodes,
        "max_steps": C1_EVAL_16X16,
        "eval_max_steps": C1_EVAL_16X16,
        "train_max_steps": C1_TRAIN_8X8,
        "goal_color": "green",
        "prompt_goal_color": "green",
        "seed": seed,
        "policy": "GRPO",
        "far_see_truncated_frac": truncated,
        **strata,
        **{f"oracle_{key}": value for key, value in oracle.items()},
    }
    if n_total:
        config["start_goal_visible_frac"] = int(strata["n_visible"] or 0) / n_total
        config["start_visible_any_dir_frac"] = (
            int(strata["n_visible"] or 0) + int(strata["n_facing_miss"] or 0)
        ) / n_total
    return {
        "config": config,
        "policies": [{"name": "GRPO", "metrics": metrics}],
        "episodes": episode_rows,
        "strata": strata,
        "oracle": oracle,
        "behavior": {
            "empty": behavior_stats(episode_rows, "empty"),
            "facing_miss": behavior_stats(episode_rows, "facing_miss"),
            "far": behavior_stats(episode_rows, "far"),
        },
    }


def paint_cell_payload(
    *,
    method: str,
    seed: int,
    episodes: int,
    sft_path: Path,
    grpo_path: Path,
    train_seed: int,
    metrics: dict,
    episode_rows: list[dict],
) -> dict:
    n_visible = sum(1 for row in episode_rows if row.get("start_goal_visible"))
    config = {
        "experiment_id": EXPERIMENT_ID,
        "c1r_method": method,
        "env_size": 8,
        "sft_adapter_path": str(sft_path),
        "grpo_adapter_path": str(grpo_path),
        "sft_epoch": C1_SFT_EPOCH,
        "grpo_episode": C1_GRPO_EPISODE,
        "train_seed": int(train_seed),
        "episodes": episodes,
        "max_steps": C1_EVAL_8X8,
        "eval_max_steps": C1_EVAL_8X8,
        "train_max_steps": C1_TRAIN_8X8,
        "goal_color": "red",
        "prompt_goal_color": "red",
        "paint_color": "green" if method == METHOD_PAINT else None,
        "seed": seed,
        "policy": "GRPO",
        "n_visible": n_visible,
    }
    return {
        "config": config,
        "policies": [{"name": "GRPO", "metrics": metrics}],
        "episodes": episode_rows,
    }


def see_cell_from_payload(payload: dict) -> SeeCell | None:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    method = str(config.get("c1s_method") or "")
    policy = str(config.get("policy") or "")
    if not method or not policy:
        return None
    metrics = None
    for row in payload.get("policies") or []:
        if str(row.get("name") or "") == policy:
            metrics = row.get("metrics")
            break
    if not isinstance(metrics, dict) or "success_rate" not in metrics:
        return None
    strata = payload.get("strata") if isinstance(payload.get("strata"), dict) else {}
    return SeeCell(
        method=method,
        policy=policy,
        seed=int(config.get("seed") or 0),
        success_rate=float(metrics.get("success_rate") or 0.0),
        n_visible=_opt_int(config.get("n_visible", strata.get("n_visible"))),
        n_facing_miss=_opt_int(config.get("n_facing_miss", strata.get("n_facing_miss"))),
        n_far=_opt_int(config.get("n_far", strata.get("n_far"))),
        n_empty=_opt_int(config.get("n_empty", strata.get("n_empty"))),
        success_visible=_opt_float(config.get("success_visible", strata.get("success_visible"))),
        success_facing_miss=_opt_float(
            config.get("success_facing_miss", strata.get("success_facing_miss"))
        ),
        success_far=_opt_float(config.get("success_far", strata.get("success_far"))),
        success_empty=_opt_float(config.get("success_empty", strata.get("success_empty"))),
        far_see_truncated_frac=_opt_float(config.get("far_see_truncated_frac")),
        inconclusive=bool(config.get("inconclusive")),
    )


def paint_cell_from_payload(payload: dict) -> PaintCell | None:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    method = str(config.get("c1r_method") or "")
    policy = str(config.get("policy") or "")
    if not method or not policy:
        return None
    metrics = None
    for row in payload.get("policies") or []:
        if str(row.get("name") or "") == policy:
            metrics = row.get("metrics")
            break
    if not isinstance(metrics, dict) or "success_rate" not in metrics:
        return None
    n_visible = config.get("n_visible")
    return PaintCell(
        method=method,
        policy=policy,
        seed=int(config.get("seed") or 0),
        success_rate=float(metrics.get("success_rate") or 0.0),
        n_visible=None if n_visible is None else int(n_visible),
        inconclusive=bool(config.get("inconclusive")),
    )


def collect_see_cells(run_dir: Path) -> list[SeeCell]:
    rows: list[SeeCell] = []
    for path in sorted(run_dir.glob("c1s_*.json")):
        cell = see_cell_from_payload(load_json(path))
        if cell is not None and cell.policy == "GRPO":
            rows.append(cell)
    return rows


def collect_paint_cells(run_dir: Path) -> list[PaintCell]:
    rows: list[PaintCell] = []
    for path in sorted(run_dir.glob("c1r_*.json")):
        cell = paint_cell_from_payload(load_json(path))
        if cell is not None and cell.policy == "GRPO":
            rows.append(cell)
    return rows


def g1_mean(cells, method: str) -> float | None:
    values = [
        float(cell.success_rate)
        for cell in cells
        if cell.method == method and cell.policy == "GRPO"
    ]
    return _mean(values)


def see_dump(cells: list[SeeCell]) -> list[dict]:
    return [
        {
            "method": cell.method,
            "policy": cell.policy,
            "seed": cell.seed,
            "success_rate": cell.success_rate,
            "n_visible": cell.n_visible,
            "n_facing_miss": cell.n_facing_miss,
            "n_far": cell.n_far,
            "success_visible": cell.success_visible,
            "success_facing_miss": cell.success_facing_miss,
            "success_far": cell.success_far,
            "success_empty": cell.success_empty,
            "far_see_truncated_frac": cell.far_see_truncated_frac,
        }
        for cell in cells
    ]


def paint_dump(cells: list[PaintCell]) -> list[dict]:
    return [
        {
            "method": cell.method,
            "policy": cell.policy,
            "seed": cell.seed,
            "success_rate": cell.success_rate,
            "n_visible": cell.n_visible,
        }
        for cell in cells
    ]


def write_train_summaries(nested: Path, train_seed: int) -> dict:
    see_cells = collect_see_cells(nested)
    paint_cells = collect_paint_cells(nested)
    see_verdict = c1_search_verdict_grpo(see_cells) if see_cells else {
        "machine": "INCONCLUSIVE",
        "scientific": "INCONCLUSIVE",
    }
    paint_verdict = (
        c1_goalred_verdict_grpo(paint_cells, train_seed=train_seed)
        if paint_cells
        else "INCONCLUSIVE"
    )
    see_g1 = g1_mean(see_cells, METHOD_G1)
    paint_g1 = g1_mean(paint_cells, METHOD_G1)
    see_ok = see_g1 is not None and g1_sanity_ok(
        suite="see", train_seed=train_seed, grpo_mean=see_g1
    )
    paint_ok = paint_g1 is not None and g1_sanity_ok(
        suite="paint", train_seed=train_seed, grpo_mean=paint_g1
    )
    see_scientific = see_verdict["scientific"] if see_ok else "INCONCLUSIVE"
    paint_scientific = paint_verdict if paint_ok else "INCONCLUSIVE"
    see_summary = {
        "train_seed": int(train_seed),
        "g1_grpo_mean": see_g1,
        "g1_sanity_ok": see_ok,
        "verdict": see_verdict,
        "scientific": see_scientific,
        "cells": see_dump(see_cells),
    }
    paint_summary = {
        "train_seed": int(train_seed),
        "g1_grpo_mean": paint_g1,
        "g1_sanity_ok": paint_ok,
        "verdict": paint_verdict,
        "scientific": paint_scientific,
        "cells": paint_dump(paint_cells),
    }
    write_json(nested / "see_summary.json", see_summary)
    write_json(nested / "paint_summary.json", paint_summary)
    return {"see": see_summary, "paint": paint_summary}


def write_root_summary(
    run_dir: Path,
    *,
    frozen_see: Path,
    frozen_paint: Path,
    train_seeds: list[int],
) -> dict:
    by_train: dict[str, dict] = {}
    see_tags: dict[int, str] = {}
    color_tags: dict[int, str] = {}

    frozen_see_cells = collect_see_cells(frozen_see)
    frozen_paint_cells = collect_paint_cells(frozen_paint)
    if frozen_see_cells:
        see42 = c1_search_verdict_grpo(frozen_see_cells)
        see_g1 = g1_mean(frozen_see_cells, METHOD_G1)
        see_ok = see_g1 is not None and g1_sanity_ok(
            suite="see", train_seed=REUSE_TRAIN_SEED, grpo_mean=see_g1
        )
        see_tags[REUSE_TRAIN_SEED] = see42["scientific"] if see_ok else "INCONCLUSIVE"
        by_train[str(REUSE_TRAIN_SEED)] = {
            "see": {
                "source": str(frozen_see),
                "g1_grpo_mean": see_g1,
                "g1_sanity_ok": see_ok,
                "verdict": see42,
                "scientific": see_tags[REUSE_TRAIN_SEED],
                "cells": see_dump(frozen_see_cells),
            }
        }
    if frozen_paint_cells:
        paint42 = c1_goalred_verdict_grpo(
            frozen_paint_cells, train_seed=REUSE_TRAIN_SEED
        )
        paint_g1 = g1_mean(frozen_paint_cells, METHOD_G1)
        paint_ok = paint_g1 is not None and g1_sanity_ok(
            suite="paint", train_seed=REUSE_TRAIN_SEED, grpo_mean=paint_g1
        )
        color_tags[REUSE_TRAIN_SEED] = paint42 if paint_ok else "INCONCLUSIVE"
        by_train.setdefault(str(REUSE_TRAIN_SEED), {})["paint"] = {
            "source": str(frozen_paint),
            "g1_grpo_mean": paint_g1,
            "g1_sanity_ok": paint_ok,
            "verdict": paint42,
            "scientific": color_tags[REUSE_TRAIN_SEED],
            "cells": paint_dump(frozen_paint_cells),
        }

    for train_seed in train_seeds:
        nested = train_run_dir(run_dir, train_seed)
        block = write_train_summaries(nested, train_seed)
        by_train[str(train_seed)] = block
        see_tags[train_seed] = str(block["see"]["scientific"])
        color_tags[train_seed] = str(block["paint"]["scientific"])

    see_axis = across_axis_verdict(see_tags, good=set(SEE_GOOD), prefix="SEE")
    color_axis = across_axis_verdict(color_tags, good=set(COLOR_GOOD), prefix="COLOR")
    joint = joint_mechanism_verdict(see_axis, color_axis)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": joint,
        "see_axis": see_axis,
        "color_axis": color_axis,
        "see_tags": {str(k): v for k, v in sorted(see_tags.items())},
        "color_tags": {str(k): v for k, v in sorted(color_tags.items())},
        "by_train_seed": by_train,
        "caps": {"paint": C1_EVAL_8X8, "see": C1_EVAL_16X16},
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Verdict: {joint} (SEE {see_axis}, COLOR {color_axis})")
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return summary


def load_grpo_model(sft_path: Path, grpo_path: Path, device: str):
    return load_vlm_model_with_adapters(
        BASE_MODEL_ID,
        [str(sft_path), str(grpo_path)],
        device,
        is_trainable=False,
    )


def require_grpo(sft_path: Path, grpo_path: Path, train_seed: int) -> None:
    if not adapter_bundle_complete(sft_path):
        raise SystemExit(f"Missing SFT adapter bundle: {sft_path}")
    if not adapter_bundle_complete(grpo_path):
        raise SystemExit(f"Missing GRPO adapter bundle: {grpo_path}")
    if not grpo_path_matches_train_seed(grpo_path, train_seed):
        raise SystemExit(
            f"GRPO path does not match train seed {train_seed}: {grpo_path}"
        )
    if ".worktrees" in sft_path.parts or ".worktrees" in grpo_path.parts:
        raise SystemExit(
            "Refusing to load adapters from a worktree path; use the primary checkpoints/."
        )


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.episodes = 4
        args.train_seeds = "123"
        args.eval_seeds = "42"
        args.arms = "PAINT,SEE"
        if args.run_dir == f"runs/{EXPERIMENT_ID}":
            args.run_dir = f"runs/_archive/{EXPERIMENT_ID}-smoke"

    train_seeds = parse_csv_ints(args.train_seeds)
    eval_seeds = parse_csv_ints(args.eval_seeds)
    arms = tuple(parse_csv(args.arms))
    unknown_arms = [arm for arm in arms if arm not in ARMS]
    if unknown_arms:
        raise SystemExit(f"Unknown arm(s): {unknown_arms}")
    policies = parse_csv(args.policies)
    if policies != ["GRPO"]:
        raise SystemExit("This run is GRPO-only; pass --policies GRPO")

    run_dir = resolve_run_dir(args.run_dir)
    try:
        assert_not_frozen_run_dir(run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sft_path = resolve_existing(args.sft_adapter_path)
    frozen_see = resolve_existing(args.frozen_see_dir)
    frozen_paint = resolve_existing(args.frozen_paint_dir)
    checkpoints_root = resolve_existing("checkpoints")

    print(
        f"{EXPERIMENT_ID}: episodes={args.episodes} train_seeds={train_seeds} "
        f"eval_seeds={eval_seeds} arms={list(arms)}"
    )
    print(f"SFT: {sft_path}")
    print(f"run-dir: {run_dir}")
    print(f"SEE cap={C1_EVAL_16X16} PAINT cap={C1_EVAL_8X8}")

    cells = planned_cells(
        train_seeds=tuple(train_seeds),
        eval_seeds=tuple(eval_seeds),
        arms=arms,
    )

    if args.dry_run:
        for train_seed, arm, method, eval_seed in cells:
            nested = train_run_dir(run_dir, train_seed)
            out = cell_out_path(nested, arm, method, eval_seed)
            cap = C1_EVAL_8X8 if arm == ARM_PAINT else C1_EVAL_16X16
            grpo = grpo_episode_dir(
                grpo_handoff_root(8, C1_SFT_EPOCH, checkpoints_root, train_seed=train_seed),
                C1_GRPO_EPISODE,
            )
            print(
                f"DRY train={train_seed} {arm} {method} eval={eval_seed} "
                f"max_steps={cap} grpo={grpo} -> {out}"
            )
        print(f"cells={len(cells)}")
        return 0

    if args.summary_only:
        write_root_summary(
            run_dir,
            frozen_see=frozen_see,
            frozen_paint=frozen_paint,
            train_seeds=train_seeds,
        )
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_see = build_navigation_prompt("green")
    prompt_paint = build_navigation_prompt("red")
    run_dir.mkdir(parents=True, exist_ok=True)

    pending: dict[int, list[tuple[str, str, int, Path]]] = defaultdict(list)
    for train_seed, arm, method, eval_seed in cells:
        nested = train_run_dir(run_dir, train_seed)
        out = cell_out_path(nested, arm, method, eval_seed)
        if args.skip_existing and out.exists():
            print(f"skip existing {out}")
            continue
        pending[train_seed].append((arm, method, eval_seed, out))

    for train_seed in train_seeds:
        jobs = pending.get(train_seed) or []
        if not jobs:
            continue
        grpo_path = grpo_episode_dir(
            grpo_handoff_root(8, C1_SFT_EPOCH, checkpoints_root, train_seed=train_seed),
            C1_GRPO_EPISODE,
        )
        require_grpo(sft_path, grpo_path, train_seed)
        print(f"GRPO train={train_seed}: {grpo_path}")
        model, tokenizer, image_processor = load_grpo_model(sft_path, grpo_path, device)
        try:
            for arm, method, eval_seed, out in jobs:
                cap = C1_EVAL_8X8 if arm == ARM_PAINT else C1_EVAL_16X16
                env_size = 8 if arm == ARM_PAINT else 16
                print(
                    f"=== train={train_seed} GRPO {arm} {method} "
                    f"eval={eval_seed} env={env_size} max_steps={cap} ==="
                )
                if arm == ARM_PAINT:
                    metrics, episode_rows = evaluate_goalred_in_env(
                        model,
                        tokenizer,
                        image_processor,
                        prompt_paint,
                        method=method,
                        env_size=8,
                        tile_size=DEFAULT_TILE_SIZE,
                        max_steps=C1_EVAL_8X8,
                        seed=eval_seed,
                        device=device,
                        model_name=f"GRPO-{method}",
                        episodes=args.episodes,
                    )
                    payload = paint_cell_payload(
                        method=method,
                        seed=eval_seed,
                        episodes=args.episodes,
                        sft_path=sft_path,
                        grpo_path=grpo_path,
                        train_seed=train_seed,
                        metrics=metrics,
                        episode_rows=episode_rows,
                    )
                else:
                    metrics, episode_rows = evaluate_search_in_env(
                        model,
                        tokenizer,
                        image_processor,
                        prompt_see,
                        method=method,
                        env_size=16,
                        tile_size=DEFAULT_TILE_SIZE,
                        max_steps=C1_EVAL_16X16,
                        seed=eval_seed,
                        device=device,
                        model_name=f"GRPO-{method}",
                        episodes=args.episodes,
                        goal_color="green",
                    )
                    payload = see_cell_payload(
                        method=method,
                        seed=eval_seed,
                        episodes=args.episodes,
                        sft_path=sft_path,
                        grpo_path=grpo_path,
                        train_seed=train_seed,
                        metrics=metrics,
                        episode_rows=episode_rows,
                    )
                write_json(out, payload)
                print(f"Saved {out}")
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    write_root_summary(
        run_dir,
        frozen_see=frozen_see,
        frozen_paint=frozen_paint,
        train_seeds=train_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
