"""Eval-only SFT goal-red by start stratum (G1, cap 12). No train, no PAINT.

Twelve cells: SFT epochs 1/5/9 and GRPO e9 control, eval seeds 42/123/456.
Does not write into frozen C1 goal-red or ood-multiseed catalogs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.c1_geometry import stratify_geometry
from vlm_minigrid_rl.c1_goalred import C1_EVAL_8X8, METHOD_G1, evaluate_goalred_in_env
from vlm_minigrid_rl.experiment_config import BASE_MODEL_ID, DEFAULT_EVAL_EPISODES, DEFAULT_TILE_SIZE
from vlm_minigrid_rl.model_utils import load_vlm_model_with_adapters
from vlm_minigrid_rl.ood_trainseed import adapter_bundle_complete
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.sft_goalred_visible import (
    EVAL_SEEDS,
    EXPERIMENT_ID,
    GRPO_ADAPTER,
    SFT_EPOCHS,
    assert_not_frozen_run_dir,
    build_summary,
    cell_out_path,
    planned_cells,
    row_from_payload,
    sft_adapter_for_epoch,
)
from vlm_minigrid_rl.training_utils import build_navigation_prompt


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only SFT goal-red strata on epochs 1/5/9 (G1, cap 12)."
    )
    parser.add_argument("--run-dir", default=f"runs/{EXPERIMENT_ID}")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in EVAL_SEEDS),
    )
    parser.add_argument(
        "--epochs",
        default=",".join(str(epoch) for epoch in SFT_EPOCHS),
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=C1_EVAL_8X8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
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


def cell_payload(
    *,
    policy: str,
    epoch: int,
    seed: int,
    episodes: int,
    max_steps: int,
    sft_path: Path,
    grpo_path: Path | None,
    metrics: dict,
    episode_rows: list[dict],
) -> dict:
    strata = stratify_geometry(episode_rows)
    return {
        "config": {
            "experiment_id": EXPERIMENT_ID,
            "policy": policy,
            "sft_epoch": int(epoch),
            "seed": int(seed),
            "episodes": int(episodes),
            "eval_max_steps": int(max_steps),
            "max_steps": int(max_steps),
            "env_size": 8,
            "goal_color": "red",
            "prompt_goal_color": "red",
            "decoder": "generate",
            "method": METHOD_G1,
            "sft_adapter_path": str(sft_path),
            "grpo_adapter_path": None if grpo_path is None else str(grpo_path),
        },
        "policies": [{"name": policy, "metrics": metrics}],
        "strata": strata,
    }


def load_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    for policy, epoch, seed in planned_cells():
        path = cell_out_path(run_dir, policy, epoch, seed)
        if not path.is_file():
            missing.append(path.name)
            continue
        rows.append(row_from_payload(load_json(path)))
    if missing:
        raise SystemExit(f"Missing cells: {', '.join(missing)}")
    return rows


def write_summary(run_dir: Path, episodes: int) -> dict:
    summary = build_summary(load_rows(run_dir), episodes=episodes)
    write_json(run_dir / "summary.json", summary)
    print(f"verdict={summary['verdict']} suite_ok={summary['suite_ok']}")
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return summary


def require_adapter(path: Path, label: str) -> None:
    if not adapter_bundle_complete(path):
        raise SystemExit(f"Missing {label} adapter bundle: {path}")
    if ".worktrees" in path.parts:
        raise SystemExit(
            "Refusing to load adapters from a worktree path; use the primary checkpoints/."
        )


def model_groups(
    epochs: list[int],
    seeds: list[int],
) -> list[tuple[str, int, str, str | None, list[int]]]:
    groups: list[tuple[str, int, str, str | None, list[int]]] = []
    for epoch in epochs:
        groups.append(("SFT", int(epoch), sft_adapter_for_epoch(epoch), None, list(seeds)))
    if 9 in epochs:
        groups.append(("GRPO", 9, sft_adapter_for_epoch(9), GRPO_ADAPTER, list(seeds)))
    return groups


def main() -> int:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    epochs = parse_csv_ints(args.epochs)
    unknown_epochs = [epoch for epoch in epochs if epoch not in SFT_EPOCHS]
    if unknown_epochs:
        raise SystemExit(f"Epochs must be in {SFT_EPOCHS}, got {unknown_epochs}")
    unknown_seeds = [seed for seed in seeds if seed not in EVAL_SEEDS]
    if unknown_seeds:
        raise SystemExit(f"Seeds must be in {EVAL_SEEDS}, got {unknown_seeds}")
    if args.max_steps != C1_EVAL_8X8:
        raise SystemExit(f"C1 goal-red cap is {C1_EVAL_8X8}; got --max-steps {args.max_steps}")

    run_dir = resolve_run_dir(args.run_dir)
    try:
        assert_not_frozen_run_dir(run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    groups = model_groups(epochs, seeds)
    print(
        f"{EXPERIMENT_ID}: episodes={args.episodes} seeds={seeds} "
        f"epochs={epochs} max_steps={args.max_steps}"
    )
    print(f"run-dir: {run_dir}")

    if args.dry_run:
        n_cells = 0
        for policy, epoch, sft_rel, grpo_rel, group_seeds in groups:
            for seed in group_seeds:
                out = cell_out_path(run_dir, policy, epoch, seed)
                print(
                    f"DRY {policy} e{epoch} eval={seed} method={METHOD_G1} "
                    f"max_steps={args.max_steps} sft={sft_rel} grpo={grpo_rel} -> {out}"
                )
                n_cells += 1
        print(f"cells={n_cells}")
        return 0

    if args.summary_only:
        write_summary(run_dir, args.episodes)
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this eval; use --dry-run or --summary-only")

    device = "cuda"
    prompt = build_navigation_prompt("red")
    run_dir.mkdir(parents=True, exist_ok=True)

    for policy, epoch, sft_rel, grpo_rel, group_seeds in groups:
        pending: list[tuple[int, Path]] = []
        for seed in group_seeds:
            out = cell_out_path(run_dir, policy, epoch, seed)
            if args.skip_existing and out.exists():
                print(f"skip existing {out}")
                continue
            pending.append((seed, out))
        if not pending:
            continue

        sft_path = resolve_existing(sft_rel)
        grpo_path = None if grpo_rel is None else resolve_existing(grpo_rel)
        require_adapter(sft_path, "SFT")
        adapters = [str(sft_path)]
        if grpo_path is not None:
            require_adapter(grpo_path, "GRPO")
            adapters.append(str(grpo_path))

        model, tokenizer, image_processor = load_vlm_model_with_adapters(
            BASE_MODEL_ID,
            adapters,
            device,
            is_trainable=False,
        )
        try:
            for seed, out in pending:
                print(
                    f"=== {policy} e{epoch} eval={seed} method={METHOD_G1} "
                    f"max_steps={args.max_steps} episodes={args.episodes} ==="
                )
                metrics, episode_rows = evaluate_goalred_in_env(
                    model,
                    tokenizer,
                    image_processor,
                    prompt,
                    method=METHOD_G1,
                    env_size=8,
                    tile_size=DEFAULT_TILE_SIZE,
                    max_steps=args.max_steps,
                    seed=seed,
                    device=device,
                    model_name=f"{policy}-e{epoch}-s{seed}",
                    episodes=args.episodes,
                )
                payload = cell_payload(
                    policy=policy,
                    epoch=epoch,
                    seed=seed,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    sft_path=sft_path,
                    grpo_path=grpo_path,
                    metrics=metrics,
                    episode_rows=episode_rows,
                )
                write_json(out, payload)
                print(f"Wrote {out}")
        finally:
            del model, tokenizer, image_processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected = {
        cell_out_path(run_dir, policy, epoch, seed)
        for policy, epoch, seed in planned_cells()
    }
    if all(path.is_file() for path in expected):
        write_summary(run_dir, args.episodes)
    else:
        missing = sorted(path.name for path in expected if not path.is_file())
        print(f"Summary deferred; missing {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
