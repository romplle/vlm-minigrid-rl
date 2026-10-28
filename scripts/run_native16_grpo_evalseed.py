"""Eval-only native Empty 16x16 GRPO far on eval seeds 123/456 (cap 40).

Does not train. Seed 42 is reused from frozen C1 far JSON.
One VLM at a time: load one GRPO train seed, run both eval seeds, unload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.c1_geometry import C1_EVAL_16X16, stratify_geometry
from vlm_minigrid_rl.c1_search import METHOD_G1, evaluate_search_in_env
from vlm_minigrid_rl.experiment_config import BASE_MODEL_ID, DEFAULT_EVAL_EPISODES, DEFAULT_TILE_SIZE
from vlm_minigrid_rl.model_utils import load_vlm_model_with_adapters
from vlm_minigrid_rl.native16_grpo_evalseed import (
    DEFAULT_SFT_ADAPTER,
    EVAL_SEEDS,
    EXPERIMENT_ID,
    FROZEN_GRPO_CELLS,
    GPU_EVAL_SEEDS,
    GRPO_EPISODE,
    REUSE_EVAL_SEED,
    SFT_EPOCH,
    SFT_SUMMARY,
    TRAIN_SEEDS,
    assert_not_frozen_far_run_dir,
    cell_out_path,
    decide_verdict,
    facing_miss_by_eval_seed,
    gpu_row_from_payload,
    grpo_adapter_for_train_seed,
    planned_gpu_cells,
    reuse_row_from_payload,
    sft_by_eval_seed,
)
from vlm_minigrid_rl.ood_trainseed import adapter_bundle_complete
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import build_navigation_prompt


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only native 16x16 GRPO far on eval seeds 123/456 (C1 cap 40)."
    )
    parser.add_argument("--run-dir", default=f"runs/{EXPERIMENT_ID}")
    parser.add_argument("--sft-adapter-path", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--sft-summary", default=SFT_SUMMARY)
    parser.add_argument(
        "--train-seeds",
        default=",".join(str(seed) for seed in TRAIN_SEEDS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in GPU_EVAL_SEEDS),
        help="Eval seeds to run on GPU. Seed 42 is reused from frozen JSON.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=C1_EVAL_16X16)
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
    train_seed: int,
    eval_seed: int,
    episodes: int,
    max_steps: int,
    sft_path: Path,
    grpo_path: Path,
    metrics: dict,
    episode_rows: list[dict],
) -> dict:
    strata = stratify_geometry(episode_rows)
    return {
        "config": {
            "experiment_id": EXPERIMENT_ID,
            "arm": "native",
            "suite": "native16",
            "env_size": 16,
            "max_steps": int(max_steps),
            "eval_max_steps": int(max_steps),
            "sft_adapter_path": str(sft_path),
            "grpo_adapter_path": str(grpo_path),
            "sft_epoch": SFT_EPOCH,
            "grpo_episode": GRPO_EPISODE,
            "train_seed": int(train_seed),
            "episodes": int(episodes),
            "seed": int(eval_seed),
            "policy": f"GRPO{int(train_seed)}",
            "goal_color": "green",
            "prompt_goal_color": "green",
            "decoder": "generate",
            "method": METHOD_G1,
            **strata,
        },
        "policies": [{"name": f"GRPO{int(train_seed)}", "metrics": metrics}],
        "strata": strata,
    }


def load_grpo_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for train_seed, rel in FROZEN_GRPO_CELLS.items():
        path = resolve_existing(rel)
        if not path.is_file():
            raise SystemExit(f"Missing frozen GRPO cell: {path}")
        rows.append(reuse_row_from_payload(load_json(path), train_seed))
    missing: list[str] = []
    for train_seed, eval_seed in planned_gpu_cells():
        path = cell_out_path(run_dir, train_seed, eval_seed)
        if not path.is_file():
            missing.append(path.name)
            continue
        rows.append(gpu_row_from_payload(load_json(path), train_seed, eval_seed))
    if missing:
        raise SystemExit(f"Missing cells: {', '.join(missing)}")
    return rows


def write_summary(run_dir: Path, sft_summary_path: Path) -> dict:
    sft_rows = sft_by_eval_seed(load_json(sft_summary_path))
    grpo_rows = load_grpo_rows(run_dir)
    decision = decide_verdict(grpo_rows=grpo_rows, sft_rows=sft_rows)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "arm": "native",
        "policy": "GRPO",
        "decoder": "generate",
        "eval_max_steps": C1_EVAL_16X16,
        "not_headline": True,
        "sft_summary": str(sft_summary_path),
        "facing_miss_grpo": facing_miss_by_eval_seed(grpo_rows),
        "sft_far_by_eval_seed": {
            str(seed): float(row["success_far"]) for seed, row in sft_rows.items()
        },
        "sft_facing_miss_by_eval_seed": {
            str(seed): _opt_float(row.get("success_facing_miss")) for seed, row in sft_rows.items()
        },
        **decision,
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"verdict={summary['verdict']} reuse_ok={summary['reuse_ok']} "
        f"reuse_mean={summary['reuse_grpo_far_mean']:.4f}"
    )
    for seed in EVAL_SEEDS:
        block = summary["paired_deltas"][str(seed)]
        print(
            f"eval {seed}: grpo {block['grpo_mean']:.4f} sft {block['sft_far']:.4f} "
            f"delta {block['delta']:.4f}"
        )
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return summary


def require_adapter(path: Path, label: str) -> None:
    if not adapter_bundle_complete(path):
        raise SystemExit(f"Missing {label} adapter bundle: {path}")
    if ".worktrees" in path.parts:
        raise SystemExit(
            "Refusing to load adapters from a worktree path; use the primary checkpoints/."
        )


def _opt_float(value):
    if value is None:
        return None
    return float(value)


def main() -> int:
    args = parse_args()
    train_seeds = parse_csv_ints(args.train_seeds)
    eval_seeds = parse_csv_ints(args.seeds)
    unknown_train = [seed for seed in train_seeds if seed not in TRAIN_SEEDS]
    if unknown_train:
        raise SystemExit(f"Train seeds must be in {TRAIN_SEEDS}, got {unknown_train}")
    unknown_eval = [seed for seed in eval_seeds if seed not in GPU_EVAL_SEEDS]
    if unknown_eval or REUSE_EVAL_SEED in eval_seeds:
        raise SystemExit(
            f"GPU eval seeds must be in {GPU_EVAL_SEEDS}; seed {REUSE_EVAL_SEED} is reused"
        )
    if args.max_steps != C1_EVAL_16X16:
        raise SystemExit(f"C1 native 16x16 cap is {C1_EVAL_16X16}; got --max-steps {args.max_steps}")

    run_dir = resolve_run_dir(args.run_dir)
    try:
        assert_not_frozen_far_run_dir(run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"{EXPERIMENT_ID}: episodes={args.episodes} train_seeds={train_seeds} "
        f"eval_seeds={eval_seeds} max_steps={args.max_steps}"
    )
    print(f"run-dir: {run_dir}")

    if args.dry_run:
        n_cells = 0
        for train_seed in train_seeds:
            adapter = grpo_adapter_for_train_seed(train_seed)
            for eval_seed in eval_seeds:
                out = cell_out_path(run_dir, train_seed, eval_seed)
                print(
                    f"DRY GRPO train={train_seed} eval={eval_seed} method={METHOD_G1} "
                    f"max_steps={args.max_steps} grpo={adapter} -> {out}"
                )
                n_cells += 1
        print(f"cells={n_cells} reuse_eval={REUSE_EVAL_SEED}")
        return 0

    if args.summary_only:
        write_summary(run_dir, resolve_existing(args.sft_summary))
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this eval; use --dry-run or --summary-only")

    sft_path = resolve_existing(args.sft_adapter_path)
    require_adapter(sft_path, "SFT")
    device = "cuda"
    prompt = build_navigation_prompt("green")
    run_dir.mkdir(parents=True, exist_ok=True)

    for train_seed in train_seeds:
        pending: list[tuple[int, Path]] = []
        for eval_seed in eval_seeds:
            out = cell_out_path(run_dir, train_seed, eval_seed)
            if args.skip_existing and out.exists():
                print(f"skip existing {out}")
                continue
            pending.append((eval_seed, out))
        if not pending:
            continue

        grpo_path = resolve_existing(grpo_adapter_for_train_seed(train_seed))
        require_adapter(grpo_path, "GRPO")
        model, tokenizer, image_processor = load_vlm_model_with_adapters(
            BASE_MODEL_ID,
            [str(sft_path), str(grpo_path)],
            device,
            is_trainable=False,
        )
        try:
            for eval_seed, out in pending:
                print(
                    f"=== GRPO train={train_seed} eval={eval_seed} method={METHOD_G1} "
                    f"max_steps={args.max_steps} episodes={args.episodes} ==="
                )
                metrics, episode_rows = evaluate_search_in_env(
                    model,
                    tokenizer,
                    image_processor,
                    prompt,
                    method=METHOD_G1,
                    env_size=16,
                    tile_size=DEFAULT_TILE_SIZE,
                    max_steps=args.max_steps,
                    seed=eval_seed,
                    device=device,
                    model_name=f"GRPO-t{train_seed}-s{eval_seed}",
                    episodes=args.episodes,
                    goal_color="green",
                )
                payload = cell_payload(
                    train_seed=train_seed,
                    eval_seed=eval_seed,
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
        cell_out_path(run_dir, train_seed, eval_seed)
        for train_seed, eval_seed in planned_gpu_cells()
    }
    if all(path.is_file() for path in expected):
        write_summary(run_dir, resolve_existing(args.sft_summary))
    else:
        missing = sorted(path.name for path in expected if not path.is_file())
        print(f"Summary deferred; missing {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
