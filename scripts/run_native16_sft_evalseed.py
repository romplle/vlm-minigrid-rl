"""Eval-only native Empty 16x16 SFT far on eval seeds 123/456 (cap 40).

Does not train. Seed 42 is reused from frozen C1 far JSON. GRPO is not re-evaled.
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
from vlm_minigrid_rl.native16_sft_evalseed import (
    DEFAULT_SFT_ADAPTER,
    EXPERIMENT_ID,
    FROZEN_FAR_DIR,
    FROZEN_SFT_CELL,
    GPU_EVAL_SEEDS,
    REUSE_EVAL_SEED,
    SFT_EPOCH,
    STABILITY_GATE_PP,
    aggregate_far,
    assert_not_frozen_far_run_dir,
    cell_out_path,
    gpu_row_from_payload,
    reuse_seed42_row,
)
from vlm_minigrid_rl.ood_trainseed import adapter_bundle_complete
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import build_navigation_prompt


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval-only native 16x16 SFT far on eval seeds 123/456 (C1 cap 40)."
    )
    parser.add_argument("--run-dir", default=f"runs/{EXPERIMENT_ID}")
    parser.add_argument("--sft-adapter-path", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in GPU_EVAL_SEEDS),
        help="Eval seeds to run on GPU. Seed 42 is always reused from frozen JSON.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=C1_EVAL_16X16)
    parser.add_argument("--frozen-sft-cell", default=FROZEN_SFT_CELL)
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
    seed: int,
    episodes: int,
    max_steps: int,
    sft_path: Path,
    metrics: dict,
    episode_rows: list[dict],
) -> dict:
    strata = stratify_geometry(episode_rows)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "arm": "native",
        "suite": "native16",
        "env_size": 16,
        "max_steps": int(max_steps),
        "eval_max_steps": int(max_steps),
        "sft_adapter_path": str(sft_path),
        "grpo_adapter_path": None,
        "sft_epoch": SFT_EPOCH,
        "episodes": episodes,
        "seed": int(seed),
        "policy": "SFT",
        "goal_color": "green",
        "prompt_goal_color": "green",
        "decoder": "generate",
        **strata,
    }
    return {
        "config": config,
        "policies": [{"name": "SFT", "metrics": metrics}],
        "episodes": episode_rows,
        "strata": strata,
    }


def write_summary(run_dir: Path, frozen_cell: Path, gpu_seeds: list[int]) -> dict:
    reuse = reuse_seed42_row(load_json(frozen_cell))
    gpu_rows: list[dict] = []
    for seed in gpu_seeds:
        path = cell_out_path(run_dir, seed)
        if not path.exists():
            raise SystemExit(f"Missing GPU cell {path}")
        gpu_rows.append(gpu_row_from_payload(load_json(path), seed))
    gpu_agg = aggregate_far(gpu_rows)
    three = aggregate_far([reuse, *gpu_rows])
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "arm": "native",
        "policy": "SFT",
        "eval_max_steps": C1_EVAL_16X16,
        "decoder": "generate",
        "not_headline": True,
        "frozen_reuse": {
            "run": FROZEN_FAR_DIR,
            "cell": str(frozen_cell),
            "seed": REUSE_EVAL_SEED,
            **{k: v for k, v in reuse.items() if k != "seed"},
        },
        "gpu_seeds": [int(seed) for seed in gpu_seeds],
        "stability_gate_pp": STABILITY_GATE_PP,
        "far_sft_seed42_reuse": reuse["success_far"],
        "far_gpu": gpu_agg,
        "far_three_eval_seed": three,
        "note": (
            "GRPO native far 73.71% is 3 train-seed at eval 42 — not the same "
            "aggregate as this SFT eval-seed mean. Do not splice them as one row."
        ),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"far SFT seed42 reuse={reuse['success_far']:.4f} "
        f"gpu mean={gpu_agg['mean']:.4f} sd={gpu_agg['sd']:.4f} "
        f"within_3pp={gpu_agg['within_3pp_of_seed42']}"
    )
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return summary


def require_sft(sft_path: Path) -> None:
    if not adapter_bundle_complete(sft_path):
        raise SystemExit(f"Missing SFT adapter bundle: {sft_path}")
    if ".worktrees" in sft_path.parts:
        raise SystemExit(
            "Refusing to load adapters from a worktree path; use the primary checkpoints/."
        )


def main() -> int:
    args = parse_args()
    gpu_seeds = parse_csv_ints(args.seeds)
    if REUSE_EVAL_SEED in gpu_seeds:
        raise SystemExit(
            f"Do not re-run seed {REUSE_EVAL_SEED}; it is reused from {FROZEN_SFT_CELL}"
        )
    if args.max_steps != C1_EVAL_16X16:
        raise SystemExit(
            f"C1 native 16x16 cap is {C1_EVAL_16X16}; got --max-steps {args.max_steps}"
        )

    run_dir = resolve_run_dir(args.run_dir)
    try:
        assert_not_frozen_far_run_dir(run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sft_path = resolve_existing(args.sft_adapter_path)
    frozen_cell = resolve_existing(args.frozen_sft_cell)

    print(
        f"{EXPERIMENT_ID}: episodes={args.episodes} gpu_seeds={gpu_seeds} "
        f"max_steps={args.max_steps}"
    )
    print(f"SFT: {sft_path}")
    print(f"run-dir: {run_dir}")
    print(f"reuse seed {REUSE_EVAL_SEED} from {frozen_cell}")

    if args.dry_run:
        for seed in gpu_seeds:
            out = cell_out_path(run_dir, seed)
            print(f"DRY SFT native16 eval={seed} max_steps={args.max_steps} -> {out}")
        print(f"cells={len(gpu_seeds)} reuse={REUSE_EVAL_SEED}")
        return 0

    if args.summary_only:
        write_summary(run_dir, frozen_cell, gpu_seeds)
        return 0

    if not frozen_cell.is_file():
        raise SystemExit(f"Missing frozen SFT cell: {frozen_cell}")
    require_sft(sft_path)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this eval; use --dry-run or --summary-only")

    device = "cuda"
    prompt = build_navigation_prompt("green")
    run_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, Path]] = []
    for seed in gpu_seeds:
        out = cell_out_path(run_dir, seed)
        if args.skip_existing and out.exists():
            print(f"skip existing {out}")
            continue
        pending.append((seed, out))

    if pending:
        model, tokenizer, image_processor = load_vlm_model_with_adapters(
            BASE_MODEL_ID,
            [str(sft_path)],
            device,
            is_trainable=False,
        )
        try:
            for seed, out in pending:
                print(
                    f"=== SFT native16 eval={seed} env=16 max_steps={args.max_steps} "
                    f"episodes={args.episodes} ==="
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
                    seed=seed,
                    device=device,
                    model_name=f"SFT-G1-s{seed}",
                    episodes=args.episodes,
                    goal_color="green",
                )
                payload = cell_payload(
                    seed=seed,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    sft_path=sft_path,
                    metrics=metrics,
                    episode_rows=episode_rows,
                )
                write_json(out, payload)
                print(f"Wrote {out}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_summary(run_dir, frozen_cell, gpu_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
