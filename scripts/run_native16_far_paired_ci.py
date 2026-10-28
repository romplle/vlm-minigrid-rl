"""Paired 95% CI for native 16x16 far. Seed 42 is CPU. GPU is eval 123/456 only.

Does not train. Does not write the frozen far catalog or the measurement-B summary.
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
from vlm_minigrid_rl.native16_far_paired_ci import (
    EXPERIMENT_ID,
    FROZEN_GRPO_42,
    N_EPISODES,
    PUBLISHED_POINT_DELTA,
    SFT_EPISODE_CELLS,
    InvalidAlign,
    assert_writable_run_dir,
    compact_episode_bits,
    gpu_interval,
    seed42_interval,
)
from vlm_minigrid_rl.native16_grpo_evalseed import (
    DEFAULT_SFT_ADAPTER,
    GPU_EVAL_SEEDS,
    GRPO_EPISODE,
    SFT_EPOCH,
    TRAIN_SEEDS,
    cell_out_path,
    grpo_adapter_for_train_seed,
    planned_gpu_cells,
)
from vlm_minigrid_rl.ood_trainseed import adapter_bundle_complete
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import build_navigation_prompt


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired 95% CI for native 16x16 far (C1 cap 40). Seed 42 is CPU."
    )
    parser.add_argument("--run-dir", default=f"runs/{EXPERIMENT_ID}")
    parser.add_argument("--sft-adapter-path", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument(
        "--train-seeds",
        default=",".join(str(seed) for seed in TRAIN_SEEDS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in GPU_EVAL_SEEDS),
        help="GPU eval seeds. Seed 42 is the frozen-bit interval.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=C1_EVAL_16X16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--seed42-only", action="store_true")
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


def require_adapter(path: Path, label: str) -> None:
    if not adapter_bundle_complete(path):
        raise SystemExit(f"Missing {label} adapter bundle: {path}")
    if ".worktrees" in path.parts:
        raise SystemExit(
            "Refusing to load adapters from a worktree path; use the primary checkpoints/."
        )


def episodes_of(payload: dict) -> list[dict]:
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise InvalidAlign(f"missing episodes in {payload.get('config', {}).get('policy')}")
    return rows


def load_seed42() -> dict:
    sft_path = resolve_existing(SFT_EPISODE_CELLS[42])
    if not sft_path.is_file():
        raise SystemExit(f"Missing frozen SFT cell: {sft_path}")
    sft_payload = load_json(sft_path)
    grpo_rows: dict[int, list[dict]] = {}
    for train_seed, rel in FROZEN_GRPO_42.items():
        path = resolve_existing(rel)
        if not path.is_file():
            raise SystemExit(f"Missing frozen GRPO cell: {path}")
        grpo_rows[int(train_seed)] = episodes_of(load_json(path))
    try:
        block = seed42_interval(episodes_of(sft_payload), grpo_rows)
    except InvalidAlign as exc:
        raise SystemExit(str(exc)) from exc
    config_delta = _config_delta(sft_payload, {
        train_seed: load_json(resolve_existing(rel))
        for train_seed, rel in FROZEN_GRPO_42.items()
    })
    if abs(float(block["point"]) - config_delta) > 1e-4:
        raise SystemExit(
            f"INVALID_ALIGN: seed 42 paired point {block['point']:.6f} "
            f"disagrees with stored far rates {config_delta:.6f}"
        )
    return block


def _config_delta(sft_payload: dict, grpo_payloads: dict[int, dict]) -> float:
    sft_far = float(sft_payload["config"]["success_far"])
    grpo_far = [float(grpo_payloads[seed]["config"]["success_far"]) for seed in TRAIN_SEEDS]
    return sum(grpo_far) / len(grpo_far) - sft_far


def write_seed42(run_dir: Path, block: dict) -> None:
    write_json(run_dir / "seed42_interval.json", block)
    _print_interval(42, block)


def gpu_cell_payload(
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
    bits = compact_episode_bits(episode_rows)
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
        "bits": bits,
    }


def write_summary(run_dir: Path, seed42: dict) -> dict:
    intervals = {"42": seed42}
    for eval_seed in GPU_EVAL_SEEDS:
        sft_path = resolve_existing(SFT_EPISODE_CELLS[eval_seed])
        if not sft_path.is_file():
            raise SystemExit(f"Missing stored SFT episodes: {sft_path}")
        sft_episodes = episodes_of(load_json(sft_path))
        grpo_bits: dict[int, list[dict]] = {}
        for train_seed in TRAIN_SEEDS:
            path = cell_out_path(run_dir, train_seed, eval_seed)
            if not path.is_file():
                raise SystemExit(f"Missing GRPO cell: {path}")
            payload = load_json(path)
            bits = payload.get("bits")
            if not isinstance(bits, list):
                raise SystemExit(f"INVALID_ALIGN: {path.name} has no compact bits")
            grpo_bits[train_seed] = bits
        try:
            intervals[str(eval_seed)] = gpu_interval(
                sft_episodes,
                grpo_bits,
                eval_seed=eval_seed,
            )
        except InvalidAlign as exc:
            raise SystemExit(str(exc)) from exc
    phrase_456 = intervals["456"]["phrase"]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "arm": "native",
        "decoder": "generate",
        "eval_max_steps": C1_EVAL_16X16,
        "not_headline": True,
        "published_point_delta": {str(seed): delta for seed, delta in PUBLISHED_POINT_DELTA.items()},
        "intervals": intervals,
        "phrase_456": phrase_456,
        "withdraw_sign_differs": phrase_456 == "ENTIRELY_BELOW",
        "note": (
            "Delta is the paired mean of three GRPO success bits minus the SFT bit "
            "on the same far start, in percentage points. "
            "Seed 42 uses the frozen episode bits of the published -7.98 cell. "
            "The GRPO re-run on eval seeds 123 and 456 used the same seed as the "
            "published cells and reproduced -2.31 / +2.80, so those intervals belong "
            "to the published points. "
            "Do not average the three intervals. "
            "Only the eval-seed 456 interval selects the phrase."
        ),
    }
    write_json(run_dir / "summary.json", summary)
    for eval_seed in GPU_EVAL_SEEDS:
        _print_interval(eval_seed, intervals[str(eval_seed)])
    print(f"phrase_456={phrase_456}")
    print(f"Saved summary to {run_dir / 'summary.json'}")
    return summary


def _print_interval(eval_seed: int, block: dict) -> None:
    print(
        f"eval {eval_seed}: point {block['point']:.4f} "
        f"95% [{block['lo']:.4f}; {block['hi']:.4f}] "
        f"n_far={block['n_far']} phrase={block['phrase']}"
    )


def expected_cells(run_dir: Path) -> set[Path]:
    return {
        cell_out_path(run_dir, train_seed, eval_seed)
        for train_seed, eval_seed in planned_gpu_cells()
    }


def main() -> int:
    args = parse_args()
    train_seeds = parse_csv_ints(args.train_seeds)
    eval_seeds = parse_csv_ints(args.seeds)
    unknown_train = [seed for seed in train_seeds if seed not in TRAIN_SEEDS]
    if unknown_train:
        raise SystemExit(f"Train seeds must be in {TRAIN_SEEDS}, got {unknown_train}")
    unknown_eval = [seed for seed in eval_seeds if seed not in GPU_EVAL_SEEDS]
    if unknown_eval or 42 in eval_seeds:
        raise SystemExit(f"GPU eval seeds must be in {GPU_EVAL_SEEDS}; seed 42 is CPU")
    if args.max_steps != C1_EVAL_16X16:
        raise SystemExit(f"C1 native 16x16 cap is {C1_EVAL_16X16}; got --max-steps {args.max_steps}")
    if args.episodes != N_EPISODES:
        raise SystemExit(f"Episode count must be {N_EPISODES} to match stored SFT bits")

    run_dir = resolve_run_dir(args.run_dir)
    try:
        assert_writable_run_dir(run_dir)
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
        print(f"cells={n_cells} seed42=cpu")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    seed42 = load_seed42()
    write_seed42(run_dir, seed42)
    if args.seed42_only:
        return 0

    if args.summary_only:
        missing = sorted(path.name for path in expected_cells(run_dir) if not path.is_file())
        if missing:
            print(f"Summary deferred; missing {missing}")
            return 0
        write_summary(run_dir, seed42)
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this eval; use --dry-run, --seed42-only, or --summary-only")

    sft_path = resolve_existing(args.sft_adapter_path)
    require_adapter(sft_path, "SFT")
    device = "cuda"
    prompt = build_navigation_prompt("green")

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
                payload = gpu_cell_payload(
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

    if all(path.is_file() for path in expected_cells(run_dir)):
        write_summary(run_dir, seed42)
    else:
        missing = sorted(path.name for path in expected_cells(run_dir) if not path.is_file())
        print(f"Summary deferred; missing {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
