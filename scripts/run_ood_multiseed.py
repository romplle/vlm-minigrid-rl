"""Empty-8x8 multi-seed OOD replication (C1).

Eval-only: transfer 8->16 and goal-red at SFT epochs 1/5/9, eval seeds
42/123/456, GRPO episode-100. Reuses A1 seed-42 JSON when config matches.
Does not train and does not write into the A1 run directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.experiment_config import DEFAULT_EVAL_EPISODES, dataset_dir
from vlm_minigrid_rl.handoff_sweep import (
    build_grpo_eval_command,
    grpo_episode_dir,
    grpo_handoff_root,
    sft_epoch_dir,
)
from vlm_minigrid_rl.ood_multiseed import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_SEEDS,
    PRIMARY_EPOCH,
    PRIMARY_GRPO_EPISODE,
    SUITES,
    TRAIN_ENV_SIZE,
    build_suite_epoch_row,
    build_summary_payload,
    ood_cell_path,
    rates_from_ood_payload,
    reuse_payload_matches_suite,
    suite_max_steps,
)
from vlm_minigrid_rl.paths import project_path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_SCRIPT = SCRIPT_DIR / "test_models.py"

ALL_PHASES = ("eval", "summary")
DEFAULT_RUN_DIR = "runs/2026-08-ood-multiseed-8x8"
DEFAULT_SFT_ROOT = "checkpoints/sft_adapter_8x8_bs32_e10"
DEFAULT_A1_OOD_DIR = "runs/2026-07-handoff-empty/8x8"
EXPERIMENT_ID = "2026-08-ood-multiseed-8x8"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_phases(raw: str) -> list[str]:
    phases = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [phase for phase in phases if phase not in ALL_PHASES]
    if unknown:
        raise SystemExit(
            f"Unknown phase(s): {', '.join(unknown)}; expected {', '.join(ALL_PHASES)}"
        )
    if not phases:
        raise SystemExit("At least one phase is required")
    return phases


def run_command(command: list[str], *, dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {path}")


def maybe_reuse_a1(
    src: Path,
    dst: Path,
    *,
    suite: dict,
    sft_epoch: int,
    dry_run: bool,
) -> bool:
    if not src.is_file():
        return False
    if src.resolve() == dst.resolve():
        return False
    try:
        payload = load_json(src)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        print(f"skip reuse (unreadable): {src}")
        return False
    if not reuse_payload_matches_suite(payload, suite=suite, sft_epoch=sft_epoch):
        print(f"skip reuse (config mismatch): {src}")
        return False
    print(f"reuse {src} -> {dst}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "C1 multi-seed OOD on Empty-8x8 (transfer 8->16 + goal-red). "
            "Eval-only; default GRPO episode-100."
        )
    )
    parser.add_argument("--phases", default="eval,summary")
    parser.add_argument(
        "--epochs",
        default=",".join(str(epoch) for epoch in DEFAULT_EPOCHS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_EVAL_SEEDS),
        help="Eval seeds.",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--sft-root", default=DEFAULT_SFT_ROOT)
    parser.add_argument(
        "--a1-ood-dir",
        default=DEFAULT_A1_OOD_DIR,
        help="A1 OOD JSON directory used to reuse seed-42 cells.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-reuse-a1",
        action="store_true",
        help="Disable reuse of A1 seed-42 OOD JSON and re-run those cells.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> None:
    epochs = parse_csv_ints(args.epochs)
    seeds = parse_csv_ints(args.seeds)
    run_dir = project_path(args.run_dir)
    sft_root = project_path(args.sft_root)
    a1_ood_dir = project_path(args.a1_ood_dir)
    checkpoints_root = project_path("checkpoints")
    python = sys.executable
    run_dir.mkdir(parents=True, exist_ok=True)

    for suite in SUITES.values():
        eval_env = int(suite["eval_env"])
        dataset = dataset_dir(eval_env)
        max_steps = suite_max_steps(suite)
        for epoch in epochs:
            sft_adapter = sft_epoch_dir(sft_root, epoch)
            grpo_adapter = grpo_episode_dir(
                grpo_handoff_root(
                    TRAIN_ENV_SIZE, epoch, checkpoints_root=checkpoints_root
                ),
                PRIMARY_GRPO_EPISODE,
            )
            if not (sft_adapter / "adapter_config.json").is_file() and not args.dry_run:
                raise SystemExit(f"Missing SFT adapter: {sft_adapter}")
            if not (grpo_adapter / "adapter_config.json").is_file() and not args.dry_run:
                raise SystemExit(f"Missing GRPO adapter: {grpo_adapter}")

            for seed in seeds:
                dest = ood_cell_path(run_dir, suite["tag"], epoch, seed)
                if args.skip_existing and dest.is_file():
                    print(f"skip existing {dest}")
                    continue
                if not args.no_reuse_a1 and not dest.is_file():
                    src = a1_ood_dir / dest.name
                    if maybe_reuse_a1(
                        src,
                        dest,
                        suite=suite,
                        sft_epoch=epoch,
                        dry_run=args.dry_run,
                    ):
                        continue
                cmd = build_grpo_eval_command(
                    python=python,
                    test_models=TEST_MODELS_SCRIPT,
                    env_size=eval_env,
                    dataset=dataset,
                    sft_adapter=sft_adapter,
                    grpo_adapter=grpo_adapter,
                    episodes=args.episodes,
                    seed=seed,
                    max_steps=max_steps,
                    output_json=dest,
                    goal_color=str(suite["goal_color"]),
                    prompt_goal_color=str(suite["prompt_goal_color"]),
                )
                code = run_command(cmd, dry_run=args.dry_run)
                if code != 0:
                    raise SystemExit(f"Eval failed ({code}): {' '.join(cmd)}")


def run_summary(args: argparse.Namespace) -> None:
    epochs = parse_csv_ints(args.epochs)
    seeds = parse_csv_ints(args.seeds)
    run_dir = project_path(args.run_dir)
    missing: list[str] = []
    rows = []
    for suite in SUITES.values():
        for epoch in epochs:
            sft_rates: list[float] = []
            grpo_rates: list[float] = []
            present_seeds: list[int] = []
            for seed in seeds:
                path = ood_cell_path(run_dir, suite["tag"], epoch, seed)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = load_json(path)
                if not reuse_payload_matches_suite(
                    payload, suite=suite, sft_epoch=epoch
                ):
                    raise SystemExit(f"Cell config mismatch: {path}")
                sft_rate, grpo_rate = rates_from_ood_payload(payload)
                present_seeds.append(seed)
                sft_rates.append(sft_rate)
                grpo_rates.append(grpo_rate)
            if len(present_seeds) != len(seeds):
                continue
            rows.append(
                build_suite_epoch_row(
                    suite_id=str(suite["id"]),
                    sft_epoch=epoch,
                    seeds=present_seeds,
                    sft_rates=sft_rates,
                    grpo_rates=grpo_rates,
                )
            )

    if missing and not rows:
        raise SystemExit(f"No complete suite/epoch rows; missing e.g. {missing[0]}")
    if missing:
        print(f"warning: {len(missing)} missing cells; summarizing complete rows only")

    payload = build_summary_payload(
        rows,
        config={
            "experiment_id": EXPERIMENT_ID,
            "train_env": TRAIN_ENV_SIZE,
            "epochs": epochs,
            "eval_seeds": seeds,
            "episodes": args.episodes,
            "grpo_episode": PRIMARY_GRPO_EPISODE,
            "primary_epoch": PRIMARY_EPOCH,
            "sft_root": str(project_path(args.sft_root)),
            "a1_ood_dir": str(project_path(args.a1_ood_dir)),
            "run_dir": str(run_dir),
            "missing_cells": missing,
        },
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    write_json(run_dir / "summary.json", payload)
    print(f"verdict={payload['verdict']} {payload['reason']}")


def main() -> None:
    args = parse_args()
    phases = parse_phases(args.phases)
    if "eval" in phases:
        run_eval(args)
    if "summary" in phases:
        run_summary(args)


if __name__ == "__main__":
    main()
