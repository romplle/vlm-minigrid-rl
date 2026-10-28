"""OOD eval across GRPO train seeds 42/123/456 (C1 caps 12/40).

Eval-only. Default --sft-epoch 9 (e9 replica). Use --sft-epoch 1 or 5 with a
new --run-dir for B1. Reuses C1 JSON for train seed 42. Does not write into
the C1 run dir or the frozen e9 run dir (unless --sft-epoch 9). Transfer
max_steps=40, goal-red max_steps=12.
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
    DEFAULT_EVAL_SEEDS,
    PRIMARY_EPOCH,
    PRIMARY_GRPO_EPISODE,
    SUITES,
    TRAIN_ENV_SIZE,
    build_suite_epoch_row,
    ood_cell_path,
    rates_from_ood_payload,
)
from vlm_minigrid_rl.ood_trainseed import (
    ALLOWED_SFT_EPOCHS,
    REUSE_TRAIN_SEED,
    adapter_bundle_complete,
    assert_run_dir_allowed,
    build_trainseed_summary_payload,
    c1_reuse_matches,
    experiment_id_for_epoch,
    grpo_path_matches_train_seed,
    suite_c1_max_steps,
    trainseed_cell_path,
)
from vlm_minigrid_rl.paths import project_path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_SCRIPT = SCRIPT_DIR / "test_models.py"

ALL_PHASES = ("eval", "summary")
DEFAULT_RUN_DIR = "runs/2026-09-ood-trainseed-e9-8x8"
DEFAULT_C1_RUN_DIR = "runs/2026-08-ood-multiseed-8x8"
DEFAULT_SFT_ROOT = "checkpoints/sft_adapter_8x8_bs32_e10"
EXPERIMENT_ID = "2026-09-ood-trainseed-e9-8x8"
DEFAULT_TRAIN_SEEDS = (42, 123, 456)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_suite_ids(raw: str) -> list[str]:
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [suite_id for suite_id in ids if suite_id not in SUITES]
    if unknown:
        raise SystemExit(
            f"Unknown suite(s): {', '.join(unknown)}; expected {', '.join(SUITES)}"
        )
    if not ids:
        raise SystemExit("At least one suite is required")
    return ids


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OOD across GRPO train seeds (transfer @40, goal-red @12). "
            "Default --sft-epoch 9. Train seed 42 is copied from C1; "
            "123/456 are eval-only."
        )
    )
    parser.add_argument("--phases", default="eval,summary")
    parser.add_argument(
        "--sft-epoch",
        type=int,
        default=PRIMARY_EPOCH,
        help="SFT epoch to replicate (default 9). Use 1 or 5 for B1; then --run-dir is required.",
    )
    parser.add_argument(
        "--train-seeds",
        default=",".join(str(seed) for seed in DEFAULT_TRAIN_SEEDS),
        help="GRPO train seeds. 42 is reused from --c1-run-dir, not re-evaled.",
    )
    parser.add_argument(
        "--eval-seeds",
        default=",".join(str(seed) for seed in DEFAULT_EVAL_SEEDS),
    )
    parser.add_argument(
        "--suites",
        default=",".join(SUITES),
        help="Comma-separated suite ids: transfer8to16,goal_red",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="New run tree. Required when --sft-epoch is not 9. Never C1 2026-08-ood-multiseed-8x8.",
    )
    parser.add_argument("--c1-run-dir", default=DEFAULT_C1_RUN_DIR)
    parser.add_argument("--sft-root", default=DEFAULT_SFT_ROOT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sft_epoch not in ALLOWED_SFT_EPOCHS:
        parser.error(f"--sft-epoch must be one of {ALLOWED_SFT_EPOCHS}")
    if args.run_dir is None:
        if args.sft_epoch != PRIMARY_EPOCH:
            parser.error("--run-dir is required when --sft-epoch is not 9")
        args.run_dir = DEFAULT_RUN_DIR
    return args


def require_adapters(sft_adapter: Path, grpo_adapter: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if not adapter_bundle_complete(sft_adapter):
        raise SystemExit(f"Missing SFT adapter bundle: {sft_adapter}")
    if not adapter_bundle_complete(grpo_adapter):
        raise SystemExit(f"Missing GRPO adapter bundle: {grpo_adapter}")


def copy_c1_cell(
    *,
    c1_run_dir: Path,
    dest: Path,
    suite: dict,
    eval_seed: int,
    episodes: int,
    sft_epoch: int,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    if skip_existing and dest.is_file():
        print(f"skip existing {dest}")
        return
    src = ood_cell_path(c1_run_dir, str(suite["tag"]), sft_epoch, eval_seed)
    if not src.is_file():
        raise SystemExit(f"Missing C1 cell for train-seed 42 reuse: {src}")
    payload = load_json(src)
    if not c1_reuse_matches(
        payload,
        suite=suite,
        eval_seed=eval_seed,
        episodes=episodes,
        sft_epoch=sft_epoch,
    ):
        raise SystemExit(f"C1 cell config mismatch (caps/path/seed): {src}")
    print(f"reuse {src} -> {dest}")
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def run_eval(args: argparse.Namespace) -> None:
    train_seeds = parse_csv_ints(args.train_seeds)
    eval_seeds = parse_csv_ints(args.eval_seeds)
    suite_ids = parse_suite_ids(args.suites)
    sft_epoch = int(args.sft_epoch)
    run_dir = project_path(args.run_dir)
    assert_run_dir_allowed(run_dir, sft_epoch)
    c1_run_dir = project_path(args.c1_run_dir)
    sft_root = project_path(args.sft_root)
    checkpoints_root = project_path("checkpoints")
    python = sys.executable
    run_dir.mkdir(parents=True, exist_ok=True)
    sft_adapter = sft_epoch_dir(sft_root, sft_epoch)

    for train_seed in train_seeds:
        grpo_adapter = grpo_episode_dir(
            grpo_handoff_root(
                TRAIN_ENV_SIZE,
                sft_epoch,
                checkpoints_root=checkpoints_root,
                train_seed=train_seed,
            ),
            PRIMARY_GRPO_EPISODE,
        )
        if train_seed != REUSE_TRAIN_SEED:
            require_adapters(sft_adapter, grpo_adapter, dry_run=args.dry_run)
            if not grpo_path_matches_train_seed(
                grpo_adapter, train_seed, sft_epoch=sft_epoch
            ):
                raise SystemExit(
                    f"GRPO path does not match train seed {train_seed} "
                    f"epoch {sft_epoch}: {grpo_adapter}"
                )
        for suite_id in suite_ids:
            suite = SUITES[suite_id]
            max_steps = suite_c1_max_steps(suite)
            eval_env = int(suite["eval_env"])
            dataset = dataset_dir(eval_env)
            for eval_seed in eval_seeds:
                dest = trainseed_cell_path(
                    run_dir,
                    train_seed,
                    str(suite["tag"]),
                    eval_seed,
                    sft_epoch=sft_epoch,
                )
                if train_seed == REUSE_TRAIN_SEED:
                    copy_c1_cell(
                        c1_run_dir=c1_run_dir,
                        dest=dest,
                        suite=suite,
                        eval_seed=eval_seed,
                        episodes=args.episodes,
                        sft_epoch=sft_epoch,
                        dry_run=args.dry_run,
                        skip_existing=args.skip_existing,
                    )
                    continue
                if args.skip_existing and dest.is_file():
                    print(f"skip existing {dest}")
                    continue
                cmd = build_grpo_eval_command(
                    python=python,
                    test_models=TEST_MODELS_SCRIPT,
                    env_size=eval_env,
                    dataset=dataset,
                    sft_adapter=sft_adapter,
                    grpo_adapter=grpo_adapter,
                    episodes=args.episodes,
                    seed=eval_seed,
                    max_steps=max_steps,
                    output_json=dest,
                    goal_color=str(suite["goal_color"]),
                    prompt_goal_color=str(suite["prompt_goal_color"]),
                )
                code = run_command(cmd, dry_run=args.dry_run)
                if code != 0:
                    raise SystemExit(f"Eval failed ({code}): {' '.join(cmd)}")


def run_summary(args: argparse.Namespace) -> None:
    train_seeds = parse_csv_ints(args.train_seeds)
    eval_seeds = parse_csv_ints(args.eval_seeds)
    suite_ids = parse_suite_ids(args.suites)
    sft_epoch = int(args.sft_epoch)
    run_dir = project_path(args.run_dir)
    assert_run_dir_allowed(run_dir, sft_epoch)
    missing: list[str] = []
    by_train: dict[str, dict[str, dict]] = {}
    for train_seed in train_seeds:
        suite_rows: dict[str, dict] = {}
        for suite_id in suite_ids:
            suite = SUITES[suite_id]
            sft_rates: list[float] = []
            grpo_rates: list[float] = []
            present: list[int] = []
            for eval_seed in eval_seeds:
                path = trainseed_cell_path(
                    run_dir,
                    train_seed,
                    str(suite["tag"]),
                    eval_seed,
                    sft_epoch=sft_epoch,
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = load_json(path)
                if train_seed == REUSE_TRAIN_SEED:
                    if not c1_reuse_matches(
                        payload,
                        suite=suite,
                        eval_seed=eval_seed,
                        episodes=args.episodes,
                        sft_epoch=sft_epoch,
                    ):
                        raise SystemExit(f"Reused C1 cell mismatch: {path}")
                else:
                    config = payload.get("config") or {}
                    if int(config.get("max_steps") or 0) != suite_c1_max_steps(suite):
                        raise SystemExit(f"max_steps mismatch (want C1 caps): {path}")
                    if not grpo_path_matches_train_seed(
                        config.get("grpo_adapter_path"),
                        train_seed,
                        sft_epoch=sft_epoch,
                    ):
                        raise SystemExit(
                            f"GRPO path does not match train seed {train_seed}: {path}"
                        )
                sft_rate, grpo_rate = rates_from_ood_payload(payload)
                present.append(eval_seed)
                sft_rates.append(sft_rate)
                grpo_rates.append(grpo_rate)
            if len(present) != len(eval_seeds):
                continue
            suite_rows[suite_id] = build_suite_epoch_row(
                suite_id=suite_id,
                sft_epoch=sft_epoch,
                seeds=present,
                sft_rates=sft_rates,
                grpo_rates=grpo_rates,
            )
        if suite_rows:
            by_train[str(train_seed)] = suite_rows

    if missing and not by_train:
        if args.dry_run:
            print(f"[dry-run] summary skipped; missing e.g. {missing[0]}")
            return
        raise SystemExit(f"No complete train-seed rows; missing e.g. {missing[0]}")
    if missing:
        print(f"warning: {len(missing)} missing cells; summarizing complete rows only")

    payload = build_trainseed_summary_payload(
        by_train,
        config={
            "experiment_id": experiment_id_for_epoch(sft_epoch),
            "train_env": TRAIN_ENV_SIZE,
            "sft_epoch": sft_epoch,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "episodes": args.episodes,
            "grpo_episode": PRIMARY_GRPO_EPISODE,
            "transfer_max_steps": suite_c1_max_steps(SUITES["transfer8to16"]),
            "goal_red_max_steps": suite_c1_max_steps(SUITES["goal_red"]),
            "sft_root": str(project_path(args.sft_root)),
            "c1_run_dir": str(project_path(args.c1_run_dir)),
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
