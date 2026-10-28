import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.experiment_config import dataset_dir
from vlm_minigrid_rl.handoff_sweep import (
    build_delta_row,
    build_grpo_eval_command,
    build_grpo_train_command,
    build_sft_eval_command,
    epochs_missing_from_picks,
    eval_json_path,
    extract_policy_metrics,
    grpo_episode_dir,
    grpo_handoff_root,
    list_sft_epochs,
    load_grpo_picks,
    merge_grpo_picks_on_disk,
    pick_best_grpo_episode,
    should_skip_grpo_train,
    sft_epoch_dir,
)
from vlm_minigrid_rl.minigrid_utils import default_eval_max_steps, default_train_max_steps
from vlm_minigrid_rl.paths import project_path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_SCRIPT = SCRIPT_DIR / "test_models.py"
GRPO_SCRIPT = SCRIPT_DIR / "grpo.py"

ALL_PHASES = ("sft-eval", "grpo", "grpo-pick", "grpo-eval", "summary")
DEFAULT_EPOCHS = "1,2,3,5,7,9"
DEFAULT_SEEDS = "42,123,456"
DEFAULT_RUN_DIR = "runs/2026-07-handoff-empty"
GRPO_PICK_EPISODES = (25, 50, 75, 100)
GRPO_TRAIN_SEED = 42


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def resolve_epochs(sft_root: Path, epochs_arg: str) -> list[int]:
    discovered = list_sft_epochs(sft_root)
    if epochs_arg.strip().lower() == "all":
        if not discovered:
            raise SystemExit(f"No SFT epochs found under {sft_root}")
        return discovered

    requested = parse_csv_ints(epochs_arg)
    missing = [epoch for epoch in requested if not sft_epoch_dir(sft_root, epoch).is_dir()]
    if missing:
        missing_str = ", ".join(str(epoch) for epoch in missing)
        raise SystemExit(f"Missing SFT adapter epoch(s) under {sft_root}: {missing_str}")
    return requested


def grpo_picks_path(run_dir: Path, env_size: int) -> Path:
    return run_dir / f"{env_size}x{env_size}" / "grpo_episode_picks.json"


def summary_path(run_dir: Path, env_size: int) -> Path:
    return run_dir / f"{env_size}x{env_size}" / "summary.json"


def should_skip_eval(output_json: Path, skip_existing: bool) -> bool:
    return skip_existing and output_json.is_file()


def run_command(command: list[str], *, dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase_sft_eval(
    *,
    python: str,
    env_size: int,
    dataset: Path,
    sft_root: Path,
    epochs: list[int],
    seeds: list[int],
    eval_episodes: int,
    max_steps: int,
    run_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    exit_code = 0
    for epoch in epochs:
        sft_adapter = sft_epoch_dir(sft_root, epoch)
        for seed in seeds:
            output_json = eval_json_path(
                run_dir,
                kind="sft",
                env_size=env_size,
                sft_epoch=epoch,
                seed=seed,
            )
            if should_skip_eval(output_json, skip_existing):
                print(f"skip existing {output_json}")
                continue
            output_json.parent.mkdir(parents=True, exist_ok=True)
            command = build_sft_eval_command(
                python=python,
                test_models=TEST_MODELS_SCRIPT,
                env_size=env_size,
                dataset=dataset,
                sft_adapter=sft_adapter,
                episodes=eval_episodes,
                seed=seed,
                max_steps=max_steps,
                output_json=output_json,
            )
            code = run_command(command, dry_run=dry_run)
            if code != 0:
                exit_code = code
    return exit_code


def phase_grpo(
    *,
    python: str,
    env_size: int,
    dataset: Path,
    sft_root: Path,
    epochs: list[int],
    train_episodes: int,
    checkpoint_interval: int,
    max_steps: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    exit_code = 0
    checkpoints_root = project_path("checkpoints")
    for epoch in epochs:
        sft_adapter = sft_epoch_dir(sft_root, epoch)
        output_dir = grpo_handoff_root(env_size, epoch, checkpoints_root=checkpoints_root)
        if should_skip_grpo_train(output_dir, train_episodes, skip_existing):
            print(f"skip existing {output_dir}")
            continue
        command = build_grpo_train_command(
            python=python,
            grpo_script=GRPO_SCRIPT,
            env_size=env_size,
            dataset=dataset,
            sft_adapter=sft_adapter,
            output_dir=output_dir,
            train_episodes=train_episodes,
            checkpoint_interval=checkpoint_interval,
            max_steps=max_steps,
            seed=GRPO_TRAIN_SEED,
        )
        code = run_command(command, dry_run=dry_run)
        if code != 0:
            exit_code = code
    return exit_code


def phase_grpo_pick(
    *,
    python: str,
    env_size: int,
    dataset: Path,
    sft_root: Path,
    epochs: list[int],
    eval_episodes: int,
    max_steps: int,
    run_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    picks_file = grpo_picks_path(run_dir, env_size)
    if skip_existing and picks_file.is_file():
        missing_epochs = epochs_missing_from_picks(picks_file, epochs)
        if not missing_epochs:
            print(f"skip existing {picks_file}")
            return 0
        epochs_to_pick = missing_epochs
        print(f"re-pick missing epochs in {picks_file}: {missing_epochs}")
    else:
        epochs_to_pick = list(epochs)

    exit_code = 0
    checkpoints_root = project_path("checkpoints")
    episode_to_success_by_epoch: dict[int, dict[int, float]] = {}

    for epoch in epochs_to_pick:
        sft_adapter = sft_epoch_dir(sft_root, epoch)
        grpo_root = grpo_handoff_root(env_size, epoch, checkpoints_root=checkpoints_root)
        episode_to_success: dict[int, float] = {}
        for grpo_episode in GRPO_PICK_EPISODES:
            output_json = eval_json_path(
                run_dir,
                kind="grpo",
                env_size=env_size,
                sft_epoch=epoch,
                seed=GRPO_TRAIN_SEED,
                grpo_episode=grpo_episode,
            )
            if should_skip_eval(output_json, skip_existing):
                payload = load_json(output_json)
                metrics = extract_policy_metrics(payload, "GRPO")
                episode_to_success[grpo_episode] = float(metrics["success_rate"])
                continue

            grpo_adapter = grpo_episode_dir(grpo_root, grpo_episode)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            command = build_grpo_eval_command(
                python=python,
                test_models=TEST_MODELS_SCRIPT,
                env_size=env_size,
                dataset=dataset,
                sft_adapter=sft_adapter,
                grpo_adapter=grpo_adapter,
                episodes=eval_episodes,
                seed=GRPO_TRAIN_SEED,
                max_steps=max_steps,
                output_json=output_json,
            )
            code = run_command(command, dry_run=dry_run)
            if code != 0:
                exit_code = code
                continue
            if dry_run:
                continue
            payload = load_json(output_json)
            metrics = extract_policy_metrics(payload, "GRPO")
            episode_to_success[grpo_episode] = float(metrics["success_rate"])

        if episode_to_success:
            episode_to_success_by_epoch[epoch] = episode_to_success

    if dry_run:
        return exit_code

    new_picks = {
        epoch: pick_best_grpo_episode(episode_to_success)
        for epoch, episode_to_success in episode_to_success_by_epoch.items()
        if episode_to_success
    }
    if new_picks:
        picks_file.parent.mkdir(parents=True, exist_ok=True)
        merged = merge_grpo_picks_on_disk(picks_file, new_picks)
        with picks_file.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return exit_code


def phase_grpo_eval(
    *,
    python: str,
    env_size: int,
    dataset: Path,
    sft_root: Path,
    epochs: list[int],
    seeds: list[int],
    eval_episodes: int,
    max_steps: int,
    run_dir: Path,
    default_grpo_episode: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    exit_code = 0
    checkpoints_root = project_path("checkpoints")
    picks = load_grpo_picks(grpo_picks_path(run_dir, env_size), epochs, default_grpo_episode)

    for epoch in epochs:
        sft_adapter = sft_epoch_dir(sft_root, epoch)
        grpo_root = grpo_handoff_root(env_size, epoch, checkpoints_root=checkpoints_root)
        grpo_episode = picks[epoch]
        grpo_adapter = grpo_episode_dir(grpo_root, grpo_episode)
        for seed in seeds:
            output_json = eval_json_path(
                run_dir,
                kind="grpo",
                env_size=env_size,
                sft_epoch=epoch,
                seed=seed,
                grpo_episode=grpo_episode,
            )
            if should_skip_eval(output_json, skip_existing):
                print(f"skip existing {output_json}")
                continue
            output_json.parent.mkdir(parents=True, exist_ok=True)
            command = build_grpo_eval_command(
                python=python,
                test_models=TEST_MODELS_SCRIPT,
                env_size=env_size,
                dataset=dataset,
                sft_adapter=sft_adapter,
                grpo_adapter=grpo_adapter,
                episodes=eval_episodes,
                seed=seed,
                max_steps=max_steps,
                output_json=output_json,
            )
            code = run_command(command, dry_run=dry_run)
            if code != 0:
                exit_code = code
    return exit_code


def phase_summary(
    *,
    env_size: int,
    epochs: list[int],
    seeds: list[int],
    run_dir: Path,
    default_grpo_episode: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    out_path = summary_path(run_dir, env_size)
    if dry_run:
        print(f"would write {out_path}")
        return 0
    if skip_existing and out_path.is_file():
        print(f"skip existing {out_path}")
        return 0

    picks = load_grpo_picks(grpo_picks_path(run_dir, env_size), epochs, default_grpo_episode)
    missing_evals: list[str] = []

    for epoch in epochs:
        grpo_episode = picks[epoch]
        for seed in seeds:
            sft_json = eval_json_path(
                run_dir,
                kind="sft",
                env_size=env_size,
                sft_epoch=epoch,
                seed=seed,
            )
            grpo_json = eval_json_path(
                run_dir,
                kind="grpo",
                env_size=env_size,
                sft_epoch=epoch,
                seed=seed,
                grpo_episode=grpo_episode,
            )
            if not sft_json.is_file():
                missing_evals.append(str(sft_json))
            if not grpo_json.is_file():
                missing_evals.append(str(grpo_json))

    if missing_evals:
        missing_list = "\n".join(f"  - {path}" for path in missing_evals)
        raise SystemExit(
            "Cannot build summary: missing eval JSON(s) for requested epoch×seed pairs:\n"
            f"{missing_list}"
        )

    rows: list[dict] = []
    for epoch in epochs:
        sft_rates: list[float] = []
        grpo_rates: list[float] = []
        sft_timeouts: list[float] = []
        grpo_timeouts: list[float] = []
        sft_invalids: list[float] = []
        grpo_invalids: list[float] = []

        grpo_episode = picks[epoch]
        for seed in seeds:
            sft_json = eval_json_path(
                run_dir,
                kind="sft",
                env_size=env_size,
                sft_epoch=epoch,
                seed=seed,
            )
            grpo_json = eval_json_path(
                run_dir,
                kind="grpo",
                env_size=env_size,
                sft_epoch=epoch,
                seed=seed,
                grpo_episode=grpo_episode,
            )
            sft_metrics = extract_policy_metrics(load_json(sft_json), "SFT")
            grpo_metrics = extract_policy_metrics(load_json(grpo_json), "GRPO")
            sft_rates.append(float(sft_metrics["success_rate"]))
            grpo_rates.append(float(grpo_metrics["success_rate"]))
            sft_timeouts.append(float(sft_metrics["timeouts"]))
            grpo_timeouts.append(float(grpo_metrics["timeouts"]))
            sft_invalids.append(float(sft_metrics["invalid_action_episodes"]))
            grpo_invalids.append(float(grpo_metrics["invalid_action_episodes"]))

        rows.append(
            build_delta_row(
                sft_epoch=epoch,
                sft_rates=sft_rates,
                grpo_rates=grpo_rates,
                sft_timeouts=sft_timeouts,
                grpo_timeouts=grpo_timeouts,
                sft_invalids=sft_invalids,
                grpo_invalids=grpo_invalids,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_size": env_size,
        "epochs": [row["sft_epoch"] for row in rows],
        "rows": rows,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate SFT-budget x GRPO handoff sweep for EmptyEnv."
    )
    parser.add_argument("--env-size", type=int, choices=(8, 16), required=True)
    parser.add_argument(
        "--sft-root",
        default=None,
        help="SFT adapter root (default checkpoints/sft_adapter_{env}_bs32_e10).",
    )
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, help="Comma list or 'all'.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--phases",
        default=",".join(ALL_PHASES),
        help=f"Subset of: {', '.join(ALL_PHASES)}",
    )
    parser.add_argument("--grpo-train-episodes", type=int, default=100)
    parser.add_argument("--grpo-checkpoint-interval", type=int, default=25)
    parser.add_argument("--grpo-eval-episode", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=250)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps for env eval (sft-eval / grpo-pick / grpo-eval). Default: env default.",
    )
    parser.add_argument(
        "--grpo-max-steps",
        type=int,
        default=None,
        help="Max steps for GRPO training only. Default: profile train_max_steps (L_max).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even when outputs or GRPO checkpoints already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_size = args.env_size
    sft_root = project_path(
        args.sft_root or f"checkpoints/sft_adapter_{env_size}x{env_size}_bs32_e10"
    )
    run_dir = project_path(args.run_dir)
    dataset = dataset_dir(env_size)
    eval_max_steps = (
        args.max_steps if args.max_steps is not None else default_eval_max_steps(env_size)
    )
    grpo_max_steps = (
        args.grpo_max_steps
        if args.grpo_max_steps is not None
        else default_train_max_steps(env_size)
    )
    seeds = parse_csv_ints(args.seeds)
    phases = [part.strip() for part in args.phases.split(",") if part.strip()]
    unknown = [phase for phase in phases if phase not in ALL_PHASES]
    if unknown:
        raise SystemExit(f"Unknown phase(s): {', '.join(unknown)}")

    epochs = resolve_epochs(sft_root, args.epochs)
    skip_existing = not args.no_skip_existing
    python = sys.executable
    exit_code = 0

    if "sft-eval" in phases:
        code = phase_sft_eval(
            python=python,
            env_size=env_size,
            dataset=dataset,
            sft_root=sft_root,
            epochs=epochs,
            seeds=seeds,
            eval_episodes=args.eval_episodes,
            max_steps=eval_max_steps,
            run_dir=run_dir,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )
        exit_code = exit_code or code

    if "grpo" in phases:
        code = phase_grpo(
            python=python,
            env_size=env_size,
            dataset=dataset,
            sft_root=sft_root,
            epochs=epochs,
            train_episodes=args.grpo_train_episodes,
            checkpoint_interval=args.grpo_checkpoint_interval,
            max_steps=grpo_max_steps,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )
        exit_code = exit_code or code

    if "grpo-pick" in phases:
        code = phase_grpo_pick(
            python=python,
            env_size=env_size,
            dataset=dataset,
            sft_root=sft_root,
            epochs=epochs,
            eval_episodes=args.eval_episodes,
            max_steps=eval_max_steps,
            run_dir=run_dir,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )
        exit_code = exit_code or code

    if "grpo-eval" in phases:
        code = phase_grpo_eval(
            python=python,
            env_size=env_size,
            dataset=dataset,
            sft_root=sft_root,
            epochs=epochs,
            seeds=seeds,
            eval_episodes=args.eval_episodes,
            max_steps=eval_max_steps,
            run_dir=run_dir,
            default_grpo_episode=args.grpo_eval_episode,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )
        exit_code = exit_code or code

    if "summary" in phases:
        code = phase_summary(
            env_size=env_size,
            epochs=epochs,
            seeds=seeds,
            run_dir=run_dir,
            default_grpo_episode=args.grpo_eval_episode,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )
        exit_code = exit_code or code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
