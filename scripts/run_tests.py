import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.experiment_config import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE,
    DEFAULT_SEED,
    DEFAULT_SFT_EPOCHS,
    dataset_dir,
    default_val_split,
    grpo_adapter_root,
    resolve_grpo_adapter_path,
    resolve_sft_adapter_path,
    sft_adapter_root,
)
from vlm_minigrid_rl.paths import project_path


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_SCRIPT = SCRIPT_DIR / "test_models.py"

PIPELINE_SUITES = {
    "8x8": [
        "native_8x8",
        "transfer_8_to_16",
        "transfer_16_to_8",
        "goal_color_red_red",
        "goal_color_red_green",
    ],
    "16x16": [
        "native_16x16",
        "transfer_8_to_16",
        "transfer_16_to_8",
        "goal_color_red_red",
        "goal_color_red_green",
    ],
    "all": [
        "native_8x8",
        "native_16x16",
        "transfer_8_to_16",
        "transfer_16_to_8",
        "goal_color_red_red",
        "goal_color_red_green",
    ],
}

SUITE_DEFINITIONS = {
    "native_8x8": {
        "eval_env": 8,
        "train_env": 8,
        "goal_color": "green",
        "prompt_goal_color": "green",
        "skip_base": False,
        "skip_majority": False,
        "skip_expert": False,
        "focal": True,
    },
    "native_16x16": {
        "eval_env": 16,
        "train_env": 16,
        "goal_color": "green",
        "prompt_goal_color": "green",
        "skip_base": False,
        "skip_majority": False,
        "skip_expert": False,
        "focal": True,
    },
    "transfer_8_to_16": {
        "eval_env": 16,
        "train_env": 8,
        "goal_color": "green",
        "prompt_goal_color": "green",
        "skip_base": True,
        "skip_majority": False,
        "skip_expert": False,
        "focal": "train",
    },
    "transfer_16_to_8": {
        "eval_env": 8,
        "train_env": 16,
        "goal_color": "green",
        "prompt_goal_color": "green",
        "skip_base": True,
        "skip_majority": False,
        "skip_expert": False,
        "focal": "train",
    },
    "goal_color_red_red": {
        "eval_env": 8,
        "train_env": None,
        "goal_color": "red",
        "prompt_goal_color": "red",
        "skip_base": True,
        "skip_majority": True,
        "skip_expert": True,
        "focal": True,
    },
    "goal_color_red_green": {
        "eval_env": 8,
        "train_env": None,
        "goal_color": "red",
        "prompt_goal_color": "green",
        "skip_base": True,
        "skip_majority": True,
        "skip_expert": True,
        "focal": True,
    },
}


@dataclass
class TestRun:
    suite: str
    eval_env: int
    train_env: int
    sft_epoch: int
    grpo_episode: int
    seed: int
    goal_color: str
    prompt_goal_color: str
    sft_adapter_path: str
    grpo_adapter_path: str
    command: list[str]
    output_json: str
    status: str = "pending"
    return_code: int | None = None
    metrics: dict | None = None
    error: str | None = None


@dataclass
class RunConfig:
    pipeline: str
    suites: list[str]
    episodes: int
    seeds: list[int]
    sft_epochs_by_env: dict[int, list[int]]
    grpo_episodes_by_env: dict[int, list[int]]
    baseline_sft_epoch: dict[int, int]
    baseline_grpo_episode: dict[int, int]
    adapter_roots: dict[tuple[str, int], Path]
    dry_run: bool
    continue_on_error: bool
    output_json: Path | None
    run_dir: Path | None


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def focal_train_envs(pipeline: str) -> set[int]:
    if pipeline == "8x8":
        return {8}
    if pipeline == "16x16":
        return {16}
    return {8, 16}


def suite_train_envs(suite_name: str, suite_def: dict, pipeline: str) -> list[int]:
    train_env = suite_def["train_env"]
    if train_env is not None:
        return [train_env]
    if pipeline == "all":
        return [8, 16]
    return [8 if pipeline == "8x8" else 16]


def is_focal_suite(suite_name: str, suite_def: dict, pipeline: str, train_env: int) -> bool:
    focal = suite_def["focal"]
    if focal is True:
        return True
    if focal == "train":
        return train_env in focal_train_envs(pipeline)
    return False


def config_to_dict(config: RunConfig) -> dict:
    data = asdict(config)
    data["adapter_roots"] = {
        f"{kind}_{train_env}": str(path) for (kind, train_env), path in config.adapter_roots.items()
    }
    data["output_json"] = str(config.output_json) if config.output_json else None
    data["run_dir"] = str(config.run_dir) if config.run_dir else None
    return data


def checkpoint_grid(
    train_env: int,
    focal: bool,
    sft_epochs_by_env: dict[int, list[int]],
    grpo_episodes_by_env: dict[int, list[int]],
    baseline_sft_epoch: dict[int, int],
    baseline_grpo_episode: dict[int, int],
) -> list[tuple[int, int]]:
    if focal:
        sft_epochs = sft_epochs_by_env[train_env]
        grpo_episodes = grpo_episodes_by_env[train_env]
        return [(epoch, episode) for epoch in sft_epochs for episode in grpo_episodes]
    return [(baseline_sft_epoch[train_env], baseline_grpo_episode[train_env])]


def build_test_models_command(
    *,
    eval_env: int,
    sft_adapter_path: Path,
    grpo_adapter_path: Path,
    episodes: int,
    seed: int,
    goal_color: str,
    prompt_goal_color: str,
    skip_base: bool,
    skip_majority: bool,
    skip_expert: bool,
    output_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(TEST_MODELS_SCRIPT),
        "--env-size",
        str(eval_env),
        "--dataset-path",
        str(dataset_dir(eval_env)),
        "--sft-adapter-path",
        str(sft_adapter_path),
        "--grpo-adapter-path",
        str(grpo_adapter_path),
        "--episodes",
        str(episodes),
        "--val-split",
        str(default_val_split(eval_env)),
        "--goal-color",
        goal_color,
        "--prompt-goal-color",
        prompt_goal_color,
        "--seed",
        str(seed),
        "--output-json",
        str(output_json),
    ]
    if skip_base:
        cmd.append("--skip-base")
    if skip_majority:
        cmd.append("--skip-majority")
    if skip_expert:
        cmd.append("--skip-expert")
    return cmd


def plan_runs(config: RunConfig) -> tuple[list[TestRun], list[str]]:
    runs: list[TestRun] = []
    warnings: list[str] = []
    run_index = 0

    for suite_name in config.suites:
        suite_def = SUITE_DEFINITIONS[suite_name]

        for train_env in suite_train_envs(suite_name, suite_def, config.pipeline):
            focal = is_focal_suite(suite_name, suite_def, config.pipeline, train_env)
            sft_root = config.adapter_roots[("sft", train_env)]
            grpo_root = config.adapter_roots[("grpo", train_env)]

            for sft_epoch, grpo_episode in checkpoint_grid(
                train_env,
                focal,
                config.sft_epochs_by_env,
                config.grpo_episodes_by_env,
                config.baseline_sft_epoch,
                config.baseline_grpo_episode,
            ):
                sft_path = resolve_sft_adapter_path(train_env, sft_epoch, root=sft_root)
                grpo_path = resolve_grpo_adapter_path(train_env, grpo_episode, root=grpo_root)
                if sft_path is None:
                    warnings.append(
                        f"Skip {suite_name} (train {train_env}x{train_env}): "
                        f"missing SFT epoch-{sft_epoch} under {sft_root}"
                    )
                    continue
                if grpo_path is None:
                    warnings.append(
                        f"Skip {suite_name} (train {train_env}x{train_env}): "
                        f"missing GRPO episode-{grpo_episode} under {grpo_root}"
                    )
                    continue

                for seed in config.seeds:
                    run_index += 1
                    if config.run_dir is not None:
                        output_json = config.run_dir / (
                            f"{run_index:03d}_{suite_name}_"
                            f"train{train_env}x{train_env}_eval{suite_def['eval_env']}x{suite_def['eval_env']}_"
                            f"sft{sft_epoch}_grpo{grpo_episode}_seed{seed}.json"
                        )
                    else:
                        output_json = project_path(
                            f"runs/_tmp_run_{run_index}_{suite_name}_seed{seed}.json"
                        )

                    command = build_test_models_command(
                        eval_env=suite_def["eval_env"],
                        sft_adapter_path=sft_path,
                        grpo_adapter_path=grpo_path,
                        episodes=config.episodes,
                        seed=seed,
                        goal_color=suite_def["goal_color"],
                        prompt_goal_color=suite_def["prompt_goal_color"],
                        skip_base=suite_def["skip_base"],
                        skip_majority=suite_def["skip_majority"],
                        skip_expert=suite_def["skip_expert"],
                        output_json=output_json,
                    )
                    runs.append(
                        TestRun(
                            suite=suite_name,
                            eval_env=suite_def["eval_env"],
                            train_env=train_env,
                            sft_epoch=sft_epoch,
                            grpo_episode=grpo_episode,
                            seed=seed,
                            goal_color=suite_def["goal_color"],
                            prompt_goal_color=suite_def["prompt_goal_color"],
                            sft_adapter_path=str(sft_path),
                            grpo_adapter_path=str(grpo_path),
                            command=command,
                            output_json=str(output_json),
                        )
                    )

    return runs, warnings


def execute_runs(config: RunConfig, runs: list[TestRun]) -> list[TestRun]:
    total = len(runs)
    for index, run in enumerate(runs, start=1):
        print(
            f"\n[{index}/{total}] {run.suite} | train {run.train_env}x{run.train_env} "
            f"-> eval {run.eval_env}x{run.eval_env} | "
            f"SFT epoch-{run.sft_epoch} GRPO episode-{run.grpo_episode} | seed={run.seed}"
        )
        print(" ".join(run.command))

        if config.dry_run:
            run.status = "dry_run"
            continue

        completed = subprocess.run(run.command, check=False)
        run.return_code = completed.returncode
        output_path = Path(run.output_json)
        if completed.returncode == 0 and output_path.is_file():
            with output_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            run.metrics = payload
            run.status = "ok"
        else:
            run.status = "failed"
            run.error = f"test_models.py exited with code {completed.returncode}"
            if not config.continue_on_error:
                break

    return runs


def summarize_runs(runs: list[TestRun]) -> None:
    ok_runs = [run for run in runs if run.status == "ok" and run.metrics]
    if not ok_runs:
        print("\nNo successful runs to summarize.")
        return

    print("\n===============================")
    print("СВОДКА RUN_TESTS")
    print("===============================")
    print(
        "| Suite | Train | Eval | SFT | GRPO | Seed | Policy | Success | Reward | Timeouts | Invalid |"
    )
    print("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for run in ok_runs:
        for policy in run.metrics["policies"]:
            metrics = policy["metrics"]
            print(
                f"| {run.suite} | {run.train_env}x{run.train_env} | {run.eval_env}x{run.eval_env} | "
                f"{run.sft_epoch} | {run.grpo_episode} | {run.seed} | {policy['name']} | "
                f"{metrics['success_rate']:.1f}% | {metrics['avg_reward']:.3f} | "
                f"{metrics['timeouts']}/{metrics['episodes']} | "
                f"{metrics['invalid_action_episodes']}/{metrics['episodes']} |"
            )
    print("===============================")


def write_report(config: RunConfig, runs: list[TestRun], warnings: list[str]) -> None:
    if config.output_json is None:
        return

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config_to_dict(config),
        "warnings": warnings,
        "runs": [
            {
                **asdict(run),
                "metrics": run.metrics,
            }
            for run in runs
        ],
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    with config.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved run report to {config.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniGrid evaluation suites with pipeline-aware defaults."
    )
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_SUITES),
        default="8x8",
        help=(
            "Which training pipeline changed. Selects default suites: "
            "8x8 -> native 8x8 + transfers + goal color; "
            "16x16 -> native 16x16 + transfers + goal color; "
            "all -> both native suites plus transfers and goal color."
        ),
    )
    parser.add_argument(
        "--suites",
        default=None,
        help="Comma-separated suite names overriding --pipeline defaults.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seeds", default=str(DEFAULT_SEED), help="Comma-separated seeds.")
    parser.add_argument(
        "--sft-epochs",
        default=str(DEFAULT_SFT_EPOCHS),
        help="Comma-separated SFT epochs for focal pipeline(s).",
    )
    parser.add_argument(
        "--grpo-episodes",
        default=None,
        help="Comma-separated GRPO episodes for focal pipeline(s). Defaults: 8x8=100, 16x16=75.",
    )
    parser.add_argument(
        "--sft-epochs-8x8",
        default=None,
        help="Override --sft-epochs for 8x8 adapters.",
    )
    parser.add_argument(
        "--sft-epochs-16x16",
        default=None,
        help="Override --sft-epochs for 16x16 adapters.",
    )
    parser.add_argument(
        "--grpo-episodes-8x8",
        default=None,
        help="Override --grpo-episodes for 8x8 adapters.",
    )
    parser.add_argument(
        "--grpo-episodes-16x16",
        default=None,
        help="Override --grpo-episodes for 16x16 adapters.",
    )
    parser.add_argument(
        "--baseline-sft-epoch",
        type=int,
        default=None,
        help="SFT epoch for non-focal transfer baselines (both envs if per-env flags omitted).",
    )
    parser.add_argument("--baseline-sft-epoch-8x8", type=int, default=DEFAULT_SFT_EPOCHS)
    parser.add_argument("--baseline-sft-epoch-16x16", type=int, default=DEFAULT_SFT_EPOCHS)
    parser.add_argument(
        "--baseline-grpo-episode",
        type=int,
        default=None,
        help="GRPO episode for non-focal transfer baselines (both envs if per-env flags omitted).",
    )
    parser.add_argument(
        "--baseline-grpo-episode-8x8",
        type=int,
        default=DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[8],
    )
    parser.add_argument(
        "--baseline-grpo-episode-16x16",
        type=int,
        default=DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[16],
    )
    parser.add_argument("--sft-root-8x8", default=None)
    parser.add_argument("--grpo-root-8x8", default=None)
    parser.add_argument("--sft-root-16x16", default=None)
    parser.add_argument("--grpo-root-16x16", default=None)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Aggregate report path (default: runs/test_report_<timestamp>.json).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory for per-run JSON files from test_models.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining tests after a failure.",
    )
    parser.add_argument("--list-suites", action="store_true", help="List available suites and exit.")
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    if args.list_suites:
        return None

    suites = (
        [item.strip() for item in args.suites.split(",") if item.strip()]
        if args.suites
        else PIPELINE_SUITES[args.pipeline]
    )
    unknown = [suite for suite in suites if suite not in SUITE_DEFINITIONS]
    if unknown:
        raise ValueError(f"Unknown suites: {unknown}. Use --list-suites to see valid names.")

    default_grpo = (
        str(DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[8])
        if args.pipeline == "8x8"
        else str(DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[16])
        if args.pipeline == "16x16"
        else f"{DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[8]},{DEFAULT_GRPO_EVAL_EPISODE_BY_ENV_SIZE[16]}"
    )
    grpo_default = args.grpo_episodes or default_grpo

    sft_epochs_default = parse_int_list(args.sft_epochs)
    grpo_episodes_default = parse_int_list(grpo_default)
    sft_epochs_by_env = {
        8: parse_int_list(args.sft_epochs_8x8) if args.sft_epochs_8x8 else sft_epochs_default,
        16: parse_int_list(args.sft_epochs_16x16) if args.sft_epochs_16x16 else sft_epochs_default,
    }
    grpo_episodes_by_env = {
        8: parse_int_list(args.grpo_episodes_8x8) if args.grpo_episodes_8x8 else grpo_episodes_default,
        16: parse_int_list(args.grpo_episodes_16x16)
        if args.grpo_episodes_16x16
        else grpo_episodes_default,
    }

    baseline_sft_epoch = {
        8: args.baseline_sft_epoch if args.baseline_sft_epoch is not None else args.baseline_sft_epoch_8x8,
        16: args.baseline_sft_epoch if args.baseline_sft_epoch is not None else args.baseline_sft_epoch_16x16,
    }
    baseline_grpo_episode = {
        8: args.baseline_grpo_episode
        if args.baseline_grpo_episode is not None
        else args.baseline_grpo_episode_8x8,
        16: args.baseline_grpo_episode
        if args.baseline_grpo_episode is not None
        else args.baseline_grpo_episode_16x16,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_path(args.run_dir) if args.run_dir else project_path(f"runs/test_run_{timestamp}")
    output_json = (
        project_path(args.output_json)
        if args.output_json
        else project_path(f"runs/test_report_{timestamp}.json")
    )

    adapter_roots = {
        ("sft", 8): project_path(args.sft_root_8x8) if args.sft_root_8x8 else sft_adapter_root(8),
        ("grpo", 8): project_path(args.grpo_root_8x8) if args.grpo_root_8x8 else grpo_adapter_root(8),
        ("sft", 16): project_path(args.sft_root_16x16) if args.sft_root_16x16 else sft_adapter_root(16),
        ("grpo", 16): project_path(args.grpo_root_16x16) if args.grpo_root_16x16 else grpo_adapter_root(16),
    }

    return RunConfig(
        pipeline=args.pipeline,
        suites=suites,
        episodes=args.episodes,
        seeds=parse_int_list(args.seeds),
        sft_epochs_by_env=sft_epochs_by_env,
        grpo_episodes_by_env=grpo_episodes_by_env,
        baseline_sft_epoch=baseline_sft_epoch,
        baseline_grpo_episode=baseline_grpo_episode,
        adapter_roots=adapter_roots,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        output_json=output_json,
        run_dir=run_dir,
    )


def list_suites() -> None:
    print("Available suites:")
    for name, definition in SUITE_DEFINITIONS.items():
        train_env = definition["train_env"]
        train_label = f"{train_env}x{train_env}" if train_env is not None else "pipeline focal env"
        print(
            f"  {name}: eval {definition['eval_env']}x{definition['eval_env']}, "
            f"train {train_label}, goal={definition['goal_color']}, "
            f"prompt={definition['prompt_goal_color']}"
        )
    print("\nPipeline defaults:")
    for pipeline, suites in PIPELINE_SUITES.items():
        print(f"  {pipeline}: {', '.join(suites)}")


def main() -> int:
    args = parse_args()
    if args.list_suites:
        list_suites()
        return 0

    config = build_run_config(args)
    runs, warnings = plan_runs(config)

    print(f"Pipeline: {config.pipeline}")
    print(f"Suites: {', '.join(config.suites)}")
    print(f"Planned runs: {len(runs)}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    if not runs:
        print("Nothing to run.")
        return 1

    runs = execute_runs(config, runs)
    summarize_runs(runs)
    write_report(config, runs, warnings)

    failed = [run for run in runs if run.status == "failed"]
    if failed and not config.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
