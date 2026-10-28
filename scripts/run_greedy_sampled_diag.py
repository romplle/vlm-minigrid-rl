"""Greedy vs sampled eval diagnostic for Empty 8×8 / 16×16 (memo Q1 / F3).

Compares env success under full-vocab sampled ``generate`` vs 3-way argmax on
``score_action_logits`` (argmax3). Does not change train≡eval defaults.
Default ``--env-size`` is 16 (historical Q1/F3 grid); pass ``--env-size 8``
for the 8×8 decoder suite.
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

from vlm_minigrid_rl.experiment_config import DEFAULT_EVAL_EPISODES, dataset_dir, env_label
from vlm_minigrid_rl.greedy_sampled_diag import (
    MODE_ARGMAX3,
    MODE_GENERATE,
    build_epoch_row,
    build_epoch_row_multiseed,
    build_summary_payload,
    cell_json_name,
)
from vlm_minigrid_rl.handoff_sweep import (
    extract_policy_metrics,
    grpo_episode_dir,
    grpo_handoff_root,
    sft_epoch_dir,
)
from vlm_minigrid_rl.minigrid_utils import default_max_steps
from vlm_minigrid_rl.paths import project_path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODELS_SCRIPT = SCRIPT_DIR / "test_models.py"

ALL_PHASES = ("eval", "summary")
DEFAULT_EPOCHS = "1,3,8"
DEFAULT_SEEDS = "42,123,456"
DEFAULT_MODES = "generate,argmax3"
DEFAULT_RUN_DIR = "runs/2026-08-greedy-sampled-16x16"
DEFAULT_SFT_ROOT = "checkpoints/sft_adapter_16x16_bs32_e10"
DEFAULT_HANDOFF_RUN_DIR = "runs/2026-07-handoff-empty"
DEFAULT_TRAINSEED_RUN_DIR = "runs/2026-07-grpo-trainseed-16x16"
DEFAULT_MATCHED_RUN_DIR = "runs/2026-08-matched-headroom-16x16"
DEFAULT_ENV_SIZE = 16
PRIMARY_GRPO_EPISODE = 100
DEFAULT_TRAIN_SEED = 42
DEFAULT_TRAIN_SEEDS = "42"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_strs(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def check_sft_root_matches_env(sft_root: str | Path, env_size: int) -> None:
    text = str(sft_root).replace("\\", "/")
    other = 16 if env_size == 8 else 8
    needle = f"{other}x{other}"
    if needle in text:
        raise SystemExit(
            f"--sft-root {sft_root} contains {needle} but --env-size={env_size}"
        )


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


def payload_env_size(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("env_size") is not None:
        return int(payload["env_size"])
    config = payload.get("config")
    if isinstance(config, dict) and config.get("env_size") is not None:
        return int(config["env_size"])
    return None


def path_has_other_env(path: Path, env_size: int) -> bool:
    other = env_label(16 if env_size == 8 else 8)
    return other in str(path).replace("\\", "/")


def maybe_reuse_json(src: Path, dst: Path, *, dry_run: bool, env_size: int) -> bool:
    if not src.is_file():
        return False
    if src.resolve() == dst.resolve():
        return False
    if path_has_other_env(src, env_size):
        return False
    try:
        src_env = payload_env_size(load_json(src))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if src_env is not None and src_env != env_size:
        return False
    print(f"reuse {src} -> {dst}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def candidate_reuse_sources(
    *,
    policy: str,
    mode: str,
    epoch: int,
    seed: int,
    train_seed: int,
    env_size: int,
    handoff_dir: Path,
    trainseed_dir: Path,
    matched_dir: Path,
    legacy_greedy_dir: Path,
) -> list[Path]:
    """Existing eval JSON produced with identical adapters/episodes/seed/mode.

    Sampled cells come from the A1, F1 and matched-headroom runs; argmax3 cells
    for training seed 42 come from the earlier single-train-seed greedy run.
    Reusing both means only genuinely new cells hit the GPU.
    """
    env = env_label(env_size)
    if mode == MODE_ARGMAX3:
        if train_seed != DEFAULT_TRAIN_SEED:
            return []
        return [
            legacy_greedy_dir
            / cell_json_name(
                policy, epoch, mode, seed, grpo_episode=PRIMARY_GRPO_EPISODE, train_seed=None
            )
        ]
    if policy == "sft":
        return [
            handoff_dir / env / f"sft_e{epoch}_seed{seed}.json",
            trainseed_dir / f"sft_e{epoch}_evalseed{seed}.json",
            matched_dir / f"sft_e{epoch}_evalseed{seed}.json",
        ]
    stem = f"grpo_e{epoch}_trainseed{train_seed}_ep{PRIMARY_GRPO_EPISODE}_evalseed{seed}.json"
    sources = [trainseed_dir / stem, matched_dir / stem]
    if train_seed == DEFAULT_TRAIN_SEED:
        sources.append(
            handoff_dir / env / f"grpo_e{epoch}_ep{PRIMARY_GRPO_EPISODE}_seed{seed}.json"
        )
    return sources


def build_eval_command(
    *,
    python: str,
    policy: str,
    sft_adapter: Path,
    grpo_adapter: Path | None,
    dataset: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    action_mode: str,
    output_json: Path,
    env_size: int,
) -> list[str]:
    cmd = [
        python,
        str(TEST_MODELS_SCRIPT),
        "--env-size",
        str(env_size),
        "--dataset-path",
        str(dataset),
        "--sft-adapter-path",
        str(sft_adapter),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--action-mode",
        action_mode,
        "--output-json",
        str(output_json),
        "--skip-base",
        "--skip-majority",
        "--skip-expert",
    ]
    if policy == "sft":
        cmd.append("--skip-grpo")
    else:
        if grpo_adapter is None:
            raise ValueError("grpo_adapter required for GRPO eval")
        cmd.extend(["--grpo-adapter-path", str(grpo_adapter), "--skip-sft"])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy vs sampled Empty diagnostic (Q1 / F3). Default --env-size 16."
    )
    parser.add_argument(
        "--env-size",
        type=int,
        choices=(8, 16),
        default=DEFAULT_ENV_SIZE,
        help="Empty grid size. Default 16 (historical F3). Pass 8 for the 8x8 decoder suite.",
    )
    parser.add_argument("--phases", default="eval,summary")
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Eval seeds.")
    parser.add_argument(
        "--train-seeds",
        default=DEFAULT_TRAIN_SEEDS,
        help=(
            "GRPO training seeds to evaluate. A single seed keeps the legacy "
            "single-train-seed filenames; two or more add a trainseed component."
        ),
    )
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--sft-root", default=DEFAULT_SFT_ROOT)
    parser.add_argument(
        "--grpo-root-pattern",
        default=None,
        help="Unused placeholder; roots resolved via grpo_handoff_root(env_size, epoch).",
    )
    parser.add_argument("--handoff-run-dir", default=DEFAULT_HANDOFF_RUN_DIR)
    parser.add_argument("--trainseed-run-dir", default=DEFAULT_TRAINSEED_RUN_DIR)
    parser.add_argument("--matched-run-dir", default=DEFAULT_MATCHED_RUN_DIR)
    parser.add_argument(
        "--legacy-greedy-run-dir",
        default=DEFAULT_RUN_DIR,
        help="Single-train-seed greedy run whose argmax3 cells (train seed 42) can be reused.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-reuse-generate",
        action="store_true",
        help="Disable reuse of existing eval JSON (both sampled and argmax3) and re-run every cell.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cell_name(
    policy: str,
    epoch: int,
    mode: str,
    seed: int,
    *,
    train_seed: int,
    multi_train_seed: bool,
) -> str:
    return cell_json_name(
        policy,
        epoch,
        mode,
        seed,
        grpo_episode=PRIMARY_GRPO_EPISODE,
        train_seed=train_seed if multi_train_seed else None,
    )


def run_eval(args: argparse.Namespace) -> None:
    epochs = parse_csv_ints(args.epochs)
    seeds = parse_csv_ints(args.seeds)
    train_seeds = parse_csv_ints(args.train_seeds)
    multi = len(train_seeds) > 1
    modes = parse_csv_strs(args.modes)
    env_size = int(args.env_size)
    check_sft_root_matches_env(args.sft_root, env_size)
    for mode in modes:
        if mode not in {MODE_GENERATE, MODE_ARGMAX3}:
            raise SystemExit(f"Unknown mode={mode!r}; expected generate or argmax3")

    run_dir = project_path(args.run_dir)
    sft_root = project_path(args.sft_root)
    handoff_dir = project_path(args.handoff_run_dir)
    trainseed_dir = project_path(args.trainseed_run_dir)
    matched_dir = project_path(args.matched_run_dir)
    legacy_greedy_dir = project_path(args.legacy_greedy_run_dir)
    dataset = dataset_dir(env_size)
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps(env_size)
    python = sys.executable

    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epochs:
        sft_adapter = sft_epoch_dir(sft_root, epoch)
        if not sft_adapter.is_dir() and not args.dry_run:
            raise SystemExit(f"Missing SFT adapter for epoch={epoch}: {sft_adapter}")

        # The SFT arm does not depend on the GRPO training seed, so evaluate it once.
        for mode in modes:
            for seed in seeds:
                out = run_dir / cell_name(
                    "sft", epoch, mode, seed, train_seed=train_seeds[0], multi_train_seed=multi
                )
                if args.skip_existing and out.is_file():
                    print(f"skip existing {out}")
                    continue
                if not args.no_reuse_generate and not out.is_file():
                    if any(
                        maybe_reuse_json(src, out, dry_run=args.dry_run, env_size=env_size)
                        for src in candidate_reuse_sources(
                            policy="sft",
                            mode=mode,
                            epoch=epoch,
                            seed=seed,
                            train_seed=train_seeds[0],
                            env_size=env_size,
                            handoff_dir=handoff_dir,
                            trainseed_dir=trainseed_dir,
                            matched_dir=matched_dir,
                            legacy_greedy_dir=legacy_greedy_dir,
                        )
                    ):
                        continue
                cmd = build_eval_command(
                    python=python,
                    policy="sft",
                    sft_adapter=sft_adapter,
                    grpo_adapter=None,
                    dataset=dataset,
                    episodes=args.episodes,
                    seed=seed,
                    max_steps=max_steps,
                    action_mode=mode,
                    output_json=out,
                    env_size=env_size,
                )
                code = run_command(cmd, dry_run=args.dry_run)
                if code != 0:
                    raise SystemExit(f"Eval failed ({code}): {' '.join(cmd)}")

        for train_seed in train_seeds:
            grpo_root = project_path(
                grpo_handoff_root(env_size, epoch, train_seed=train_seed)
            )
            grpo_adapter = grpo_episode_dir(grpo_root, PRIMARY_GRPO_EPISODE)
            if not (grpo_adapter / "adapter_config.json").is_file() and not args.dry_run:
                raise SystemExit(
                    f"Missing GRPO adapter for epoch={epoch} train_seed={train_seed}: "
                    f"{grpo_adapter}"
                )

            for mode in modes:
                for seed in seeds:
                    out = run_dir / cell_name(
                        "grpo", epoch, mode, seed, train_seed=train_seed, multi_train_seed=multi
                    )
                    if args.skip_existing and out.is_file():
                        print(f"skip existing {out}")
                        continue

                    if not args.no_reuse_generate and not out.is_file():
                        if any(
                            maybe_reuse_json(src, out, dry_run=args.dry_run, env_size=env_size)
                            for src in candidate_reuse_sources(
                                policy="grpo",
                                mode=mode,
                                epoch=epoch,
                                seed=seed,
                                train_seed=train_seed,
                                env_size=env_size,
                                handoff_dir=handoff_dir,
                                trainseed_dir=trainseed_dir,
                                matched_dir=matched_dir,
                                legacy_greedy_dir=legacy_greedy_dir,
                            )
                        ):
                            continue

                    cmd = build_eval_command(
                        python=python,
                        policy="grpo",
                        sft_adapter=sft_adapter,
                        grpo_adapter=grpo_adapter,
                        dataset=dataset,
                        episodes=args.episodes,
                        seed=seed,
                        max_steps=max_steps,
                        action_mode=mode,
                        output_json=out,
                        env_size=env_size,
                    )
                    code = run_command(cmd, dry_run=args.dry_run)
                    if code != 0:
                        raise SystemExit(f"Eval failed ({code}): {' '.join(cmd)}")


def success_rate_from_cell(path: Path, policy: str) -> float:
    payload = load_json(path)
    name = "SFT" if policy == "sft" else "GRPO"
    metrics = extract_policy_metrics(payload, name)
    return float(metrics["success_rate"])


def run_summary(args: argparse.Namespace) -> None:
    epochs = parse_csv_ints(args.epochs)
    seeds = parse_csv_ints(args.seeds)
    train_seeds = parse_csv_ints(args.train_seeds)
    multi = len(train_seeds) > 1
    env_size = int(args.env_size)
    check_sft_root_matches_env(args.sft_root, env_size)
    run_dir = project_path(args.run_dir)

    epoch_rows = []
    missing: list[str] = []
    for epoch in epochs:
        sft: dict[str, list[float]] = {"sampled": [], "greedy": []}
        grpo: dict[int, dict[str, list[float]]] = {
            ts: {"sampled": [], "greedy": []} for ts in train_seeds
        }

        for seed in seeds:
            for mode, key in ((MODE_GENERATE, "sampled"), (MODE_ARGMAX3, "greedy")):
                path = run_dir / cell_name(
                    "sft", epoch, mode, seed, train_seed=train_seeds[0], multi_train_seed=multi
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue
                sft[key].append(success_rate_from_cell(path, "sft"))

            for train_seed in train_seeds:
                for mode, key in ((MODE_GENERATE, "sampled"), (MODE_ARGMAX3, "greedy")):
                    path = run_dir / cell_name(
                        "grpo", epoch, mode, seed, train_seed=train_seed, multi_train_seed=multi
                    )
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    grpo[train_seed][key].append(success_rate_from_cell(path, "grpo"))

        complete = all(len(sft[k]) == len(seeds) for k in sft) and all(
            len(grpo[ts][k]) == len(seeds) for ts in train_seeds for k in ("sampled", "greedy")
        )
        if not complete:
            continue

        if multi:
            epoch_rows.append(
                build_epoch_row_multiseed(
                    epoch,
                    sft_sampled=sft["sampled"],
                    sft_greedy=sft["greedy"],
                    grpo_by_train_seed=grpo,
                )
            )
        else:
            epoch_rows.append(
                build_epoch_row(
                    epoch,
                    sft_sampled=sft["sampled"],
                    sft_greedy=sft["greedy"],
                    grpo_sampled=grpo[train_seeds[0]]["sampled"],
                    grpo_greedy=grpo[train_seeds[0]]["greedy"],
                )
            )

    if missing and not epoch_rows:
        raise SystemExit(f"No complete epoch rows; missing e.g. {missing[0]}")
    if missing:
        print(f"warning: {len(missing)} missing cells; summarizing complete epochs only")

    payload = build_summary_payload(
        epoch_rows,
        config={
            "env_size": env_size,
            "epochs": epochs,
            "seeds": seeds,
            "train_seeds": train_seeds,
            "modes": parse_csv_strs(args.modes),
            "episodes": args.episodes,
            "max_steps": args.max_steps
            if args.max_steps is not None
            else default_max_steps(env_size),
            "sft_root": str(project_path(args.sft_root)),
            "grpo_episode": PRIMARY_GRPO_EPISODE,
            "train_seed": train_seeds[0] if not multi else None,
            "run_dir": str(run_dir),
            "missing_cells": missing,
        },
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    write_json(run_dir / "summary.json", payload)
    f3 = payload["f3"]
    print(
        f"F3 verdict={f3['verdict']} "
        f"slope_sampled={f3['slope_sampled']} slope_greedy={f3['slope_greedy']}"
    )


def main() -> None:
    args = parse_args()
    phases = parse_phases(args.phases)
    if "eval" in phases:
        run_eval(args)
    if "summary" in phases:
        run_summary(args)


if __name__ == "__main__":
    main()
