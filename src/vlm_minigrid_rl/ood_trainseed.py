"""e9 OOD replication across GRPO train seeds (C1 eval caps 12/40)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_minigrid_rl.handoff_sweep import DEFAULT_GRPO_TRAIN_SEED, eval_json_path
from vlm_minigrid_rl.ood_multiseed import (
    PRIMARY_EPOCH,
    PRIMARY_GRPO_EPISODE,
    SUITE_GOAL_RED,
    SUITE_TRANSFER,
    SUITES,
    TRAIN_ENV_SIZE,
    VERDICT_HURTS,
    VERDICT_INCONCLUSIVE,
    VERDICT_MIXED,
    VERDICT_NOT_HURTS,
    _path_parts,
    mean_sd_sample,
    payload_config,
    reuse_payload_matches_suite,
)

C1_TRANSFER_MAX_STEPS = 40
C1_GOAL_RED_MAX_STEPS = 12
C1_SUITE_MAX_STEPS = {
    SUITE_TRANSFER: C1_TRANSFER_MAX_STEPS,
    SUITE_GOAL_RED: C1_GOAL_RED_MAX_STEPS,
}

REUSE_TRAIN_SEED = DEFAULT_GRPO_TRAIN_SEED
ALLOWED_SFT_EPOCHS = (1, 5, 9)
FROZEN_E9_RUN_NAME = "2026-09-ood-trainseed-e9-8x8"
FROZEN_C1_RUN_NAME = "2026-08-ood-multiseed-8x8"


def experiment_id_for_epoch(sft_epoch: int) -> str:
    if int(sft_epoch) == PRIMARY_EPOCH:
        return FROZEN_E9_RUN_NAME
    return f"2026-09-ood-trainseed-e{int(sft_epoch)}-8x8"


def assert_run_dir_allowed(run_dir: Path, sft_epoch: int) -> None:
    names = {part.lower() for part in Path(run_dir).parts}
    if FROZEN_C1_RUN_NAME.lower() in names:
        raise ValueError(f"refusing to write into frozen C1 run dir {run_dir}")
    if int(sft_epoch) != PRIMARY_EPOCH and FROZEN_E9_RUN_NAME.lower() in names:
        raise ValueError(
            f"refusing to write e{int(sft_epoch)} cells into frozen e9 run dir {run_dir}"
        )


def suite_c1_max_steps(suite: dict[str, Any]) -> int:
    suite_id = str(suite.get("id") or "")
    if suite_id not in C1_SUITE_MAX_STEPS:
        raise KeyError(f"No C1 max_steps for suite {suite_id!r}")
    return int(C1_SUITE_MAX_STEPS[suite_id])


def trainseed_cell_path(
    run_dir: Path,
    train_seed: int,
    tag: str,
    eval_seed: int,
    sft_epoch: int = PRIMARY_EPOCH,
) -> Path:
    nested = Path(run_dir) / f"train{int(train_seed)}"
    return eval_json_path(
        nested,
        kind="ood",
        env_size=TRAIN_ENV_SIZE,
        sft_epoch=sft_epoch,
        seed=eval_seed,
        tag=tag,
    )


def handoff_dir_name(sft_epoch: int, train_seed: int) -> str:
    base = (
        f"grpo_adapter_{TRAIN_ENV_SIZE}x{TRAIN_ENV_SIZE}_handoff_from_e10_e{int(sft_epoch)}"
    )
    if int(train_seed) != REUSE_TRAIN_SEED:
        base = f"{base}_seed{int(train_seed)}"
    return base


def grpo_path_matches_train_seed(
    path: object,
    train_seed: int,
    sft_epoch: int = PRIMARY_EPOCH,
) -> bool:
    return handoff_dir_name(int(sft_epoch), int(train_seed)) in _path_parts(path)


def c1_reuse_matches(
    payload: dict,
    *,
    suite: dict[str, Any],
    eval_seed: int,
    episodes: int = 250,
    sft_epoch: int = PRIMARY_EPOCH,
) -> bool:
    if not reuse_payload_matches_suite(
        payload, suite=suite, sft_epoch=sft_epoch, episodes=episodes
    ):
        return False
    config = payload_config(payload)
    if int(config.get("max_steps") or 0) != suite_c1_max_steps(suite):
        return False
    if int(config.get("seed") or -1) != int(eval_seed):
        return False
    return grpo_path_matches_train_seed(
        config.get("grpo_adapter_path"),
        REUSE_TRAIN_SEED,
        sft_epoch=sft_epoch,
    )


def adapter_bundle_complete(adapter_dir: Path) -> bool:
    root = Path(adapter_dir)
    return (root / "adapter_config.json").is_file() and (
        root / "adapter_model.safetensors"
    ).is_file()


def trainseed_ood_verdict(
    *,
    transfer_deltas: dict[int, float],
    goal_deltas: dict[int, float],
) -> dict[str, Any]:
    t_seeds = sorted(int(s) for s in transfer_deltas)
    g_seeds = sorted(int(s) for s in goal_deltas)
    if not t_seeds or t_seeds != g_seeds:
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "Need matching train-seed keys on both OOD suites",
            "transfer_n_train_negative": 0,
            "goal_n_train_negative": 0,
        }
    t_vals = [float(transfer_deltas[s]) for s in t_seeds]
    g_vals = [float(goal_deltas[s]) for s in g_seeds]
    t_neg = sum(1 for v in t_vals if v < 0.0)
    g_neg = sum(1 for v in g_vals if v < 0.0)
    t_all = t_neg == len(t_vals)
    g_all = g_neg == len(g_vals)
    t_mean = mean_sd_sample(t_vals)["mean"]
    g_mean = mean_sd_sample(g_vals)["mean"]
    if t_all and g_all:
        verdict = VERDICT_HURTS
        reason = (
            f"all {len(t_seeds)} train seeds have mean delta<0 on transfer "
            f"({t_mean:+.2f} pp) and goal-red ({g_mean:+.2f} pp)"
        )
    elif t_mean >= 0.0 and g_mean >= 0.0:
        verdict = VERDICT_NOT_HURTS
        reason = (
            f"across-train mean delta>=0 on both suites: transfer {t_mean:+.2f} pp, "
            f"goal-red {g_mean:+.2f} pp"
        )
    else:
        verdict = VERDICT_MIXED
        reason = (
            f"train-seed sign split: transfer mean {t_mean:+.2f} pp "
            f"({t_neg}/{len(t_vals)} negative), goal-red mean {g_mean:+.2f} pp "
            f"({g_neg}/{len(g_vals)} negative)"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "transfer_n_train_negative": t_neg,
        "goal_n_train_negative": g_neg,
        "n_train_seeds": len(t_seeds),
        "transfer_delta_mean": t_mean,
        "goal_delta_mean": g_mean,
    }


def build_trainseed_summary_payload(
    by_train_seed: dict[str, dict[str, dict[str, Any]]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    transfer_deltas: dict[int, float] = {}
    goal_deltas: dict[int, float] = {}
    across: dict[str, dict[str, Any]] = {}
    for train_key, suites in by_train_seed.items():
        train_seed = int(train_key)
        transfer = suites.get(SUITE_TRANSFER)
        goal = suites.get(SUITE_GOAL_RED)
        if transfer is not None:
            transfer_deltas[train_seed] = float(transfer["delta_mean"])
        if goal is not None:
            goal_deltas[train_seed] = float(goal["delta_mean"])

    for suite_id in (SUITE_TRANSFER, SUITE_GOAL_RED):
        deltas = [
            float(by_train_seed[str(seed)][suite_id]["delta_mean"])
            for seed in sorted(int(k) for k in by_train_seed)
            if suite_id in by_train_seed[str(seed)]
        ]
        stats = mean_sd_sample(deltas)
        across[suite_id] = {
            **stats,
            "delta_mean": stats["mean"],
            "delta_sd": stats["sd"],
            "n_negative": sum(1 for value in deltas if value < 0.0),
            "deltas": deltas,
        }

    reading = trainseed_ood_verdict(
        transfer_deltas=transfer_deltas,
        goal_deltas=goal_deltas,
    )
    return {
        "config": config,
        "by_train_seed": by_train_seed,
        "across_train_seed": across,
        "primary_epoch": int(config.get("sft_epoch", PRIMARY_EPOCH)),
        "grpo_episode": PRIMARY_GRPO_EPISODE,
        **reading,
    }
