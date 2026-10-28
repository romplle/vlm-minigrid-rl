"""Helpers for the 8x8 multi-seed OOD replication (C1).

Replicates the A1 OOD slice (transfer 8->16 and goal-red) across eval seeds
42/123/456 at GRPO episode-100. No new training.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from typing import Any

from vlm_minigrid_rl.env_profiles import profile_for_env_size
from vlm_minigrid_rl.handoff_sweep import eval_json_path, extract_policy_metrics

PRIMARY_EPOCH = 9
PRIMARY_GRPO_EPISODE = 100
DEFAULT_EPOCHS = (1, 5, 9)
DEFAULT_EVAL_SEEDS = (42, 123, 456)
TRAIN_ENV_SIZE = 8

SUITE_TRANSFER = "transfer8to16"
SUITE_GOAL_RED = "goal_red"

SUITES: dict[str, dict[str, Any]] = {
    SUITE_TRANSFER: {
        "id": SUITE_TRANSFER,
        "tag": "transfer8to16",
        "eval_env": 16,
        "goal_color": "green",
        "prompt_goal_color": "green",
    },
    SUITE_GOAL_RED: {
        "id": SUITE_GOAL_RED,
        "tag": "goalred",
        "eval_env": 8,
        "goal_color": "red",
        "prompt_goal_color": "red",
    },
}

VERDICT_HURTS = "HURTS_REPLICATED"
VERDICT_MIXED = "MIXED"
VERDICT_NOT_HURTS = "NOT_HURTS"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def ood_cell_path(run_dir: Path, tag: str, sft_epoch: int, seed: int) -> Path:
    """A1-compatible path: ``<run_dir>/8x8/ood_<tag>_e<E>_seed<S>.json``."""
    return eval_json_path(
        run_dir,
        kind="ood",
        env_size=TRAIN_ENV_SIZE,
        sft_epoch=sft_epoch,
        seed=seed,
        tag=tag,
    )


def suite_max_steps(suite: dict[str, Any]) -> int:
    return profile_for_env_size(int(suite["eval_env"])).eval_max_steps


def payload_config(payload: dict) -> dict:
    config = payload.get("config")
    if isinstance(config, dict):
        return config
    return payload


def payload_env_size(payload: dict) -> int | None:
    config = payload_config(payload)
    if config.get("env_size") is not None:
        return int(config["env_size"])
    if payload.get("env_size") is not None:
        return int(payload["env_size"])
    return None


def _normalized_path(raw: object) -> str:
    return str(raw).replace("\\", "/")


def _path_parts(raw: object) -> list[str]:
    return [part for part in _normalized_path(raw).split("/") if part]


def _has_epoch(path: object, epoch: int) -> bool:
    return f"epoch-{epoch}" in _path_parts(path)


def _has_episode(path: object, episode: int) -> bool:
    return f"episode-{episode}" in _path_parts(path)


def reuse_payload_matches_suite(
    payload: dict,
    *,
    suite: dict[str, Any],
    sft_epoch: int,
    episodes: int = 250,
    grpo_episode: int = PRIMARY_GRPO_EPISODE,
) -> bool:
    """True if an A1 OOD JSON is safe to copy into the C1 run dir."""
    config = payload_config(payload)
    env_size = payload_env_size(payload)
    if env_size != int(suite["eval_env"]):
        return False
    if str(config.get("goal_color") or "").lower() != str(suite["goal_color"]).lower():
        return False
    if str(config.get("prompt_goal_color") or "").lower() != str(
        suite["prompt_goal_color"]
    ).lower():
        return False
    if int(config.get("episodes") or 0) != int(episodes):
        return False
    if not _has_epoch(config.get("sft_adapter_path"), sft_epoch):
        return False
    if not _has_episode(config.get("grpo_adapter_path"), grpo_episode):
        return False
    return True


def mean_sd_sample(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "sd": 0.0, "n": 1}
    return {
        "mean": float(mean(values)),
        "sd": float(stdev(values)),
        "n": len(values),
    }


def build_suite_epoch_row(
    *,
    suite_id: str,
    sft_epoch: int,
    seeds: list[int],
    sft_rates: list[float],
    grpo_rates: list[float],
) -> dict[str, Any]:
    if len(seeds) != len(sft_rates) or len(seeds) != len(grpo_rates):
        raise ValueError("seeds, sft_rates, and grpo_rates must have the same length")
    deltas = [float(g) - float(s) for g, s in zip(grpo_rates, sft_rates)]
    sft_stats = mean_sd_sample([float(v) for v in sft_rates])
    grpo_stats = mean_sd_sample([float(v) for v in grpo_rates])
    delta_stats = mean_sd_sample(deltas)
    by_seed = {
        str(seed): {
            "sft": float(sft),
            "grpo": float(grpo),
            "delta": float(grpo) - float(sft),
        }
        for seed, sft, grpo in zip(seeds, sft_rates, grpo_rates)
    }
    return {
        "suite": suite_id,
        "sft_epoch": int(sft_epoch),
        "sft_mean": sft_stats["mean"],
        "grpo_mean": grpo_stats["mean"],
        "delta_mean": delta_stats["mean"],
        "delta_sd": delta_stats["sd"],
        "sft_rates": [float(v) for v in sft_rates],
        "grpo_rates": [float(v) for v in grpo_rates],
        "n_grpo_worse": sum(1 for delta in deltas if delta < 0),
        "n_seeds": len(seeds),
        "by_seed": by_seed,
    }


def rates_from_ood_payload(payload: dict) -> tuple[float, float]:
    sft = extract_policy_metrics(payload, "SFT")
    grpo = extract_policy_metrics(payload, "GRPO")
    return float(sft["success_rate"]), float(grpo["success_rate"])


def ood_verdict(
    rows: list[dict[str, Any]],
    *,
    primary_epoch: int = PRIMARY_EPOCH,
) -> dict[str, Any]:
    """Pre-registered C1 reading: both e9 suites must have mean Δ < 0."""
    primary = [row for row in rows if int(row["sft_epoch"]) == int(primary_epoch)]
    by_suite = {row["suite"]: row for row in primary}
    transfer = by_suite.get(SUITE_TRANSFER)
    goal = by_suite.get(SUITE_GOAL_RED)
    if transfer is None or goal is None:
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": (
                f"Need both {SUITE_TRANSFER} and {SUITE_GOAL_RED} at epoch {primary_epoch}"
            ),
            "primary_epoch": primary_epoch,
            "e9_transfer_delta": None if transfer is None else transfer["delta_mean"],
            "e9_goal_red_delta": None if goal is None else goal["delta_mean"],
        }

    transfer_hurts = float(transfer["delta_mean"]) < 0.0
    goal_hurts = float(goal["delta_mean"]) < 0.0
    if transfer_hurts and goal_hurts:
        verdict = VERDICT_HURTS
        reason = (
            f"e{primary_epoch} transfer delta={transfer['delta_mean']:+.2f} pp and "
            f"goal-red delta={goal['delta_mean']:+.2f} pp; both mean delta < 0"
        )
    elif (not transfer_hurts) and (not goal_hurts):
        verdict = VERDICT_NOT_HURTS
        reason = (
            f"e{primary_epoch} transfer delta={transfer['delta_mean']:+.2f} pp and "
            f"goal-red delta={goal['delta_mean']:+.2f} pp; neither mean delta < 0"
        )
    else:
        verdict = VERDICT_MIXED
        reason = (
            f"e{primary_epoch} suites disagree: transfer delta="
            f"{transfer['delta_mean']:+.2f} pp, "
            f"goal-red delta={goal['delta_mean']:+.2f} pp"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "primary_epoch": primary_epoch,
        "e9_transfer_delta": float(transfer["delta_mean"]),
        "e9_goal_red_delta": float(goal["delta_mean"]),
        "e9_transfer_n_grpo_worse": int(transfer["n_grpo_worse"]),
        "e9_goal_red_n_grpo_worse": int(goal["n_grpo_worse"]),
    }


def build_summary_payload(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    suites: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite_block = suites.setdefault(row["suite"], {})
        suite_block[str(row["sft_epoch"])] = {
            "sft_mean": row["sft_mean"],
            "grpo_mean": row["grpo_mean"],
            "delta_mean": row["delta_mean"],
            "delta_sd": row["delta_sd"],
            "sft_rates": row["sft_rates"],
            "grpo_rates": row["grpo_rates"],
            "n_grpo_worse": row["n_grpo_worse"],
            "by_seed": row["by_seed"],
        }
    reading = ood_verdict(rows, primary_epoch=int(config.get("primary_epoch", PRIMARY_EPOCH)))
    return {
        "config": config,
        "suites": suites,
        "rows": rows,
        **reading,
    }
