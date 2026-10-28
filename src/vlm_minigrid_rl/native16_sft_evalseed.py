"""Eval-only native Empty 16x16 SFT far across eval seeds. No train, no GRPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_minigrid_rl.c1_geometry import C1_EVAL_16X16
from vlm_minigrid_rl.ood_multiseed import mean_sd_sample

EXPERIMENT_ID = "2026-09-native16-sft-evalseed"
FROZEN_FAR_RUN = "2026-08-c1-far-16x16"
FROZEN_FAR_DIR = f"runs/{FROZEN_FAR_RUN}"
FROZEN_SFT_CELL = f"{FROZEN_FAR_DIR}/c1f_native_native16_SFT_s42.json"
DEFAULT_SFT_ADAPTER = "checkpoints/sft_adapter_16x16_bs32_e10/epoch-8"
REUSE_EVAL_SEED = 42
GPU_EVAL_SEEDS = (123, 456)
SFT_EPOCH = 8
STABILITY_GATE_PP = 3.0
C1_FAR_FLOOR_GRPO = 4.61


def cell_out_path(run_dir: Path, seed: int) -> Path:
    return Path(run_dir) / f"sft_native16_s{int(seed)}.json"


def assert_not_frozen_far_run_dir(path: Path) -> None:
    resolved = Path(path)
    parts = set(resolved.parts)
    if FROZEN_FAR_RUN in parts or resolved.name == FROZEN_FAR_RUN:
        raise ValueError(f"refusing to write into frozen run dir {FROZEN_FAR_RUN}")


def reuse_seed42_row(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    metrics = None
    for row in payload.get("policies") or []:
        if str(row.get("name") or "") == "SFT":
            metrics = row.get("metrics")
            break
    if not isinstance(metrics, dict) or "success_rate" not in metrics:
        raise ValueError("frozen SFT cell is missing SFT metrics")
    far = config.get("success_far")
    n_far = config.get("n_far")
    if far is None or n_far is None:
        raise ValueError("frozen SFT cell is missing success_far / n_far")
    return {
        "seed": REUSE_EVAL_SEED,
        "source": "reuse",
        "success_rate": float(metrics["success_rate"]),
        "success_far": float(far),
        "success_visible": _opt_float(config.get("success_visible")),
        "success_facing_miss": _opt_float(config.get("success_facing_miss")),
        "n_visible": _opt_int(config.get("n_visible")),
        "n_facing_miss": _opt_int(config.get("n_facing_miss")),
        "n_far": int(n_far),
        "eval_max_steps": int(config.get("eval_max_steps") or C1_EVAL_16X16),
        "sft_adapter_path": str(config.get("sft_adapter_path") or ""),
    }


def gpu_row_from_payload(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    metrics = None
    for row in payload.get("policies") or []:
        if str(row.get("name") or "") == "SFT":
            metrics = row.get("metrics")
            break
    strata = payload.get("strata") if isinstance(payload.get("strata"), dict) else {}
    if not isinstance(metrics, dict) or "success_rate" not in metrics:
        raise ValueError(f"SFT cell seed {seed} is missing metrics")
    far = config.get("success_far", strata.get("success_far"))
    n_far = config.get("n_far", strata.get("n_far"))
    if far is None or n_far is None:
        raise ValueError(f"SFT cell seed {seed} is missing success_far / n_far")
    return {
        "seed": int(seed),
        "source": "gpu",
        "success_rate": float(metrics["success_rate"]),
        "success_far": float(far),
        "success_visible": _opt_float(
            config.get("success_visible", strata.get("success_visible"))
        ),
        "success_facing_miss": _opt_float(
            config.get("success_facing_miss", strata.get("success_facing_miss"))
        ),
        "n_visible": _opt_int(config.get("n_visible", strata.get("n_visible"))),
        "n_facing_miss": _opt_int(config.get("n_facing_miss", strata.get("n_facing_miss"))),
        "n_far": int(n_far),
        "eval_max_steps": int(config.get("eval_max_steps") or C1_EVAL_16X16),
        "sft_adapter_path": str(config.get("sft_adapter_path") or ""),
    }


def aggregate_far(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["success_far"]) for row in rows]
    stats = mean_sd_sample(values)
    reuse_far = next(
        (float(row["success_far"]) for row in rows if int(row["seed"]) == REUSE_EVAL_SEED),
        None,
    )
    delta = None
    within_gate = None
    if reuse_far is not None and stats["n"]:
        delta = float(stats["mean"]) - reuse_far
        within_gate = abs(delta) <= STABILITY_GATE_PP
    above_c1_floor = all(float(row["success_far"]) > C1_FAR_FLOOR_GRPO + 10 for row in rows)
    return {
        "seeds": [int(row["seed"]) for row in rows],
        "n": int(stats["n"]),
        "mean": float(stats["mean"]) if stats["n"] else None,
        "sd": float(stats["sd"]) if stats["n"] else None,
        "delta_vs_seed42_pp": delta,
        "within_3pp_of_seed42": within_gate,
        "all_far_above_c1_floor_plus_10": above_c1_floor,
        "rows": rows,
    }


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
