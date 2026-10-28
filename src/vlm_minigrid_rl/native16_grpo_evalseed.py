"""Eval-only native Empty 16x16 GRPO far on eval seeds 123/456.

Seed 42 is reused from frozen C1 far JSON. No train. Cap stays 40.
Paired delta is mean GRPO far minus SFT far on the same eval seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_minigrid_rl.c1_geometry import C1_EVAL_16X16
from vlm_minigrid_rl.c1_search import METHOD_G1

EXPERIMENT_ID = "2026-09-native16-grpo-evalseed"
FROZEN_FAR_RUN = "2026-08-c1-far-16x16"
FROZEN_FAR_DIR = f"runs/{FROZEN_FAR_RUN}"
SFT_SUMMARY = "runs/2026-09-native16-sft-evalseed/summary.json"
DEFAULT_SFT_ADAPTER = "checkpoints/sft_adapter_16x16_bs32_e10/epoch-8"
SFT_EPOCH = 8
GRPO_EPISODE = 100

TRAIN_SEEDS = (42, 123, 456)
EVAL_SEEDS = (42, 123, 456)
REUSE_EVAL_SEED = 42
GPU_EVAL_SEEDS = (123, 456)

GRPO_ADAPTERS = {
    42: "checkpoints/grpo_adapter_16x16_handoff_from_e10_e8/episode-100",
    123: "checkpoints/grpo_adapter_16x16_handoff_from_e10_e8_seed123/episode-100",
    456: "checkpoints/grpo_adapter_16x16_handoff_from_e10_e8_seed456/episode-100",
}
FROZEN_GRPO_CELLS = {
    42: f"{FROZEN_FAR_DIR}/c1f_native_native16_GRPO42_s42.json",
    123: f"{FROZEN_FAR_DIR}/c1f_native_native16_GRPO123_s42.json",
    456: f"{FROZEN_FAR_DIR}/c1f_native_native16_GRPO456_s42.json",
}

REGISTERED_GRPO_FAR_MEAN = 73.71
REUSE_TOL_PP = 0.05
C1_FAR_FLOOR = 4.61
FAR_CELL_FLOOR = C1_FAR_FLOOR + 10.0
TRANSFER_HOLE_PP = -21.0

VERDICT_INVALID = "INVALID_REUSE"
VERDICT_SOFT = "SOFT_MINUS_HOLDS"
VERDICT_SIGN = "SIGN_DIFFERS"
VERDICT_TRANSFER = "TRANSFER_SCALE"
VERDICT_FLOOR = "FLOOR_BREACH"


def grpo_adapter_for_train_seed(train_seed: int) -> str:
    try:
        return GRPO_ADAPTERS[int(train_seed)]
    except KeyError as exc:
        raise ValueError(f"train seed must be one of {TRAIN_SEEDS}, got {train_seed}") from exc


def cell_out_path(run_dir: Path, train_seed: int, eval_seed: int) -> Path:
    name = f"grpo_t{int(train_seed)}_s{int(eval_seed)}.json"
    return Path(run_dir) / name


def planned_gpu_cells() -> list[tuple[int, int]]:
    return [
        (int(train_seed), int(eval_seed))
        for train_seed in TRAIN_SEEDS
        for eval_seed in GPU_EVAL_SEEDS
    ]


def assert_not_frozen_far_run_dir(path: Path) -> None:
    resolved = Path(path)
    parts = set(resolved.parts)
    if FROZEN_FAR_RUN in parts or resolved.name == FROZEN_FAR_RUN:
        raise ValueError(f"refusing to write into frozen run dir {FROZEN_FAR_RUN}")


def reuse_row_from_payload(payload: dict[str, Any], train_seed: int) -> dict[str, Any]:
    return _row_from_payload(payload, train_seed=train_seed, eval_seed=REUSE_EVAL_SEED, source="reuse")


def gpu_row_from_payload(payload: dict[str, Any], train_seed: int, eval_seed: int) -> dict[str, Any]:
    return _row_from_payload(payload, train_seed=train_seed, eval_seed=eval_seed, source="gpu")


def sft_by_eval_seed(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    block = summary.get("far_three_eval_seed")
    if not isinstance(block, dict):
        raise ValueError("SFT summary is missing far_three_eval_seed")
    rows = block.get("rows")
    if not isinstance(rows, list):
        raise ValueError("SFT summary is missing far rows")
    by_seed = {int(row["seed"]): row for row in rows}
    if sorted(by_seed) != sorted(EVAL_SEEDS):
        raise ValueError(f"SFT far seeds {sorted(by_seed)} != {list(EVAL_SEEDS)}")
    return by_seed


def paired_deltas(
    grpo_rows: list[dict[str, Any]],
    sft_rows: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for eval_seed in EVAL_SEEDS:
        cells = [row for row in grpo_rows if int(row["eval_seed"]) == int(eval_seed)]
        trains = sorted(int(row["train_seed"]) for row in cells)
        if trains != sorted(TRAIN_SEEDS):
            raise ValueError(f"eval seed {eval_seed} train seeds {trains} != {list(TRAIN_SEEDS)}")
        grpo_mean = _mean([float(row["success_far"]) for row in cells])
        sft_far = float(sft_rows[int(eval_seed)]["success_far"])
        by_train = {
            int(row["train_seed"]): float(row["success_far"])
            for row in sorted(cells, key=lambda row: int(row["train_seed"]))
        }
        out[int(eval_seed)] = {
            "grpo_mean": grpo_mean,
            "sft_far": sft_far,
            "delta": grpo_mean - sft_far,
            "grpo_far_by_train": by_train,
        }
    return out


def decide_verdict(
    *,
    grpo_rows: list[dict[str, Any]],
    sft_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    _require_cells(grpo_rows)
    deltas = paired_deltas(grpo_rows, sft_rows)
    reuse_fars = [
        float(row["success_far"])
        for row in grpo_rows
        if int(row["eval_seed"]) == REUSE_EVAL_SEED
    ]
    reuse_mean = _mean(reuse_fars)
    reuse_ok = _within_pp(reuse_mean, REGISTERED_GRPO_FAR_MEAN, REUSE_TOL_PP)
    floor_cells = [
        {
            "train_seed": int(row["train_seed"]),
            "eval_seed": int(row["eval_seed"]),
            "success_far": float(row["success_far"]),
        }
        for row in grpo_rows
        if float(row["success_far"]) <= FAR_CELL_FLOOR + 1e-9
    ]
    delta_values = {seed: float(block["delta"]) for seed, block in deltas.items()}
    if not reuse_ok:
        verdict = VERDICT_INVALID
    elif floor_cells:
        verdict = VERDICT_FLOOR
    elif any(delta >= -1e-9 for delta in delta_values.values()):
        verdict = VERDICT_SIGN
    elif any(delta <= TRANSFER_HOLE_PP + 1e-6 for delta in delta_values.values()):
        verdict = VERDICT_TRANSFER
    else:
        verdict = VERDICT_SOFT
    return {
        "verdict": verdict,
        "reuse_ok": reuse_ok,
        "reuse_grpo_far_mean": reuse_mean,
        "registered_grpo_far_mean": REGISTERED_GRPO_FAR_MEAN,
        "reuse_tol_pp": REUSE_TOL_PP,
        "far_cell_floor": FAR_CELL_FLOOR,
        "transfer_hole_pp": TRANSFER_HOLE_PP,
        "floor_cells": floor_cells,
        "paired_deltas": {str(seed): block for seed, block in deltas.items()},
        "note": (
            "Delta is mean GRPO far minus SFT far on the same eval seed. "
            "Do not average the three deltas into one number. "
            "Do not subtract 73.71 from the SFT eval-seed mean 77.01."
        ),
    }


def facing_miss_by_eval_seed(grpo_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reported beside the far verdict. Not an input to the verdict."""
    out: dict[str, Any] = {}
    for eval_seed in EVAL_SEEDS:
        cells = [row for row in grpo_rows if int(row["eval_seed"]) == int(eval_seed)]
        rates = [row.get("success_facing_miss") for row in cells]
        defined = [float(rate) for rate in rates if rate is not None]
        out[str(eval_seed)] = {
            "mean": None if not defined else _mean(defined),
            "by_train": {
                int(row["train_seed"]): _opt_float(row.get("success_facing_miss"))
                for row in cells
            },
        }
    return out


def _row_from_payload(
    payload: dict[str, Any],
    *,
    train_seed: int,
    eval_seed: int,
    source: str,
) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    strata = payload.get("strata") if isinstance(payload.get("strata"), dict) else {}
    if config.get("method") not in (None, METHOD_G1):
        raise ValueError("native GRPO cells must be G1")
    cap = int(config.get("eval_max_steps") or config.get("max_steps") or 0)
    if cap != C1_EVAL_16X16:
        raise ValueError(f"eval cap must be {C1_EVAL_16X16}")
    if int(config.get("seed") or -1) != int(eval_seed):
        raise ValueError(f"cell seed {config.get('seed')} != {eval_seed}")
    metrics = None
    for row in payload.get("policies") or []:
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
            metrics = row["metrics"]
            break
    if metrics is None or "success_rate" not in metrics:
        raise ValueError("native GRPO cell is missing success_rate")
    far = config.get("success_far", strata.get("success_far"))
    n_far = config.get("n_far", strata.get("n_far"))
    if far is None or n_far is None:
        raise ValueError("native GRPO cell is missing success_far / n_far")
    return {
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "source": source,
        "success_rate": float(metrics["success_rate"]),
        "success_far": float(far),
        "success_visible": _opt_float(config.get("success_visible", strata.get("success_visible"))),
        "success_facing_miss": _opt_float(
            config.get("success_facing_miss", strata.get("success_facing_miss"))
        ),
        "n_visible": _opt_int(config.get("n_visible", strata.get("n_visible"))),
        "n_facing_miss": _opt_int(config.get("n_facing_miss", strata.get("n_facing_miss"))),
        "n_far": int(n_far),
        "eval_max_steps": C1_EVAL_16X16,
    }


def _require_cells(rows: list[dict[str, Any]]) -> None:
    present = {(int(row["train_seed"]), int(row["eval_seed"])) for row in rows}
    expected = {(train, eval_seed) for train in TRAIN_SEEDS for eval_seed in EVAL_SEEDS}
    if present != expected:
        missing = sorted(expected - present)
        raise ValueError(f"GRPO cells missing {missing}")


def _within_pp(value: float, target: float, tol: float) -> bool:
    return abs(float(value) - float(target)) <= float(tol) + 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return float(sum(values) / len(values))


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
