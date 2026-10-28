"""Paired 95% interval for native 16x16 far: mean GRPO minus SFT on the same starts.

Seed 42 uses frozen episode bits. Eval seeds 123 and 456 pair a new GRPO draw
with the SFT bits already stored. No train. Cap stays 40.
"""

from __future__ import annotations

import random
from statistics import mean
from typing import Any

from vlm_minigrid_rl.c1_geometry import STRATUM_FAR, episode_stratum
from vlm_minigrid_rl.native16_grpo_evalseed import (
    FROZEN_GRPO_CELLS,
    TRAIN_SEEDS,
    assert_not_frozen_far_run_dir,
)
from vlm_minigrid_rl.native16_sft_evalseed import FROZEN_SFT_CELL


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(float(mean(sample)))
    boot_means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return (boot_means[lo_idx], boot_means[hi_idx])

EXPERIMENT_ID = "2026-09-native16-far-paired-ci"
PRIOR_EXPERIMENT = "2026-09-native16-grpo-evalseed"
N_EPISODES = 250
N_BOOT = 10_000
BOOT_SEED = 0

SFT_EPISODE_CELLS = {
    42: FROZEN_SFT_CELL,
    123: "runs/2026-09-native16-sft-evalseed/sft_native16_s123.json",
    456: "runs/2026-09-native16-sft-evalseed/sft_native16_s456.json",
}
FROZEN_GRPO_42 = FROZEN_GRPO_CELLS

PHRASE_HOLDS = "ENTIRELY_ABOVE"
PHRASE_COVERS_ABOVE = "COVERS_ZERO_POINT_ABOVE"
PHRASE_COVERS = "COVERS_ZERO"
PHRASE_BELOW = "ENTIRELY_BELOW"
PHRASE_INVALID = "INVALID_ALIGN"

PHRASE_RU = {
    PHRASE_HOLDS: "целиком выше нуля",
    PHRASE_COVERS_ABOVE: "точечно выше, интервал накрывает ноль",
    PHRASE_COVERS: "интервал накрывает ноль",
    PHRASE_BELOW: "целиком ниже нуля",
}

PUBLISHED_POINT_DELTA = {42: -7.98, 123: -2.31, 456: 2.80}
COMPACT_KEYS = ("index", "success", "stratum", "start_pos", "start_dir")


class InvalidAlign(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"{PHRASE_INVALID}: {detail}")


def assert_writable_run_dir(path: Any) -> None:
    resolved_parts = set(getattr(path, "parts", ()))
    name = getattr(path, "name", "")
    assert_not_frozen_far_run_dir(path)
    if PRIOR_EXPERIMENT in resolved_parts or name == PRIOR_EXPERIMENT:
        raise ValueError(f"refusing to overwrite {PRIOR_EXPERIMENT}")


def compact_episode_bits(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bits: list[dict[str, Any]] = []
    for index, row in enumerate(episode_rows):
        bits.append(
            {
                "index": index,
                "success": bool(row["success"]),
                "stratum": row_stratum(row),
                "start_pos": [int(value) for value in row["start_pos"]],
                "start_dir": int(row["start_dir"]),
            }
        )
    return bits


def row_stratum(row: dict[str, Any]) -> str:
    if "start_goal_visible" in row and "start_visible_any_dir" in row:
        return episode_stratum(row)
    if row.get("stratum"):
        return str(row["stratum"])
    raise InvalidAlign("episode row has no stratum")


def assert_seed42_aligned(
    sft_episodes: list[dict[str, Any]],
    grpo_episodes_by_train: dict[int, list[dict[str, Any]]],
) -> None:
    _require_train_seeds(grpo_episodes_by_train)
    reference = _stratum_sequence(sft_episodes, label="sft")
    for train_seed in TRAIN_SEEDS:
        other = _stratum_sequence(grpo_episodes_by_train[train_seed], label=f"grpo {train_seed}")
        if other != reference:
            raise InvalidAlign(f"train {train_seed} stratum sequence differs from SFT")


def assert_gpu_aligned(
    sft_episodes: list[dict[str, Any]],
    grpo_bits_by_train: dict[int, list[dict[str, Any]]],
) -> None:
    _require_train_seeds(grpo_bits_by_train)
    if len(sft_episodes) != N_EPISODES:
        raise InvalidAlign(f"SFT episode count {len(sft_episodes)} != {N_EPISODES}")
    for train_seed in TRAIN_SEEDS:
        bits = grpo_bits_by_train[train_seed]
        if len(bits) != N_EPISODES:
            raise InvalidAlign(f"train {train_seed} bit count {len(bits)} != {N_EPISODES}")
        for index, (sft, bit) in enumerate(zip(sft_episodes, bits)):
            if int(bit["index"]) != index:
                raise InvalidAlign(f"train {train_seed} bit index {bit['index']} != {index}")
            if row_stratum(sft) != str(bit["stratum"]):
                raise InvalidAlign(f"train {train_seed} stratum mismatch at episode {index}")
            sft_pos = [int(value) for value in sft["start_pos"]]
            bit_pos = [int(value) for value in bit["start_pos"]]
            if sft_pos != bit_pos or int(sft["start_dir"]) != int(bit["start_dir"]):
                raise InvalidAlign(f"train {train_seed} start mismatch at episode {index}")


def paired_far_deltas_pp(
    sft_rows: list[dict[str, Any]],
    grpo_rows_by_train: dict[int, list[dict[str, Any]]],
) -> list[float]:
    _require_train_seeds(grpo_rows_by_train)
    deltas: list[float] = []
    for index, sft in enumerate(sft_rows):
        if row_stratum(sft) != STRATUM_FAR:
            continue
        grpo_bits: list[float] = []
        for train_seed in TRAIN_SEEDS:
            row = grpo_rows_by_train[train_seed][index]
            if _stored_stratum(row) != STRATUM_FAR:
                raise InvalidAlign(f"train {train_seed} far mismatch at episode {index}")
            grpo_bits.append(1.0 if row["success"] else 0.0)
        sft_bit = 1.0 if sft["success"] else 0.0
        deltas.append(100.0 * (sum(grpo_bits) / 3.0 - sft_bit))
    if not deltas:
        raise InvalidAlign("no far episodes")
    return deltas


def classify_interval(point: float, lo: float, hi: float) -> str:
    if float(lo) > 0.0:
        return PHRASE_HOLDS
    if float(hi) < 0.0:
        return PHRASE_BELOW
    if float(point) > 0.0:
        return PHRASE_COVERS_ABOVE
    return PHRASE_COVERS


def interval_from_deltas(deltas: list[float]) -> dict[str, Any]:
    point = float(sum(deltas) / len(deltas))
    lo, hi = bootstrap_mean_ci(deltas, n_boot=N_BOOT, seed=BOOT_SEED)
    phrase = classify_interval(point, lo, hi)
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_far": len(deltas),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "phrase": phrase,
        "phrase_ru": PHRASE_RU[phrase],
    }


def seed42_interval(
    sft_episodes: list[dict[str, Any]],
    grpo_episodes_by_train: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    assert_seed42_aligned(sft_episodes, grpo_episodes_by_train)
    block = interval_from_deltas(paired_far_deltas_pp(sft_episodes, grpo_episodes_by_train))
    block["source"] = "frozen_bits"
    block["eval_seed"] = 42
    return block


def gpu_interval(
    sft_episodes: list[dict[str, Any]],
    grpo_bits_by_train: dict[int, list[dict[str, Any]]],
    *,
    eval_seed: int,
) -> dict[str, Any]:
    assert_gpu_aligned(sft_episodes, grpo_bits_by_train)
    block = interval_from_deltas(paired_far_deltas_pp(sft_episodes, grpo_bits_by_train))
    block["source"] = "new_grpo_stored_sft"
    block["eval_seed"] = int(eval_seed)
    return block


def _require_train_seeds(rows_by_train: dict[int, list[dict[str, Any]]]) -> None:
    present = sorted(int(seed) for seed in rows_by_train)
    if present != sorted(TRAIN_SEEDS):
        raise InvalidAlign(f"train seeds {present} != {list(TRAIN_SEEDS)}")


def _stratum_sequence(rows: list[dict[str, Any]], *, label: str) -> list[str]:
    if len(rows) != N_EPISODES:
        raise InvalidAlign(f"{label} episode count {len(rows)} != {N_EPISODES}")
    return [row_stratum(row) for row in rows]


def _stored_stratum(row: dict[str, Any]) -> str:
    if row.get("stratum"):
        return str(row["stratum"])
    return row_stratum(row)
