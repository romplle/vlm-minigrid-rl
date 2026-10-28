"""Eval-only SFT goal-red success by start stratum. No train, no PAINT.

Caps stay C1: Empty 8x8 eval 12. Table 3.2 aggregates are not replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_minigrid_rl.c1_goalred import (
    C1_EVAL_8X8,
    C1_GRPO_EPISODE,
    C1_GRPO_GOALRED_MEAN,
    C1_SANITY_TOL_PP,
    C1_SFT_GOALRED_MEAN,
    METHOD_G1,
)
from vlm_minigrid_rl.ood_multiseed import mean_sd_sample

EXPERIMENT_ID = "2026-09-sft-goalred-visible"
FROZEN_GOALRED_RUN = "2026-08-c1-goalred-8x8"
FROZEN_OOD_RUN = "2026-08-ood-multiseed-8x8"
FROZEN_RUNS = (FROZEN_GOALRED_RUN, FROZEN_OOD_RUN)

SFT_ROOT = "checkpoints/sft_adapter_8x8_bs32_e10"
SFT_EPOCHS = (1, 5, 9)
PRIMARY_EPOCHS = (5, 9)
EVAL_SEEDS = (42, 123, 456)
GRPO_ADAPTER = (
    "checkpoints/grpo_adapter_8x8_handoff_from_e10_e9/episode-100"
)

VISIBLE_DROP_PP = 10.0
VISIBLE_FLAT_PP = 3.0
CONTRIB_FLOOR_PP = 10.0

VERDICT_INVALID = "INVALID_SUITE"
VERDICT_VISIBLE = "VISIBLE_CARRIES"
VERDICT_OTHER = "OTHER_STRATUM_CARRIES"
VERDICT_UNLOCALIZED = "UNLOCALIZED"

_STRATA = (
    ("visible", "success_visible", "n_visible"),
    ("facing_miss", "success_facing_miss", "n_facing_miss"),
    ("far", "success_far", "n_far"),
)


def sft_adapter_for_epoch(epoch: int) -> str:
    if int(epoch) not in SFT_EPOCHS:
        raise ValueError(f"SFT epoch must be one of {SFT_EPOCHS}, got {epoch}")
    return f"{SFT_ROOT}/epoch-{int(epoch)}"


def cell_out_path(run_dir: Path, policy: str, epoch: int, seed: int) -> Path:
    name = f"{str(policy).lower()}_e{int(epoch)}_s{int(seed)}.json"
    return Path(run_dir) / name


def assert_not_frozen_run_dir(path: Path) -> None:
    resolved = Path(path)
    parts = set(resolved.parts)
    for frozen in FROZEN_RUNS:
        if frozen in parts or resolved.name == frozen:
            raise ValueError(f"refusing to write into frozen run dir {frozen}")


def planned_cells() -> list[tuple[str, int, int]]:
    cells: list[tuple[str, int, int]] = []
    for epoch in SFT_EPOCHS:
        for seed in EVAL_SEEDS:
            cells.append(("SFT", int(epoch), int(seed)))
    for seed in EVAL_SEEDS:
        cells.append(("GRPO", 9, int(seed)))
    return cells


def _within_pp(value: float, target: float, tol: float) -> bool:
    return abs(float(value) - float(target)) <= float(tol) + 1e-6


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return float(sum(values) / len(values))


def row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    strata = payload.get("strata") if isinstance(payload.get("strata"), dict) else {}
    metrics = None
    for row in payload.get("policies") or []:
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
            metrics = row["metrics"]
            break
    if metrics is None or "success_rate" not in metrics:
        raise ValueError("goal-red cell is missing success_rate")
    if config.get("method") not in (None, METHOD_G1):
        raise ValueError("goal-red visible cells must be G1, not PAINT")
    if int(config.get("eval_max_steps") or config.get("max_steps") or 0) != C1_EVAL_8X8:
        raise ValueError(f"eval cap must be {C1_EVAL_8X8}")
    return {
        "policy": str(config.get("policy") or ""),
        "epoch": int(config.get("sft_epoch")),
        "seed": int(config.get("seed")),
        "success_rate": float(metrics["success_rate"]),
        "success_visible": _opt_float(strata.get("success_visible", config.get("success_visible"))),
        "success_facing_miss": _opt_float(
            strata.get("success_facing_miss", config.get("success_facing_miss"))
        ),
        "success_far": _opt_float(strata.get("success_far", config.get("success_far"))),
        "n_visible": _opt_int(strata.get("n_visible", config.get("n_visible"))),
        "n_facing_miss": _opt_int(strata.get("n_facing_miss", config.get("n_facing_miss"))),
        "n_far": _opt_int(strata.get("n_far", config.get("n_far"))),
        "eval_max_steps": C1_EVAL_8X8,
    }


def stratum_change(
    e5_rows: list[dict[str, Any]],
    e9_rows: list[dict[str, Any]],
    *,
    rate_key: str,
    n_key: str,
    episodes: int,
) -> dict[str, Any]:
    by5 = {int(row["seed"]): row for row in e5_rows}
    by9 = {int(row["seed"]): row for row in e9_rows}
    rates5: list[float] = []
    rates9: list[float] = []
    ns: list[float] = []
    for seed in sorted(set(by5) & set(by9)):
        rate5 = by5[seed].get(rate_key)
        rate9 = by9[seed].get(rate_key)
        n5 = by5[seed].get(n_key)
        n9 = by9[seed].get(n_key)
        if rate5 is None or rate9 is None or n5 is None or n9 is None:
            continue
        rates5.append(float(rate5))
        rates9.append(float(rate9))
        ns.append(0.5 * (float(n5) + float(n9)))
    if not rates5:
        return {"delta": None, "mean_n": None, "contribution_pp": None, "n_seeds": 0}
    delta = _mean(rates9) - _mean(rates5)
    mean_n = _mean(ns)
    return {
        "delta": delta,
        "mean_n": mean_n,
        "contribution_pp": (mean_n / float(episodes)) * delta,
        "n_seeds": len(rates5),
    }


def decide_verdict(
    *,
    sft_e5: list[dict[str, Any]],
    sft_e9: list[dict[str, Any]],
    grpo_e9: list[dict[str, Any]],
    episodes: int = 250,
) -> dict[str, Any]:
    _require_seeds(sft_e5, "SFT e5")
    _require_seeds(sft_e9, "SFT e9")
    _require_seeds(grpo_e9, "GRPO e9")
    sft_mean = _mean([float(row["success_rate"]) for row in sft_e9])
    grpo_mean = _mean([float(row["success_rate"]) for row in grpo_e9])
    suite_ok = _within_pp(sft_mean, C1_SFT_GOALRED_MEAN, C1_SANITY_TOL_PP) and _within_pp(
        grpo_mean, C1_GRPO_GOALRED_MEAN, C1_SANITY_TOL_PP
    )
    changes = {
        name: stratum_change(
            sft_e5,
            sft_e9,
            rate_key=rate_key,
            n_key=n_key,
            episodes=episodes,
        )
        for name, rate_key, n_key in _STRATA
    }
    carriers: list[str] = []
    if not suite_ok:
        verdict = VERDICT_INVALID
    else:
        visible_delta = changes["visible"]["delta"]
        if visible_delta is not None and visible_delta <= -VISIBLE_DROP_PP + 1e-6:
            verdict = VERDICT_VISIBLE
            carriers = ["visible"]
        elif visible_delta is not None and abs(visible_delta) <= VISIBLE_FLAT_PP + 1e-6:
            verdict, carriers = _other_or_unlocalized(changes)
        else:
            verdict = VERDICT_UNLOCALIZED
    return {
        "verdict": verdict,
        "suite_ok": suite_ok,
        "sft_e9_mean": sft_mean,
        "grpo_e9_mean": grpo_mean,
        "sft_e9_gate": C1_SFT_GOALRED_MEAN,
        "grpo_e9_gate": C1_GRPO_GOALRED_MEAN,
        "gate_pp": C1_SANITY_TOL_PP,
        "strata_e9_minus_e5": changes,
        "carrier_strata": carriers,
        "grpo_episode": C1_GRPO_EPISODE,
    }


def build_summary(rows: list[dict[str, Any]], *, episodes: int = 250) -> dict[str, Any]:
    grouped = _group_rows(rows)
    sft_e5 = grouped[("SFT", 5)]
    sft_e9 = grouped[("SFT", 9)]
    grpo_e9 = grouped[("GRPO", 9)]
    decision = decide_verdict(
        sft_e5=sft_e5,
        sft_e9=sft_e9,
        grpo_e9=grpo_e9,
        episodes=episodes,
    )
    by_arm: dict[str, Any] = {}
    for (policy, epoch), arm_rows in sorted(grouped.items()):
        key = f"{policy.lower()}_e{epoch}"
        by_arm[key] = _arm_block(arm_rows)
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": METHOD_G1,
        "decoder": "generate",
        "eval_max_steps": C1_EVAL_8X8,
        "episodes": int(episodes),
        "not_headline": True,
        "by_arm": by_arm,
        **decision,
        "note": (
            "Verdict compares SFT e9 minus e5 by start stratum. "
            "Do not replace C1 table goal-red aggregates. "
            "GRPO e9 is the suite gate only."
        ),
    }


def _other_or_unlocalized(changes: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    total = 0.0
    counted = False
    carriers: list[str] = []
    for name in ("facing_miss", "far"):
        contrib = changes[name]["contribution_pp"]
        if contrib is None:
            continue
        counted = True
        total += float(contrib)
        if float(contrib) <= -CONTRIB_FLOOR_PP + 1e-6:
            carriers.append(name)
    if counted and total <= -CONTRIB_FLOOR_PP + 1e-6:
        if not carriers:
            carriers = ["facing_miss", "far"]
        return VERDICT_OTHER, carriers
    return VERDICT_UNLOCALIZED, []


def _require_seeds(rows: list[dict[str, Any]], label: str) -> None:
    seeds = sorted(int(row["seed"]) for row in rows)
    expected = sorted(EVAL_SEEDS)
    if seeds != expected:
        raise ValueError(f"{label} seeds {seeds} != {expected}")


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), int(row["epoch"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _arm_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["seed"]))
    success = mean_sd_sample([float(row["success_rate"]) for row in ordered])
    strata: dict[str, Any] = {}
    for name, rate_key, n_key in _STRATA:
        rates = [row.get(rate_key) for row in ordered]
        ns = [row.get(n_key) for row in ordered]
        defined = [float(rate) for rate in rates if rate is not None]
        strata[name] = {
            "success_mean": None if not defined else _mean(defined),
            "n_by_seed": [None if n is None else int(n) for n in ns],
            "success_by_seed": [None if rate is None else float(rate) for rate in rates],
        }
    return {
        "seeds": [int(row["seed"]) for row in ordered],
        "success": success,
        "strata": strata,
    }


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
