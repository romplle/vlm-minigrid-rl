"""C1 start-stratum helpers for SEE/PAINT. No SPIN/LOOK eval loop."""

from __future__ import annotations

STRATUM_VISIBLE = "visible"
STRATUM_FACING_MISS = "facing_miss"
STRATUM_FAR = "far"
C1_EVAL_16X16 = 40
C1_TRAIN_8X8 = 12
C1_SFT_EPOCH = 9
C1_GRPO_EPISODE = 100
C1_SFT_TRANSFER_MEAN = 57.47
C1_GRPO_TRANSFER_MEAN = 45.07
C1_SANITY_TOL_PP = 3.0
LOW_N_MIN = 20
DEFAULT_SFT_ADAPTER = "checkpoints/sft_adapter_8x8_bs32_e10/epoch-9"
DEFAULT_GRPO_ADAPTER = "checkpoints/grpo_adapter_8x8_handoff_from_e10_e9/episode-100"


def episode_stratum(row: dict) -> str:
    if row.get("start_goal_visible"):
        return STRATUM_VISIBLE
    if row.get("start_visible_any_dir"):
        return STRATUM_FACING_MISS
    return STRATUM_FAR


def _rate(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return 100.0 * sum(1 for row in rows if row.get("success")) / len(rows)


def stratify_geometry(episode_rows: list[dict]) -> dict[str, float | int | None]:
    visible = [row for row in episode_rows if episode_stratum(row) == STRATUM_VISIBLE]
    facing = [row for row in episode_rows if episode_stratum(row) == STRATUM_FACING_MISS]
    far = [row for row in episode_rows if episode_stratum(row) == STRATUM_FAR]
    empty = facing + far
    return {
        "n_visible": len(visible),
        "n_facing_miss": len(facing),
        "n_far": len(far),
        "n_empty": len(empty),
        "success_visible": _rate(visible),
        "success_facing_miss": _rate(facing),
        "success_far": _rate(far),
        "success_empty": _rate(empty),
    }


def behavior_stats(episode_rows: list[dict], stratum: str) -> dict:
    if stratum == "empty":
        selected = [row for row in episode_rows if episode_stratum(row) != STRATUM_VISIBLE]
    else:
        selected = [row for row in episode_rows if episode_stratum(row) == stratum]
    failed = [row for row in selected if not row.get("success")]
    failed_actions: list[str] = []
    for row in failed:
        failed_actions.extend(row.get("actions") or [])
    n_fail_act = len(failed_actions)
    right_share = (
        None
        if n_fail_act == 0
        else 100.0 * sum(1 for name in failed_actions if name == "right") / n_fail_act
    )
    n_sel = len(selected)
    n_fail = len(failed)

    def _frac(pred) -> float | None:
        if not failed:
            return None
        return 100.0 * sum(1 for row in failed if pred(row)) / n_fail

    return {
        "stratum": stratum,
        "n": n_sel,
        "n_failed": n_fail,
        "right_share_failed": right_share,
        "right_collapse": bool(right_share is not None and right_share >= 70.0),
        "spin_in_place_failed_frac": _frac(lambda row: bool(row.get("pos_unchanged_first8"))),
        "no_search_turn_failed_frac": _frac(lambda row: bool(row.get("no_search_turn"))),
        "never_visible_failed_frac": _frac(lambda row: row.get("first_visible_step") is None),
    }
