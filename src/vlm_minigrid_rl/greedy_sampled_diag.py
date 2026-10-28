"""Helpers for greedy-vs-sampled (Q1 / F3) diagnostic summary."""

from __future__ import annotations

from statistics import mean, pstdev, stdev
from typing import Any


MODE_GENERATE = "generate"
MODE_ARGMAX3 = "argmax3"
MODE_TO_LABEL = {
    MODE_GENERATE: "sampled",
    MODE_ARGMAX3: "greedy",
}


def cell_json_name(
    policy: str,
    epoch: int,
    mode: str,
    seed: int,
    *,
    grpo_episode: int = 100,
    train_seed: int | None = None,
) -> str:
    """Filename for one eval cell under the diagnostic run dir.

    ``train_seed=None`` reproduces the single-train-seed layout of
    ``runs/2026-08-greedy-sampled-16x16``. Passing a train seed adds a
    ``trainseed{S}`` component so GRPO cells from different training runs
    cannot overwrite each other.
    """
    if policy == "sft":
        return f"sft_e{epoch}_mode{mode}_seed{seed}.json"
    if policy == "grpo":
        if train_seed is None:
            return f"grpo_e{epoch}_ep{grpo_episode}_mode{mode}_seed{seed}.json"
        return (
            f"grpo_e{epoch}_trainseed{int(train_seed)}"
            f"_ep{grpo_episode}_mode{mode}_seed{seed}.json"
        )
    raise ValueError(f"Unknown policy={policy!r}")


def mean_sd(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "sd": 0.0, "n": 1}
    return {
        "mean": float(mean(values)),
        "sd": float(pstdev(values)),
        "n": len(values),
    }


def mean_sd_sample(values: list[float]) -> dict[str, float]:
    """Like :func:`mean_sd` but with the sample standard deviation.

    Used for spread across training seeds, so the numbers line up with the
    train-seed sds reported by the F1 replication.
    """
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "sd": 0.0, "n": 1}
    return {
        "mean": float(mean(values)),
        "sd": float(stdev(values)),
        "n": len(values),
    }


def build_epoch_row(
    epoch: int,
    *,
    sft_sampled: list[float],
    sft_greedy: list[float],
    grpo_sampled: list[float],
    grpo_greedy: list[float],
) -> dict[str, Any]:
    sft_s = mean_sd(sft_sampled)
    sft_g = mean_sd(sft_greedy)
    grpo_s = mean_sd(grpo_sampled)
    grpo_g = mean_sd(grpo_greedy)
    delta_sampled = grpo_s["mean"] - sft_s["mean"]
    delta_greedy = grpo_g["mean"] - sft_g["mean"]
    return {
        "sft_epoch": epoch,
        "sft_sampled": sft_s,
        "sft_greedy": sft_g,
        "grpo_sampled": grpo_s,
        "grpo_greedy": grpo_g,
        "delta_sampled": delta_sampled,
        "delta_greedy": delta_greedy,
        "sft_sampled_rates": list(sft_sampled),
        "sft_greedy_rates": list(sft_greedy),
        "grpo_sampled_rates": list(grpo_sampled),
        "grpo_greedy_rates": list(grpo_greedy),
    }


def build_epoch_row_multiseed(
    epoch: int,
    *,
    sft_sampled: list[float],
    sft_greedy: list[float],
    grpo_by_train_seed: dict[int, dict[str, list[float]]],
) -> dict[str, Any]:
    """Epoch row where GRPO is replicated over training seeds.

    ``sft_*`` are per-eval-seed rates for the (single) SFT checkpoint.
    ``grpo_by_train_seed`` maps train seed -> {"sampled": rates, "greedy": rates}.

    The pooled ``grpo_*`` entry averages the per-train-seed eval means, so its
    ``sd`` is the spread **across training seeds** — the quantity the chapter
    needs — not across eval seeds.
    """
    sft_s = mean_sd(sft_sampled)
    sft_g = mean_sd(sft_greedy)

    per_seed = []
    for train_seed in sorted(grpo_by_train_seed):
        rates = grpo_by_train_seed[train_seed]
        g_s = mean_sd(rates["sampled"])
        g_g = mean_sd(rates["greedy"])
        per_seed.append(
            {
                "train_seed": int(train_seed),
                "grpo_sampled": g_s,
                "grpo_greedy": g_g,
                "delta_sampled": g_s["mean"] - sft_s["mean"],
                "delta_greedy": g_g["mean"] - sft_g["mean"],
                "grpo_sampled_rates": list(rates["sampled"]),
                "grpo_greedy_rates": list(rates["greedy"]),
            }
        )

    pooled_s = mean_sd_sample([c["grpo_sampled"]["mean"] for c in per_seed])
    pooled_g = mean_sd_sample([c["grpo_greedy"]["mean"] for c in per_seed])
    return {
        "sft_epoch": epoch,
        "sft_sampled": sft_s,
        "sft_greedy": sft_g,
        "grpo_sampled": pooled_s,
        "grpo_greedy": pooled_g,
        "delta_sampled": pooled_s["mean"] - sft_s["mean"],
        "delta_greedy": pooled_g["mean"] - sft_g["mean"],
        "sft_sampled_rates": list(sft_sampled),
        "sft_greedy_rates": list(sft_greedy),
        "by_train_seed": per_seed,
    }


def build_f3_reading(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """F3 heuristic: compare e1→e8 Δ curves under sampled vs greedy.

    If Δ_sampled changes a lot across budgets while Δ_greedy stays near-flat,
    treat headline Δ as largely a sampling/sharpening effect. If Δ_greedy tracks
    Δ_sampled, treat as restricted 3-way policy improvement.
    """
    by_epoch = {int(r["sft_epoch"]): r for r in epoch_rows}
    early = min(by_epoch) if by_epoch else None
    late = max(by_epoch) if by_epoch else None
    if early is None or late is None or early == late:
        return {
            "early_epoch": early,
            "late_epoch": late,
            "delta_sampled_early": None,
            "delta_sampled_late": None,
            "delta_greedy_early": None,
            "delta_greedy_late": None,
            "slope_sampled": None,
            "slope_greedy": None,
            "temperature_artifact_suspected": None,
            "policy_improvement_suspected": None,
            "verdict": "insufficient_epochs",
        }

    d_s_early = float(by_epoch[early]["delta_sampled"])
    d_s_late = float(by_epoch[late]["delta_sampled"])
    d_g_early = float(by_epoch[early]["delta_greedy"])
    d_g_late = float(by_epoch[late]["delta_greedy"])
    slope_sampled = d_s_late - d_s_early
    slope_greedy = d_g_late - d_g_early

    # Greedy nearly flat (|slope| < 1 pp) while sampled |slope| ≥ 2× greedy and ≥ 2 pp.
    temperature = (
        abs(slope_greedy) < 1.0
        and abs(slope_sampled) >= 2.0
        and abs(slope_sampled) >= 2.0 * max(abs(slope_greedy), 0.25)
    )
    both_flat = abs(slope_sampled) < 1.0 and abs(slope_greedy) < 1.0
    # Δ_greedy tracks Δ_sampled: similar slope sign and magnitude within 1.5 pp,
    # or both move substantially in the same direction (greedy may amplify).
    same_direction = (slope_sampled * slope_greedy) > 0
    tracks = (not both_flat) and (
        abs(slope_sampled - slope_greedy) <= 1.5
        and (
            slope_sampled == 0.0
            or slope_greedy == 0.0
            or same_direction
        )
    )
    amplifies = (
        not both_flat
        and same_direction
        and abs(slope_sampled) >= 2.0
        and abs(slope_greedy) >= 2.0
    )
    policy = (tracks or amplifies) and not temperature

    if temperature:
        verdict = "temperature_artifact"
    elif both_flat:
        verdict = "both_flat"
    elif policy:
        verdict = "policy_improvement"
    else:
        verdict = "mixed"

    return {
        "early_epoch": early,
        "late_epoch": late,
        "delta_sampled_early": d_s_early,
        "delta_sampled_late": d_s_late,
        "delta_greedy_early": d_g_early,
        "delta_greedy_late": d_g_late,
        "slope_sampled": slope_sampled,
        "slope_greedy": slope_greedy,
        "temperature_artifact_suspected": temperature,
        "policy_improvement_suspected": policy,
        "verdict": verdict,
    }


def build_summary_payload(
    epoch_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted(epoch_rows, key=lambda r: int(r["sft_epoch"]))
    return {
        "config": config or {},
        "epochs": ordered,
        "f3": build_f3_reading(ordered),
    }
