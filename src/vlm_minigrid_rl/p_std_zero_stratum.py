"""CPU audit: GRPO group all-fail rate by start stratum. No VLM, no new train.

Join reconstructed layout seeds with stdout Success% from existing C1 GRPO logs.
CPU spin/search on the same layouts is a Lu/Bayes control, not p_std_zero
(deterministic copies of one trajectory always have σ=0).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from vlm_minigrid_rl.memoryless_bayes import (
    POLICY_SEARCH,
    POLICY_SPIN,
    STRATA,
    run_episode,
    start_stratum,
)
from vlm_minigrid_rl.minigrid_utils import (
    apply_episode_horizon,
    create_minigrid_env,
    reset_env_with_goal,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = ROOT / "thesis" / "data" / "p_std_zero_stratum.json"
DEFAULT_LOGS = (
    ROOT / "runs" / "2026-07-handoff-empty" / "8x8" / "grpo-train.log",
    ROOT / "runs" / "2026-08-grpo-trainseed-8x8" / "train.log",
)

C1_E9_MAX_STEPS = 12
C1_E9_EPISODES = 100
G = 16
RAW_STD_EPS = 1e-8
LAYOUT_RANDINT_HI = 100000

GROUP_ALL_FAIL = "all_fail"
GROUP_MIXED = "mixed"
GROUP_ALL_SUCCESS = "all_success"

VERDICT_FAR_ALLFAIL_DOMINATES = "FAR_ALLFAIL_DOMINATES"
# Machine tag only: p_fail(far) > p_fail(miss) and CPU-spin far >= 0.95.
# Does not mean all-fail dominates real train (join far 2/12 mixed 10/12; frequency 12/300).
VERDICT_MIXED = "MIXED"
VERDICT_NOT_SHOWN = "NOT_SHOWN"

NOTE = (
    "Join: reconstructed GRPO layout seeds × CPU start_stratum × stdout Success%. "
    "Success=0% iff all G=16 returns are 0 iff p_std_zero=1. Success=100% does not "
    "imply σ=0. CPU spin/search: deterministic policy, one episode = group; report "
    "p_all_fail not p_std_zero. Caps C1 train max_steps=12. No VLM."
)

_EP_RE = re.compile(
    r"Ep\s+(\d+)/(\d+)\s+\|\s+Mean Return:\s+([0-9.]+)\s+\|\s+Success:\s+([0-9.]+)%"
)
_CONFIG_SEED_RE = re.compile(r"^Config:.*\bseed=(\d+)\b", re.MULTILINE)
_E9_EP100_RE = re.compile(
    r"(grpo_adapter_8x8_handoff_from_e10_e9(?:_seed(\d+))?)"
    r"[/\\]episode-100"
)
_ANY_HANDOFF_RE = re.compile(
    r"grpo_adapter_8x8_handoff_from_e10_e\d+(?:_seed\d+)?"
)

ProgressCallback = Callable[[str], None]


def grpo_layout_seeds(train_seed: int, n_episodes: int = C1_E9_EPISODES) -> list[int]:
    rng = random.Random(int(train_seed))
    return [rng.randint(0, LAYOUT_RANDINT_HI) for _ in range(int(n_episodes))]


def raw_std_zero(returns: list[float], eps: float = RAW_STD_EPS) -> bool:
    if len(returns) <= 1:
        return True
    mean = sum(returns) / float(len(returns))
    var = sum((x - mean) ** 2 for x in returns) / float(len(returns))
    return math.sqrt(var) < float(eps)


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def parse_grpo_episode_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _EP_RE.finditer(text):
        rows.append(
            {
                "episode": int(match.group(1)),
                "n_episodes": int(match.group(2)),
                "mean_return": float(match.group(3)),
                "success_rate": float(match.group(4)),
            }
        )
    return rows


def _adapter_from_chunk(chunk: str) -> str | None:
    ep100 = list(_E9_EP100_RE.finditer(chunk))
    if ep100:
        return ep100[-1].group(1)
    any_handoff = list(_ANY_HANDOFF_RE.finditer(chunk))
    if any_handoff:
        return any_handoff[-1].group(0)
    return None


def split_log_runs(text: str) -> list[dict[str, Any]]:
    chunks = re.split(r"(?=^Config:)", text, flags=re.MULTILINE)
    runs: list[dict[str, Any]] = []
    for chunk in chunks:
        seed_match = _CONFIG_SEED_RE.search(chunk)
        episodes = parse_grpo_episode_rows(chunk)
        adapter = _adapter_from_chunk(chunk)
        if seed_match is None and not episodes:
            continue
        runs.append(
            {
                "seed": int(seed_match.group(1)) if seed_match else None,
                "adapter": adapter,
                "episodes": episodes,
                "text": chunk,
            }
        )
    return runs


def _is_c1_e9_adapter(adapter: str | None, train_seed: int) -> bool:
    if not adapter:
        return False
    if int(train_seed) == 42:
        return adapter.endswith("handoff_from_e10_e9") and "_seed" not in adapter
    return adapter.endswith(f"handoff_from_e10_e9_seed{int(train_seed)}")


def select_c1_e9_run(runs: list[dict[str, Any]], train_seed: int) -> dict[str, Any] | None:
    matched = [
        run
        for run in runs
        if run.get("seed") == int(train_seed) and _is_c1_e9_adapter(run.get("adapter"), train_seed)
    ]
    if matched:
        return matched[-1]
    fallback = [run for run in runs if run.get("seed") == int(train_seed) and run.get("episodes")]
    if fallback:
        return fallback[-1]
    return None


def group_outcome(success_rate: float) -> str:
    rate = float(success_rate)
    if rate <= 1e-9:
        return GROUP_ALL_FAIL
    if rate >= 100.0 - 1e-9:
        return GROUP_ALL_SUCCESS
    return GROUP_MIXED


def join_rows(strata: list[str], episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(strata) != len(episode_rows):
        raise ValueError(
            f"stratum/episode length mismatch: {len(strata)} vs {len(episode_rows)}"
        )
    joined: list[dict[str, Any]] = []
    for stratum, episode in zip(strata, episode_rows):
        rate = float(episode["success_rate"])
        joined.append(
            {
                "episode": int(episode["episode"]),
                "stratum": str(stratum),
                "mean_return": float(episode["mean_return"]),
                "success_rate": rate,
                "outcome": group_outcome(rate),
            }
        )
    return joined


def _empty_stratum_tally() -> dict[str, Any]:
    return {
        "n": 0,
        "n_all_fail": 0,
        "n_mixed": 0,
        "n_all_success": 0,
        "p_all_fail": None,
        "p_all_fail_ci": None,
        "mean_success": None,
    }


def tally_by_stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["stratum"])].append(row)
    out: dict[str, Any] = {}
    for key in STRATA:
        items = buckets.get(key, [])
        n = len(items)
        n_fail = sum(1 for row in items if row["outcome"] == GROUP_ALL_FAIL)
        n_mixed = sum(1 for row in items if row["outcome"] == GROUP_MIXED)
        n_success = sum(1 for row in items if row["outcome"] == GROUP_ALL_SUCCESS)
        p_fail = (n_fail / n) if n else None
        ci = wilson_ci(n_fail, n) if n else None
        mean_success = (
            sum(float(row["success_rate"]) for row in items) / n if n else None
        )
        out[key] = {
            "n": n,
            "n_all_fail": n_fail,
            "n_mixed": n_mixed,
            "n_all_success": n_success,
            "p_all_fail": None if p_fail is None else round(p_fail, 6),
            "p_all_fail_ci": None if ci is None else [round(ci[0], 6), round(ci[1], 6)],
            "mean_success": None if mean_success is None else round(mean_success, 4),
        }
    for key, items in buckets.items():
        if key not in out:
            extra = _empty_stratum_tally()
            extra["n"] = len(items)
            out[key] = extra
    return out


def verdict_from_rates(
    *,
    p_far: float | None,
    p_miss: float | None,
    cpu_spin_far: float | None,
) -> str:
    """Compare all-fail *rates*. Tag is not «A=0 dominates real train» (see gloss)."""
    if p_far is None or p_miss is None:
        return VERDICT_NOT_SHOWN
    if cpu_spin_far is None:
        return VERDICT_NOT_SHOWN
    if float(p_far) > float(p_miss) and float(cpu_spin_far) >= 0.95:
        return VERDICT_FAR_ALLFAIL_DOMINATES
    return VERDICT_MIXED


def classify_layout(
    env_size: int,
    layout_seed: int,
    max_steps: int = C1_E9_MAX_STEPS,
) -> str:
    env = create_minigrid_env(int(env_size), max_steps=int(max_steps))
    try:
        reset_env_with_goal(env, int(layout_seed), goal_color="green")
        return start_stratum(env)
    finally:
        env.close()


def cpu_group_all_fail(
    env_size: int,
    layout_seed: int,
    policy: str,
    max_steps: int = C1_E9_MAX_STEPS,
) -> bool:
    env = create_minigrid_env(int(env_size), max_steps=int(max_steps))
    try:
        apply_episode_horizon(env, int(max_steps))
        reset_env_with_goal(env, int(layout_seed), goal_color="green")
        success, _stratum = run_episode(env, policy, int(max_steps))
        return not success
    finally:
        env.close()


def classify_layouts(
    layout_seeds: list[int],
    *,
    env_size: int = 8,
    max_steps: int = C1_E9_MAX_STEPS,
) -> list[str]:
    env = create_minigrid_env(int(env_size), max_steps=int(max_steps))
    try:
        labels: list[str] = []
        for seed in layout_seeds:
            reset_env_with_goal(env, int(seed), goal_color="green")
            labels.append(start_stratum(env))
        return labels
    finally:
        env.close()


def cpu_eval_layouts(
    layout_seeds: list[int],
    policy: str,
    *,
    env_size: int = 8,
    max_steps: int = C1_E9_MAX_STEPS,
) -> list[dict[str, Any]]:
    env = create_minigrid_env(int(env_size), max_steps=int(max_steps))
    apply_episode_horizon(env, int(max_steps))
    rows: list[dict[str, Any]] = []
    try:
        for seed in layout_seeds:
            reset_env_with_goal(env, int(seed), goal_color="green")
            success, stratum = run_episode(env, policy, int(max_steps))
            rows.append(
                {
                    "layout_seed": int(seed),
                    "stratum": stratum,
                    "success_rate": 100.0 if success else 0.0,
                    "mean_return": 1.0 if success else 0.0,
                    "episode": len(rows) + 1,
                    "outcome": GROUP_ALL_SUCCESS if success else GROUP_ALL_FAIL,
                }
            )
        return rows
    finally:
        env.close()


def _pool_tallies(tallies: list[dict[str, Any]]) -> dict[str, Any]:
    pooled_rows: list[dict[str, Any]] = []
    for tally in tallies:
        for key in STRATA:
            cell = tally[key]
            for _ in range(int(cell["n_all_fail"])):
                pooled_rows.append({"stratum": key, "outcome": GROUP_ALL_FAIL, "success_rate": 0.0})
            for _ in range(int(cell["n_mixed"])):
                pooled_rows.append({"stratum": key, "outcome": GROUP_MIXED, "success_rate": 50.0})
            for _ in range(int(cell["n_all_success"])):
                pooled_rows.append(
                    {"stratum": key, "outcome": GROUP_ALL_SUCCESS, "success_rate": 100.0}
                )
    pooled = tally_by_stratum(pooled_rows)
    for key in STRATA:
        weight = 0
        total = 0.0
        for tally in tallies:
            cell = tally[key]
            n = int(cell["n"])
            mean_success = cell.get("mean_success")
            if n and mean_success is not None:
                total += float(mean_success) * n
                weight += n
        pooled[key]["mean_success"] = (
            None if weight == 0 else round(total / weight, 4)
        )
    return pooled


def decode_log_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


def load_log_text(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if path.is_file():
            parts.append(decode_log_bytes(path.read_bytes()))
    return "\n".join(parts)


def join_one_seed(
    train_seed: int,
    episode_rows: list[dict[str, Any]],
    *,
    env_size: int = 8,
    max_steps: int = C1_E9_MAX_STEPS,
) -> dict[str, Any]:
    n = len(episode_rows)
    seeds = grpo_layout_seeds(int(train_seed), n)
    strata = classify_layouts(seeds, env_size=env_size, max_steps=max_steps)
    rows = join_rows(strata, episode_rows)
    for row, layout_seed in zip(rows, seeds):
        row["layout_seed"] = int(layout_seed)
    tally = tally_by_stratum(rows)
    return {
        "train_seed": int(train_seed),
        "n": n,
        "by_stratum": tally,
        "n_far": int(tally["far"]["n"]),
        "rows": rows,
    }


def cpu_one_policy(
    layout_seeds: list[int],
    policy: str,
    *,
    env_size: int = 8,
    max_steps: int = C1_E9_MAX_STEPS,
) -> dict[str, Any]:
    rows = cpu_eval_layouts(
        layout_seeds, policy, env_size=env_size, max_steps=max_steps
    )
    return {
        "policy": policy,
        "n": len(rows),
        "by_stratum": tally_by_stratum(rows),
    }


def build_report(
    *,
    join_by_seed: dict[str, dict[str, Any]] | None,
    cpu_spin_prior: dict[str, Any],
    cpu_search_prior: dict[str, Any],
    cpu_spin_layouts: dict[str, dict[str, Any]],
    cpu_search_layouts: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    join_block: dict[str, Any] | None = None
    p_far = None
    p_miss = None
    if join_by_seed:
        tallies = [cell["by_stratum"] for cell in join_by_seed.values()]
        pooled = _pool_tallies(tallies)
        join_block = {
            "by_train_seed": {
                key: {
                    "n": cell["n"],
                    "by_stratum": cell["by_stratum"],
                    "n_far": cell["n_far"],
                }
                for key, cell in join_by_seed.items()
            },
            "pooled": pooled,
        }
        p_far = pooled["far"]["p_all_fail"]
        p_miss = pooled["facing_miss"]["p_all_fail"]
    cpu_spin_far = cpu_spin_prior["by_stratum"]["far"]["p_all_fail"]
    verdict = verdict_from_rates(
        p_far=p_far, p_miss=p_miss, cpu_spin_far=cpu_spin_far
    )
    reason_join = "no GRPO stdout join"
    if join_block is not None:
        far_cell = join_block["pooled"]["far"]
        miss_cell = join_block["pooled"]["facing_miss"]
        reason_join = (
            f"join p_all_fail far={far_cell['n_all_fail']}/{far_cell['n']} "
            f"facing_miss={miss_cell['n_all_fail']}/{miss_cell['n']}"
        )
    spin_far = cpu_spin_prior["by_stratum"]["far"]
    reason = (
        f"{reason_join}; CPU spin far all-fail "
        f"{spin_far['n_all_fail']}/{spin_far['n']}"
    )
    return {
        "note": NOTE,
        "config": config,
        "join": join_block,
        "cpu": {
            "prior": {
                POLICY_SPIN: cpu_spin_prior,
                POLICY_SEARCH: cpu_search_prior,
            },
            "grpo_layouts": {
                POLICY_SPIN: cpu_spin_layouts,
                POLICY_SEARCH: cpu_search_layouts,
            },
        },
        "verdict": verdict,
        "reason": reason,
    }


def run_audit(
    *,
    log_paths: list[Path],
    train_seeds: tuple[int, ...] = (42, 123, 456),
    env_size: int = 8,
    max_steps: int = C1_E9_MAX_STEPS,
    n_episodes: int = C1_E9_EPISODES,
    cpu_prior_n: int = 2000,
    cpu_prior_seed: int = 42,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    config = {
        "env_size": int(env_size),
        "max_steps": int(max_steps),
        "n_episodes": int(n_episodes),
        "train_seeds": [int(s) for s in train_seeds],
        "cpu_prior_n": int(cpu_prior_n),
        "cpu_prior_seed": int(cpu_prior_seed),
        "G": G,
        "goal_color": "green",
        "log_paths": [str(path) for path in log_paths],
    }
    text = load_log_text(list(log_paths))
    runs = split_log_runs(text) if text else []
    join_by_seed: dict[str, dict[str, Any]] = {}
    for seed in train_seeds:
        picked = select_c1_e9_run(runs, int(seed)) if runs else None
        if picked is None or len(picked["episodes"]) < int(n_episodes):
            log(f"join skip seed={seed}: no C1 e9 log block")
            continue
        episodes = picked["episodes"][: int(n_episodes)]
        if len(episodes) != int(n_episodes):
            log(f"join skip seed={seed}: {len(episodes)} episode rows")
            continue
        log(f"join classify seed={seed} n={n_episodes}")
        join_by_seed[str(seed)] = join_one_seed(
            int(seed),
            episodes,
            env_size=env_size,
            max_steps=max_steps,
        )

    prior_seeds = [int(cpu_prior_seed) + i for i in range(int(cpu_prior_n))]
    log(f"CPU prior n={cpu_prior_n} policy=spin")
    cpu_spin_prior = cpu_one_policy(
        prior_seeds, POLICY_SPIN, env_size=env_size, max_steps=max_steps
    )
    log(f"CPU prior n={cpu_prior_n} policy=search")
    cpu_search_prior = cpu_one_policy(
        prior_seeds, POLICY_SEARCH, env_size=env_size, max_steps=max_steps
    )

    cpu_spin_layouts: dict[str, dict[str, Any]] = {}
    cpu_search_layouts: dict[str, dict[str, Any]] = {}
    for seed in train_seeds:
        layout_seeds = grpo_layout_seeds(int(seed), int(n_episodes))
        log(f"CPU GRPO layouts seed={seed} spin")
        cpu_spin_layouts[str(seed)] = cpu_one_policy(
            layout_seeds, POLICY_SPIN, env_size=env_size, max_steps=max_steps
        )
        log(f"CPU GRPO layouts seed={seed} search")
        cpu_search_layouts[str(seed)] = cpu_one_policy(
            layout_seeds, POLICY_SEARCH, env_size=env_size, max_steps=max_steps
        )

    return build_report(
        join_by_seed=join_by_seed or None,
        cpu_spin_prior=cpu_spin_prior,
        cpu_search_prior=cpu_search_prior,
        cpu_spin_layouts=cpu_spin_layouts,
        cpu_search_layouts=cpu_search_layouts,
        config=config,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU p_all_fail by stratum: join C1 GRPO logs + spin/search control."
    )
    parser.add_argument("--max-steps", type=int, default=C1_E9_MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=C1_E9_EPISODES)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--cpu-prior-n", type=int, default=2000)
    parser.add_argument("--cpu-prior-seed", type=int, default=42)
    parser.add_argument("--env-size", type=int, default=8)
    parser.add_argument(
        "--log",
        action="append",
        default=None,
        help="GRPO stdout log (repeatable). Default: C1 e9 handoff + trainseed logs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    log_paths = [Path(p) for p in args.log] if args.log else list(DEFAULT_LOGS)
    report = run_audit(
        log_paths=log_paths,
        train_seeds=seeds,
        env_size=int(args.env_size),
        max_steps=int(args.max_steps),
        n_episodes=int(args.episodes),
        cpu_prior_n=int(args.cpu_prior_n),
        cpu_prior_seed=int(args.cpu_prior_seed),
        progress=print,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"verdict={report['verdict']} {report['reason']}")
    join = report.get("join")
    if join:
        pooled = join["pooled"]
        far = pooled["far"]
        miss = pooled["facing_miss"]
        print(
            f"join far {far['n_all_fail']}/{far['n']} "
            f"facing_miss {miss['n_all_fail']}/{miss['n']}"
        )
    spin_far = report["cpu"]["prior"][POLICY_SPIN]["by_stratum"]["far"]
    print(f"cpu spin far {spin_far['n_all_fail']}/{spin_far['n']}")


if __name__ == "__main__":
    cli_main()
