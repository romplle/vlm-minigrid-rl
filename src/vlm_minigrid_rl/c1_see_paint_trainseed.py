"""e9 SEE/PAINT replication across GRPO train seeds (C1 caps 12/40)."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_ID = "2026-09-c1-see-paint-trainseed"
FROZEN_SEE_RUN = "runs/2026-08-c1-search-8x8"
FROZEN_PAINT_RUN = "runs/2026-08-c1-goalred-8x8"
EVAL_SEEDS = (42, 123, 456)
TRAIN_SEEDS_GPU = (123, 456)
REUSE_TRAIN_SEED = 42

G1_TRANSFER_GRPO = {42: 45.07, 123: 46.93, 456: 45.47}
G1_GOALRED_GRPO = {42: 2.40, 123: 4.27, 456: 2.93}
G1_SANITY_TOL_PP = 3.0

FROZEN_RUN_NAMES = (
    "2026-08-c1-search-8x8",
    "2026-08-c1-goalred-8x8",
    "2026-08-ood-multiseed-8x8",
    "2026-08-c1-geometry-8x8",
)

SEE_GOOD = frozenset({"SEARCH_PARTIAL", "SEARCH_RECOVERS"})
COLOR_GOOD = frozenset({"COLOR_RECOVERS"})


def train_run_dir(run_dir: Path, train_seed: int) -> Path:
    return Path(run_dir) / f"train{int(train_seed)}"


def g1_sanity_ok(*, suite: str, train_seed: int, grpo_mean: float) -> bool:
    table = G1_TRANSFER_GRPO if suite == "see" else G1_GOALRED_GRPO
    return abs(float(grpo_mean) - table[int(train_seed)]) <= G1_SANITY_TOL_PP


def across_axis_verdict(tags: dict[int, str], *, good: set[str], prefix: str) -> str:
    """tags: train_seed -> scientific tag."""
    seeds = (42, 123, 456)
    if any(seed not in tags for seed in seeds):
        return "INCONCLUSIVE"
    hits = [seed for seed in seeds if tags[seed] in good]
    if set(hits) == set(seeds):
        return f"{prefix}_REPLICATED"
    if hits == [42]:
        return f"{prefix}_SEED42_ONLY"
    return f"{prefix}_MIXED"


def joint_mechanism_verdict(see_axis: str, color_axis: str) -> str:
    if see_axis == "SEE_REPLICATED" and color_axis == "COLOR_REPLICATED":
        return "MECHANISM_REPLICATED"
    if see_axis == "SEE_REPLICATED" or color_axis == "COLOR_REPLICATED":
        return "MECHANISM_PARTIAL"
    if see_axis == "SEE_SEED42_ONLY" and color_axis == "COLOR_SEED42_ONLY":
        return "MECHANISM_SEED42"
    return "MECHANISM_MIXED"


def assert_not_frozen_run_dir(path: Path) -> None:
    resolved = Path(path)
    parts = set(resolved.parts)
    for name in FROZEN_RUN_NAMES:
        if name in parts or resolved.name == name:
            raise ValueError(f"refusing to write into frozen run dir {name}")


def planned_cells(
    *,
    train_seeds: tuple[int, ...] = (123, 456),
    eval_seeds: tuple[int, ...] = (42, 123, 456),
    arms: tuple[str, ...] = ("PAINT", "SEE"),
) -> list[tuple[int, str, str, int]]:
    """(train_seed, arm, method, eval_seed) in GPU order."""
    rows: list[tuple[int, str, str, int]] = []
    methods = {"PAINT": ("G1", "PAINT"), "SEE": ("G1", "SEE")}
    for train_seed in train_seeds:
        for arm in arms:
            if arm not in methods:
                raise ValueError(f"Unknown arm {arm!r}; expected PAINT or SEE")
            for method in methods[arm]:
                for eval_seed in eval_seeds:
                    rows.append((train_seed, arm, method, eval_seed))
    return rows

