from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

HEADLINE_GRPO_HP = {
    "lr": 5e-6,
    "epsilon": 0.1,
    "beta": 0.1,
    "lora_dropout": 0.0,
}


def list_sft_epochs(sft_root: Path) -> list[int]:
    epochs: list[int] = []
    if not sft_root.is_dir():
        return epochs
    for child in sft_root.iterdir():
        if not child.is_dir() or not child.name.startswith("epoch-"):
            continue
        if not (child / "adapter_config.json").exists():
            continue
        suffix = child.name.removeprefix("epoch-")
        if suffix.isdigit():
            epochs.append(int(suffix))
    return sorted(epochs)


def sft_epoch_dir(sft_root: Path, epoch: int) -> Path:
    return sft_root / f"epoch-{epoch}"


DEFAULT_GRPO_TRAIN_SEED = 42


def grpo_handoff_root(
    env_size: int,
    sft_epoch: int,
    checkpoints_root: Path | None = None,
    train_seed: int | None = None,
) -> Path:
    """GRPO handoff checkpoint root.

    Seed 42 (and ``train_seed=None``) keep the legacy A1 path without a seed
    suffix. Other training seeds append ``_seed{S}``.
    """
    root = checkpoints_root if checkpoints_root is not None else Path("checkpoints")
    base = f"grpo_adapter_{env_size}x{env_size}_handoff_from_e10_e{sft_epoch}"
    if train_seed is not None and int(train_seed) != DEFAULT_GRPO_TRAIN_SEED:
        base = f"{base}_seed{int(train_seed)}"
    return root / base


def grpo_episode_dir(grpo_root: Path, episode: int) -> Path:
    return grpo_root / f"episode-{episode}"


def eval_json_path(
    run_dir: Path,
    *,
    kind: str,
    env_size: int,
    sft_epoch: int,
    seed: int,
    grpo_episode: int | None = None,
    tag: str | None = None,
) -> Path:
    env = f"{env_size}x{env_size}"
    if kind == "sft":
        name = f"sft_e{sft_epoch}_seed{seed}.json"
    elif kind == "grpo":
        if grpo_episode is None:
            raise ValueError("grpo_episode is required when kind='grpo'")
        name = f"grpo_e{sft_epoch}_ep{grpo_episode}_seed{seed}.json"
    elif kind == "ood":
        if tag is None:
            raise ValueError("tag is required when kind='ood'")
        name = f"ood_{tag}_e{sft_epoch}_seed{seed}.json"
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return run_dir / env / name


def extract_policy_metrics(eval_payload: dict, policy_name: str) -> dict:
    policies = eval_payload.get("policies") or eval_payload.get("results") or []
    for item in policies:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("policy")
        if name == policy_name or (isinstance(name, str) and policy_name in name):
            metrics = item.get("metrics")
            if isinstance(metrics, dict):
                return metrics
            return item
    for item in policies:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == policy_name:
            metrics = item[1]
            if isinstance(metrics, dict):
                return metrics
            raise KeyError(f"Policy {policy_name!r} metrics are not a dict")
    raise KeyError(f"Policy {policy_name!r} not found in eval payload")


def build_delta_row(
    sft_epoch: int,
    sft_rates: list[float],
    grpo_rates: list[float],
    sft_timeouts: list[float],
    grpo_timeouts: list[float],
    sft_invalids: list[float],
    grpo_invalids: list[float],
) -> dict:
    sft_mean = float(mean(sft_rates))
    grpo_mean = float(mean(grpo_rates))
    return {
        "sft_epoch": sft_epoch,
        "sft_mean": sft_mean,
        "grpo_mean": grpo_mean,
        "delta_pp": grpo_mean - sft_mean,
        "sft_timeouts_mean": float(mean(sft_timeouts)),
        "grpo_timeouts_mean": float(mean(grpo_timeouts)),
        "sft_invalids_mean": float(mean(sft_invalids)),
        "grpo_invalids_mean": float(mean(grpo_invalids)),
        "sft_rates": list(sft_rates),
        "grpo_rates": list(grpo_rates),
    }


def select_ood_sft_epochs(delta_rows: list[dict], k: int = 3) -> list[int]:
    if not delta_rows:
        return []
    ordered = sorted(delta_rows, key=lambda r: int(r["sft_epoch"]))
    if len(ordered) <= k:
        return [int(r["sft_epoch"]) for r in ordered]
    under = int(ordered[0]["sft_epoch"])
    over = int(ordered[-1]["sft_epoch"])
    mid = int(ordered[len(ordered) // 2]["sft_epoch"])
    out: list[int] = []
    for e in (under, mid, over):
        if e not in out:
            out.append(e)
    return out[:k]


def pick_best_grpo_episode(episode_to_success: dict[int, float]) -> int:
    return max(episode_to_success.items(), key=lambda kv: (kv[1], kv[0]))[0]


def load_grpo_picks(
    picks_file: Path,
    epochs: list[int],
    default_episode: int,
) -> dict[int, int]:
    merged = {epoch: default_episode for epoch in epochs}
    if picks_file.is_file():
        with picks_file.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        merged.update({int(k): int(v) for k, v in raw.items()})
        merged = {e: merged.get(e, default_episode) for e in epochs}
    return merged


def read_grpo_picks_file(picks_file: Path) -> dict[int, int]:
    if not picks_file.is_file():
        return {}
    with picks_file.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): int(v) for k, v in raw.items()}


def epochs_missing_from_picks(picks_file: Path, epochs: list[int]) -> list[int]:
    existing = read_grpo_picks_file(picks_file)
    return [epoch for epoch in epochs if epoch not in existing]


def merge_grpo_picks_on_disk(picks_file: Path, new_picks: dict[int, int]) -> dict[str, int]:
    merged = {str(k): v for k, v in read_grpo_picks_file(picks_file).items()}
    for epoch, episode in new_picks.items():
        merged[str(epoch)] = episode
    return merged


def grpo_train_complete(grpo_root: Path, train_episodes: int) -> bool:
    return (grpo_episode_dir(grpo_root, train_episodes) / "adapter_config.json").is_file()


def should_skip_grpo_train(
    grpo_root: Path,
    train_episodes: int,
    skip_existing: bool,
) -> bool:
    if not skip_existing:
        return False
    return grpo_train_complete(grpo_root, train_episodes)


def build_sft_eval_command(
    *,
    python: str,
    test_models: Path,
    env_size: int,
    dataset: Path,
    sft_adapter: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    output_json: Path,
) -> list[str]:
    return [
        python,
        str(test_models),
        "--env-size",
        str(env_size),
        "--dataset-path",
        str(dataset),
        "--sft-adapter-path",
        str(sft_adapter),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-json",
        str(output_json),
        "--skip-base",
        "--skip-majority",
        "--skip-expert",
        "--skip-grpo",
    ]


def build_grpo_train_command(
    *,
    python: str,
    grpo_script: Path,
    env_size: int,
    dataset: Path,
    sft_adapter: Path,
    output_dir: Path,
    train_episodes: int,
    checkpoint_interval: int,
    max_steps: int,
    seed: int,
) -> list[str]:
    hp = HEADLINE_GRPO_HP
    return [
        python,
        str(grpo_script),
        "--env-size",
        str(env_size),
        "--dataset-path",
        str(dataset),
        "--sft-adapter-path",
        str(sft_adapter),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(train_episodes),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--lr",
        str(hp["lr"]),
        "--epsilon",
        str(hp["epsilon"]),
        "--beta",
        str(hp["beta"]),
        "--lora-dropout",
        str(hp["lora_dropout"]),
        "--no-wandb",
    ]


def build_grpo_eval_command(
    *,
    python: str,
    test_models: Path,
    env_size: int,
    dataset: Path,
    sft_adapter: Path,
    grpo_adapter: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    output_json: Path,
    goal_color: str | None = None,
    prompt_goal_color: str | None = None,
) -> list[str]:
    cmd = [
        python,
        str(test_models),
        "--env-size",
        str(env_size),
        "--dataset-path",
        str(dataset),
        "--sft-adapter-path",
        str(sft_adapter),
        "--grpo-adapter-path",
        str(grpo_adapter),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-json",
        str(output_json),
        "--skip-base",
        "--skip-majority",
        "--skip-expert",
    ]
    if goal_color is not None:
        cmd.extend(["--goal-color", goal_color])
    if prompt_goal_color is not None:
        cmd.extend(["--prompt-goal-color", prompt_goal_color])
    return cmd
