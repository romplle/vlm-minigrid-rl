import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch


ACTION_TO_ID = {"left": 0, "right": 1, "forward": 2}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _episode_action_counts(dataset, episode_column: str = "episode_id", action_column: str = "action"):
    counts_by_episode = defaultdict(Counter)
    for episode_id, action in zip(dataset[episode_column], dataset[action_column]):
        counts_by_episode[int(episode_id)][action] += 1
    return dict(counts_by_episode)


def _distribution_score(counts: Counter, target_proportions: dict[str, float]) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("inf")

    return sum(
        (counts.get(action, 0) / total - target_proportion) ** 2
        for action, target_proportion in target_proportions.items()
    )


def _select_stratified_val_episodes(
    counts_by_episode: dict[int, Counter],
    val_count: int,
    seed: int,
    swap_iterations: int = 20000,
) -> set[int]:
    episodes = sorted(counts_by_episode)
    rng = random.Random(seed)
    candidates = episodes[:]
    rng.shuffle(candidates)

    full_counts = Counter()
    for counts in counts_by_episode.values():
        full_counts.update(counts)
    full_total = sum(full_counts.values())
    target_proportions = {
        action: count / full_total
        for action, count in full_counts.items()
    }

    val_count = min(val_count, len(candidates))
    val_episodes = set(candidates[:val_count])
    val_counts = Counter()
    for episode_id in val_episodes:
        val_counts.update(counts_by_episode[episode_id])

    best_score = _distribution_score(val_counts, target_proportions)

    for _ in range(swap_iterations):
        remove_episode = rng.choice(tuple(val_episodes))
        add_episode = rng.choice(candidates)
        if add_episode in val_episodes:
            continue

        candidate_counts = val_counts.copy()
        candidate_counts.subtract(counts_by_episode[remove_episode])
        candidate_counts.update(counts_by_episode[add_episode])

        score = _distribution_score(candidate_counts, target_proportions)
        if score < best_score:
            val_episodes.remove(remove_episode)
            val_episodes.add(add_episode)
            val_counts = candidate_counts
            best_score = score

    return val_episodes


def action_distribution(actions):
    counts = Counter(actions)
    return {
        action: counts.get(action, 0)
        for action in ACTION_TO_ID
    }


def split_dataset_by_episode(dataset, test_size: float = 0.1, seed: int = 42, episode_column: str = "episode_id"):
    episodes = sorted({int(ep) for ep in dataset[episode_column]})
    val_count = max(1, int(round(len(episodes) * test_size)))
    counts_by_episode = _episode_action_counts(dataset, episode_column=episode_column)
    val_episodes = _select_stratified_val_episodes(counts_by_episode, val_count, seed)

    train_indices = []
    val_indices = []
    for idx, episode_id in enumerate(dataset[episode_column]):
        if int(episode_id) in val_episodes:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return dataset.select(train_indices), dataset.select(val_indices), val_episodes


def majority_action_baseline(train_dataset, eval_dataset):
    action_counts = Counter(train_dataset["action"])
    majority_action, majority_count = action_counts.most_common(1)[0]
    correct = sum(1 for action in eval_dataset["action"] if action == majority_action)

    return {
        "action": majority_action,
        "train_count": majority_count,
        "accuracy": correct / len(eval_dataset) if len(eval_dataset) else 0.0,
        "train_distribution": action_distribution(train_dataset["action"]),
        "eval_distribution": action_distribution(eval_dataset["action"]),
    }


def parse_action(generated_text: str):
    text = generated_text.strip().lower()
    if "left" in text:
        return "left", ACTION_TO_ID["left"]
    if "right" in text:
        return "right", ACTION_TO_ID["right"]
    if "forward" in text:
        return "forward", ACTION_TO_ID["forward"]
    return None, ACTION_TO_ID["forward"]
