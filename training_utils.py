import os
import random
from collections import Counter

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


def split_dataset_by_episode(dataset, test_size: float = 0.1, seed: int = 42, episode_column: str = "episode_id"):
    episodes = sorted({int(ep) for ep in dataset[episode_column]})
    rng = random.Random(seed)
    rng.shuffle(episodes)

    val_count = max(1, int(round(len(episodes) * test_size)))
    val_episodes = set(episodes[:val_count])

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
        "train_distribution": dict(action_counts),
        "eval_distribution": dict(Counter(eval_dataset["action"])),
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
