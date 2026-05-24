import argparse

import torch
from datasets import load_from_disk

from _bootstrap import bootstrap
bootstrap()

from vlm_minigrid_rl.minigrid_utils import (
    default_max_steps,
    evaluate_expert_in_env,
    evaluate_fixed_action_in_env,
    evaluate_model_in_env,
    print_comparison_table,
)
from vlm_minigrid_rl.model_utils import load_vlm_model, load_vlm_model_with_adapters
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import (
    ACTION_TO_ID,
    GOAL_COLORS,
    build_navigation_prompt,
    majority_action_baseline,
    set_global_seed,
    split_dataset_by_episode,
)


BASE_MODEL_ID = "lusxvr/nanoVLM-222M"
SFT_ADAPTER_PATH = "checkpoints/sft_adapter_8x8"
GRPO_ADAPTER_PATH = "checkpoints/grpo_adapter_8x8"
DATASET_PATH = "datasets/dataset_8x8"

ENV_SIZE = 8
TILE_SIZE = 32
TEST_EPISODES = 100
MAX_STEPS = 12
SEED = 42
VAL_SPLIT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT/GRPO MiniGrid policies.")
    parser.add_argument("--env-size", type=int, default=8, choices=[8, 16])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument("--grpo-adapter-path", default=None)
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--goal-color", default="green", choices=GOAL_COLORS)
    parser.add_argument("--prompt-goal-color", default=None, choices=GOAL_COLORS)
    parser.add_argument("--skip-grpo", action="store_true")
    return parser.parse_args()


def default_dataset_path(env_size):
    return f"datasets/dataset_{env_size}x{env_size}"


def default_sft_adapter_path(env_size):
    return f"checkpoints/sft_adapter_{env_size}x{env_size}"


def default_grpo_adapter_path(env_size):
    return f"checkpoints/grpo_adapter_{env_size}x{env_size}"


args = parse_args()
ENV_SIZE = args.env_size
DATASET_PATH = str(project_path(args.dataset_path or default_dataset_path(ENV_SIZE)))
SFT_ADAPTER_PATH = str(project_path(args.sft_adapter_path or default_sft_adapter_path(ENV_SIZE)))
GRPO_ADAPTER_PATH = str(project_path(args.grpo_adapter_path or default_grpo_adapter_path(ENV_SIZE)))
TEST_EPISODES = args.episodes
MAX_STEPS = args.max_steps if args.max_steps is not None else default_max_steps(ENV_SIZE)
VAL_SPLIT = args.val_split
GOAL_COLOR = args.goal_color
PROMPT_GOAL_COLOR = args.prompt_goal_color or args.goal_color

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_global_seed(SEED)


if __name__ == "__main__":
    full_ds = load_from_disk(DATASET_PATH)
    train_ds, val_ds, _ = split_dataset_by_episode(full_ds, test_size=VAL_SPLIT, seed=SEED)
    majority = majority_action_baseline(train_ds, val_ds)
    test_prompt = build_navigation_prompt(PROMPT_GOAL_COLOR)

    print("=== Dataset baselines ===")
    print(f"Evaluation goal color: {GOAL_COLOR} | prompt goal color: {PROMPT_GOAL_COLOR}")
    print(f"Episode-level train rows: {len(train_ds)}, val rows: {len(val_ds)}")
    print(
        f"Majority action from train: {majority['action']} | "
        f"validation accuracy: {majority['accuracy']:.4f}"
    )
    print(f"Train action distribution: {majority['train_distribution']}")
    print(f"Val action distribution: {majority['eval_distribution']}")

    print("=== Оценка SFT Модели ===")
    sft_model, tokenizer, image_processor = load_vlm_model(
        BASE_MODEL_ID, SFT_ADAPTER_PATH, DEVICE, is_trainable=False
    )
    sft_result = evaluate_model_in_env(
        sft_model,
        tokenizer,
        image_processor,
        test_prompt,
        ENV_SIZE,
        TILE_SIZE,
        MAX_STEPS,
        SEED,
        DEVICE,
        "SFT",
        episodes=TEST_EPISODES,
        goal_color=GOAL_COLOR,
    )

    results = [("SFT", sft_result)]

    if not args.skip_grpo:
        print("\n=== Оценка GRPO Модели ===")
        grpo_model, _, _ = load_vlm_model_with_adapters(
            BASE_MODEL_ID,
            [SFT_ADAPTER_PATH, GRPO_ADAPTER_PATH],
            DEVICE,
            is_trainable=False,
        )
        grpo_result = evaluate_model_in_env(
            grpo_model,
            tokenizer,
            image_processor,
            test_prompt,
            ENV_SIZE,
            TILE_SIZE,
            MAX_STEPS,
            SEED,
            DEVICE,
            "GRPO",
            episodes=TEST_EPISODES,
            goal_color=GOAL_COLOR,
        )
        results.append(("GRPO", grpo_result))

    majority_result = evaluate_fixed_action_in_env(
        majority["action"],
        ACTION_TO_ID[majority["action"]],
        ENV_SIZE,
        TILE_SIZE,
        MAX_STEPS,
        SEED,
        episodes=TEST_EPISODES,
        goal_color=GOAL_COLOR,
    )
    expert_result = evaluate_expert_in_env(
        ENV_SIZE,
        TILE_SIZE,
        MAX_STEPS,
        SEED,
        episodes=TEST_EPISODES,
        goal_color=GOAL_COLOR,
    )

    print_comparison_table([
        ("Majority baseline", majority_result),
        *results,
        ("Expert BFS upper bound", expert_result),
    ])
