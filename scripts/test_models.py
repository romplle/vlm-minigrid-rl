import argparse
import json

import torch
from datasets import load_from_disk

from _bootstrap import bootstrap
bootstrap()

from vlm_minigrid_rl.env_profiles import add_profile_cli_args, resolve_profile
from vlm_minigrid_rl.experiment_config import (
    BASE_MODEL_ID,
    DEFAULT_ENV_SIZE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_SEED,
    DEFAULT_TILE_SIZE,
    dataset_dir_for_profile,
    grpo_adapter_episode_dir,
    sft_adapter_epoch_dir,
)
from vlm_minigrid_rl.minigrid_utils import (
    default_eval_max_steps,
    evaluate_expert_in_env,
    evaluate_fixed_action_in_env,
    evaluate_model_in_env,
    print_comparison_table,
)
from vlm_minigrid_rl.model_utils import load_base_vlm_model, load_vlm_model, load_vlm_model_with_adapters
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import (
    ACTION_TO_ID,
    GOAL_COLORS,
    majority_action_baseline,
    set_global_seed,
    split_dataset_by_episode,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT/GRPO MiniGrid policies.")
    parser.add_argument("--env-size", type=int, default=DEFAULT_ENV_SIZE, choices=[8, 16])
    add_profile_cli_args(parser)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument("--grpo-adapter-path", default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Env eval horizon. Default: profile eval_max_steps.",
    )
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--goal-color", default="green", choices=GOAL_COLORS)
    parser.add_argument("--prompt-goal-color", default=None, choices=GOAL_COLORS)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    parser.add_argument("--skip-majority", action="store_true")
    parser.add_argument("--skip-expert", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", default=None, help="Write policy metrics to a JSON file.")
    return parser.parse_args()


args = parse_args()
PROFILE = resolve_profile(args.env_size, args.env_profile, args.env_id)
ENV_SIZE = PROFILE.env_size
DATASET_PATH = str(
    project_path(args.dataset_path) if args.dataset_path else dataset_dir_for_profile(PROFILE)
)
SFT_ADAPTER_PATH = str(
    project_path(args.sft_adapter_path) if args.sft_adapter_path else sft_adapter_epoch_dir(ENV_SIZE)
)
GRPO_ADAPTER_PATH = str(
    project_path(args.grpo_adapter_path) if args.grpo_adapter_path else grpo_adapter_episode_dir(ENV_SIZE)
)
TEST_EPISODES = args.episodes
MAX_STEPS = (
    args.max_steps
    if args.max_steps is not None
    else default_eval_max_steps(ENV_SIZE, profile=PROFILE)
)
VAL_SPLIT = args.val_split if args.val_split is not None else PROFILE.val_split
GOAL_COLOR = args.goal_color
PROMPT_GOAL_COLOR = args.prompt_goal_color or args.goal_color
SEED = args.seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_global_seed(SEED)


if __name__ == "__main__":
    full_ds = load_from_disk(DATASET_PATH)
    train_ds, val_ds, _ = split_dataset_by_episode(full_ds, test_size=VAL_SPLIT, seed=SEED)
    majority = majority_action_baseline(train_ds, val_ds)
    test_prompt = PROFILE.prompt(PROMPT_GOAL_COLOR)

    print("=== Dataset baselines ===")
    print(f"Evaluation goal color: {GOAL_COLOR} | prompt goal color: {PROMPT_GOAL_COLOR}")
    print(f"Eval max_steps={MAX_STEPS} (train_max_steps={PROFILE.train_max_steps})")
    print(f"Episode-level train rows: {len(train_ds)}, val rows: {len(val_ds)}")
    print(
        f"Majority action from train: {majority['action']} | "
        f"validation accuracy: {majority['accuracy']:.4f}"
    )
    print(f"Train action distribution: {majority['train_distribution']}")
    print(f"Val action distribution: {majority['eval_distribution']}")

    results = []

    if not args.skip_base:
        print("=== Оценка чистой NanoVLM ===")
        base_model, base_tokenizer, base_image_processor = load_base_vlm_model(
            BASE_MODEL_ID, DEVICE, is_trainable=False
        )
        base_result = evaluate_model_in_env(
            base_model,
            base_tokenizer,
            base_image_processor,
            test_prompt,
            ENV_SIZE,
            DEFAULT_TILE_SIZE,
            MAX_STEPS,
            SEED,
            DEVICE,
            "Base NanoVLM",
            episodes=TEST_EPISODES,
            goal_color=GOAL_COLOR,
        )
        results.append(("Base NanoVLM", base_result))
        del base_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    tokenizer = None
    image_processor = None

    if not args.skip_sft:
        print("\n=== Оценка SFT Модели ===")
        sft_model, tokenizer, image_processor = load_vlm_model(
            BASE_MODEL_ID, SFT_ADAPTER_PATH, DEVICE, is_trainable=False
        )
        sft_result = evaluate_model_in_env(
            sft_model,
            tokenizer,
            image_processor,
            test_prompt,
            ENV_SIZE,
            DEFAULT_TILE_SIZE,
            MAX_STEPS,
            SEED,
            DEVICE,
            "SFT",
            episodes=TEST_EPISODES,
            goal_color=GOAL_COLOR,
        )
        results.append(("SFT", sft_result))
        del sft_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_grpo:
        print("\n=== Оценка GRPO Модели ===")
        grpo_model, grpo_tokenizer, grpo_image_processor = load_vlm_model_with_adapters(
            BASE_MODEL_ID,
            [SFT_ADAPTER_PATH, GRPO_ADAPTER_PATH],
            DEVICE,
            is_trainable=False,
        )
        tokenizer = tokenizer or grpo_tokenizer
        image_processor = image_processor or grpo_image_processor
        grpo_result = evaluate_model_in_env(
            grpo_model,
            tokenizer,
            image_processor,
            test_prompt,
            ENV_SIZE,
            DEFAULT_TILE_SIZE,
            MAX_STEPS,
            SEED,
            DEVICE,
            "GRPO",
            episodes=TEST_EPISODES,
            goal_color=GOAL_COLOR,
        )
        results.append(("GRPO", grpo_result))
        del grpo_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    comparison_rows = []
    if not args.skip_majority:
        majority_result = evaluate_fixed_action_in_env(
            majority["action"],
            ACTION_TO_ID[majority["action"]],
            ENV_SIZE,
            DEFAULT_TILE_SIZE,
            MAX_STEPS,
            SEED,
            episodes=TEST_EPISODES,
            goal_color=GOAL_COLOR,
        )
        comparison_rows.append(("Majority baseline", majority_result))
    comparison_rows.extend(results)
    if not args.skip_expert:
        expert_result = evaluate_expert_in_env(
            ENV_SIZE,
            DEFAULT_TILE_SIZE,
            MAX_STEPS,
            SEED,
            episodes=TEST_EPISODES,
            goal_color=GOAL_COLOR,
        )
        comparison_rows.append(("Expert BFS upper bound", expert_result))

    if comparison_rows:
        print_comparison_table(comparison_rows)

    if args.output_json:
        payload = {
            "config": {
                "env_size": ENV_SIZE,
                "env_profile": PROFILE.name,
                "dataset_path": DATASET_PATH,
                "sft_adapter_path": SFT_ADAPTER_PATH,
                "grpo_adapter_path": GRPO_ADAPTER_PATH,
                "episodes": TEST_EPISODES,
                "max_steps": MAX_STEPS,
                "eval_max_steps": MAX_STEPS,
                "train_max_steps": PROFILE.train_max_steps,
                "val_split": VAL_SPLIT,
                "goal_color": GOAL_COLOR,
                "prompt_goal_color": PROMPT_GOAL_COLOR,
                "seed": SEED,
            },
            "policies": [
                {"name": name, "metrics": metrics}
                for name, metrics in comparison_rows
            ],
        }
        output_path = project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to {output_path}")
