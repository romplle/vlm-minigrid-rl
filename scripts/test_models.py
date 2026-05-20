import argparse

import torch
import gymnasium as gym
from tqdm import tqdm

from datasets import load_from_disk
from transformers import GenerationConfig

from minigrid.wrappers import RGBImgPartialObsWrapper
from minigrid.core.world_object import Goal

from _bootstrap import bootstrap
bootstrap()

from vlm_minigrid_rl.expert import get_shortest_path_actions, turn_balance
from vlm_minigrid_rl.model_utils import load_vlm_model, load_vlm_model_with_adapters
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import majority_action_baseline, parse_action, set_global_seed, split_dataset_by_episode


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
    parser.add_argument("--skip-grpo", action="store_true")
    return parser.parse_args()


def default_dataset_path(env_size):
    return f"datasets/dataset_{env_size}x{env_size}"


def default_sft_adapter_path(env_size):
    return f"checkpoints/sft_adapter_{env_size}x{env_size}"


def default_grpo_adapter_path(env_size):
    return f"checkpoints/grpo_adapter_{env_size}x{env_size}"


def default_max_steps(env_size):
    return 12 if env_size == 8 else 40


args = parse_args()
ENV_SIZE = args.env_size
DATASET_PATH = str(project_path(args.dataset_path or default_dataset_path(ENV_SIZE)))
SFT_ADAPTER_PATH = str(project_path(args.sft_adapter_path or default_sft_adapter_path(ENV_SIZE)))
GRPO_ADAPTER_PATH = str(project_path(args.grpo_adapter_path or default_grpo_adapter_path(ENV_SIZE)))
TEST_EPISODES = args.episodes
MAX_STEPS = args.max_steps if args.max_steps is not None else default_max_steps(ENV_SIZE)
VAL_SPLIT = args.val_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_global_seed(SEED)

def create_env():
    env = gym.make(f"MiniGrid-Empty-{ENV_SIZE}x{ENV_SIZE}-v0", render_mode="rgb_array")
    return RGBImgPartialObsWrapper(env, tile_size=TILE_SIZE)

def reset_env_with_goal(env, seed):
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped.place_agent()
    for x in range(unwrapped.grid.width):
        for y in range(unwrapped.grid.height):
            cell = unwrapped.grid.get(x, y)
            if cell and cell.type == "goal":
                unwrapped.grid.set(x, y, None)
    unwrapped.place_obj(Goal())
    return env.observation(unwrapped.gen_obs())


def empty_metrics():
    metrics = {
        "successes": 0,
        "total_reward": 0.0,
        "total_steps_in_success": 0,
        "timeouts": 0,
        "actions": {"left": 0, "right": 0, "forward": 0}
    }
    return metrics


def finalize_metrics(metrics, episodes):
    success_rate = (metrics["successes"] / episodes) * 100
    avg_reward = metrics["total_reward"] / episodes
    avg_steps = (metrics["total_steps_in_success"] / metrics["successes"]) if metrics["successes"] > 0 else 0
    total_actions = sum(metrics["actions"].values())
    action_dist = {k: (v / total_actions) * 100 for k, v in metrics["actions"].items()} if total_actions > 0 else metrics["actions"]

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps_success": avg_steps,
        "timeouts": metrics["timeouts"],
        "episodes": episodes,
        "action_dist": action_dist,
    }


def print_metrics(model_name, result):
    action_dist = result["action_dist"]
    print(f"\n--- Результаты {model_name} ---")
    print(f"Success Rate:    {result['success_rate']:.1f}%")
    print(f"Average Reward:  {result['avg_reward']:.3f}")
    print(f"Avg Steps (Win): {result['avg_steps_success']:.1f}")
    print(f"Timeouts:        {result['timeouts']}/{result['episodes']}")
    print(f"Action Dist:     L:{action_dist['left']:.1f}% | R:{action_dist['right']:.1f}% | F:{action_dist['forward']:.1f}%")


def evaluate_model_in_env(model, tokenizer, image_processor, prompt, model_name="Model", episodes=50):
    set_global_seed(SEED)
    env = create_env()
    model.eval()
    model.generation_config = GenerationConfig()
    metrics = empty_metrics()

    print(f"\n[{model_name}] Запуск симуляции ({episodes} эпизодов)...")
    
    for episode in tqdm(range(episodes), desc=f"Testing {model_name}"):
        obs = reset_env_with_goal(env, SEED + episode)
        episode_reward = 0.0
        
        for step in range(MAX_STEPS):
            ego_image = obs["image"]
            
            text = f"User: <image>\n{prompt}\nAssistant: "
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            
            image_inputs = image_processor(
                ego_image, return_tensors="pt", do_resize=True, size={"height": 224, "width": 224}
            )
            pixel_values = image_inputs.pixel_values.to(torch.float32).to(DEVICE)

            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"], 
                    pixel_values, 
                    max_new_tokens=1
                )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()

            action_name, action_idx = parse_action(generated_text)
            if action_name is None:
                action_name = "forward"
                
            metrics["actions"][action_name] += 1

            obs, reward, terminated, truncated, _ = env.step(action_idx)
            episode_reward += reward

            if terminated:
                metrics["successes"] += 1
                metrics["total_reward"] += episode_reward
                metrics["total_steps_in_success"] += (step + 1)
                break
            elif truncated or step == MAX_STEPS - 1:
                metrics["timeouts"] += 1
                break

    env.close()
    result = finalize_metrics(metrics, episodes)
    print_metrics(model_name, result)
    return result


def evaluate_fixed_action_in_env(action_name, action_idx, episodes=50):
    env = create_env()
    metrics = empty_metrics()
    print(f"\n[Majority baseline: {action_name}] Запуск симуляции ({episodes} эпизодов)...")

    for episode in tqdm(range(episodes), desc=f"Testing majority-{action_name}"):
        obs = reset_env_with_goal(env, SEED + episode)
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            metrics["actions"][action_name] += 1
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            episode_reward += reward

            if terminated:
                metrics["successes"] += 1
                metrics["total_reward"] += episode_reward
                metrics["total_steps_in_success"] += (step + 1)
                break
            if truncated or step == MAX_STEPS - 1:
                metrics["timeouts"] += 1
                break

    env.close()
    result = finalize_metrics(metrics, episodes)
    print_metrics(f"Majority-{action_name}", result)
    return result


def evaluate_expert_in_env(episodes=50):
    env = create_env()
    metrics = empty_metrics()
    action_balance = 0
    print(f"\n[Expert BFS] Запуск симуляции ({episodes} эпизодов)...")

    for episode in tqdm(range(episodes), desc="Testing expert"):
        reset_env_with_goal(env, SEED + episode)
        candidate_paths = [
            get_shortest_path_actions(env, action_order=(0, 1, 2)),
            get_shortest_path_actions(env, action_order=(1, 0, 2)),
        ]
        candidate_paths = [candidate for candidate in candidate_paths if candidate]
        path = min(
            candidate_paths,
            key=lambda candidate: (abs(action_balance + turn_balance(candidate)), turn_balance(candidate)),
        ) if candidate_paths else []
        action_balance += turn_balance(path)
        episode_reward = 0.0

        for step, action_idx in enumerate(path[:MAX_STEPS]):
            action_name = {0: "left", 1: "right", 2: "forward"}[action_idx]
            metrics["actions"][action_name] += 1
            _, reward, terminated, truncated, _ = env.step(action_idx)
            episode_reward += reward

            if terminated:
                metrics["successes"] += 1
                metrics["total_reward"] += episode_reward
                metrics["total_steps_in_success"] += (step + 1)
                break
            if truncated or step == MAX_STEPS - 1:
                metrics["timeouts"] += 1
                break
        else:
            if not path or len(path) > MAX_STEPS:
                metrics["timeouts"] += 1

    env.close()
    result = finalize_metrics(metrics, episodes)
    print_metrics("Expert BFS", result)
    return result


def print_comparison_table(results):
    print("\n===============================")
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("===============================")
    print("| Policy | Success Rate | Avg Reward | Avg Steps (Win) | Timeouts |")
    print("|---|---:|---:|---:|---:|")
    for name, result in results:
        print(
            f"| {name} | {result['success_rate']:.1f}% | {result['avg_reward']:.3f} | "
            f"{result['avg_steps_success']:.1f} | {result['timeouts']}/{result['episodes']} |"
        )
    print("===============================")

if __name__ == "__main__":
    full_ds = load_from_disk(DATASET_PATH)
    train_ds, val_ds, _ = split_dataset_by_episode(full_ds, test_size=VAL_SPLIT, seed=SEED)
    majority = majority_action_baseline(train_ds, val_ds)
    test_prompt = full_ds[0]["prompt"]

    print("=== Dataset baselines ===")
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
    sft_result = evaluate_model_in_env(sft_model, tokenizer, image_processor, test_prompt, "SFT", TEST_EPISODES)

    results = [("SFT", sft_result)]

    if not args.skip_grpo:
        print("\n=== Оценка GRPO Модели ===")
        grpo_model, _, _ = load_vlm_model_with_adapters(
            BASE_MODEL_ID,
            [SFT_ADAPTER_PATH, GRPO_ADAPTER_PATH],
            DEVICE,
            is_trainable=False,
        )
        grpo_result = evaluate_model_in_env(grpo_model, tokenizer, image_processor, test_prompt, "GRPO", TEST_EPISODES)
        results.append(("GRPO-action", grpo_result))

    majority_result = evaluate_fixed_action_in_env(
        majority["action"],
        {"left": 0, "right": 1, "forward": 2}[majority["action"]],
        TEST_EPISODES,
    )
    expert_result = evaluate_expert_in_env(TEST_EPISODES)

    print_comparison_table([
        ("Majority baseline", majority_result),
        *results,
        ("Expert BFS upper bound", expert_result),
    ])
