import argparse
import shutil

import numpy as np
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value
from tqdm import tqdm

from _bootstrap import bootstrap
bootstrap()

from vlm_minigrid_rl.experiment_config import (
    DEFAULT_ENV_SIZE,
    DEFAULT_NUM_DATASET_EPISODES,
    DEFAULT_SEED,
    DEFAULT_TILE_SIZE,
    dataset_dir,
)
from vlm_minigrid_rl.minigrid_utils import choose_balanced_shortest_path, create_minigrid_env, reset_env_with_goal, turn_balance
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import GOAL_COLORS, ID_TO_ACTION, build_navigation_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MiniGrid expert trajectories.")
    parser.add_argument("--env-size", type=int, default=DEFAULT_ENV_SIZE, choices=[8, 16])
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_DATASET_EPISODES)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--goal-color", default="green", choices=GOAL_COLORS)
    parser.add_argument("--prompt-goal-color", default=None, choices=GOAL_COLORS)
    return parser.parse_args()


def main():
    args = parse_args()
    env_size = args.env_size
    prompt_goal_color = args.prompt_goal_color or args.goal_color
    save_path = project_path(args.save_path) if args.save_path else dataset_dir(env_size)
    env_id = f"MiniGrid-Empty-{env_size}x{env_size}-v0"
    prompt = build_navigation_prompt(prompt_goal_color)
    print(f"Создаём датасет: {env_id}, goal_color={args.goal_color}, prompt_goal_color={prompt_goal_color}")

    wrapper = create_minigrid_env(env_size, tile_size=args.tile_size)
    env = wrapper.unwrapped

    data = []
    action_balance = 0

    for episode in tqdm(range(args.num_episodes), desc="Генерация траекторий"):
        seed = args.seed_base + episode
        obs = reset_env_with_goal(wrapper, seed, goal_color=args.goal_color)
        unwrapped = wrapper.unwrapped

        path = choose_balanced_shortest_path(wrapper, action_balance)
        action_balance += turn_balance(path)
        if not path:
            continue

        global_img = env.render()

        for step_idx, action in enumerate(path):
            ego_img = np.asarray(obs["image"], dtype=np.uint8)
            
            data.append({
                "ego_image": Image.fromarray(ego_img),
                "global_image": Image.fromarray(np.asarray(global_img, dtype=np.uint8)),
                "prompt": prompt,
                "action": ID_TO_ACTION[action],
                "action_id": int(action),
                "episode_id": int(episode),
                "step": int(step_idx),
                "env_size": int(env_size),
                "agent_pos": str(unwrapped.agent_pos),
                "agent_dir": int(unwrapped.agent_dir),
            })

            obs, _, terminated, truncated, _ = wrapper.step(action)
            global_img = env.render()

            if terminated or truncated:
                break

    wrapper.close()
    env.close()

    print(f"Собрано {len(data)} примеров")

    features = Features({
        "ego_image": HFImage(),
        "global_image": HFImage(),
        "prompt": Value("string"),
        "action": Value("string"),
        "action_id": Value("int64"),
        "episode_id": Value("int64"),
        "step": Value("int64"),
        "env_size": Value("int64"),
        "agent_pos": Value("string"),
        "agent_dir": Value("int64"),
    })

    dataset = Dataset.from_list(data)
    dataset = dataset.cast(features)

    tmp_path = save_path.with_name(f"{save_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))
    if save_path.exists():
        shutil.rmtree(save_path)
    tmp_path.replace(save_path)
    print(f"Датасет сохранён: {save_path}")


if __name__ == "__main__":
    main()
