from collections import deque

import gymnasium as gym
import numpy as np
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value
from tqdm import tqdm

from minigrid.wrappers import RGBImgPartialObsWrapper
from minigrid.core.world_object import Goal

ENV_SIZE = 8
NUM_EPISODES = 1000
TILE_SIZE = 32
SEED_BASE = 42
SAVE_PATH = "dataset"


def get_shortest_path_actions(env):
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir

    goal_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
                break
        if goal_pos:
            break
    if not goal_pos:
        return []

    dir_delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    start = (agent_pos[0], agent_pos[1], agent_dir)
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (x, y, d), path = queue.popleft()
        if (x, y) == goal_pos:
            return path
        for a in range(3):
            nx, ny, nd = x, y, d
            if a == 0:
                nd = (d - 1) % 4
            elif a == 1:
                nd = (d + 1) % 4
            else:
                dx, dy = dir_delta[d]
                nx = x + dx
                ny = y + dy
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue
                cell = grid.get(nx, ny)
                if cell is not None and cell.type == "wall":
                    continue
            new_state = (nx, ny, nd)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [a]))
    return []


def main():
    env_id = f"MiniGrid-Empty-{ENV_SIZE}x{ENV_SIZE}-v0"
    print(f"Создаём датасет: {env_id}")

    env = gym.make(env_id, render_mode="rgb_array")
    wrapper = RGBImgPartialObsWrapper(env, tile_size=TILE_SIZE)

    data = []

    for episode in tqdm(range(NUM_EPISODES), desc="Генерация траекторий"):
        seed = SEED_BASE + episode
        obs, _ = wrapper.reset(seed=seed)
        unwrapped = wrapper.unwrapped
        unwrapped.place_agent()

        grid = unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    grid.set(x, y, None)
        
        unwrapped.place_obj(Goal())
        obs = wrapper.observation(unwrapped.gen_obs())

        path = get_shortest_path_actions(wrapper)
        if not path:
            continue

        if episode < 5:
            print(f"[DEBUG] ep={episode} | agent={unwrapped.agent_pos} dir={unwrapped.agent_dir} "
                  f"| goal={unwrapped.goal_pos if hasattr(unwrapped, 'goal_pos') else 'N/A'}")

        global_img = env.render()

        for step_idx, action in enumerate(path):
            ego_img = np.asarray(obs["image"], dtype=np.uint8)
            
            prompt = (
                "You are a robot in a 2D grid world. You see a 7x7 partial RGB view in front of you.\n"
                "Your mission: get to the green goal square as quickly as possible.\n"
                "Choose the next action: left, right or forward."
            )

            action_map = {0: "left", 1: "right", 2: "forward"}
            data.append({
                "ego_image": Image.fromarray(ego_img),
                "global_image": Image.fromarray(np.asarray(global_img, dtype=np.uint8)),
                "prompt": prompt,
                "action": action_map[action],
                "action_id": int(action),
                "episode_id": int(episode),
                "step": int(step_idx),
                "env_size": int(ENV_SIZE),
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
    dataset.save_to_disk(SAVE_PATH)
    print(f"Датасет сохранён → {SAVE_PATH}")


if __name__ == "__main__":
    main()
