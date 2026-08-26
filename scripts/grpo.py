import os
import random
import argparse

import torch
import torch.nn.functional as F
import wandb

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

from _bootstrap import bootstrap
bootstrap()

from vlm_minigrid_rl.env_profiles import add_profile_cli_args, resolve_profile
from vlm_minigrid_rl.experiment_config import (
    BASE_MODEL_ID,
    DEFAULT_ENV_SIZE,
    DEFAULT_GRPO_CHECKPOINT_INTERVAL,
    DEFAULT_GRPO_TRAIN_EPISODES,
    DEFAULT_SEED,
    DEFAULT_TILE_SIZE,
    WANDB_PROJECT,
    dataset_dir_for_profile,
    grpo_adapter_root_for_profile,
    sft_adapter_epoch_dir,
)
from vlm_minigrid_rl.minigrid_utils import create_minigrid_env, default_train_max_steps, reset_env_with_goal
from vlm_minigrid_rl.model_utils import (
    action_token_ids,
    evaluate_action_accuracy,
    generate_action,
    get_action_distribution,
    load_vlm_model,
    save_model_bundle,
    score_action_logits,
    score_action_logits_batched,
    seq_logprob_given_prefix,
    single_token_action_ids,
)
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import (
    GOAL_COLORS,
    majority_action_baseline,
    set_global_seed,
    split_dataset_by_episode,
)


G = 16
UPDATE_BATCH_SIZE = 32
LR = 2e-5
EPSILON = 0.2
BETA = 0.05
LORA_DROPOUT = 0.05
USE_WANDB = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train NanoVLM with GRPO-style RL in MiniGrid.")
    parser.add_argument("--env-size", type=int, default=DEFAULT_ENV_SIZE, choices=[8, 16])
    add_profile_cli_args(parser)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_GRPO_TRAIN_EPISODES)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_GRPO_CHECKPOINT_INTERVAL)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="GRPO rollout horizon. Default: profile train_max_steps (L_max).",
    )
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--update-batch-size",type=int, default=UPDATE_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--goal-color", default="green", choices=GOAL_COLORS)
    parser.add_argument("--prompt-goal-color", default=None, choices=GOAL_COLORS)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
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
OUTPUT_DIR = str(
    project_path(args.output_dir) if args.output_dir else grpo_adapter_root_for_profile(PROFILE)
)
EPISODES = args.episodes
CHECKPOINT_INTERVAL = args.checkpoint_interval
MAX_STEPS = (
    args.max_steps
    if args.max_steps is not None
    else default_train_max_steps(ENV_SIZE, profile=PROFILE)
)
VAL_SPLIT = args.val_split if args.val_split is not None else PROFILE.val_split
LR = args.lr
EPSILON = args.epsilon
BETA = args.beta
LORA_DROPOUT = args.lora_dropout
UPDATE_BATCH_SIZE = args.update_batch_size
SEED = args.seed
GOAL_COLOR = args.goal_color
PROMPT_GOAL_COLOR = args.prompt_goal_color or args.goal_color
USE_WANDB = USE_WANDB and not args.no_wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_global_seed(SEED)

if USE_WANDB:
    wandb.init(project=WANDB_PROJECT, name=args.wandb_name or f"grpo-{PROFILE.name}")

ref_model, tokenizer, image_processor = load_vlm_model(
    BASE_MODEL_ID, SFT_ADAPTER_PATH, DEVICE, is_trainable=False
)

active_model, _, _ = load_vlm_model(
    BASE_MODEL_ID, SFT_ADAPTER_PATH, DEVICE, is_trainable=True
)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none", 
    task_type="CAUSAL_LM"
)

active_model = get_peft_model(active_model, lora_config).to(DEVICE)
active_model.train()

optimizer = AdamW(active_model.parameters(), lr=LR)

action_ids_list = action_token_ids(tokenizer)
action_single_ids = single_token_action_ids(action_ids_list)

full_ds = load_from_disk(DATASET_PATH)
train_ds, val_ds, val_episodes = split_dataset_by_episode(full_ds, test_size=VAL_SPLIT, seed=SEED)
majority_baseline = majority_action_baseline(train_ds, val_ds)
print(f"Episode-level split: train={len(train_ds)}, val={len(val_ds)}, val episodes={len(val_episodes)}")
print(f"Majority baseline on val: {majority_baseline['action']} -> {majority_baseline['accuracy']:.4f}")
print(f"Rollout goal color: {GOAL_COLOR} | prompt goal color: {PROMPT_GOAL_COLOR}")
print(
    f"Config: lr={LR}, epsilon={EPSILON}, beta={BETA}, lora_dropout={LORA_DROPOUT}, "
    f"seed={SEED}, rollout=generate, update_batch_size={UPDATE_BATCH_SIZE}, "
    f"train_max_steps={MAX_STEPS} (eval_max_steps={PROFILE.eval_max_steps})"
)

prompt = PROFILE.prompt(PROMPT_GOAL_COLOR)


def select_rollout_action(active_model, ego_img):
    action_name, action_idx, _ = generate_action(
        active_model,
        tokenizer,
        image_processor,
        ego_img,
        prompt,
        DEVICE,
    )
    if action_name is None or action_idx is None:
        return None
    with torch.no_grad():
        logits, input_ids, pixel_values = get_action_distribution(
            active_model,
            tokenizer,
            image_processor,
            ego_img,
            prompt,
            DEVICE,
            action_ids_list,
            action_single_ids=action_single_ids,
        )
        probs = F.softmax(logits, dim=-1)
        action_log_prob = torch.log(probs[action_idx] + 1e-12)
    return action_idx, action_log_prob, input_ids, pixel_values


def save_checkpoint(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_model_bundle(active_model, tokenizer, image_processor, save_dir)
    print(f"Сохранено в {save_dir}")


# GRPO
global_step = 0
env = create_minigrid_env(ENV_SIZE, tile_size=DEFAULT_TILE_SIZE, profile=PROFILE, max_steps=MAX_STEPS)
rng = random.Random(SEED)

for episode in range(EPISODES):
    seed = rng.randint(0, 100000)
    
    group_trajectories = []
    group_returns = []

    if episode % 25 == 0:
        val_acc = evaluate_action_accuracy(
            active_model, tokenizer, image_processor, val_ds, num_samples=100, seed=SEED, device=DEVICE
        )
        print(f"Validation Accuracy: {val_acc:.4f}")

    for g in range(G):
        obs = reset_env_with_goal(env, seed, goal_color=GOAL_COLOR)

        trajectory = []
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            ego_img = obs["image"]

            with torch.no_grad():
                rollout = select_rollout_action(active_model, ego_img)
                if rollout is None:
                    break
                action_idx, action_log_prob, input_ids, pixel_values = rollout

                if action_single_ids is not None:
                    ref_action_logits = score_action_logits(
                        ref_model,
                        tokenizer,
                        input_ids,
                        pixel_values,
                        action_ids_list,
                        action_single_ids=action_single_ids,
                    )
                    ref_probs = F.softmax(ref_action_logits, dim=-1)
                    ref_log_prob = torch.log(ref_probs[action_idx] + 1e-12)
                else:
                    ref_log_prob = seq_logprob_given_prefix(
                        ref_model, tokenizer, input_ids, pixel_values, action_ids_list[action_idx]
                    )

            obs, reward, terminated, truncated, _ = env.step(action_idx)
            
            trajectory.append({
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "action_idx": action_idx,
                "old_log_prob": action_log_prob,
                "ref_log_prob": ref_log_prob
            })

            episode_reward += reward
            if terminated or truncated:
                break
                
        group_trajectories.append(trajectory)
        group_returns.append(episode_reward)

    returns_tensor = torch.tensor(group_returns, dtype=torch.float32).to(DEVICE)
    mean_return = returns_tensor.mean()
    std_return = returns_tensor.std() + 1e-8
    advantages = (returns_tensor - mean_return) / std_return

    success_rate = (returns_tensor > 0).float().mean().item()
    if USE_WANDB:
        wandb.log({
            "train/mean_return": mean_return.item(),
            "train/success_rate": success_rate,
            "episode": episode
        })

    print(f"Ep {episode+1}/{EPISODES} | Mean Return: {mean_return.item():.3f} | Success: {success_rate*100:.1f}%")

    flat_input_ids = []
    flat_pixel_values = []
    flat_action_idx = []
    flat_old_log_prob = []
    flat_ref_log_prob = []
    flat_advantages = []
    flat_scales = []

    for g in range(G):
        trajectory = group_trajectories[g]
        traj_len = len(trajectory)
        if traj_len == 0:
            continue
        scale = 1.0 / (traj_len * G)
        adv = advantages[g]
        for step_data in trajectory:
            flat_input_ids.append(step_data["input_ids"])
            flat_pixel_values.append(step_data["pixel_values"])
            flat_action_idx.append(int(step_data["action_idx"]))
            flat_old_log_prob.append(step_data["old_log_prob"].detach())
            flat_ref_log_prob.append(step_data["ref_log_prob"].detach())
            flat_advantages.append(adv)
            flat_scales.append(scale)

    optimizer.zero_grad()
    episode_loss = 0.0
    num_steps = len(flat_input_ids)

    for start in range(0, num_steps, UPDATE_BATCH_SIZE):
        end = min(start + UPDATE_BATCH_SIZE, num_steps)
        input_ids = torch.cat(flat_input_ids[start:end], dim=0)
        pixel_values = torch.cat(flat_pixel_values[start:end], dim=0)
        action_idx = torch.tensor(flat_action_idx[start:end], dtype=torch.long, device=DEVICE)
        old_log_prob = torch.stack(flat_old_log_prob[start:end]).to(DEVICE)
        ref_log_prob = torch.stack(flat_ref_log_prob[start:end]).to(DEVICE)
        adv = torch.stack(flat_advantages[start:end]).to(DEVICE)
        scales = torch.tensor(flat_scales[start:end], dtype=torch.float32, device=DEVICE)

        new_action_logits = score_action_logits_batched(
            active_model,
            tokenizer,
            input_ids,
            pixel_values,
            action_ids_list,
            action_single_ids=action_single_ids,
        )
        new_probs = F.softmax(new_action_logits, dim=-1)
        new_log_prob = torch.log(new_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1) + 1e-12)

        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * adv
        kl = torch.exp(ref_log_prob - new_log_prob) - (ref_log_prob - new_log_prob) - 1.0
        step_loss = -(torch.min(surr1, surr2) - BETA * kl)
        batch_loss = (step_loss * scales).sum()

        batch_loss.backward()
        episode_loss += batch_loss.item()

    if num_steps > 0:
        optimizer.step()

    if USE_WANDB:
        wandb.log({"train/grpo_loss": episode_loss, "episode": episode})

    if (episode + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(f"{OUTPUT_DIR}/episode-{episode+1}")

val_acc = evaluate_action_accuracy(active_model, tokenizer, image_processor, val_ds, num_samples=100, seed=SEED, device=DEVICE)
print(f"Validation Accuracy: {val_acc:.4f}")

save_checkpoint(OUTPUT_DIR)
print(f"GRPO-обучение завершено. Модель сохранена в {OUTPUT_DIR}")

if USE_WANDB:
    wandb.finish()
