import os
import sys
import random

import torch
import torch.nn.functional as F
import gymnasium as gym
import wandb
from tqdm import tqdm

from datasets import load_from_disk
from transformers import GenerationConfig
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit

from minigrid.wrappers import RGBImgPartialObsWrapper
from minigrid.core.world_object import Goal

sys.path.append("./nanoVLM")
from nanoVLM.models.vision_language_model import VisionLanguageModel
from model_utils import load_vlm_model


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

OUTPUT_DIR = "checkpoints/grpo_action_adapter"
SFT_ADAPTER_PATH = "checkpoints/sft_adapter"
DATASET_PATH = "dataset"
ENV_SIZE = 8
TILE_SIZE = 32

G = 4
EPISODES = 100
MAX_STEPS = 25
LR = 5e-6
EPSILON = 0.2
BETA = 0.05
USE_WANDB = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if USE_WANDB:
    wandb.init(project="nanoVLM-minigrid", name="grpo-action-direct")

BASE_MODEL_ID = "lusxvr/nanoVLM-222M"

ref_model, tokenizer, image_processor = load_vlm_model(
    BASE_MODEL_ID, SFT_ADAPTER_PATH, DEVICE, is_trainable=False
)

active_model, _, _ = load_vlm_model(
    BASE_MODEL_ID, SFT_ADAPTER_PATH, DEVICE, is_trainable=True
)

lora_config = LoraConfig(
    r=32, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

active_model = get_peft_model(active_model, lora_config).to(DEVICE)
active_model.train()

optimizer = AdamW8bit(active_model.parameters(), lr=LR)

action_texts = ["left", "right", "forward"]
action_ids_list = [tokenizer.encode(a, add_special_tokens=False) for a in action_texts]

if all(len(ids) == 1 for ids in action_ids_list):
    action_single_ids = [ids[0] for ids in action_ids_list]

def get_logits(model, input_ids, pixel_values, attention_mask=None):
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
    logits = outputs[1] if outputs[0].dim() == 0 else outputs[0]
    head = model.decoder.head
    logits = head(logits)
                
    return logits

def get_vocab_last_logits(model, input_ids, pixel_values, attention_mask=None):
    logits = get_logits(model, input_ids, pixel_values, attention_mask)
    return logits[0, -1, :]

def seq_logprob_given_prefix(model, tokenizer, input_ids_prefix, pixel_values, action_token_ids):
    device = next(model.parameters()).device
    prefix = input_ids_prefix.to(device)
    action = torch.tensor([action_token_ids], dtype=torch.long, device=device)
    full = torch.cat([prefix, action], dim=1)
    
    logits = get_logits(model, full, pixel_values)
    
    shift_logits = logits[:, :-1, :].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    L = prefix.size(1)
    K = action.size(1)
    total = torch.tensor(0.0, device=device)
    for k in range(K):
        label_pos = L + k
        tok = action[0, k]
        total = total + log_probs[0, label_pos - 1, tok]
    return total

def get_action_distribution(model, tokenizer, ego_image, prompt):
    text = f"User: <image>\n{prompt}\nAssistant: "
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    image_inputs = image_processor(ego_image, return_tensors="pt", do_resize=True, size={"height": 224, "width": 224})
    pixel_values = image_inputs["pixel_values"].to(torch.float32).to(DEVICE)

    vocab_last = get_vocab_last_logits(model, input_ids, pixel_values, attention_mask=attention_mask)
    vocab_size = vocab_last.size(0)

    if action_single_ids is not None:
        max_id = max(action_single_ids)
        if max_id < vocab_size:
            aid_tensor = torch.tensor(action_single_ids, dtype=torch.long, device=DEVICE)
            action_logits = vocab_last.index_select(0, aid_tensor)
            return action_logits, input_ids, pixel_values

    action_logits = []
    for txt in action_texts:
        token_ids = tokenizer.encode(txt, add_special_tokens=False)
        if len(token_ids) == 0:
            action_logits.append(torch.tensor(-1e9, device=DEVICE))  
            continue
        lp = seq_logprob_given_prefix(model, tokenizer, input_ids, pixel_values, token_ids)
        action_logits.append(lp.detach())
    action_logits = torch.stack(action_logits)
    return action_logits, input_ids, pixel_values

def create_env():
    env = gym.make(f"MiniGrid-Empty-{ENV_SIZE}x{ENV_SIZE}-v0", render_mode="rgb_array")
    return RGBImgPartialObsWrapper(env, tile_size=TILE_SIZE)

def evaluate_accuracy(model, dataset, num_samples=100, seed=42):
    model.eval()
    rng = random.Random(seed)
    correct = 0
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.generation_config = GenerationConfig()

    eval_size = min(len(dataset), num_samples)
    indices = rng.sample(range(len(dataset)), eval_size)
    
    for idx in tqdm(indices, desc="Оценка Accuracy"):
        item = dataset[idx]
        image = item["ego_image"].convert("RGB")
        prompt = item["prompt"]
        true_action = item["action"]

        text = f"User: <image>\n{prompt}\nAssistant: "
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]

        image_inputs = image_processor(
            image, 
            return_tensors="pt", 
            do_resize=True, 
            size={"height": 224, "width": 224},
        )
        pixel_values = image_inputs.pixel_values.to(torch.float32).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                pixel_values, 
                max_new_tokens=1,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()

        pred_action = None
        if "left" in generated_text:
            pred_action = "left"
        elif "right" in generated_text:
            pred_action = "right"
        elif "forward" in generated_text:
            pred_action = "forward"
            
        if pred_action == true_action:
            correct += 1

    model.train()
    return correct / eval_size

full_ds = load_from_disk(DATASET_PATH)

split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
val_ds = split_ds["test"]

prompt = full_ds[0]["prompt"]

# GRPO
global_step = 0
env = create_env()

for episode in range(EPISODES):
    seed = random.randint(0, 100000)
    
    group_trajectories = []
    group_returns = []

    if episode % 25 == 0:
        val_acc = evaluate_accuracy(active_model, val_ds, num_samples=100)
        print(f"Validation Accuracy: {val_acc:.4f}")

    for g in range(G):
        obs, _ = env.reset(seed=seed)
        unwrapped = env.unwrapped
        
        unwrapped.place_agent()
        for x in range(unwrapped.grid.width):
            for y in range(unwrapped.grid.height):
                cell = unwrapped.grid.get(x, y)
                if cell and cell.type == "goal":
                    unwrapped.grid.set(x, y, None)
        unwrapped.place_obj(Goal())
        obs = env.observation(unwrapped.gen_obs())
        ego_img = obs["image"]
        action_logits, input_ids, pixel_values = get_action_distribution(ref_model, tokenizer, ego_img, prompt)

        trajectory = []
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            ego_img = obs["image"]
            
            with torch.no_grad():
                logits, input_ids, pixel_values = get_action_distribution(
                    active_model, tokenizer, ego_img, prompt
                )
                
                probs = F.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
                action_log_prob = torch.log(probs[action_idx] + 1e-12)
                
                ref_full_logits = get_logits(ref_model, input_ids, pixel_values)
                ref_vocab_last = ref_full_logits[0, -1, :]
                
                aid_tensor = torch.tensor(action_single_ids, dtype=torch.long, device=DEVICE)
                ref_action_logits = ref_vocab_last.index_select(0, aid_tensor)
                ref_probs = F.softmax(ref_action_logits, dim=-1)
                ref_log_prob = torch.log(ref_probs[action_idx] + 1e-12)

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

    optimizer.zero_grad()
    loss_total = 0.0
    steps_count = 0

    for g in range(G):
        adv = advantages[g]
        for step_data in group_trajectories[g]:
            input_ids = step_data["input_ids"]
            pixel_values = step_data["pixel_values"]
            action_idx = step_data["action_idx"]
            old_log_prob = step_data["old_log_prob"]
            ref_log_prob = step_data["ref_log_prob"]

            logits = get_logits(active_model, input_ids, pixel_values)
            vocab_last = logits[0, -1, :]
            
            if action_single_ids is not None:
                aid_tensor = torch.tensor(action_single_ids, dtype=torch.long, device=DEVICE)
                new_action_logits = vocab_last.index_select(0, aid_tensor)
            else:
                new_action_logits = []
                for txt in action_texts:
                    token_ids = tokenizer.encode(txt, add_special_tokens=False)
                    lp = seq_logprob_given_prefix(active_model, tokenizer, input_ids, pixel_values, token_ids)
                    new_action_logits.append(lp)
                new_action_logits = torch.stack(new_action_logits)

            new_probs = F.softmax(new_action_logits, dim=-1)
            new_log_prob = torch.log(new_probs[action_idx] + 1e-12)

            ratio = torch.exp(new_log_prob - old_log_prob)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * adv
            
            kl = torch.exp(ref_log_prob - new_log_prob) - (ref_log_prob - new_log_prob) - 1.0

            loss = - (torch.min(surr1, surr2) - BETA * kl)
            loss.backward()
            
            loss_total += loss.item()
            steps_count += 1

    optimizer.step()

    if USE_WANDB:
        wandb.log({"train/grpo_loss": loss_total / steps_count})

val_acc = evaluate_accuracy(active_model, val_ds, num_samples=100)
print(f"Validation Accuracy: {val_acc:.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
active_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
image_processor.save_pretrained(OUTPUT_DIR)
print(f"GRPO-обучение завершено. Модель сохранена в {OUTPUT_DIR}")

if USE_WANDB:
    wandb.finish()
