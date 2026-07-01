import argparse
import os

import torch
import wandb
from bitsandbytes.optim import AdamW8bit
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import bootstrap
bootstrap()

from vlm_minigrid_rl.model_utils import (
    evaluate_action_accuracy,
    load_sft_training_model,
    make_sft_collate_fn,
    save_model_bundle,
)
from vlm_minigrid_rl.paths import project_path
from vlm_minigrid_rl.training_utils import majority_action_baseline, set_global_seed, split_dataset_by_episode


MODEL_ID = "lusxvr/nanoVLM-222M"
DATASET_PATH = "datasets/dataset_8x8"
OUTPUT_DIR = "checkpoints/sft_adapter_8x8"

BATCH_SIZE = 6
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-5
MAX_SEQ_LEN = 256
USE_WANDB = True
VAL_SAMPLES = 100
SEED = 42
VAL_SPLIT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Train NanoVLM with SFT on MiniGrid expert data.")
    parser.add_argument("--env-size", type=int, default=8, choices=[8, 16])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--val-samples", type=int, default=VAL_SAMPLES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def default_dataset_path(env_size):
    return f"datasets/dataset_{env_size}x{env_size}"


def default_output_dir(env_size):
    return f"checkpoints/sft_adapter_{env_size}x{env_size}/epoch-3"

args = parse_args()
experiment_name = f"{args.env_size}x{args.env_size}"
DATASET_PATH = str(project_path(args.dataset_path or default_dataset_path(args.env_size)))
OUTPUT_DIR = str(project_path(args.output_dir or default_output_dir(args.env_size)))
EPOCHS = args.epochs if args.epochs is not None else EPOCHS
VAL_SPLIT = args.val_split if args.val_split is not None else VAL_SPLIT
VAL_SAMPLES = args.val_samples
LR = args.lr
USE_WANDB = USE_WANDB and not args.no_wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_global_seed(SEED)

if USE_WANDB:
    wandb.init(project="nanoVLM-minigrid", name=args.wandb_name or f"sft-{experiment_name}")

model, tokenizer, image_processor = load_sft_training_model(MODEL_ID)

lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(DEVICE)

full_ds = load_from_disk(DATASET_PATH)
train_ds, val_ds, val_episodes = split_dataset_by_episode(full_ds, test_size=VAL_SPLIT, seed=SEED)
majority_baseline = majority_action_baseline(train_ds, val_ds)

print(f"Размер train: {len(train_ds)}, Размер val: {len(val_ds)}")
print(f"Episode-level split: train episodes={len(set(train_ds['episode_id']))}, val episodes={len(val_episodes)}")
print(
    "Majority baseline: "
    f"{majority_baseline['action']} -> {majority_baseline['accuracy']:.4f} "
    f"on validation"
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=make_sft_collate_fn(tokenizer, image_processor, max_seq_len=MAX_SEQ_LEN),
)

optimizer = AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
global_step = 0
model.train()

print("\n[Baseline] Валидация до начала обучения...")
baseline_acc = evaluate_action_accuracy(
    model, tokenizer, image_processor, val_ds, num_samples=VAL_SAMPLES, seed=SEED, device=DEVICE
)
print(f"Baseline Accuracy: {baseline_acc:.4f}")
if USE_WANDB:
    wandb.log({"val_accuracy": baseline_acc, "epoch": 0})

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in pbar:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}

        outputs = model(**batch)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss = loss / GRAD_ACCUM
        loss.backward()

        if (global_step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * GRAD_ACCUM
        epoch_loss += current_loss
        global_step += 1

        pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        if USE_WANDB and global_step % 10 == 0:
            wandb.log({"train_loss": current_loss, "step": global_step})

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1} finished. Avg train loss: {avg_loss:.4f}")

    print(f"Запуск оценки Accuracy (Epoch {epoch + 1})...")
    val_acc = evaluate_action_accuracy(
        model, tokenizer, image_processor, val_ds, num_samples=VAL_SAMPLES, seed=SEED, device=DEVICE
    )
    print(f"Epoch {epoch + 1} Validation Accuracy: {val_acc:.4f}")

    if USE_WANDB:
        wandb.log({"val_accuracy": val_acc, "epoch": epoch + 1})

    save_dir = f"{OUTPUT_DIR}/epoch-{epoch + 1}"
    os.makedirs(save_dir, exist_ok=True)
    save_model_bundle(model, tokenizer, image_processor, save_dir)
    print(f"Сохранено в {save_dir}")

print(f"SFT-обучение завершено. Последний checkpoint сохранён в {OUTPUT_DIR}/epoch-{EPOCHS}")

if USE_WANDB:
    wandb.finish()
