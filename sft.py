import os
import sys
import inspect
import types
import random

import torch
import wandb
from bitsandbytes.optim import AdamW8bit
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor, GenerationConfig

sys.path.append("./nanoVLM")
from nanoVLM.models.vision_language_model import VisionLanguageModel


def dummy_create_or_update_model_card(self, save_directory):
    return

PeftModel.create_or_update_model_card = dummy_create_or_update_model_card

MODEL_ID = "lusxvr/nanoVLM-222M"
TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M" 
DATASET_PATH = "dataset"
OUTPUT_DIR = "checkpoints/sft_adapter"

BATCH_SIZE = 6
GRAD_ACCUM = 8 
EPOCHS = 3
LR = 2e-5
MAX_SEQ_LEN = 256
IMAGE_TOKEN = "<image>"
USE_WANDB = True
VAL_SAMPLES = 100

if USE_WANDB:
    wandb.init(project="nanoVLM-minigrid", name="sft")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"

model = VisionLanguageModel.from_pretrained(MODEL_ID)
model.tokenizer = tokenizer

model.config = getattr(model, "cfg", type('Config', (), {})) 
model.config.model_type = "nanovlm"
model.original_forward = model.forward

def patched_forward(self, **kwargs):
    sig = inspect.signature(self.original_forward)
    accepted_keys = list(sig.parameters.keys())

    kwargs['image'] = kwargs.pop('pixel_values')
    kwargs['targets'] = kwargs.pop('labels')

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
    return self.original_forward(**filtered_kwargs)

model.forward = types.MethodType(patched_forward, model)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model.prepare_inputs_for_generation = lambda *args, **kwargs: kwargs

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.cuda()

image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")

full_ds = load_from_disk(DATASET_PATH)

split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
val_ds = split_ds["test"]

print(f"Размер train: {len(train_ds)}, Размер val: {len(val_ds)}")

def collate_fn(batch):
    images = []
    for item in batch:
        img = item["ego_image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    prompts = [item["prompt"] for item in batch]
    actions = [str(item["action"]) for item in batch] 

    conversations = []
    for prompt, action in zip(prompts, actions):
        conv = [
            {"role": "user", "content": f"{IMAGE_TOKEN}\n{prompt}"},
            {"role": "assistant", "content": action}
        ]
        conversations.append(conv)

    texts = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN
    )

    processed = image_processor(
        images,
        return_tensors="pt",
        do_resize=True,
        size={"height": 224, "width": 224},
    )

    pixel_values = processed.pixel_values.to(dtype=torch.float32).contiguous()

    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)

    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": tokenized["input_ids"],
        "pixel_values": pixel_values,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

def evaluate_accuracy(model, dataset, num_samples=100):
    model.eval()
    correct = 0
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.generation_config = GenerationConfig()

    eval_size = min(len(dataset), num_samples)
    indices = random.sample(range(len(dataset)), eval_size)
    
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

optimizer = AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
global_step = 0
model.train()

print("\n[Baseline] Валидация до начала обучения...")
baseline_acc = evaluate_accuracy(model, val_ds, num_samples=VAL_SAMPLES)
print(f"Baseline Accuracy: {baseline_acc:.4f}")
if USE_WANDB:
    wandb.log({"val_accuracy": baseline_acc, "epoch": 0})

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        batch = {k: v.cuda() for k, v in batch.items()}

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
    print(f"Epoch {epoch+1} finished. Avg train loss: {avg_loss:.4f}")

    print(f"Запуск оценки Accuracy (Epoch {epoch+1})...")
    val_acc = evaluate_accuracy(model, val_ds, num_samples=VAL_SAMPLES)
    print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.4f}")
    
    if USE_WANDB:
        wandb.log({"val_accuracy": val_acc, "epoch": epoch+1})

    save_dir = f"{OUTPUT_DIR}/epoch-{epoch+1}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)

    print(f"Сохранено в {save_dir}")

save_dir = OUTPUT_DIR
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
image_processor.save_pretrained(save_dir)

if USE_WANDB:
    wandb.finish()
