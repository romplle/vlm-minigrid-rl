import inspect
import random
import types

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, GenerationConfig

from .paths import ensure_project_paths
from .training_utils import ACTION_NAMES, parse_action


ensure_project_paths()
from nanoVLM.models.vision_language_model import VisionLanguageModel


TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
IMAGE_PROCESSOR_ID = "google/siglip-base-patch16-224"
IMAGE_TOKEN = "<image>"
NANOVLM_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"


def disable_peft_model_card():
    def dummy_create_or_update_model_card(self, save_directory):
        return

    PeftModel.create_or_update_model_card = dummy_create_or_update_model_card


def patch_nanovlm(model):
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward

    def patched_forward(self, **kwargs):
        sig = inspect.signature(self.original_forward)
        accepted_keys = list(sig.parameters.keys())

        if "pixel_values" in kwargs:
            kwargs["image"] = kwargs.pop("pixel_values")
        if "labels" in kwargs:
            kwargs["targets"] = kwargs.pop("labels")

        filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted_keys}
        return self.original_forward(**filtered_kwargs)

    model.forward = types.MethodType(patched_forward, model)
    return model


def configure_nanovlm_for_peft(model):
    model.prepare_inputs_for_generation = lambda *args, **kwargs: kwargs
    model.config = getattr(model, "cfg", type("Config", (), {}))
    model.config.model_type = "nanovlm"
    return model


def clear_peft_metadata(model):
    for attr in ("peft_config", "_hf_peft_config_loaded"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except AttributeError:
                pass
    return model


def load_project_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = NANOVLM_CHAT_TEMPLATE
    return tokenizer


def load_project_image_processor():
    return AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_ID)


def load_sft_training_model(model_id):
    disable_peft_model_card()

    tokenizer = load_project_tokenizer()

    model = VisionLanguageModel.from_pretrained(model_id)
    model.tokenizer = tokenizer
    model = configure_nanovlm_for_peft(model)
    model = patch_nanovlm(model)
    model = prepare_model_for_kbit_training(model)

    image_processor = load_project_image_processor()
    return model, tokenizer, image_processor


def load_base_vlm_model(model_id, device="cuda", is_trainable=False):
    tokenizer = load_project_tokenizer()

    model = VisionLanguageModel.from_pretrained(model_id)
    model.tokenizer = tokenizer
    model = configure_nanovlm_for_peft(model)
    model = patch_nanovlm(model)

    image_processor = load_project_image_processor()

    if not is_trainable:
        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer, image_processor


def load_vlm_model(base_model_or_id, adapter_path, device="cuda", is_trainable=False):
    disable_peft_model_card()

    if isinstance(base_model_or_id, str):
        model = VisionLanguageModel.from_pretrained(base_model_or_id)
        model = configure_nanovlm_for_peft(model)
    else:
        model = base_model_or_id

    tokenizer = load_project_tokenizer()
    image_processor = load_project_image_processor()

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = clear_peft_metadata(model)
    model = patch_nanovlm(model)

    if not is_trainable:
        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer, image_processor


def load_vlm_model_with_adapters(base_model_or_id, adapter_paths, device="cuda", is_trainable=False):
    if isinstance(adapter_paths, (str, bytes)):
        adapter_paths = [adapter_paths]

    model = base_model_or_id
    tokenizer = None
    image_processor = None

    for idx, adapter_path in enumerate(adapter_paths):
        trainable_step = is_trainable and idx == len(adapter_paths) - 1
        model, tokenizer, image_processor = load_vlm_model(
            model,
            adapter_path,
            device=device,
            is_trainable=trainable_step,
        )

    return model, tokenizer, image_processor


def save_model_bundle(model, tokenizer, image_processor, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)


def format_action_prefix(prompt):
    return f"User: {IMAGE_TOKEN}\n{prompt}\nAssistant:"


def preprocess_images(image_processor, images, device=None):
    rgb_images = [image.convert("RGB") if hasattr(image, "mode") and image.mode != "RGB" else image for image in images]
    processed = image_processor(
        rgb_images,
        return_tensors="pt",
        do_resize=True,
        size={"height": 224, "width": 224},
    )
    pixel_values = processed.pixel_values.to(dtype=torch.float32).contiguous()
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if device is not None:
        pixel_values = pixel_values.to(device)
    return pixel_values


def preprocess_image(image_processor, image, device=None):
    return preprocess_images(image_processor, [image], device=device)


def make_sft_collate_fn(tokenizer, image_processor, max_seq_len=256):
    def collate_fn(batch):
        images = [item["ego_image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        actions = [str(item["action"]) for item in batch]

        prefix_texts = [format_action_prefix(prompt) for prompt in prompts]
        full_texts = [
            f"{prefix} {action}{tokenizer.eos_token}"
            for prefix, action in zip(prefix_texts, actions)
        ]

        tokenized = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )

        labels = tokenized["input_ids"].clone()
        for row_idx, prefix in enumerate(prefix_texts):
            prefix_ids = tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len,
            )["input_ids"]
            labels[row_idx, : min(len(prefix_ids), labels.size(1))] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "pixel_values": preprocess_images(image_processor, images),
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    return collate_fn


def build_inference_inputs(tokenizer, image_processor, image, prompt, device):
    encoded = tokenizer(format_action_prefix(prompt), return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    pixel_values = preprocess_image(image_processor, image, device=device)
    return input_ids, pixel_values, attention_mask


def generate_action(model, tokenizer, image_processor, image, prompt, device, max_new_tokens=1):
    input_ids, pixel_values, _ = build_inference_inputs(tokenizer, image_processor, image, prompt, device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, pixel_values, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
    action_name, action_idx = parse_action(generated_text)
    return action_name, action_idx, generated_text


def evaluate_action_accuracy(model, tokenizer, image_processor, dataset, num_samples=100, seed=42, device="cuda"):
    was_training = model.training
    model.eval()
    model.generation_config = GenerationConfig()

    rng = random.Random(seed)
    eval_size = min(len(dataset), num_samples)
    indices = rng.sample(range(len(dataset)), eval_size)
    correct = 0

    for idx in tqdm(indices, desc="Оценка Accuracy"):
        item = dataset[idx]
        action_name, _, _ = generate_action(
            model,
            tokenizer,
            image_processor,
            item["ego_image"].convert("RGB"),
            item["prompt"],
            device,
        )
        if action_name == item["action"]:
            correct += 1

    if was_training:
        model.train()
    return correct / eval_size


def get_logits(model, input_ids, pixel_values, attention_mask=None):
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
    hidden_states = outputs[1] if outputs[0].dim() == 0 else outputs[0]
    return model.decoder.head(hidden_states)


def get_vocab_last_logits(model, input_ids, pixel_values, attention_mask=None):
    logits = get_logits(model, input_ids, pixel_values, attention_mask)
    return logits[0, -1, :]


def action_token_texts(action_names=ACTION_NAMES):
    return [f" {action}" for action in action_names]


def action_token_ids(tokenizer, action_texts=ACTION_NAMES):
    return [
        tokenizer.encode(action, add_special_tokens=False)
        for action in action_token_texts(action_texts)
    ]


def single_token_action_ids(action_ids_list):
    if all(len(ids) == 1 for ids in action_ids_list):
        return [ids[0] for ids in action_ids_list]
    return None


def seq_logprob_given_prefix(model, tokenizer, input_ids_prefix, pixel_values, action_token_ids_):
    device = next(model.parameters()).device
    prefix = input_ids_prefix.to(device)
    action = torch.tensor([action_token_ids_], dtype=torch.long, device=device)
    full = torch.cat([prefix, action], dim=1)

    logits = get_logits(model, full, pixel_values)
    log_probs = F.log_softmax(logits[:, :-1, :].contiguous(), dim=-1)

    prefix_len = prefix.size(1)
    total = torch.tensor(0.0, device=device)
    for offset, token_id in enumerate(action[0]):
        label_pos = prefix_len + offset
        total = total + log_probs[0, label_pos - 1, token_id]
    return total


def score_action_logits(
    model,
    tokenizer,
    input_ids,
    pixel_values,
    action_ids_list,
    action_single_ids=None,
    attention_mask=None,
):
    vocab_last = get_vocab_last_logits(model, input_ids, pixel_values, attention_mask=attention_mask)
    if action_single_ids is not None and max(action_single_ids) < vocab_last.size(0):
        action_id_tensor = torch.tensor(action_single_ids, dtype=torch.long, device=vocab_last.device)
        return vocab_last.index_select(0, action_id_tensor)

    return torch.stack([
        seq_logprob_given_prefix(model, tokenizer, input_ids, pixel_values, ids)
        if ids else torch.tensor(-1e9, device=vocab_last.device)
        for ids in action_ids_list
    ])


def get_action_distribution(
    model,
    tokenizer,
    image_processor,
    image,
    prompt,
    device,
    action_ids_list,
    action_single_ids=None,
):
    input_ids, pixel_values, attention_mask = build_inference_inputs(tokenizer, image_processor, image, prompt, device)
    action_logits = score_action_logits(
        model,
        tokenizer,
        input_ids,
        pixel_values,
        action_ids_list,
        action_single_ids=action_single_ids,
        attention_mask=attention_mask,
    )
    return action_logits, input_ids, pixel_values
