import inspect
import types
from transformers import AutoTokenizer, AutoImageProcessor
from peft import PeftModel

from .paths import ensure_project_paths


ensure_project_paths()
from nanoVLM.models.vision_language_model import VisionLanguageModel


def disable_peft_model_card():
    def dummy_create_or_update_model_card(self, save_directory):
        return
    PeftModel.create_or_update_model_card = dummy_create_or_update_model_card

def patch_nanovlm(m):
    if not hasattr(m, "original_forward"):
        m.original_forward = m.forward

    def patched_forward(self, **kwargs):
        sig = inspect.signature(self.original_forward)
        accepted_keys = list(sig.parameters.keys())
        
        kwargs['image'] = kwargs.pop('pixel_values')
            
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
        return self.original_forward(**filtered_kwargs)
    
    m.forward = types.MethodType(patched_forward, m)
    return m

def clear_peft_metadata(model):
    for attr in ("peft_config", "_hf_peft_config_loaded"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except AttributeError:
                pass
    return model

def load_vlm_model(base_model_or_id, adapter_path, device="cuda", is_trainable=False):
    disable_peft_model_card()
    
    if isinstance(base_model_or_id, str):
        model = VisionLanguageModel.from_pretrained(base_model_or_id)
        model.prepare_inputs_for_generation = lambda *args, **kwargs: kwargs
        model.config = getattr(model, "cfg", type('Config', (), {})) 
        model.config.model_type = "nanovlm"
    else:
        model = base_model_or_id

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    image_processor = AutoImageProcessor.from_pretrained(adapter_path)
    
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
