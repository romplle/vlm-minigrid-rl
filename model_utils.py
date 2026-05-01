import inspect
import types
from transformers import AutoTokenizer, AutoImageProcessor
from peft import PeftModel
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
        
        if 'pixel_values' in kwargs:
            kwargs['image'] = kwargs.pop('pixel_values')
            
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
        return self.original_forward(**filtered_kwargs)
    
    m.forward = types.MethodType(patched_forward, m)
    return m

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
    
    model = patch_nanovlm(model)
    
    if not is_trainable:
        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
            
    return model, tokenizer, image_processor
