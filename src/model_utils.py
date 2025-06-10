import logging
import torch

logger = logging.getLogger(__name__)

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
MLLAMA = 'mllama'
MODEL2BACKBONE = {  # keys are from hf_config.model_type
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'mllama': MLLAMA,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

vlm_image_tokens = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|vision_start|><|image_pad|><|vision_end|>",
    MLLAMA: "<image>",
}

def get_backbone_name(hf_config):
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]

