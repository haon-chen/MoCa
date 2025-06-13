from transformers import AutoProcessor, AutoConfig
from src.vlm_backbone.qwen2_5_vl_embed.qwen2_5_vl_embed import Qwen2_5ForEmbedding
import torch
from PIL import Image
import torch.nn.functional as F


def compute_similarity(q_reps, p_reps):
    return torch.matmul(q_reps, p_reps.transpose(0, 1))

from huggingface_hub import login
login("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")
model_name = "Haon-Chen/mmE5-qwen25-7B"
processor_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# Load Processor and Model
processor = AutoProcessor.from_pretrained(processor_name)
config = AutoConfig.from_pretrained(model_name)
model = Qwen2_5ForEmbedding.from_pretrained(
    model_name, config=config, 
    torch_dtype=torch.bfloat16
).to("cuda")
model.eval()
# Image + Text -> Text
inputs = processor(text='<|vision_start|><|image_pad|><|vision_end|>Represent the given image with the following question: What is in the image\n', images=[Image.open(
    'figures/example.jpg')], return_tensors="pt").to("cuda")
qry_output = F.normalize(model(**inputs, return_dict=True, output_hidden_states=True), dim=-1)

string = 'A cat and a dog'
text_inputs = processor(text=string, return_tensors="pt").to("cuda")
tgt_output = F.normalize(model(**text_inputs, return_dict=True, output_hidden_states=True), dim=-1)
print(string, '=', compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.4531]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
text_inputs = processor(text=string, return_tensors="pt").to("cuda")
tgt_output = F.normalize(model(**text_inputs, return_dict=True, output_hidden_states=True), dim=-1)
print(string, '=', compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.4551]], device='cuda:0', dtype=torch.bfloat16)

# Text -> Image
inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a dog.\n', return_tensors="pt").to("cuda")
qry_output = F.normalize(model(**inputs, return_dict=True, output_hidden_states=True), dim=-1)

string = '<|vision_start|><|image_pad|><|vision_end|>Represent the given image.\n'
tgt_inputs = processor(text=string, images=[Image.open('figures/example.jpg')], return_tensors="pt").to("cuda")
tgt_output = F.normalize(model(**tgt_inputs, return_dict=True, output_hidden_states=True), dim=-1)
print(string, '=', compute_similarity(qry_output, tgt_output))
## <|vision_start|><|image_pad|><|vision_end|>Represent the given image. = tensor([[0.1885]], device='cuda:0', dtype=torch.bfloat16)

inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a tiger.\n', return_tensors="pt").to("cuda")
qry_output = F.normalize(model(**inputs, return_dict=True, output_hidden_states=True), dim=-1)

string = '<|vision_start|><|image_pad|><|vision_end|>Represent the given image.\n'
tgt_inputs = processor(text=string, images=[Image.open('figures/example.jpg')], return_tensors="pt").to("cuda")
tgt_output = F.normalize(model(**tgt_inputs, return_dict=True, output_hidden_states=True), dim=-1)
print(string, '=', compute_similarity(qry_output, tgt_output))
## <|vision_start|><|image_pad|><|vision_end|>Represent the given image. = tensor([[0.1621]], device='cuda:0', dtype=torch.bfloat16)
