import logging
from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments
import torch
from PIL import ImageFile
from src.vlm_backbone.qwen2_5_vl_embed.utils_new import Truncation

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

@dataclass
class QWEN25TrainCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        neg_inputs = self._get_batch_inputs(examples, 4, 5)

        return qry_inputs, pos_inputs, neg_inputs

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        if text_idx == 4:
            texts = [t for exp in examples for t in exp[text_idx] if t is not None]
            images = [i for exp in examples for i in exp[image_idx] if i is not None]
        else:
            texts = [exp[text_idx] for exp in examples if exp[text_idx] is not None]
            images = [exp[image_idx] for exp in examples if exp[image_idx] is not None]

        if len(texts) == 0:
            return None

        if len(images) == 0:
            inputs = self.processor(text=texts, images=None, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        else:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        
        trucation = Truncation(train=True)
        inputs = trucation.truncate(inputs, self.data_args.max_len)
        
        return inputs

@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        logger.debug(f"Collating batch of size: {len(examples)}")
        inputs = self._get_batch_inputs(examples)
        logger.debug(f"Collated input keys: {inputs.keys()}")
        logger.debug(f"Collated input shapes: {[(k, v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)]}")
        return inputs

    def _get_batch_inputs(self, examples):
        texts = [exp[0] for exp in examples if exp[0] is not None]
        images = [exp[1] for exp in examples if exp[1] is not None]
        
        if len(images) == 0:
            inputs = self.processor(text=texts, images=None, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        else:
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt", pad_to_multiple_of=32)
        trucation = Truncation(train=False)
        inputs = trucation.truncate(inputs, self.data_args.max_len)

        return inputs
