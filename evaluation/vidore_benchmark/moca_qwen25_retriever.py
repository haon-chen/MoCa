from __future__ import annotations

import os
import logging
from typing import List, Optional, Union, Dict, Any
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.data_utils import ListDataset
from vidore_benchmark.utils.torch_utils import get_torch_device
from model import MMEBModel
from arguments import ModelArguments
from model_utils import QWEN2_VL, QWEN2_5_VL, vlm_image_tokens
from transformers import AutoProcessor

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@register_vision_retriever("moca-qwen25")
class MoCaQwen25Retriever(BaseVisionRetriever):
    """
    Vision retriever based on MMEBModel with Qwen2_5ForEmbedding.
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        *,
        pretrained_model_name_or_path: str = None,
        device: str = "auto",
        use_bidirectional: bool = True,
        use_linear_projection: bool = False,
        normalize: bool = True,
        pooling: str = "last",
        image_resolution: str = "original",  # "original", "high" or "low"
        num_workers: int = 4,
        max_length: int = 512,
        processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
        # Handle both model_name_or_path and pretrained_model_name_or_path
        self.model_path = model_name_or_path or pretrained_model_name_or_path
        if not self.model_path:
            raise ValueError("Either model_name_or_path or pretrained_model_name_or_path must be provided")
            
        self.device = get_torch_device(device)
        self.normalize = normalize
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.max_length = max_length
        self.processor_name = processor_name
        self.model_backbone = QWEN2_5_VL  # Use the constant from model_utils
        
        # Log all parameters for debugging
        logger.info(f"Initializing MoCaQwen25Retriever with parameters:")
        logger.info(f"  model_path: {self.model_path}")
        logger.info(f"  device: {device} (resolved to: {self.device})")
        logger.info(f"  use_bidirectional: {use_bidirectional}")
        logger.info(f"  normalize: {normalize}")
        logger.info(f"  pooling: {pooling}")
        logger.info(f"  image_resolution: {image_resolution}")
        logger.info(f"  num_workers: {num_workers}")
        logger.info(f"  max_length: {max_length}")
        logger.info(f"  Additional kwargs: {kwargs}")
        
        try:
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            model_args = ModelArguments(
                model_name=self.model_path,
                checkpoint_path=self.model_path,
                pooling=pooling,
                normalize=normalize,
                bidirectional=use_bidirectional,
                use_linear_projection=use_linear_projection,
                model_backbone=self.model_backbone
            )
            
            # Load the model
            self.model = MMEBModel.load(model_args)
            self.model.to(self.device, dtype=torch.bfloat16)
            self.model.eval()
            
            # Load processor from same checkpoint
            try:
                if self.processor_name:
                    logger.info(f"Attempting to load processor from processor_name: {self.processor_name}")
                else:
                    logger.info("Attempting to load processor directly from checkpoint")
                min_pixels = 256*28*28
                max_pixels = 1024*28*28
                self.processor = AutoProcessor.from_pretrained(
                    self.processor_name if self.processor_name else model_args.model_name,
                    trust_remote_code=True,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                logger.info("Successfully loaded processor")
            except Exception as e:
                logger.warning(f"Could not load processor directly: {str(e)}")
                
                # Try to load from a known model name
                try:
                    logger.info("Attempting to load processor from Qwen/Qwen2.5-VL-3B-Instruct")
                    self.processor = AutoProcessor.from_pretrained(
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded processor from Qwen/Qwen2.5-VL-3B-Instruct")
                except Exception as e2:
                    logger.warning(f"Could not load processor from Qwen/Qwen2.5-VL-3B-Instruct: {str(e2)}")
                    
                    # Try to get processor from encoder
                    if hasattr(self.model.encoder, "tokenizer"):
                        logger.info("Using encoder as processor")
                        self.processor = self.model.encoder
                    else:
                        # Final fallback
                        logger.info("Falling back to tokenizer-only processor")
                        from transformers import AutoTokenizer
                        self.processor = AutoTokenizer.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True
                        )
                        logger.info("Using tokenizer as processor")
            
            # Get the image token
            self.image_token = vlm_image_tokens.get(self.model_backbone, "<image>")
            logger.info(f"Using image token: {self.image_token}")
            
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load MMEBModel from {self.model_path}: {str(e)}")

    def _process_image(self, image, resolution="original"):
        """Process image following the same approach as in dataset.py"""
        if image is None:
            return None
            
        image = image.convert('RGB')
        
        if resolution == "original":
            # Ensure minimum size for Qwen models
            if self.model_backbone in [QWEN2_VL, QWEN2_5_VL]:
                if image.size[0] < 28:
                    image = image.resize((28, image.size[1]))
                if image.size[1] < 28:
                    image = image.resize((image.size[0], 28))
            return image
        elif resolution == "high":
            image = image.resize((512, 512))
        else:  # low
            image = image.resize((336, 336))
            
        return image
    
    def process_query(self, query: str):
        """Process a single text query."""
        if hasattr(self.processor, "__call__"):
            inputs = self.processor(
                text=query,
                images=None,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                pad_to_multiple_of=32
            )
        else:
            # Fallback to basic tokenization
            assert False, "Processor is not callable"
            inputs = self.processor.tokenizer(
                query,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def process_queries(self, queries: List[str]):
        """Process a batch of text queries."""
        try:
            if hasattr(self.processor, "__call__"):
                inputs = self.processor(
                    text=queries,
                    images=None,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    pad_to_multiple_of=32
                )
            else:
                # Fallback to basic tokenization
                assert False, "Processor is not callable"
                inputs = self.processor.tokenizer(
                    queries,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
            
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            logger.error(f"Error processing queries: {str(e)}")
            raise RuntimeError(f"Failed to process queries: {str(e)}")
    
    def process_image(self, image: Image.Image):
        """Process a single image with vision token."""
        try:
            # Process the image with appropriate resolution handling
            processed_image = self._process_image(image, self.image_resolution)
            if processed_image is None:
                logger.warning("Image is None, returning text-only input")
                return self.process_query("")
            
            # Process the image with appropriate token
            text = self.image_token
            
            # Log image processing info
            logger.debug(f"Processing image with size: {processed_image.size}, using token: {text}")
            
            if hasattr(self.processor, "__call__"):
                try:
                    # Try processor with both text and image
                    inputs = self.processor(
                        text=text,
                        images=processed_image,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        pad_to_multiple_of=32
                    )
                    logger.debug(f"Processor output keys: {inputs.keys()}")
                except Exception as e:
                    logger.warning(f"Error with processor for both text and image: {str(e)}. Trying image only.")
                    # Try with just the image
                    try:
                        image_inputs = self.processor(
                            images=processed_image,
                            return_tensors="pt"
                        )
                        text_inputs = self.processor.tokenizer(
                            text,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
                        # Combine the inputs
                        inputs = {**text_inputs, **image_inputs}
                        logger.debug(f"Combined processor output keys: {inputs.keys()}")
                    except Exception as inner_e:
                        logger.error(f"Error with separate processing: {str(inner_e)}. Falling back to basic processing.")
                        raise inner_e
            else:
                assert False, "Processor is not callable"
                # Fallback processing using tokenizer and manual image processing
                inputs = self.processor.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Process the image
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = preprocess(processed_image).unsqueeze(0)
                
                # Add to inputs
                inputs["pixel_values"] = pixel_values
                
                # Generate image grid info if needed
                height, width = processed_image.height, processed_image.width
                image_grid_thw = torch.tensor([[1, height, width]], dtype=torch.long)
                inputs["image_grid_thw"] = image_grid_thw
            
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(f"Image type: {type(image)}, Image properties: {getattr(image, 'size', 'Unknown size')}")
            raise RuntimeError(f"Failed to process image: {str(e)}")
    
    def process_images(self, images: List[Image.Image]):
        """Process a batch of images."""
        processed_images = []
        
        for image in images:
            if image is not None:
                processed_images.append(self._process_image(image, self.image_resolution))
            
        if len(processed_images) == 0:
            logger.warning("No valid images to process")
            return None
            
        prompt_text = f"{self.image_token}\nRepresent the given document image."
        # from IPython import embed
        # embed()
        # input()
        try:
            if hasattr(self.processor, "__call__"):
                # Similar approach to collator.py
                texts = [prompt_text] * len(processed_images)
                print(f"texts: {texts}")
                inputs = self.processor(
                    text=texts, 
                    images=processed_images, 
                    padding=True, 
                    return_tensors="pt", 
                    pad_to_multiple_of=32
                )
                return {k: v.to(self.device) for k, v in inputs.items()}
            else:
                assert False, "Processor is not callable"
                # Process individually and return as list
                return [self.process_image(img) for img in images]
        except Exception as e:
            logger.error(f"Error batch processing images: {str(e)}")
            raise RuntimeError(f"Failed to process images: {str(e)}")

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Process text queries and generate embeddings.
        """
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
            num_workers=self.num_workers,
        )

        query_embeddings = []

        with torch.no_grad():
            for batch_input in tqdm(dataloader, desc="Forward pass queries...", leave=False):
                embeddings = self.model.encode_input(batch_input)
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                query_embeddings.extend(list(torch.unbind(embeddings.cpu())))

        return query_embeddings

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Process image passages and generate embeddings.
        """
        # Process images in batches using a collator-like approach
        batch_size = min(batch_size, 8)  # Limit batch size for image processing
        
        # Using DataLoader with the processor as collate_fn 
        # Similar to how collator.py works
        if hasattr(self.processor, "__call__"):
            # Create a custom collate function
            def process_image_batch(images):
                processed_images = [self._process_image(img, self.image_resolution) for img in images]
                # texts = [self.image_token] * len(processed_images)
                prompt_text = f"{self.image_token}\nRepresent the given document image."
                texts = [prompt_text] * len(processed_images)
                # print(f"texts: {texts}")
                return self.processor(
                    text=texts, 
                    images=processed_images, 
                    padding=True, 
                    return_tensors="pt", 
                    pad_to_multiple_of=32
                )
            
            dataloader = DataLoader(
                dataset=ListDataset[Image.Image](passages),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=process_image_batch,
                num_workers=self.num_workers,
            )
            
            passage_embeddings = []
            with torch.no_grad():
                for batch_inputs in tqdm(dataloader, desc="Processing images...", leave=False):
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                    embeddings = self.model.encode_input(batch_inputs)
                    if self.normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    passage_embeddings.extend(list(torch.unbind(embeddings.cpu())))
        else:
            assert False, "Processor is not callable"
            # Process each image individually
            passage_embeddings = []
            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i+batch_size]
                batch_inputs = []
                
                for image in tqdm(batch_passages, desc=f"Processing images {i}-{i+len(batch_passages)-1}", leave=False):
                    batch_inputs.append(self.process_image(image))
                
                with torch.no_grad():
                    for inputs in batch_inputs:
                        embedding = self.model.encode_input(inputs)
                        if self.normalize:
                            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                        passage_embeddings.append(embedding.squeeze(0).cpu())

        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Calculate similarity scores between queries and passages.
        """
        try:
            # Convert embeddings to the same format
            if isinstance(query_embeddings, list):
                query_embeddings = torch.stack(query_embeddings)
            
            if isinstance(passage_embeddings, list):
                passage_embeddings = torch.stack(passage_embeddings)
            
            # Ensure embeddings are on CPU for large matrix multiplications
            query_embeddings = query_embeddings.cpu()
            passage_embeddings = passage_embeddings.cpu()
            
            # If batch_size is specified and matrices are large, compute scores in batches
            if batch_size is not None and query_embeddings.shape[0] * passage_embeddings.shape[0] > 1000000:
                logger.info(f"Computing similarity scores in batches with batch_size={batch_size}")
                scores = torch.zeros(
                    (query_embeddings.shape[0], passage_embeddings.shape[0]), 
                    dtype=torch.float32
                )
                
                # Process in batches to avoid OOM
                for i in range(0, query_embeddings.shape[0], batch_size):
                    end_idx = min(i + batch_size, query_embeddings.shape[0])
                    batch_query_embeddings = query_embeddings[i:end_idx]
                    
                    # Compute similarity for this batch
                    batch_scores = torch.mm(batch_query_embeddings, passage_embeddings.T)
                    scores[i:end_idx] = batch_scores
            else:
                # Compute similarity matrix (queries x passages)
                scores = torch.mm(query_embeddings, passage_embeddings.T)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error computing similarity scores: {str(e)}")
            raise RuntimeError(f"Failed to compute similarity scores: {str(e)}") 