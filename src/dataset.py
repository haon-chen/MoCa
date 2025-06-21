import random
from typing import List
import datasets
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from PIL import Image
import os
from PIL import ImageFile
from src.model_utils import PHI3V, vlm_image_tokens, QWEN2_VL, QWEN2_5_VL
import logging
import math
logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TaskBatchDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        self.negative_ratio = self.data_args.negative_ratio
        self.batch_size = 1
        self.subset_datasets = []
        self.subset_names = []
        
        if self.data_args.dataset_name or self.data_args.dataset_path:
            print(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
            for subset in data_args.subset_name:
                num_sample = data_args.num_sample_per_subset
                if self.data_args.dataset_name:
                    subset_data = load_dataset(
                        self.data_args.dataset_name,
                        subset,
                        split=f"{self.data_args.dataset_split}[:{num_sample}]",
                    )
                elif self.data_args.dataset_path:
                    subset_path = os.path.join(self.data_args.dataset_path, subset) 
                    subset_data = load_from_disk(subset_path)
                    if len(subset_data) > num_sample and num_sample != -1:
                        subset_data = subset_data.select(range(num_sample))
                self.subset_datasets.append(subset_data)
                self.subset_names.append(subset)
        
        self.initialize_dataset_mapping()
        
        print(f"Task batch dataset created with {len(self)} samples from {len(self.subset_datasets)} subsets")
        for name, size in zip(self.subset_names, [len(ds) for ds in self.subset_datasets]):
            print(f"  - {name}: {size} samples")
    
    def initialize_dataset_mapping(self):
        """Create a mapping from global indices to (subset_idx, local_idx)"""
        self.subset_sizes = [len(ds) for ds in self.subset_datasets]
        self.total_raw_samples = sum(self.subset_sizes)
        
        self.padded_subset_sizes = []
        for size in self.subset_sizes:
            padded_size = math.ceil(size / self.batch_size) * self.batch_size
            self.padded_subset_sizes.append(padded_size)
        
        self.total_size = sum(self.padded_subset_sizes)
        
        self.idx_mapping = []
        for subset_idx, subset_size in enumerate(self.subset_sizes):
            for local_idx in range(subset_size):
                self.idx_mapping.append((subset_idx, local_idx))
            
            padding_size = self.padded_subset_sizes[subset_idx] - subset_size
            if padding_size > 0:
                last_idx = subset_size - 1 if subset_size > 0 else 0
                for _ in range(padding_size):
                    self.idx_mapping.append((subset_idx, last_idx))
    
    def set_batch_size(self, batch_size):
        """Update the batch size and reinitialize the dataset mapping"""
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.initialize_dataset_mapping()
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx >= len(self.idx_mapping):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.idx_mapping)} samples")
        
        subset_idx, local_idx = self.idx_mapping[idx]
        dataset = self.subset_datasets[subset_idx]
        item_data = dataset[local_idx]
        
        qry_text, qry_image_path, pos_text, pos_image_path = (
            item_data["qry"], item_data["qry_image_path"],
            item_data["pos_text"], item_data["pos_image_path"],
        )
        neg_texts, neg_image_paths, neg_images = [], [], []
        if self.negative_ratio > 0:
            neg_text_list, neg_image_path_list = (
                item_data["neg_text"], item_data["neg_image_path"],
            )
            neg_texts = self.filter_hard_negtives(neg_text_list, pos_text, self.negative_ratio)
            neg_image_paths = self.filter_hard_negtives(neg_image_path_list, pos_image_path, self.negative_ratio)

        for ind, neg in enumerate(neg_texts):
            if neg == '':
                if len(set(eval(neg_text_list))) == 1:
                    neg_texts[ind] = pos_text
                else:
                    neg_texts[ind] = random.choice([text for text in eval(neg_text_list) if text != ""])
        if self.model_args.model_backbone != PHI3V:
            qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_args.model_backbone])
            pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_args.model_backbone])
            for ind, neg in enumerate(neg_texts):
                neg_texts[ind] = neg.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_args.model_backbone])
        for neg_img in neg_image_paths:
            neg_images.append(self._get_image(neg_img))
        all_texts = [qry_text, pos_text] + neg_texts
        all_image_paths = [qry_image_path, pos_image_path] + neg_image_paths
        
        for idx, (text, img_path) in enumerate(zip(all_texts, all_image_paths)):
            if vlm_image_tokens[self.model_args.model_backbone] in text and img_path == "":
                all_texts[idx] = text.replace(vlm_image_tokens[self.model_args.model_backbone], "")
        
        qry_text, pos_text, *neg_texts = all_texts
        return (qry_text, self._get_image(qry_image_path),
                pos_text, self._get_image(pos_image_path),
                neg_texts, neg_images,
                subset_idx)
    
    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "original":
            if self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL]:
                if image.size[0] < 28:
                    image = image.resize((28, image.size[1]))
                if image.size[1] < 28:
                    image = image.resize((image.size[0], 28))
            return image
        elif resolution == "high":
            image = image.resize((512, 512))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        if not os.path.exists(full_img_path):
            full_img_path = os.path.join(self.data_args.laion_image_dir, img_path)
        try:
            image = Image.open(full_img_path)
        except Exception as e:
            print(f"Error loading image {full_img_path}: {str(e)}, returning black image")
            return Image.new("RGB", (28, 28), (0, 0, 0))
            
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return self._process_image(image, self.data_args.image_resolution)
        else:
            return image

    def filter_hard_negtives(self, negs, pos, negative_ratio):
        negs = eval(negs)
        if not isinstance(negs, list):
            negs = [negs]

        if len(negs) < negative_ratio and len(negs) > 0:
            negs += [negs[-1]] * (negative_ratio - len(negs))

        negs = negs[:negative_ratio]
        return negs

class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        logger.info(f"Initializing EvalDataset for {subset}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Model args: {model_args}")
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args

        if self.data_args.dataset_name:
            self.eval_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=self.data_args.dataset_split,
            )
        elif self.data_args.dataset_path:
            subset_path = os.path.join(self.data_args.dataset_path, subset) 
            self.eval_data = load_from_disk(subset_path)
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

        logger.info(f"Dataset loaded with {len(self.eval_data)} samples")
        logger.info(f"Unique paired data: {len(self.paired_data)}")

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.model_args.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_args.model_backbone])
        if vlm_image_tokens[self.model_args.model_backbone] in text and img_path == "":
            text = text.replace(vlm_image_tokens[self.model_args.model_backbone], "")
        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "original":
            if image.size[0] < 28:
                image = image.resize((28, image.size[1]))
            if image.size[1] < 28:
                image = image.resize((image.size[0], 28))
            return image
        elif resolution == "high":
            image = image.resize((512, 512))
            # image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            logger.debug("Empty image path")
            return None
            
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        logger.debug(f"Loading image from: {full_img_path}")
        
        try:
            image = Image.open(full_img_path)
            logger.debug(f"Image loaded with size: {image.size}")
            
            if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
                processed_image = self._process_image(image, self.data_args.image_resolution)
                logger.debug(f"Processed image size: {processed_image.size}")
                return processed_image
            return image
        except Exception as e:
            logger.error(f"Error loading image {full_img_path}: {str(e)}")
            raise

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data


