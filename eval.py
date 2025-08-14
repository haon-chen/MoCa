import logging
import sys
import os
import json
from transformers import (
    HfArgumentParser,
)

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import AutoProcessor

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset, load_from_disk
from evaluation.eval_utils import get_pred

data_group = {
    "IND": ["ImageNet-1K", "N24News", "HatefulMemes", "SUN397", "VOC2007", "InfographicsVQA", "ChartQA", "A-OKVQA", "DocVQA", "OK-VQA", "Visual7W", "VisDial", "CIRR", "NIGHTS", "WebQA", "VisualNews_i2t", "VisualNews_t2i", "MSCOCO_i2t", "MSCOCO_t2i", "MSCOCO"],
    "OOD": ["Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211", "ScienceQA", "VizWiz", "GQA", "TextVQA", "OVEN", "FashionIQ", "EDIS", "Wiki-SS-NQ", "Visual7W-Pointing", "RefCOCO", "RefCOCO-Matching"],
}

data_group_class = {
    "Classification": [
        "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", 
        "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"
    ],
    "VQA": [
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", 
        "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"
    ],
    "Retrieval": [
        "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", 
        "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", 
        "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"
    ],
    "Visual Grounding": [
        "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"
    ]
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Environment Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("============================")

    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if model_args.checkpoint_path:
        base_name = os.path.basename(model_args.checkpoint_path) 
        if base_name.startswith("checkpoint"):
            dir_name = os.path.basename(os.path.dirname(model_args.checkpoint_path)).split('-')[-1]
        elif base_name.startswith("contrastive_learning"):
            dir_name = model_args.checkpoint_path.split('/')[-2].split('-')[-1]
        elif base_name.startswith("run-"):
            dir_name = model_args.checkpoint_path.split('/')[-2].split('-')[-1]+'/'+base_name
        else:
            dir_name = os.path.basename(model_args.checkpoint_path).split('-')[-1]
        output_path = f"{data_args.encode_output_path}/{dir_name}/{base_name}/" if base_name.startswith("checkpoint") else f"{data_args.encode_output_path}/{dir_name}/"
    else:
        output_path = data_args.encode_output_path

    os.makedirs(output_path, exist_ok=True)

    if model_args.model_backbone == 'qwen2_vl' or model_args.model_backbone == 'qwen2_5_vl':
        min_pixels = model_args.min_patch_size*28*28
        max_pixels = model_args.max_patch_size*28*28
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )

    processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    logger.info(f"Loading model with backbone: {model_args.model_backbone}")
    logger.info(f"Model num_labels: {model.config.num_labels}")
    # logger.info(f"Model config: {model.config}")

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            continue
        
        dataset_func = EvalDataset
        
        eval_qry_dataset = dataset_func(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = dataset_func(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        if not os.path.exists(encode_qry_path):
            logger.info(f"Encoding query embeddings for {subset}")
            encoded_tensor = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(eval_qry_loader)):
                    batch = {key: value.to(training_args.device) if value is not None else value for key, value in batch.items()}
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=batch)
                    encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_qry_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

            logger.info(f"Query tensor shape: {encoded_tensor.shape}")

        if not os.path.exists(encode_tgt_path):
            encoded_tensor = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(eval_tgt_loader, desc="Encode target")):
                    batch = {key: value.to(training_args.device) if value is not None else value for key, value in batch.items()}
                    output = model(tgt=batch)
                    encoded_tensor.append(output["tgt_reps"].cpu().detach().float().numpy())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_tgt_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    accuracy_ood = []
    accuracy_ind = []

    accuracy_cla = []
    accuracy_ret = []
    accuracy_vqa = []
    accuracy_vg = []

    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        if data_args.dataset_name:
            eval_data = load_dataset(
                data_args.dataset_name,
                subset,
                split=data_args.dataset_split,
            )
        elif data_args.dataset_path:
            subset_path = os.path.join(data_args.dataset_path, subset) 
            eval_data = load_from_disk(subset_path)
        n_correct = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
        with open(os.path.join(output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        score_path = os.path.join(output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        if subset in data_group["IND"]:
            accuracy_ind.append(n_correct/len(eval_data))
        elif subset in data_group["OOD"]:
            accuracy_ood.append(n_correct/len(eval_data))
        else:
            print(f"Unknown subset: {subset}")
        if subset in data_group_class["Classification"]:
            accuracy_cla.append(n_correct/len(eval_data))
        elif subset in data_group_class["VQA"]:
            accuracy_vqa.append(n_correct/len(eval_data))
        elif subset in data_group_class["Retrieval"]:
            accuracy_ret.append(n_correct/len(eval_data))
        elif subset in data_group_class["Visual Grounding"]:
            accuracy_vg.append(n_correct/len(eval_data))
        else:
            print(f"Unknown subset: {subset}")
        # accuracy.append(n_correct/len(eval_data))
        logger.info(f"Computing scores for {subset}")
        logger.info(f"Number of correct predictions: {n_correct}/{len(eval_data)}")
        logger.info(f"Accuracy: {n_correct/len(eval_data):.4f}")
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")
    print(f"IND accuracy: {np.mean(accuracy_ind)*100}, dataset num: {len(accuracy_ind)}")
    print(f"OOD accuracy: {np.mean(accuracy_ood)*100}, dataset num: {len(accuracy_ood)}")
    print(f"All accuracy: {np.mean(accuracy_ind + accuracy_ood)*100}, dataset num: {len(accuracy_ind + accuracy_ood)}")

    print(f"Classification accuracy: {np.mean(accuracy_cla)*100}, dataset num: {len(accuracy_cla)}")
    print(f"VQA accuracy: {np.mean(accuracy_vqa)*100}, dataset num: {len(accuracy_vqa)}")
    print(f"Retrieval accuracy: {np.mean(accuracy_ret)*100}, dataset num: {len(accuracy_ret)}")
    print(f"Visual Grounding accuracy: {np.mean(accuracy_vg)*100}, dataset num: {len(accuracy_vg)}")
    print(f"Printed format: {round(np.mean(accuracy_cla)*100, 1)},{round(np.mean(accuracy_vqa)*100, 1)},{round(np.mean(accuracy_ret)*100, 1)},{round(np.mean(accuracy_vg)*100, 1)},{round(np.mean(accuracy_ind)*100, 1)},{round(np.mean(accuracy_ood)*100, 1)},{round(np.mean(accuracy_ind + accuracy_ood)*100, 1)}")

if __name__ == "__main__":
    main()
