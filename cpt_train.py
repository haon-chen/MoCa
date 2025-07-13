# Adapted from Tevatron code
import logging
import sys
import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from transformers import AutoProcessor
from transformers import (
    HfArgumentParser,
)
import torch.distributed as dist

from src.dataloader import Dataset, MLMMAECollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModelForMLMMAE
from src.trainer import MMEBMLMMAETrainer
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
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
    use_mae = model_args.use_mae
    use_mlm = model_args.use_mlm

    force_download = True
    if dist.get_world_size() == 1:
        force_download = False
        
    # Initialize processor
    if model_args.model_backbone == 'qwen2_vl' or model_args.model_backbone == 'qwen2_5_vl':
        min_pixels = model_args.min_patch_size*28*28
        max_pixels = model_args.max_patch_size*28*28
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            force_download=force_download,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
            force_download=force_download
        )
        processor.tokenizer.padding_side = "right"
    
    # Add mask token if not present
    if processor.tokenizer.mask_token is None:
        processor.tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    
    # Initialize dataset
    if data_args.dataset_name:
        dataset_names = [
            {"name": data_name, "split": "train", "source": "hf", "sample": data_args.num_sample_per_subset} 
            for data_name in data_args.dataset_name.split(",")
        ]
    else:
        dataset_names = [
            {"name": data_path, "split": "train", "source": "local", "sample": data_args.num_sample_per_subset}
            for data_path in data_args.dataset_path.split(",")
        ]
    print(f"world_size: {dist.get_world_size()}")
    train_dataset = Dataset(
        dataset_names,
        data_args.stats_dir,
        micro_batch_size=training_args.micro_batch_size,
        max_equiv_tokens=data_args.max_len,
        world_size=dist.get_world_size(),
        tokenizer_id=model_args.processor_name if model_args.processor_name else model_args.model_name,
    )
    train_dataset.set_epoch(0)
    
    # Always use MLMMAECollator, control by use_mae/use_mlm
    collator = MLMMAECollator(
        processor=processor,
        mask_prob=data_args.mask_prob,
        mae_mask_prob=data_args.mae_mask_prob,
        max_equiv_tokens=data_args.max_len,
        use_mae=use_mae,
        use_mlm=use_mlm,
    )

    model = MMEBModelForMLMMAE.build(model_args, training_args, use_mae=use_mae, use_mlm=use_mlm)
    
    # Initialize trainer
    training_args.shuffle = False

    trainer = MMEBMLMMAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
        
    # Resume from checkpoint if specified
    resume_from_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        if os.path.exists(training_args.resume_from_checkpoint):
            resume_from_checkpoint = get_last_checkpoint(training_args.resume_from_checkpoint)
            print(f"Restarting from checkpoint {resume_from_checkpoint}")
        else:
            print(f"Checkpoint {training_args.resume_from_checkpoint} not found")

    # Train model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
