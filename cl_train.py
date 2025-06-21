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

from src.dataset import TrainDataset, TaskBatchDataset
from src.collator import QWEN25TrainCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.trainer import MMEBTrainer
import torch.distributed as dist
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
    
    # Configure logging
    if data_args.use_task_batch and training_args.local_rank == 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info("=" * 80)
        logger.info("TASK-SPECIFIC BATCHING ENABLED")
        logger.info("This will ensure each batch contains data from only one subset")
        logger.info("Batch consistency will be verified during training")
        logger.info("=" * 80)

    force_download = True
    if dist.get_world_size() == 1:
        force_download = False
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

    if data_args.use_task_batch:
        logger.info("Using task-specific batching (each batch contains data from only one subset)")
        train_dataset = TaskBatchDataset(data_args, model_args)
        train_dataset.set_batch_size(training_args.per_device_train_batch_size)
    else:
        logger.info("Using standard batching (mixed subsets)")
        train_dataset = TrainDataset(data_args, model_args)
    
    collator = QWEN25TrainCollator(data_args, model_args, processor)

    model = MMEBModel.build(model_args, training_args)

    trainer = MMEBTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataset.trainer = trainer

    if training_args.local_rank == 0:  # Only print on main process
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    if training_args.resume_from_checkpoint is not None:
        if os.path.exists(training_args.resume_from_checkpoint):
            resume_from_checkpoint = get_last_checkpoint(training_args.resume_from_checkpoint)
            print(f"Restarting from checkpoint {resume_from_checkpoint}")
        else:
            print(f"Checkpoint {training_args.resume_from_checkpoint} not found")
            resume_from_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)    
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    main()
