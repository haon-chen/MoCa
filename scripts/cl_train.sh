#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="./checkpoint/cpt_xxx"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/cl_$(date +%F-%H%M.%S)"
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/"
fi

DS_CONFIG_PATH="ds_config_stage3.json"

if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=24
fi

if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="qwen2_5_vl"
fi

if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
fi

if [ -z "$BIDIRECTIONAL" ]; then
  BIDIRECTIONAL=True
fi

if [ -z "$IMAGE_RESOLUTION" ]; then
  IMAGE_RESOLUTION=original
fi

if [ -z "$USE_TASK_BATCH" ]; then
  USE_TASK_BATCH=True
fi

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

deepspeed --master_port 18271 train.py --deepspeed "${DS_CONFIG_PATH}" \
    --subset_name visrag_ind visrag_syn tevatron_colpali TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    --dataset_name "intfloat/MoCa-CL-Pairs" \
    --synthetic_dataset_name "intfloat/mmE5-synthetic" \
    --synthetic_subset_name Classification Retrieval VQA \
    --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
    --num_sample_per_subset 50000 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True --gradient_accumulation_steps 1 \
    --image_dir "./" \
    --num_train_epochs 1 \
    --max_len ${MAX_LEN} --output_dir "${OUTPUT_DIR}" --logging_steps 5 \
    --lr_scheduler_type linear --learning_rate 2e-5 --max_grad_norm 5.0 \
    --warmup_ratio 0.05 --save_steps 200 --save_total_limit 10 --normalize True \
    --temperature 0.03 --per_device_train_batch_size ${BATCH_SIZE} \
    --model_backbone "${MODEL_BACKBONE}" \
    --processor_name "${PROCESSOR_NAME}" \
    --resume_from_checkpoint "${OUTPUT_DIR}" \
    --bidirectional ${BIDIRECTIONAL} \
    --negative_ratio 2 \
    --image_resolution "${IMAGE_RESOLUTION}" \
    --min_patch_size 256 --max_patch_size 1024 \
    --use_task_batch ${USE_TASK_BATCH} \
    --report_to none "$@"
