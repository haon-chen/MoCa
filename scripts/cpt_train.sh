#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$MODEL_NAME_OR_PATH" ]; then
  MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./checkpoint/cpt_$(date +%F-%H%M.%S)"
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/"
fi

DS_CONFIG_PATH="ds_config_stage3.json"

if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="qwen2_5_vl"
fi

if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
fi

if [ -z "$BIDIRECTIONAL" ]; then
  BIDIRECTIONAL=True
fi

if [ -z "$USE_MLM" ]; then
  USE_MLM=True
fi

if [ -z "$MASK_PROB" ]; then
  MASK_PROB=0.4
fi

if [ -z "$USE_MAE" ]; then
  USE_MAE=True
fi

if [ -z "$MAE_MASK_PROB" ]; then
  MAE_MASK_PROB=0.5
fi

if [ -z "$MICRO_BATCH_SIZE" ]; then
  MICRO_BATCH_SIZE=12800
fi


if [ -z "$MAE_LOSS_WEIGHT" ]; then
  MAE_LOSS_WEIGHT=0.1
fi

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

deepspeed --master_port 18273 cpt_train.py --deepspeed "${DS_CONFIG_PATH}" \
    --dataset_name "moca-embed/dclm_20b,moca-embed/pixelprose_commonpool,moca-embed/pixelprose_cc12m_sub_005,moca-embed/pixelprose_redcaps_sub_010,moca-embed/MAmmoTH-VL-Instruct-12M,moca-embed/MMEB-train,moca-embed/docmatix,moca-embed/visrag_ind,moca-embed/visrag_syn,moca-embed/tevatron_colpali" \
    --num_sample_per_subset 500000 \
    --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True --gradient_accumulation_steps 4 \
    --max_len ${MAX_LEN} --output_dir "${OUTPUT_DIR}" --logging_steps 5 \
    --lr_scheduler_type linear --learning_rate 2e-6 --max_grad_norm 5.0 \
    --warmup_ratio 0.1 --save_steps 200 --save_total_limit 10 --normalize True \
    --min_patch_size 256 --max_patch_size 1024 \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_train_batch_size 1 \
    --model_backbone "${MODEL_BACKBONE}" \
    --processor_name "${PROCESSOR_NAME}" \
    --resume_from_checkpoint "${OUTPUT_DIR}" \
    --bidirectional ${BIDIRECTIONAL} \
    --mask_prob ${MASK_PROB} \
    --mae_mask_prob ${MAE_MASK_PROB} \
    --stats_dir "${DATA_DIR}/cache/stats" \
    --num_train_epochs 1 \
    --report_to none \
    --mae_loss_weight ${MAE_LOSS_WEIGHT} \
    --use_mae ${USE_MAE} --use_mlm ${USE_MLM} \
    "$@"
