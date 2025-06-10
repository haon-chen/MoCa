#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME_OR_PATH="/home/v-chenhaonan/multimodal/mmE5-qwen25/checkpoint/Qwen2.5-VL-3B-Instruct-MAE-Init"

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/mlm_pretrain_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/home/v-chenhaonan/teamdrive/multimodal/data/pretraining"
fi

DS_CONFIG_PATH="ds_config_stage3.json"
if [ "$(nvidia-smi --list-gpus | wc -l)" = "1" ]; then
  DS_CONFIG_PATH="ds_config_stage3.json"
fi

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
if [ "$PROC_PER_NODE" != "1" ] && [ "$RANK" != "0" ]; then
  exit 0
fi

if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=1
fi

if [ "${WORLD_SIZE}" -gt 1 ]; then
  # Sleep 300 seconds on node 0 before starting deepspeed
  if [ "${RANK}" = "0" ]; then
    echo "Multiple nodes detected. Node 0 sleeping for 300 seconds..."
    sleep 300
  fi
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

if [ -z "$MASK_PROB" ]; then
  MASK_PROB=0.4
fi

if [ -z "$MAE_MASK_PROB" ]; then
  MAE_MASK_PROB=0.9
fi

if [ -z "$MICRO_BATCH_SIZE" ]; then
  MICRO_BATCH_SIZE=16384
#   MICRO_BATCH_SIZE=12800
#   MICRO_BATCH_SIZE=5120
fi

if [ -z "$USE_MAE" ]; then
  USE_MAE=True
#   USE_MAE=False
fi
if [ -z "$USE_MLM" ]; then
  USE_MLM=True
fi

if [ -z "$MAE_LOSS_WEIGHT" ]; then
  MAE_LOSS_WEIGHT=0.1
fi

if [ -z "$COLLATOR_TYPE" ]; then
  COLLATOR_TYPE="mlmmae"
fi

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

    # --dataset_name "${DATA_DIR}/pixelprose,${DATA_DIR}/test_dclm_20b_2" \
    # --dataset_path "${DATA_DIR}/pixelprose,${DATA_DIR}/dclm_20b" \
    # --dataset_path "${DATA_DIR}/pixelprose_commonpool_sampled_1M,${DATA_DIR}/dclm_20b" \
    # --dataset_path "${DATA_DIR}/test_vqa,${DATA_DIR}/docmatix" \
    # --dataset_path "${DATA_DIR}/mammoth_full_dataset_cleaned" \
    # --dataset_path "${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m" \
    # --dataset_path "${DATA_DIR}/pixelprose_redcaps_sub_010" \
    # --dataset_path "${DATA_DIR}/dclm_20b" \
    # --dataset_path "${DATA_DIR}/MMEB-train/full_dataset_cleaned" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/MMEB-train/full_dataset_cleaned" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m" \
    # --dataset_path "${DATA_DIR}/pixelprose_redcaps_sub_010,${DATA_DIR}/pixelprose_cc12m_sub_005" \
    # --dataset_path "${DATA_DIR}/pixelprose_redcaps_sub_010" \
    # --dataset_path "${DATA_DIR}/visrag_syn_unfiltered,${DATA_DIR}/visrag_ind_filtered" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/pixelprose_commonpool" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/pixelprose_commonpool_sampled_1M,${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/pixelprose_commonpool,${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m,${DATA_DIR}/MMEB-train/full_dataset_cleaned,${DATA_DIR}/docmatix" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/pixelprose_commonpool,${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m,${DATA_DIR}/MMEB-train/full_dataset_cleaned,${DATA_DIR}/docmatix,${DATA_DIR}/pixelprose_redcaps_sub_010,${DATA_DIR}/pixelprose_cc12m_sub_005,${DATA_DIR}/visrag_ind_unfiltered,${DATA_DIR}/visrag_syn_unfiltered,${DATA_DIR}/tevatron_colpali_pre_valid,${DATA_DIR}/vdr_multi_500k_unfiltered" \
    # --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/pixelprose_commonpool,${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m,${DATA_DIR}/MMEB-train/full_dataset_cleaned,${DATA_DIR}/docmatix,${DATA_DIR}/visrag_ind_unfiltered,${DATA_DIR}/visrag_syn_unfiltered,${DATA_DIR}/tevatron_colpali_pre_valid" \
    # --dataset_path "${DATA_DIR}/vdr_multi_500k_unfiltered" \
    # --dataset_path "${DATA_DIR}/vdr_multi_500k_unfiltered,${DATA_DIR}/pixelprose_redcaps_sub_010" \

    # --dataset_path "${DATA_DIR}/pixelprose_commonpool_sampled_1M,${DATA_DIR}/dclm_20b" \
# deepspeed --master_port 18273 pretrain_mlm.py --deepspeed "${DS_CONFIG_PATH}" \
deepspeed --num_nodes "${WORLD_SIZE}" --master_port "${MASTER_PORT}" --master_addr "${MASTER_ADDR}" pretrain_mlm.py --deepspeed "${DS_CONFIG_PATH}" \
    --dataset_path "${DATA_DIR}/dclm_20b,${DATA_DIR}/pixelprose_commonpool,${DATA_DIR}/mammoth_full_dataset_cleaned_sampled_2m,${DATA_DIR}/MMEB-train/full_dataset_cleaned,${DATA_DIR}/docmatix,${DATA_DIR}/visrag_ind_unfiltered,${DATA_DIR}/visrag_syn_unfiltered,${DATA_DIR}/tevatron_colpali_pre_valid" \
    --num_sample_per_subset 500000 \
    --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True --gradient_accumulation_steps 4 \
    --max_len ${MAX_LEN} --output_dir "${OUTPUT_DIR}" --logging_steps 5 \
    --lr_scheduler_type linear --learning_rate 2e-6 --max_grad_norm 5.0 \
    --warmup_ratio 0.1 --save_steps 200 --save_total_limit 10 --normalize True \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
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
    --collator_type ${COLLATOR_TYPE} \
    "$@"

#  micro_batch_size128000 grad4 5e-5 epoch1 ratio0.4 maxlen4096
#  micro_batch_size64000 grad4 5e-5 epoch1 ratio0.4 maxlen4096
#  pixel dclm micro_batch_size32768 grad4 5e-5 epoch1 ratio0.4 maxlen4096
#  micro_batch_size25600 grad4 5e-6 epoch1 ratio0.4 maxlen4096 vqa
#  micro_batch_size25600 grad4 1e-5 epoch1 ratio0.4 maxlen2048 vqa
#  micro_batch_size10240 grad4 5e-6 epoch1 ratio0.4 maxlen2048

#  micro_batch_size8192 grad4 2e-6 epoch1 ratio0.4 maxlen2048 2M
#  micro_batch_size8192 grad4 2e-6 epoch1 ratio0.4 maxlen4096 mammoth 2M
#  micro_batch_size8192 grad4 2e-6 epoch1 ratio0.4 maxlen4096 pixelprose_commonpool_sampled_1M dclm_20b 1M MAE
#  micro_batch_size25600 grad4 2e-6 epoch1 ratio0.4 maxlen4096 pixelprose_commonpool_sampled_1M dclm_20b 1M MAE
#  micro_batch_size6480 grad4 2e-6 epoch1 MLM_0.4 MAE_0.5 maxlen2048 pixelprose_commonpool_sampled_1M mammoth_full_dataset_cleaned dclm_20b MMEB docmatix tevatron_colpali_filtered 0.5M
#  micro_batch_size6480 grad4 2e-6 epoch1 MLM_0.4 MAE_0.5_0.05 maxlen2048 pixelprose_commonpool_sampled_1M mammoth_full_dataset_cleaned dclm_20b MMEB docmatix 0.5M
#  micro_batch_size6480 grad4 2e-6 epoch1 ratio0.4 maxlen2048 pixelprose_commonpool_sampled_1M mammoth_full_dataset_cleaned dclm_20b 1M
#  micro_batch_size5120 grad4 7B warm0.1 1e-6 epoch1 MLM_0.5 MAE_0.6_0.5 maxlen2048 pixelprose_commonpool mammoth_full_dataset_cleaned dclm_20b MMEB docmatix tevatron_colpali_pre_valid visrag_ind_unfiltered visrag_syn_unfiltered 0.5M
#  micro_batch_size8192 grad4 2e-6 epoch1 MLM_0.4 MAE_0.5_0.5 maxlen2048 pixelprose_commonpool mammoth_full_dataset_cleaned dclm_20b MMEB docmatix 0.5M
#  micro_batch_size5120 grad4 7B len2280 warm0.1 2e-6 epoch1 MLM_0.6 noMAE_0.5_0.5 maxlen2048 pixelprose_commonpool mammoth_full_dataset_cleaned dclm_20b MMEB docmatix pixelprose_redcaps_sub_010 pixelprose_cc12m_sub_005 tevatron_colpali_pre_valid visrag_ind_unfiltered visrag_syn_unfiltered 0.5M

# export HF_HOME="/home/v-chenhaonan/teamdrive/multimodal/data/hf_cache"