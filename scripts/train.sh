#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME_OR_PATH="./checkpoint/Qwen2.5-VL-3B-Instruct-Embed-Init"
MODEL_NAME_OR_PATH="/home/v-chenhaonan/teamdrive/projects/multimodal/amlt-results/7251206589.55459-e97332a9-274f-4e11-b230-0d070bda4790/checkpoint-17000"

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/ft_$(date +%F-%H%M.%S)"
#   OUTPUT_DIR="/home/v-chenhaonan/teamdrive/projects/multimodal/amlt-results/7252261233.44673-3d7e8eee-e832-4cf2-a0dd-10910889bd87"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/home/v-chenhaonan/teamdrive/multimodal/data/"
fi

DS_CONFIG_PATH="ds_config_stage3.json"
# DS_CONFIG_PATH="ds_config.json"
if [ "$(nvidia-smi --list-gpus | wc -l)" = "1" ]; then
  DS_CONFIG_PATH="ds_config_stage3.json"
  # DS_CONFIG_PATH="ds_config.json"
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

if [ -z "$IMAGE_RESOLUTION" ]; then
#   IMAGE_RESOLUTION=high
  IMAGE_RESOLUTION=original
fi

if [ -z "$USE_TASK_BATCH" ]; then
  USE_TASK_BATCH=True
fi

if [ -z "$USE_LINEAR_PROJECTION" ]; then
  USE_LINEAR_PROJECTION=False
fi

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

    # --dataset_name "intfloat/mmE5-MMEB-hardneg" \
    # --gradient_checkpointing True --gradient_accumulation_steps 8 \
    # --negative_ratio 1 \
    # --lora --lora_r 8 \
    # wiki_pairs arxiv_questions_pairs mldr_pairs miracl_pairs
    # --subset_name wiki_pairs arxiv_questions_pairs mldr_pairs miracl_pairs trivia t2ranking squad s2orc quora orcas nq nli msmarco msmarco-doc mrtydi miracl kilt eli5 dureader codesearchnet bitext TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --subset_name trivia t2ranking squad s2orc quora orcas nq nli msmarco msmarco-doc mrtydi miracl kilt eli5 dureader codesearchnet bitext TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --subset_name TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --subset_name tevatron_colpali TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --subset_name visrag_ind visrag_syn_patched tevatron_colpali TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --subset_name visrag_ind visrag_syn_patched tevatron_colpali trivia t2ranking squad s2orc quora orcas nq nli msmarco msmarco-doc mrtydi miracl kilt eli5 dureader codesearchnet bitext TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --subset_name vdr_multi visrag_ind visrag_syn_patched tevatron_colpali trivia t2ranking squad s2orc quora orcas nq nli msmarco msmarco-doc mrtydi miracl kilt eli5 dureader codesearchnet bitext TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    # --synthetic_dataset_name "intfloat/mmE5-synthetic" \
    # --synthetic_subset_name Classification Retrieval VQA \

# deepspeed --master_port 18271 train.py --deepspeed "${DS_CONFIG_PATH}" \
deepspeed --num_nodes "${WORLD_SIZE}" --master_port "${MASTER_PORT}" --master_addr "${MASTER_ADDR}" train.py --deepspeed "${DS_CONFIG_PATH}" \
    --subset_name visrag_ind visrag_syn_patched tevatron_colpali trivia t2ranking squad s2orc quora orcas nq nli msmarco msmarco-doc mrtydi miracl kilt eli5 dureader codesearchnet bitext TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
    --dataset_path "${DATA_DIR}/contrastive_learning/MMEB-hardneg" \
    --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
    --num_sample_per_subset 50000 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True --gradient_accumulation_steps 1 \
    --image_dir "${DATA_DIR}/MMEB-train" \
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
    --use_task_batch ${USE_TASK_BATCH} \
    --use_linear_projection ${USE_LINEAR_PROJECTION} \
    --report_to none "$@"

#  Qwen-3B-Embed-Init-Full bsz24 grad1 2e-5 epoch1 temp0.03 maxlen2048+1536
#  Qwen-3B-Embed-Init-Full bsz24 grad1 2e-5 epoch1 temp0.03 maxlen2048 fix reso
#  Qwen-3B-Embed-Init-Full bsz40 grad1 2e-5 epoch1 temp0.03 maxlen1280
#  Qwen-3B-Embed-Init-Full bsz16 grad1 1e-5 epoch1 temp0.03 maxlen1280
#  1035b8f676c0 b12 grad1 2e-5 epoch1 temp0.03 maxlen1280
#  72d92b874a88 b12 grad1 1e-5 epoch1 temp0.03 maxlen1280
#  1035b8f676c0 b32 grad1 1e-5 epoch1 temp0.03 maxlen1280
#  9db652d5cfa1 b32 grad1 1e-5 epoch1 temp0.03 maxlen1280

#  e3f69a27072c b32 grad1 1e-5 epoch1 temp0.03 maxlen1280
#  7148f9e72ead b12 grad1 1e-5 epoch1 temp0.03 maxlen1280 colpali_T2I
#  9db652d5cfa1 b16 grad1 1e-5 epoch1 temp0.03 maxlen1280

#  157713d41203 b16 grad1 1e-5 epoch1 temp0.03 maxlen1280

#  Qwen-3B-Embed-Init-Ful-full b16 grad1 2e-5 epoch1 temp0.03 maxlen1280
#  135728fbefd9 b12 grad1 2e-5 epoch1 temp0.03 maxlen1280
#  135728fbefd9 b16 grad1 2e-5 epoch1 temp0.03 maxlen1280
# pip install "vidore-benchmark[qwen]"
# 3B USE_TASK_BATCH nolinear texts visrag_ind visrag_syn_patched tevatron_colpali