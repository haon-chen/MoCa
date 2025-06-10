#!/usr/bin/env bash

set -x
set -e


if [ -z "$CKP_PATH" ]; then
    CKP_PATH="/home/v-chenhaonan/teamdrive/projects/multimodal/amlt-results/7250685354.27569-3e87ec0f-97a7"
    # CKP_PATH="/home/v-chenhaonan/teamdrive/projects/multimodal/amlt-results/7257768871.74213-140161f0-0600-4656-a4ac-9882e42671af"
fi
# if [ -z "$MODEL_NAME_OR_PATH" ]; then
#   MODEL_NAME_OR_PATH="intfloat/mmE5-mllama-11b-instruct"
# fi
if [ -z "$OUTPUT_DIR" ]; then
#   OUTPUT_DIR="./outputs/mmE5-qwen25"
  OUTPUT_DIR="/home/v-chenhaonan/teamdrive/multimodal/outputs/mmE5-qwen1"
#   OUTPUT_DIR="./outputs/mmE5-mllama-original-resolution"
fi
if [ -z "$DATA_DIR" ]; then
# change to local image dir
  DATA_DIR="/home/v-chenhaonan/teamdrive/multimodal/data/"
fi
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=16
fi
if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE=qwen2_5_vl
#   MODEL_BACKBONE=mllama
fi

if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
#   PROCESSOR_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
#   PROCESSOR_NAME="meta-llama/Llama-3.2-11B-Vision"
fi

if [ -z "$BIDIRECTIONAL" ]; then
  BIDIRECTIONAL=True
fi

if [ -z "$IMAGE_RESOLUTION" ]; then
#   IMAGE_RESOLUTION=high
  IMAGE_RESOLUTION=original
fi

if [ -z "$USE_LINEAR_PROJECTION" ]; then
  USE_LINEAR_PROJECTION=False
fi

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

#   --subset_name Wiki-SS-NQ Visual7W-Pointing RefCOCO RefCOCO-Matching ImageNet-1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 ScienceQA VizWiz GQA TextVQA OVEN FashionIQ EDIS \
#   --subset_name EDIS FashionIQ OVEN TextVQA GQA VizWiz ScienceQA Country211 ObjectNet ImageNet-R ImageNet-A Place365 MSCOCO MSCOCO_t2i MSCOCO_i2t VisualNews_t2i VisualNews_i2t WebQA NIGHTS CIRR VisDial Visual7W OK-VQA DocVQA A-OKVQA ChartQA InfographicsVQA VOC2007 SUN397 HatefulMemes N24News ImageNet-1K RefCOCO-Matching RefCOCO Visual7W-Pointing Wiki-SS-NQ \
#   --subset_name GQA Country211 ObjectNet \
#   --subset_name EDIS \
#  --model_name "${MODEL_NAME_OR_PATH}"
#   --dataset_name "TIGER-Lab/MMEB-eval" \

# PYTHONPATH=src/ python ./eval.py > eval.log 2>&1 \
PYTHONPATH=src/ python ./eval.py \
  --processor_name "${PROCESSOR_NAME}" \
  --checkpoint_path "${CKP_PATH}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --max_len ${MAX_LEN} \
  --pooling last --normalize True \
  --dataloader_num_workers 4 \
  --dataset_name "TIGER-Lab/MMEB-eval" \
  --subset_name EDIS FashionIQ OVEN TextVQA GQA VizWiz ScienceQA Country211 ObjectNet ImageNet-R ImageNet-A Place365 MSCOCO MSCOCO_t2i MSCOCO_i2t VisualNews_t2i VisualNews_i2t WebQA NIGHTS CIRR VisDial Visual7W OK-VQA DocVQA A-OKVQA ChartQA InfographicsVQA VOC2007 SUN397 HatefulMemes N24News ImageNet-1K RefCOCO-Matching RefCOCO Visual7W-Pointing Wiki-SS-NQ \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --image_dir "${DATA_DIR}/eval_images/" \
  --image_resolution "${IMAGE_RESOLUTION}" \
  --model_backbone "${MODEL_BACKBONE}" \
  --bidirectional ${BIDIRECTIONAL} \
  --use_linear_projection ${USE_LINEAR_PROJECTION}

echo "done"

# screen -S 2004471.mteb -X quit