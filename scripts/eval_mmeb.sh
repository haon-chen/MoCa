#!/usr/bin/env bash

set -x
set -e


if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="intfloat/MoCa-Qwen25VL-3B"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./outputs"
fi
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=16
fi
if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE=qwen2_5_vl
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

if [ -z "$MAX_LEN" ]; then
  MAX_LEN=1280
fi

PYTHONPATH=src/ python ./eval.py \
  --processor_name "${PROCESSOR_NAME}" \
  --model_name "${MODEL_NAME}" \
  --encode_output_path "${OUTPUT_DIR}" \
  --max_len ${MAX_LEN} \
  --pooling last --normalize True \
  --dataloader_num_workers 4 \
  --dataset_name "TIGER-Lab/MMEB-eval" \
  --subset_name Wiki-SS-NQ Visual7W-Pointing RefCOCO RefCOCO-Matching ImageNet-1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 ScienceQA VizWiz GQA TextVQA OVEN FashionIQ EDIS \
  --dataset_split test --per_device_eval_batch_size ${BATCH_SIZE} \
  --image_dir "images/eval_images/" \
  --image_resolution "${IMAGE_RESOLUTION}" \
  --model_backbone "${MODEL_BACKBONE}" \
  --bidirectional ${BIDIRECTIONAL}

echo "done"
