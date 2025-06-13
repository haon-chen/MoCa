#!/usr/bin/env bash

set -x
set -e

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="intfloat/mmE5-qwen25-7B"
fi

if [ -z "$COLLECTION_NAME" ]; then
    COLLECTION_NAME="Haon-Chen/vidore-v2-full-683e7a451417d107337b45d2"
fi

if [ -z "$DATASET_FORMAT" ]; then
    DATASET_FORMAT="beir"
fi

if [ -z "$SPLIT" ]; then
    SPLIT="test"
fi

vidore-benchmark evaluate-retriever \
    --model-class mmeb-qwen25 \
    --model-name "$MODEL_NAME" \
    --collection-name "$COLLECTION_NAME" \
    --dataset-format "$DATASET_FORMAT" \
    --split "$SPLIT" 

echo "Evaluation complete." 