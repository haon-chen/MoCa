#!/usr/bin/env bash
set -xe

# Get the project root directory (parent directory of the script's location)
DIR="$(cd "$(dirname "$0")" && cd .. && pwd)"
echo "Working directory: ${DIR}"

mkdir -p images
mkdir -p images/image_zips

# Download the Synthetic dataset images from mmE5

LAION_SYNTHETIC="LAION_Synthetic.tar.gz"
if [ ! -e images/image_zips/${LAION_SYNTHETIC} ]; then
  wget -O images/image_zips/${LAION_SYNTHETIC} "https://huggingface.co/datasets/intfloat/mmE5-synthetic/resolve/main/${LAION_SYNTHETIC}"
fi
tar -I "pigz -d -p 8" -xf images/image_zips/${LAION_SYNTHETIC} -C images/


# Download the Infoseek dataset (M-BEIR) images (download parts, then merge and extract)

INFOSEEK="mbeir_images"
PARTS=("00" "01" "02" "03")
for PART in "${PARTS[@]}"; do
    if [ ! -e images/image_zips/${INFOSEEK}.tar.gz.part-${PART} ]; then
        wget -O images/image_zips/${INFOSEEK}.tar.gz.part-${PART} "https://huggingface.co/datasets/TIGER-Lab/M-BEIR/resolve/main/${INFOSEEK}.tar.gz.part-${PART}"
    fi
done

if [ ! -e images/image_zips/${INFOSEEK}.tar.gz ]; then
  echo "Combining parts into ${INFOSEEK}.tar.gz..."
  cat images/image_zips/${INFOSEEK}.tar.gz.part-00 images/image_zips/${INFOSEEK}.tar.gz.part-01 images/image_zips/${INFOSEEK}.tar.gz.part-02 images/image_zips/${INFOSEEK}.tar.gz.part-03 > images/image_zips/${INFOSEEK}.tar.gz
  echo "Extracting ${INFOSEEK}/oven_images from ${INFOSEEK}.tar.gz..."  
  cd images
  tar -I "pigz -d -p 8" -xf image_zips/${INFOSEEK}.tar.gz --wildcards 'mbeir_images/oven_images/*'
  cd ..
fi


# Download MMEB Training dataset images

MMEB_DATASETS=("ImageNet_1K" "N24News" "HatefulMemes" "SUN397" "VOC2007" "InfographicsVQA" "ChartQA" "A-OKVQA" "DocVQA" "OK-VQA" "Visual7W" "VisDial" "CIRR" "NIGHTS" "WebQA" "VisualNews_i2t" "VisualNews_t2i" "MSCOCO_i2t" "MSCOCO_t2i" "MSCOCO")
for DATASET in "${MMEB_DATASETS[@]}"; do
    if [ ! -e images/image_zips/${DATASET}.zip ]; then
        wget -O images/image_zips/${DATASET}.zip "https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/${DATASET}.zip"
        unzip -q images/image_zips/${DATASET}.zip -d images/
    fi
done

# Download Supplement MMEB Training dataset images
SUPP_MMEB_DATASETS=("TAT-DQA" "ArxivQA" "colpali_tevatron" "visrag_ind")
for DATASET in "${SUPP_MMEB_DATASETS[@]}"; do
    if [ ! -e images/image_zips/${DATASET}.tar.gz ]; then
        wget -O images/image_zips/${DATASET}.tar.gz "https://huggingface.co/datasets/moca-embed/MoCa-CL-Pairs/resolve/main/supplement_images_zip/${DATASET}.tar.gz"
        tar -I "pigz -d -p 8" -xf images/image_zips/${DATASET}.tar.gz -C images/
    fi
done

# Download visrag_syn dataset images (download parts, then merge and extract)
VISRAG_SYN="visrag_syn.tar.gz"
VISRAG_SYN_PARTS=("aa" "ab" "ac" "ad" "ae")
for PART in "${VISRAG_SYN_PARTS[@]}"; do
    if [ ! -e images/image_zips/${VISRAG_SYN}.part_${PART} ]; then
        wget -O images/image_zips/${VISRAG_SYN}.part_${PART} "https://huggingface.co/datasets/moca-embed/MoCa-CL-Pairs/resolve/main/supplement_images_zip/${VISRAG_SYN}.part_${PART}"
    fi
done

if [ ! -e images/image_zips/${VISRAG_SYN} ]; then
  echo "Combining parts into ${VISRAG_SYN}..."
  cat images/image_zips/${VISRAG_SYN}.part_* > images/image_zips/${VISRAG_SYN}
  echo "Extracting from ${VISRAG_SYN}..."  
  tar -I "pigz -d -p 8" -xf images/image_zips/${VISRAG_SYN} -C images/
fi

# Download MMEB Eval dataset images
MMEB_EVAL="eval_images"
if [ ! -e images/image_zips/${MMEB_EVAL}.zip ]; then
  wget -O images/image_zips/${MMEB_EVAL}.zip "https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip"
  unzip -q images/image_zips/${MMEB_EVAL}.zip -d images/${MMEB_EVAL}/
fi

echo "All images downloaded!"