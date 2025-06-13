# [Transforming VLMs into Powerful Multimodal Encoders via Continual Pre‑training](https://arxiv.org/abs/XXXX.XXXXX)

This repo hosts **code, checkpoints, and data scripts** for our paper **xxx**.
We convert causal Vision‑Language Models into *bidirectional* multimodal embedders through a two‑stage pipeline:

1. **Modality‑aware Continual Pre‑training** MLM + MAE jointly reconstruct text tokens and image patches.
2. **Heterogeneous Contrastive Fine‑tuning** Aligns the encoder on captions, long‑form docs, and text‑only pairs.

---

## Updates

- **2025‑06‑10:** Initial release – paper, training scripts, checkpoints, and evaluation notebooks.

## Quick Start
```bash
git clone https://github.com/<user>/tme-cp && cd tme-cp
pip install -r requirements.txt
# Download images (≈1 TB)
bash scripts/prepare_images.sh
# Training
bash scripts/train/train.sh
# Evaluation
bash scripts/eval/eval_mmeb.sh
# Encode your own data
python demo.py --text "Pepper the aussie pup" --image path/to/img.jpg
```

## Resources

- **Synthetic + Real Data** covering captions, VQA, long‑form docs (MMEB, ViDoRe, etc.).
- **Checkpoints**: 3 B & 7 B bidirectional encoders (FP16 & INT8).
- **Evaluation**: Scripts for MMEB, ViDoRe‑v1/v2, XTD‑10.

## Reproducing Table 1
All MMEB scores can be reproduced with:
```bash
bash scripts/reproduce/mmeb_7b.sh  # 1×A100‑80G, <3 h
```

## Acknowledgement
Our code builds on **mmE5**, **VLM2Vec**, and **Qwen‑2.5‑VL**.

## Citation
```bibtex
@article{xxx,
  title={Transforming VLMs into Powerful Multimodal Encoders via Continual Pre-training},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```