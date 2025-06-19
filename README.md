# [Transforming VLMs into Powerful Multimodal Encoders via Continual Pre‑training](https://arxiv.org/abs/XXXX.XXXXX)

This repo hosts **code, checkpoints, and data scripts** for our paper **xxx**.
We convert causal Vision‑Language Models into *bidirectional* multimodal embedders through a two‑stage pipeline:

1. **Modality‑aware Continual Pre‑training** MLM + MAE jointly reconstruct text tokens and image patches.
2. **Heterogeneous Contrastive Fine‑tuning** Aligns the encoder on captions, long‑form docs, and text‑only pairs.

---

## Updates

- **2025‑06‑10:** Initial release – paper, training scripts, checkpoints, and evaluation notebooks.

## Quick Start

```
pip install -r requirements.txt
pip install flash-attn==2.5.8
```

- Preparation

```
bash scripts/prepare_images.sh
```

This script will download images from [Synthetic Dataset](https://huggingface.co/datasets/intfloat/mmE5-synthetic), [MMEB with Hard Negative](https://huggingface.co/datasets/intfloat/mmE5-MMEB-hardneg), [MMEB-eval](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval).

**Caution:** This could take a while as the images are large in size. Make sure you have enough disk space (at least 1T).

We have provided example scripts in the `scripts/` directory to help you get started with training and evaluation.

- Continual Pre-training
```
bash scripts/cpt_train.sh
```
- Contrastive Learning
```
bash scripts/cl_train.sh
```
- Test MMEB
```
bash scripts/eval_full.sh
```
- Test ViDoRe-v2

1. Install vidore-benchmark package following [this repo](https://github.com/illuin-tech/vidore-benchmark).

2. Move __init__.py and mmeb_qwen25_retriever.py from /evaluation/vidore_benchmark/ to the vidore-benchmark repo (src/vidore_benchmark/evaluation).

3. Run
```
bash scripts/eval_vidore.sh
```

You can also use `demo.py` to embed your own text and images.
```
python demo.py
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