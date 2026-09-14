# LLM Fine-Tuning & Alignment Research Lab

A reproducible research lab for studying **Base vs Full SFT vs LoRA vs QLoRA vs DPO** on text extracted from document-question-answering data.

> **Research status:** infrastructure and experiment protocol are implemented; empirical metrics are intentionally not claimed until runs are executed.

## Research question

How much task performance and factual reliability can be gained from supervised and preference-based adaptation of a small instruction model, and what is the compute/parameter trade-off between full fine-tuning and parameter-efficient methods?

## Experimental matrix

| Variant | Trainable parameters | Quantization | Objective |
|---|---:|---|---|
| Base | 0 | No | Zero-shot baseline |
| Full SFT | All | No | Supervised fine-tuning |
| LoRA | Adapter only | No | Parameter-efficient SFT |
| QLoRA | Adapter only | 4-bit | Memory-efficient SFT |
| DPO | Adapter/full configurable | Optional | Preference alignment |

## Dataset

The default data source is `nielsr/docvqa_1200_examples`. The lab converts its English questions plus OCR tokens into a **text-only document QA** instruction format. The source dataset contains 1,000 training and 200 test examples and includes images, questions, human answers, OCR words and bounding boxes. This lab deliberately uses the OCR-text path so the comparison isolates language-model adaptation rather than visual encoder training.

## Metrics

- Exact Match (EM)
- token-level F1
- normalized edit similarity
- answerability / abstention behavior
- hallucination rate on adversarially perturbed contexts
- trainable parameter count
- peak GPU memory
- wall-clock training time
- checkpoint size

## Reproducibility

```bash
cd finetuning-lab
python -m venv .venv
# activate the environment
pip install -e .

python scripts/prepare_data.py --output-dir data/processed
python scripts/train.py --config configs/full_sft.yaml
python scripts/train.py --config configs/lora.yaml
python scripts/train.py --config configs/qlora.yaml
python scripts/train_dpo.py --config configs/dpo.yaml
python scripts/evaluate.py --model outputs/lora --split test
```

For GPU experiments, use the QLoRA configuration first. The repository records configurations and evaluation outputs rather than inventing results.

## Directory layout

```text
finetuning-lab/
├── configs/
├── data/
├── scripts/
├── src/finetuning_lab/
├── tests/
├── reports/
└── README.md
```

## Research integrity

No accuracy, speedup, VRAM reduction, publication, or benchmark claim is considered a result until it is produced by the scripts and stored under `reports/results/` with the exact configuration used.