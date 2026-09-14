# Experiment Protocol

## Hypotheses

H1. Task-specific SFT improves document QA over the frozen base model.

H2. LoRA approaches full SFT performance with substantially fewer trainable parameters.

H3. QLoRA reduces memory requirements while preserving most of the LoRA quality.

H4. DPO improves preference for evidence-grounded answers and reduces unsupported answers under adversarial evaluation.

## Controls

- Same base checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`
- Same English DocVQA-derived train/test split
- Fixed seed: 42
- Same prompt template
- Greedy decoding for evaluation
- No test examples in training
- Record hardware, CUDA/PyTorch/Transformers/TRL versions

## Required result table

| Variant | EM | Token F1 | Edit similarity | Hallucination rate | Trainable params | Peak VRAM | Train time | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base | TBD | TBD | TBD | TBD | 0 | TBD | 0 | N/A |
| Full SFT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| QLoRA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Ablations

1. LoRA rank: 4 / 8 / 16 / 32
2. Context length: 512 / 1024 / 2048
3. Training size: 10% / 25% / 50% / 100%
4. Learning rate around the selected baseline
5. DPO beta: 0.05 / 0.1 / 0.2

## Failure analysis

For every evaluated variant, retain examples where:

- the answer is unsupported by OCR evidence;
- the answer is semantically correct but string-mismatched;
- the model copies irrelevant document text;
- the model refuses despite sufficient evidence;
- the model follows a contradictory preference example.

The final research conclusion must be based on these stored measurements, not on expected outcomes.