from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning_lab.metrics import exact_match, normalized_edit_similarity, token_f1


def load_model(model_path: str):
    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (path / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(path, torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")
    return model.eval(), tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main(model_path: str, data_path: str, output: str, limit: int | None) -> None:
    model, tokenizer = load_model(model_path)
    rows = [json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]

    results = []
    for row in rows:
        pred = generate(model, tokenizer, row["prompt"])
        results.append(
            {
                "id": row["id"],
                "prediction": pred,
                "reference": row["answer"],
                "exact_match": exact_match(pred, row["answer"]),
                "token_f1": token_f1(pred, row["answer"]),
                "normalized_edit_similarity": normalized_edit_similarity(pred, row["answer"]),
            }
        )

    aggregate = {
        "model": model_path,
        "examples": len(results),
        "exact_match": sum(r["exact_match"] for r in results) / max(1, len(results)),
        "token_f1": sum(r["token_f1"] for r in results) / max(1, len(results)),
        "normalized_edit_similarity": sum(r["normalized_edit_similarity"] for r in results) / max(1, len(results)),
    }
    payload = {"aggregate": aggregate, "examples": results}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/docvqa_test.jsonl")
    parser.add_argument("--output", default="reports/results/evaluation.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.model, args.data, args.output, args.limit)
