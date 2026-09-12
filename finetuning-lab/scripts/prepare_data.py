from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def build_prompt(words: list[str], question: str) -> str:
    context = " ".join(str(w) for w in words if str(w).strip())
    context = " ".join(context.split())[:9000]
    return (
        "You are a document intelligence assistant. Answer only from the OCR text. "
        "If the evidence is insufficient, say so.\n\n"
        f"DOCUMENT OCR:\n{context}\n\nQUESTION:\n{question}"
    )


def convert(split: str, output: Path) -> list[dict]:
    ds = load_dataset("nielsr/docvqa_1200_examples", split=split)
    rows = []
    for ex in ds:
        question = ex["query"].get("en", "").strip()
        answers = ex["answers"]
        if not question or not answers:
            continue
        prompt = build_prompt(ex["words"], question)
        answer = str(answers[0]).strip()
        rows.append(
            {
                "id": ex["id"],
                "prompt": prompt,
                "answer": answer,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def make_preferences(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            # Deterministic hard negative: a correct answer from a different document.
            rejected = rows[(i + 1) % len(rows)]["answer"] if len(rows) > 1 else "I cannot determine this from the document."
            if rejected == row["answer"]:
                rejected = "I cannot determine this from the document."
            record = {"prompt": row["prompt"], "chosen": row["answer"], "rejected": rejected}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    out = Path(args.output_dir)
    train = convert("train", out / "docvqa_train.jsonl")
    convert("test", out / "docvqa_test.jsonl")
    make_preferences(train, out / "docvqa_preferences.jsonl")
    print(f"prepared {len(train)} training examples")
