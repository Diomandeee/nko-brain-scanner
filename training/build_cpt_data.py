#!/usr/bin/env python3
"""Build continued pre-training (CPT) data from N'Ko Wikipedia corpus.

Creates chat-format JSONL compatible with MLX LoRA training.
Strategy: chunk articles into overlapping windows and create
text-completion tasks (given context, predict continuation).
"""

import json
import random
import os

WIKI_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nko_wikipedia_corpus.jsonl")
OUTPUT_DIR = os.path.dirname(__file__)
MIN_NKO_CHARS = 50
CHUNK_SIZE = 300       # chars per chunk (N'Ko is ~1 token/char)
OVERLAP = 50           # overlap between chunks
MIN_CHUNK_NKO = 20     # minimum N'Ko chars in a chunk
CONTEXT_RATIO = 0.6    # 60% context, 40% completion target
SEED = 42


def count_nko(text: str) -> int:
    return sum(1 for c in text if "\u07C0" <= c <= "\u07FF")


def chunk_article(text: str, title: str) -> list[dict]:
    """Split article into overlapping chunks for text completion training."""
    examples = []
    # Clean: remove very short lines, keep paragraph structure
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    full_text = "\n".join(lines)

    if len(full_text) < 80:
        return []

    # Strategy 1: Full article as reading comprehension (for short articles)
    if len(full_text) < CHUNK_SIZE * 2:
        nko_count = count_nko(full_text)
        if nko_count >= MIN_CHUNK_NKO:
            split_point = int(len(full_text) * CONTEXT_RATIO)
            # Find a sentence boundary near the split point
            for i in range(split_point, min(split_point + 80, len(full_text))):
                if full_text[i] in ".!?\u061F\u061B\u060C\u0964\u002E\u002C ":
                    split_point = i + 1
                    break
            context = full_text[:split_point].strip()
            continuation = full_text[split_point:].strip()
            if context and continuation and count_nko(continuation) >= 5:
                examples.append({
                    "messages": [
                        {"role": "user", "content": f"ߞߊ߬ ߣߌ߲߬ ߟߊߓߊ߲߬: {context}"},
                        {"role": "assistant", "content": continuation}
                    ]
                })
        return examples

    # Strategy 2: Sliding window chunks (for longer articles)
    pos = 0
    while pos < len(full_text) - 80:
        end = min(pos + CHUNK_SIZE, len(full_text))
        chunk = full_text[pos:end]

        # Find a clean break point
        if end < len(full_text):
            for i in range(len(chunk) - 1, max(len(chunk) - 80, 0), -1):
                if chunk[i] in ".!?\u061F\u061B ":
                    chunk = chunk[:i + 1]
                    break

        nko_count = count_nko(chunk)
        if nko_count >= MIN_CHUNK_NKO:
            split_point = int(len(chunk) * CONTEXT_RATIO)
            # Find sentence boundary
            for i in range(split_point, min(split_point + 60, len(chunk))):
                if chunk[i] in ".!?\u061F\u061B\u060C ":
                    split_point = i + 1
                    break

            context = chunk[:split_point].strip()
            continuation = chunk[split_point:].strip()

            if context and continuation and len(continuation) > 20:
                examples.append({
                    "messages": [
                        {"role": "user", "content": f"ߞߊ߬ ߣߌ߲߬ ߟߊߓߊ߲߬: {context}"},
                        {"role": "assistant", "content": continuation}
                    ]
                })

        pos += CHUNK_SIZE - OVERLAP

    # Strategy 3: Title-based completion (use title as prompt)
    first_para = lines[0] if lines else ""
    if count_nko(first_para) >= MIN_CHUNK_NKO and len(first_para) > 40:
        examples.append({
            "messages": [
                {"role": "user", "content": f"{title}: "},
                {"role": "assistant", "content": first_para}
            ]
        })

    return examples


def main():
    random.seed(SEED)
    articles = []

    with open(WIKI_PATH) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("nko_chars", 0) >= MIN_NKO_CHARS:
                articles.append(obj)

    print(f"Loaded {len(articles)} articles with >={MIN_NKO_CHARS} N'Ko chars")

    all_examples = []
    for art in articles:
        examples = chunk_article(art["text"], art["title"])
        all_examples.extend(examples)

    random.shuffle(all_examples)

    # Split: 90% train, 10% valid
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    valid = all_examples[split_idx:]

    # Write CPT-only files
    train_path = os.path.join(OUTPUT_DIR, "cpt_train.jsonl")
    valid_path = os.path.join(OUTPUT_DIR, "cpt_valid.jsonl")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"CPT train: {len(train)} examples -> {train_path}")
    print(f"CPT valid: {len(valid)} examples -> {valid_path}")

    # Also build combined (CPT + SFT) dataset
    sft_train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    sft_valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")

    sft_train = []
    sft_valid = []
    if os.path.exists(sft_train_path):
        with open(sft_train_path) as f:
            for line in f:
                sft_train.append(json.loads(line))
    if os.path.exists(sft_valid_path):
        with open(sft_valid_path) as f:
            for line in f:
                sft_valid.append(json.loads(line))

    combined_train = train + sft_train
    combined_valid = valid + sft_valid
    random.shuffle(combined_train)
    random.shuffle(combined_valid)

    combined_train_path = os.path.join(OUTPUT_DIR, "combined_train.jsonl")
    combined_valid_path = os.path.join(OUTPUT_DIR, "combined_valid.jsonl")

    with open(combined_train_path, "w") as f:
        for ex in combined_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(combined_valid_path, "w") as f:
        for ex in combined_valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nCombined train: {len(combined_train)} ({len(train)} CPT + {len(sft_train)} SFT)")
    print(f"Combined valid: {len(combined_valid)} ({len(valid)} CPT + {len(sft_valid)} SFT)")

    # Stats
    total_nko = sum(
        count_nko(ex["messages"][0]["content"] + ex["messages"][1]["content"])
        for ex in combined_train
    )
    print(f"Total N'Ko chars in combined train: {total_nko:,}")


if __name__ == "__main__":
    main()
