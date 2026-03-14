#!/usr/bin/env python3
"""
Measure the 'translation tax' before and after fine-tuning.

Computes perplexity on N'Ko vs English text for base and fine-tuned models.
The gap between N'Ko and English perplexity IS the translation tax.

Usage:
  python3 eval_translation_tax.py \
    --model mlx-community/Qwen3-8B-8bit \
    --adapter adapters \
    --data training/valid.jsonl \
    --output results/translation_tax.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
except ImportError:
    print("ERROR: mlx and mlx_lm required")
    sys.exit(1)


def compute_perplexity(model, tokenizer, texts, max_tokens=256):
    """Compute average perplexity over a list of texts."""
    total_loss = 0.0
    total_tokens = 0
    num_processed = 0

    for i, text in enumerate(texts):
        try:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]

            x = mx.array([tokens[:-1]])
            y = mx.array([tokens[1:]])

            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Cross-entropy loss
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="sum"
            )
            mx.eval(loss)

            total_loss += float(loss)
            total_tokens += len(tokens) - 1
            num_processed += 1

        except Exception as e:
            if i < 3:
                print(f"  Skip example {i}: {e}")
            continue

    if total_tokens == 0:
        return float("inf"), 0, 0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap at exp(20) to avoid overflow
    return perplexity, avg_loss, num_processed


def split_by_language(data_path, max_per_lang=100):
    """Split validation data into N'Ko-containing and English-only examples."""
    nko_texts = []
    en_texts = []

    with open(data_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            msgs = entry.get("messages", [])
            if len(msgs) < 2:
                continue

            # Combine user + assistant text
            full_text = msgs[0]["content"] + " " + msgs[1]["content"]
            has_nko = any(0x07C0 <= ord(ch) <= 0x07FF for ch in full_text)

            if has_nko:
                nko_texts.append(full_text)
            else:
                en_texts.append(full_text)

    # Cap to max_per_lang
    import random
    random.seed(42)
    if len(nko_texts) > max_per_lang:
        nko_texts = random.sample(nko_texts, max_per_lang)
    if len(en_texts) > max_per_lang:
        en_texts = random.sample(en_texts, max_per_lang)

    return nko_texts, en_texts


def evaluate_model(model, tokenizer, nko_texts, en_texts, label=""):
    """Evaluate perplexity on both language sets."""
    print(f"\n{'='*50}")
    print(f"  Evaluating: {label}")
    print(f"{'='*50}")

    print(f"\n  N'Ko ({len(nko_texts)} examples)...")
    nko_ppl, nko_loss, nko_n = compute_perplexity(model, tokenizer, nko_texts)
    print(f"    Perplexity: {nko_ppl:.2f}")
    print(f"    Avg loss: {nko_loss:.4f}")
    print(f"    Processed: {nko_n}")

    print(f"\n  English ({len(en_texts)} examples)...")
    en_ppl, en_loss, en_n = compute_perplexity(model, tokenizer, en_texts)
    print(f"    Perplexity: {en_ppl:.2f}")
    print(f"    Avg loss: {en_loss:.4f}")
    print(f"    Processed: {en_n}")

    tax = nko_ppl / en_ppl if en_ppl > 0 else float("inf")
    print(f"\n  Translation Tax (N'Ko/English PPL ratio): {tax:.2f}x")

    return {
        "label": label,
        "nko_perplexity": nko_ppl,
        "nko_loss": nko_loss,
        "nko_examples": nko_n,
        "english_perplexity": en_ppl,
        "english_loss": en_loss,
        "english_examples": en_n,
        "translation_tax": tax,
    }


def main():
    parser = argparse.ArgumentParser(description="Translation Tax Evaluator")
    parser.add_argument("--model", required=True, help="Model path or HF repo")
    parser.add_argument("--adapter", default=None, help="Adapter path")
    parser.add_argument("--data", required=True, help="Validation JSONL")
    parser.add_argument("--output", default="results/translation_tax.json", help="Output JSON")
    parser.add_argument("--max-per-lang", type=int, default=100, help="Max examples per language")
    args = parser.parse_args()

    # Split data
    print("Splitting data by language...")
    nko_texts, en_texts = split_by_language(args.data, args.max_per_lang)
    print(f"  N'Ko: {len(nko_texts)} examples")
    print(f"  English: {len(en_texts)} examples")

    results = {}

    # Evaluate base model
    print("\nLoading base model...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    results["base"] = evaluate_model(model, tokenizer, nko_texts, en_texts, "Base Model")
    del model

    # Evaluate fine-tuned model
    if args.adapter:
        print("\nLoading fine-tuned model...")
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter,
            tokenizer_config={"trust_remote_code": True}
        )
        results["finetuned"] = evaluate_model(
            model, tokenizer, nko_texts, en_texts, "Fine-Tuned Model"
        )
        del model

    # Summary
    if "base" in results and "finetuned" in results:
        b = results["base"]
        f = results["finetuned"]
        print("\n" + "=" * 50)
        print("  BEFORE vs AFTER COMPARISON")
        print("=" * 50)
        print(f"\n  N'Ko Perplexity:")
        print(f"    Before: {b['nko_perplexity']:.2f}")
        print(f"    After:  {f['nko_perplexity']:.2f}")
        nko_change = ((f['nko_perplexity'] - b['nko_perplexity']) / b['nko_perplexity']) * 100
        print(f"    Change: {nko_change:+.1f}%")

        print(f"\n  Translation Tax (N'Ko/English PPL):")
        print(f"    Before: {b['translation_tax']:.2f}x")
        print(f"    After:  {f['translation_tax']:.2f}x")
        tax_change = ((f['translation_tax'] - b['translation_tax']) / b['translation_tax']) * 100
        print(f"    Change: {tax_change:+.1f}%")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
