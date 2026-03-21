#!/usr/bin/env python3
"""
Compare English cognitive twin vs N'Ko cognitive twin.

Loads both LoRA adapters, runs the same 50 evaluation prompts through each,
and measures perplexity, token compression, and behavioral divergence.

Usage:
    python3 compare_twins.py \
        --model mlx-community/Qwen3-8B-8bit \
        --english-adapter ~/adapters-english/ \
        --nko-adapter ~/adapters-nko/ \
        --prompts eval_prompts.jsonl \
        --output results/twin_comparison.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
except ImportError:
    print("ERROR: mlx and mlx_lm required. pip install mlx mlx-lm")
    sys.exit(1)


def compute_perplexity(model, tokenizer, text, max_tokens=256):
    """Compute perplexity for a single text."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 4:
        return float("inf"), len(tokens)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    x = mx.array([tokens[:-1]])
    y = mx.array([tokens[1:]])

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]

    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="mean",
    )
    mx.eval(loss)

    ppl = math.exp(min(float(loss), 20))
    return ppl, len(tokens)


def generate_response(model, tokenizer, prompt, max_tokens=200):
    """Generate a response from the model given a prompt."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Simple greedy generation
    generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        token_id = int(next_token[0])

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

    return tokenizer.decode(generated)


def evaluate_adapter(model_name, adapter_path, prompts, label):
    """Evaluate a single adapter on all prompts."""
    print(f"\nLoading {label}: {model_name}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")
        model, tokenizer = load(model_name, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_name)

    results = []
    total_ppl = 0
    total_tokens = 0
    valid = 0

    for i, prompt_entry in enumerate(prompts):
        prompt_text = prompt_entry["prompt"]
        category = prompt_entry.get("category", "general")

        # Compute perplexity on prompt
        ppl, num_tokens = compute_perplexity(model, tokenizer, prompt_text)

        # Generate response
        try:
            response = generate_response(model, tokenizer, prompt_text, max_tokens=150)
            response_tokens = len(tokenizer.encode(response))
        except Exception as e:
            response = f"[generation failed: {e}]"
            response_tokens = 0

        result = {
            "prompt_idx": i,
            "category": category,
            "perplexity": ppl if ppl != float("inf") else 999999,
            "prompt_tokens": num_tokens,
            "response_tokens": response_tokens,
            "response_preview": response[:200],
        }
        results.append(result)

        if ppl != float("inf"):
            total_ppl += ppl
            total_tokens += response_tokens
            valid += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(prompts)}] avg_ppl={total_ppl / valid:.1f}" if valid else "")

    avg_ppl = total_ppl / valid if valid else float("inf")
    avg_tokens = total_tokens / valid if valid else 0

    print(f"  {label}: avg_ppl={avg_ppl:.1f}, avg_response_tokens={avg_tokens:.1f}, valid={valid}")

    del model
    mx.clear_cache()

    return {
        "label": label,
        "avg_perplexity": avg_ppl,
        "avg_response_tokens": avg_tokens,
        "valid_prompts": valid,
        "per_prompt": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare English vs N'Ko cognitive twins")
    parser.add_argument("--model", required=True, help="Base model name/path")
    parser.add_argument("--english-adapter", required=True, help="English adapter path")
    parser.add_argument("--nko-adapter", required=True, help="N'Ko adapter path")
    parser.add_argument("--prompts", required=True, help="Evaluation prompts JSONL")
    parser.add_argument("--output", default="results/twin_comparison.json")
    args = parser.parse_args()

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    print(f"Loaded {len(prompts)} evaluation prompts")

    # Evaluate English twin
    en_results = evaluate_adapter(args.model, args.english_adapter, prompts, "English Twin")

    # Evaluate N'Ko twin
    nko_results = evaluate_adapter(args.model, args.nko_adapter, prompts, "N'Ko Twin")

    # Comparison
    print("\n" + "=" * 60)
    print("TWIN COMPARISON")
    print("=" * 60)

    en_ppl = en_results["avg_perplexity"]
    nko_ppl = nko_results["avg_perplexity"]
    en_tokens = en_results["avg_response_tokens"]
    nko_tokens = nko_results["avg_response_tokens"]

    print(f"\n  Perplexity:")
    print(f"    English: {en_ppl:.1f}")
    print(f"    N'Ko:    {nko_ppl:.1f}")
    if en_ppl > 0:
        print(f"    Ratio:   {nko_ppl / en_ppl:.2f}x")

    print(f"\n  Token Economy:")
    print(f"    English avg response tokens: {en_tokens:.1f}")
    print(f"    N'Ko avg response tokens:    {nko_tokens:.1f}")
    if en_tokens > 0:
        compression = en_tokens / nko_tokens if nko_tokens > 0 else float("inf")
        print(f"    Compression ratio:           {compression:.2f}x")

    # Per-category breakdown
    categories = set()
    for p in prompts:
        categories.add(p.get("category", "general"))

    print(f"\n  Per-Category Perplexity:")
    for cat in sorted(categories):
        en_cat = [r for r, p in zip(en_results["per_prompt"], prompts) if p.get("category") == cat]
        nko_cat = [r for r, p in zip(nko_results["per_prompt"], prompts) if p.get("category") == cat]

        en_avg = sum(r["perplexity"] for r in en_cat) / len(en_cat) if en_cat else 0
        nko_avg = sum(r["perplexity"] for r in nko_cat) / len(nko_cat) if nko_cat else 0
        print(f"    {cat:<25} EN={en_avg:>8.1f}  NKo={nko_avg:>8.1f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "english": en_results,
        "nko": nko_results,
        "summary": {
            "perplexity_ratio": nko_ppl / en_ppl if en_ppl > 0 else None,
            "compression_ratio": en_tokens / nko_tokens if nko_tokens > 0 else None,
            "english_avg_ppl": en_ppl,
            "nko_avg_ppl": nko_ppl,
        },
    }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
