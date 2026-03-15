#!/usr/bin/env python3
"""V3 generative evaluation: check for mode collapse.

Generates N'Ko text from 20 prompts and measures:
- Unique token ratio (1.0 = no repetition, 0.0 = total collapse)
- Distinct-1/Distinct-2 (unigram/bigram diversity)
- N'Ko character count and ratio
- Whether response is degenerate (same token repeated)

Usage:
    python3 eval/run_v3_generation.py
    python3 eval/run_v3_generation.py --model ~/nko-brain-scanner/fused-v3-nko-qwen3
"""

import argparse
import json
import os
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

NKO_RANGE = range(0x07C0, 0x0800)
OUTPUT_PATH = os.path.expanduser("~/nko-brain-scanner/results/v3_generation.json")

# Same 20 prompts used for V2 evaluation
PROMPTS = [
    "ߌ ߕߐ߮ ߦߋ߫ ߡߎ߲߬",
    "ߡߊ߲߬ߕߎ ߞߊ߲ ߦߋ߫ ߡߎ߲߬",
    "ߒ ߞߊ߬ ߟߊ߫ ߘߋ߬ ߟߊ߫",
    "ߒ ߦߋ߫ ߛߓߍ ߞߍ߫ ߟߊ߫",
    "ߞߊ߬ ߝߘߊ ߘߏ߫ ߘߊߡߌߘߊ",
    "ߘߎ߰ ߓߌ߬ ߦߋ߫ ߡߎ߲߬",
    "ߊߟߎ߫ ߛߊ ߦߋ߫ ߘߏ߫",
    "ߡߐ߰ ߞߋ߬ߟߋ",
    "ߓߊ߲ߡߊ߬ ߞߊ߲ ߠߊ߬ ߘߐ߬ ߖߊ߬ ߓߌ",
    "ߊ߬ ߞߊ߬ ߓߊ߲ ߞߎ߲߬ ߞߊ߬ ߝߘߊ",
    "ߒߞߏ ߛߓߍ ߟߊ ߞߊ߲ ߠߊ",
    "ߝߏ߬ ߡߊ߬ ߓߊ߲ ߡ ߛ ߞ",
    "ߡߊ߲߬ߘߋ߲ ߘߐ ߛߌ",
    "ߒ ߓߊ ߦ ߝ",
    "ߊ߬ ߓ ߊ߬ ߡ ߊ",
    "ߛߌ߬ ߞ ߊ ߡ ߊ ߝ",
    "ߝߘ ߊ ߞ ߊ ߡ ߛ ߓ",
    "ߞ ߊ ߝ ߊ ߡ ߊ ߟ",
    "ߒ ߡ ߊ ߓ ߊ ߟ ߊ",
    "ߓ ߊ ߛ ߌ ߞ ߊ ߡ ߊ",
]


def has_nko(text):
    return any(ord(c) in NKO_RANGE for c in text)


def nko_char_count(text):
    return sum(1 for c in text if ord(c) in NKO_RANGE)


def is_degenerate(text, threshold=0.7):
    """Check if text is degenerate (dominated by repeated tokens)."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 5:
        return True
    counter = Counter(chars)
    most_common_freq = counter.most_common(1)[0][1] / len(chars) if chars else 0
    return most_common_freq > threshold


def distinct_n(tokens, n):
    """Distinct-N: ratio of unique n-grams to total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def unique_token_ratio(tokens):
    """Ratio of unique tokens to total tokens."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser(
        "~/nko-brain-scanner/fused-v3-nko-qwen3"))
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--rep-penalty", type=float, default=1.3)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    results = []
    total_degenerate = 0
    total_nko_chars = 0
    all_distinct1 = []
    all_distinct2 = []
    all_unique_ratios = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/20] Prompt: {prompt}")

        # Generate with temperature sampling (no rep penalty — tests raw model quality)
        response_tokens = []
        response_text = ""
        sampler = make_sampler(temp=args.temp)

        for token_result in stream_generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            response_text += token_result.text
            response_tokens.append(token_result.token)

        # Analyze
        nko_count = nko_char_count(response_text)
        degenerate = is_degenerate(response_text)
        d1 = distinct_n(response_tokens, 1)
        d2 = distinct_n(response_tokens, 2)
        utr = unique_token_ratio(response_tokens)

        total_nko_chars += nko_count
        if degenerate:
            total_degenerate += 1
        all_distinct1.append(d1)
        all_distinct2.append(d2)
        all_unique_ratios.append(utr)

        status = "DEGENERATE" if degenerate else "OK"
        print(f"  Response ({len(response_text)} chars, {nko_count} N'Ko): "
              f"{response_text[:80]}...")
        print(f"  D1={d1:.3f} D2={d2:.3f} UTR={utr:.3f} [{status}]")

        results.append({
            "id": i + 1,
            "prompt": prompt,
            "response": response_text,
            "nko_chars": nko_count,
            "total_chars": len(response_text),
            "degenerate": degenerate,
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "unique_token_ratio": round(utr, 4),
            "num_tokens": len(response_tokens),
        })

    # Summary
    print("\n" + "=" * 70)
    print("V3 GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total prompts: 20")
    print(f"Degenerate responses: {total_degenerate}/20 "
          f"({total_degenerate/20*100:.0f}%)")
    print(f"Avg Distinct-1: {sum(all_distinct1)/len(all_distinct1):.3f}")
    print(f"Avg Distinct-2: {sum(all_distinct2)/len(all_distinct2):.3f}")
    print(f"Avg Unique Token Ratio: "
          f"{sum(all_unique_ratios)/len(all_unique_ratios):.3f}")
    print(f"Total N'Ko chars generated: {total_nko_chars}")

    collapse_verdict = "MODE COLLAPSE" if total_degenerate >= 15 else \
        "PARTIAL COLLAPSE" if total_degenerate >= 5 else \
        "NO MODE COLLAPSE"
    print(f"\nVerdict: {collapse_verdict}")

    output = {
        "samples": results,
        "summary": {
            "total_prompts": 20,
            "degenerate_count": total_degenerate,
            "degenerate_pct": round(total_degenerate / 20 * 100, 1),
            "avg_distinct_1": round(sum(all_distinct1) / len(all_distinct1), 4),
            "avg_distinct_2": round(sum(all_distinct2) / len(all_distinct2), 4),
            "avg_unique_token_ratio": round(
                sum(all_unique_ratios) / len(all_unique_ratios), 4),
            "total_nko_chars": total_nko_chars,
            "verdict": collapse_verdict,
        },
        "metadata": {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "repetition_penalty": args.rep_penalty,
            "timestamp": datetime.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
