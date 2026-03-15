#!/usr/bin/env python3
"""
Evaluate admissibility-constrained decoding on N'Ko generation.

Generates N text samples with and without the syllable FSM constraint,
then measures:
  - % valid N'Ko syllables in output
  - Perplexity difference (does constraint hurt fluency?)
  - Generation speed overhead
  - GK boost effect (if GK available)

Usage:
    python3 constrained/eval_admissibility.py \
        --model ~/nko-brain-scanner/fused-extended-nko-qwen3/ \
        --num-samples 50
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.generate import generate_step

# Add project root to path
_SCANNER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_SCANNER_ROOT))

from constrained.nko_fsm import NKoSyllableFSM, NKO_START, NKO_END
from constrained.logits_processor import NKoAdmissibilityProcessor, constrained_generate

# Try importing GK scorer (optional)
try:
    from constrained.gk_scorer import GKScoringProcessor
    GK_AVAILABLE = True
except ImportError:
    GK_AVAILABLE = False


def has_nko(text: str) -> bool:
    return any(NKO_START <= ord(ch) <= NKO_END for ch in text)


def clean_output(text: str) -> str:
    """Strip Qwen3 thinking tags and special tokens from generation output."""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>.*?$', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    return text.strip()


def generate_sample(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    constrained: bool = False,
) -> tuple:
    """Generate a single sample, optionally with FSM constraint."""
    start_time = time.time()

    if constrained:
        generated_text = constrained_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        n_tokens = len(tokenizer.encode(generated_text))
    else:
        generated_text = ""
        n_tokens = 0
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
        ):
            generated_text += response.text
            n_tokens += 1

    generated_text = clean_output(generated_text)
    elapsed = time.time() - start_time
    return generated_text, elapsed, n_tokens


def compute_ppl(model, tokenizer, text: str) -> float:
    """Compute perplexity of text under the model."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 4:
        return float("inf")

    tokens = tokens[:256]
    x = mx.array(tokens[:-1])[None]
    y = mx.array(tokens[1:])
    logits = model(x)[0]
    log_probs = nn.log_softmax(logits, axis=-1)
    token_losses = -log_probs[mx.arange(len(y)), y]
    avg_loss = mx.mean(token_losses).item()
    return math.exp(min(avg_loss, 20))


def wrap_chat_prompt(nko_text: str) -> str:
    """Wrap N'Ko text in Qwen3 chat template for better generation."""
    return (
        f"<|im_start|>system\n"
        f"You are an N'Ko language expert. Continue writing in N'Ko script.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Continue this N'Ko text: {nko_text}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_nko_prompts(eval_path: str, n: int, use_chat: bool = True) -> list:
    """Load N'Ko prompts from eval set."""
    prompts = []
    path = Path(eval_path)
    if not path.exists():
        nko_starters = [
            "ߒ ߓߍ ߕߊ",       # n bɛ taa (I am going)
            "ߊ߬ ߞߊ",          # a ka (he/she [completive])
            "ߞߊ߲ ߕߍ߫",       # kan te (language is not)
            "ߒߞߏ ߞߊ߲",       # nkɔ kan (N'Ko language)
            "ߡߊ߲߬ߘߍ߲ ߞߊ",   # manden ka (Manding [completive])
            "ߒ ߕߍ",            # n te (I am not)
            "ߊ ߓߍ",            # a bɛ (he/she is)
            "ߖߊ߬ߡߊ߬ߣߊ",     # jamaana (country)
            "ߘߎ߲ߞߎ߲ ߓߍ",    # dunkun bɛ (hope is)
            "ߟߊ߬ߡߌ߬ߣߌ",     # lamini (around)
        ]
        for i in range(n):
            raw = nko_starters[i % len(nko_starters)]
            prompts.append(wrap_chat_prompt(raw) if use_chat else raw)
        return prompts

    with open(path) as f:
        for line in f:
            try:
                ex = json.loads(line)
                msgs = ex.get("messages", [])
                if msgs:
                    content = msgs[0]["content"]
                    if has_nko(content):
                        nko_parts = []
                        for ch in content:
                            if NKO_START <= ord(ch) <= NKO_END or ch == " ":
                                nko_parts.append(ch)
                        nko_text = "".join(nko_parts).strip()
                        if len(nko_text) >= 3:
                            raw = nko_text[:50]
                            prompts.append(
                                wrap_chat_prompt(raw) if use_chat else raw
                            )
            except (json.JSONDecodeError, KeyError):
                continue

            if len(prompts) >= n:
                break

    return prompts[:n]


def main():
    parser = argparse.ArgumentParser(description="Evaluate admissibility-constrained N'Ko decoding")
    parser.add_argument("--model", type=str,
                        default=os.path.expanduser("~/nko-brain-scanner/fused-nko-qwen3-v2/"),
                        help="Path to fused model")
    parser.add_argument("--eval-data", type=str,
                        default=os.path.expanduser("~/nko-brain-scanner/eval/nko_eval.jsonl"),
                        help="Path to N'Ko eval JSONL")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to generate")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--use-gk", action="store_true",
                        help="Enable GK semantic scoring (requires GK at localhost:8001)")
    parser.add_argument("--chat", action="store_true", default=True,
                        help="Use chat template for prompts (default: True)")
    parser.add_argument("--no-chat", action="store_false", dest="chat",
                        help="Use raw N'Ko prompts without chat template")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load(args.model)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading {args.num_samples} N'Ko prompts (chat={args.chat})...")
    prompts = load_nko_prompts(args.eval_data, args.num_samples, use_chat=args.chat)
    print(f"  Loaded {len(prompts)} prompts")

    # FSM validator for measuring output quality
    fsm = NKoSyllableFSM()

    # Run both modes: unconstrained (baseline) and constrained (FSM)
    all_results = {}
    for mode in ["unconstrained", "constrained"]:
        is_constrained = mode == "constrained"
        print(f"\n{'='*60}")
        print(f"  Generating {len(prompts)} samples — {mode.upper()}")
        print(f"{'='*60}")

        results = {
            "mode": mode,
            "samples": [],
            "total_time": 0.0,
            "total_tokens": 0,
            "valid_syllable_ratios": [],
            "ppls": [],
        }

        for i, prompt in enumerate(prompts):
            text, elapsed, n_tokens = generate_sample(
                model, tokenizer, prompt, args.max_tokens,
                args.temperature, constrained=is_constrained,
            )

            ratio = fsm.valid_syllable_ratio(text)

            if has_nko(text) and len(text) > 5:
                ppl = compute_ppl(model, tokenizer, prompt + " " + text)
            else:
                ppl = float("inf")

            results["samples"].append({
                "prompt": prompt,
                "generated": text[:200],
                "tokens": n_tokens,
                "time": round(elapsed, 3),
                "valid_syllable_ratio": round(ratio, 4),
                "ppl": round(ppl, 2) if ppl != float("inf") else None,
            })
            results["total_time"] += elapsed
            results["total_tokens"] += n_tokens
            results["valid_syllable_ratios"].append(ratio)
            if ppl != float("inf"):
                results["ppls"].append(ppl)

            if (i + 1) % 10 == 0:
                avg_r = sum(results["valid_syllable_ratios"]) / len(results["valid_syllable_ratios"])
                print(f"  [{i+1}/{len(prompts)}] avg valid_syllable={avg_r:.3f}, "
                      f"tokens/s={results['total_tokens']/results['total_time']:.1f}")

        # Aggregate
        avg_ratio = sum(results["valid_syllable_ratios"]) / max(len(results["valid_syllable_ratios"]), 1)
        avg_ppl = sum(results["ppls"]) / max(len(results["ppls"]), 1) if results["ppls"] else float("inf")
        tokens_per_sec = results["total_tokens"] / max(results["total_time"], 0.001)

        high = sum(1 for r in results["valid_syllable_ratios"] if r >= 0.8)
        mid = sum(1 for r in results["valid_syllable_ratios"] if 0.4 <= r < 0.8)
        low = sum(1 for r in results["valid_syllable_ratios"] if r < 0.4)

        results["summary"] = {
            "avg_valid_syllable_ratio": round(avg_ratio, 4),
            "avg_ppl": round(avg_ppl, 2) if avg_ppl != float("inf") else None,
            "tokens_per_second": round(tokens_per_sec, 1),
            "total_samples": len(prompts),
            "high_validity": high,
            "mid_validity": mid,
            "low_validity": low,
        }
        all_results[mode] = results

    # Print comparison table
    print(f"\n{'='*80}")
    print("N'KO ADMISSIBILITY-CONSTRAINED DECODING — COMPARISON")
    print(f"{'='*80}")
    print(f"  Model: {args.model}")
    print(f"  Samples per mode: {len(prompts)}")
    print(f"\n  {'Metric':<30} {'Unconstrained':>15} {'Constrained':>15} {'Delta':>10}")
    print(f"  {'-'*70}")

    u = all_results["unconstrained"]["summary"]
    c = all_results["constrained"]["summary"]

    syl_u = u["avg_valid_syllable_ratio"]
    syl_c = c["avg_valid_syllable_ratio"]
    print(f"  {'Valid syllable ratio':<30} {syl_u:>15.4f} {syl_c:>15.4f} {syl_c-syl_u:>+10.4f}")

    ppl_u = u["avg_ppl"] or float("inf")
    ppl_c = c["avg_ppl"] or float("inf")
    if ppl_u != float("inf") and ppl_c != float("inf"):
        print(f"  {'Perplexity':<30} {ppl_u:>15.2f} {ppl_c:>15.2f} {ppl_c-ppl_u:>+10.2f}")

    tps_u = u["tokens_per_second"]
    tps_c = c["tokens_per_second"]
    print(f"  {'Tokens/sec':<30} {tps_u:>15.1f} {tps_c:>15.1f} {tps_c-tps_u:>+10.1f}")

    n = len(prompts)
    print(f"\n  {'Validity distribution':<30} {'Unconstrained':>15} {'Constrained':>15}")
    print(f"  {'-'*60}")
    print(f"  {'High (>=0.8)':<30} {u['high_validity']:>12}/{n} {c['high_validity']:>12}/{n}")
    print(f"  {'Mid (0.4-0.8)':<30} {u['mid_validity']:>12}/{n} {c['mid_validity']:>12}/{n}")
    print(f"  {'Low (<0.4)':<30} {u['low_validity']:>12}/{n} {c['low_validity']:>12}/{n}")

    # Save results
    output_path = args.output or str(_SCANNER_ROOT / "results" / "admissibility_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {}
    for mode, results in all_results.items():
        r = dict(results)
        r.pop("valid_syllable_ratios", None)
        r.pop("ppls", None)
        save_data[mode] = r

    with open(output_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
