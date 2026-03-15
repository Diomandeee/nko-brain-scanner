#!/usr/bin/env python3
"""V3 profiler: evaluate fused-v3-nko-qwen3 on frozen 100+100 eval sets.

Loads the V3 fused model (extended vocab + V3 adapter) and evaluates
PPL, accuracy, translation tax. Compares against stored V1/V2/V3 results.

Usage:
    python3 eval/run_v3_profiler.py
    python3 eval/run_v3_profiler.py --model ~/nko-brain-scanner/fused-v3-nko-qwen3
"""

import argparse
import json
import math
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

ENG_EVAL = os.path.expanduser("~/nko-brain-scanner/eval/english_eval.jsonl")
NKO_EVAL = os.path.expanduser("~/nko-brain-scanner/eval/nko_eval.jsonl")
PREV_RESULTS = os.path.expanduser("~/nko-brain-scanner/results/profiler_corrected.json")
OUTPUT_PATH = os.path.expanduser("~/nko-brain-scanner/results/profiler_v3.json")
MAX_SEQ_LEN = 384  # V3 trained with 384
NKO_RANGE = range(0x07C0, 0x0800)


def load_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except Exception:
                continue
    return examples


def evaluate(model, tokenizer, examples, label):
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_nko_correct = 0
    total_nko_tokens = 0
    valid_examples = 0

    for i, ex in enumerate(examples):
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            continue
        text = msgs[0]["content"] + " " + msgs[1]["content"]
        tokens = tokenizer.encode(text)
        if len(tokens) < 4:
            continue
        tokens = tokens[:MAX_SEQ_LEN]
        x = mx.array(tokens[:-1])[None]
        y = mx.array(tokens[1:])
        logits = model(x)
        logits = logits[0]
        log_probs = nn.log_softmax(logits, axis=-1)
        token_losses = -log_probs[mx.arange(len(y)), y]
        total_loss += mx.sum(token_losses).item()
        total_tokens += len(y)
        preds = mx.argmax(logits, axis=-1)
        correct = mx.sum(preds == y).item()
        total_correct += correct
        valid_examples += 1

        y_list = y.tolist()
        pred_list = preds.tolist()
        for j, tid in enumerate(tokens[1:]):
            decoded = tokenizer.decode([tid])
            if any(ord(c) in NKO_RANGE for c in decoded):
                total_nko_tokens += 1
                if pred_list[j] == y_list[j]:
                    total_nko_correct += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    acc = total_correct / max(total_tokens, 1)
    nko_acc = total_nko_correct / max(total_nko_tokens, 1) if total_nko_tokens > 0 else 0.0

    print(f"  [{label}] {valid_examples} examples, {total_tokens} tokens, "
          f"{total_nko_tokens} N'Ko tokens")

    return {
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "top1_accuracy": round(acc, 4),
        "nko_token_accuracy": round(nko_acc, 4),
        "nko_tokens_evaluated": total_nko_tokens,
        "total_tokens": total_tokens,
        "valid_examples": valid_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser(
        "~/nko-brain-scanner/fused-v3-nko-qwen3"))
    args = parser.parse_args()

    print(f"V3 Model: {args.model}")
    print("Loading frozen eval sets...")
    eng_examples = load_examples(ENG_EVAL)
    nko_examples = load_examples(NKO_EVAL)
    print(f"  English: {len(eng_examples)} examples")
    print(f"  N'Ko: {len(nko_examples)} examples")

    # Load V3 fused model (no adapter_path — it's already fused)
    print(f"\nLoading V3 fused model from {args.model}...")
    model, tokenizer = load(args.model)

    print("\nEvaluating on English...")
    eng_result = evaluate(model, tokenizer, eng_examples, "v3_eng")
    print(f"  Loss={eng_result['avg_loss']}, PPL={eng_result['perplexity']}, "
          f"Top1={eng_result['top1_accuracy']}")

    print("\nEvaluating on N'Ko...")
    nko_result = evaluate(model, tokenizer, nko_examples, "v3_nko")
    print(f"  Loss={nko_result['avg_loss']}, PPL={nko_result['perplexity']}, "
          f"Top1={nko_result['top1_accuracy']}, NKo={nko_result['nko_token_accuracy']}")

    v3_tax = nko_result["perplexity"] / max(eng_result["perplexity"], 0.01)

    # Load previous results for comparison
    prev = {}
    if os.path.exists(PREV_RESULTS):
        with open(PREV_RESULTS) as f:
            prev = json.load(f)

    # Print comparison
    print("\n" + "=" * 90)
    print("V3 COMPARISON (100 English + 100 N'Ko examples)")
    print("=" * 90)

    headers = ["Metric", "Base", "2-Stage", "3-Stage", "V3"]
    print(f"{'Metric':<28} {'Base':>10} {'2-Stage':>10} {'3-Stage':>10} {'V3':>10}")
    print("-" * 90)

    for metric in ["perplexity", "top1_accuracy", "nko_token_accuracy"]:
        for lang in ["english", "nko"]:
            label = f"{lang.title()} {metric.replace('_', ' ').title()}"
            vals = []
            for key in ["base", "two_stage", "three_stage_bpe"]:
                if key in prev and lang in prev[key]:
                    vals.append(prev[key][lang].get(metric, "N/A"))
                else:
                    vals.append("N/A")
            v3_val = eng_result[metric] if lang == "english" else nko_result[metric]
            if isinstance(v3_val, float):
                row = f"{label:<28}"
                for v in vals:
                    row += f" {v:>10.4f}" if isinstance(v, float) else f" {str(v):>10}"
                row += f" {v3_val:>10.4f}"
                print(row)
            else:
                print(f"{label:<28} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {v3_val:>10}")

    # Translation tax
    base_tax = prev.get("translation_tax", {}).get("base", "N/A")
    ft_tax = prev.get("translation_tax", {}).get("three_stage", "N/A")
    print(f"\n{'Translation Tax':<28} {base_tax:>10} {'':>10} {ft_tax:>10} {v3_tax:>10.2f}")

    results = {
        "v3": {"english": eng_result, "nko": nko_result},
        "translation_tax_v3": round(v3_tax, 4),
        "metadata": {
            "model_path": args.model,
            "english_examples": len(eng_examples),
            "nko_examples": len(nko_examples),
            "max_seq_len": MAX_SEQ_LEN,
            "timestamp": datetime.now().isoformat(),
            "training_config": {
                "examples": 92184,
                "iterations": 3000,
                "batch_size": 1,
                "grad_accum": 8,
                "lr": 1e-5,
                "lora_layers": 8,
                "lora_rank": 8,
                "max_seq_len": 384,
            },
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
