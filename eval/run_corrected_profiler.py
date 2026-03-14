#!/usr/bin/env python3
"""Corrected brain scan profiler using frozen 100+100 eval sets.

Compares base, two-stage, and three-stage models on identical eval data.
This replaces all previous profiler runs that had inconsistent eval sets.
"""

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

MODEL_ID = "mlx-community/Qwen3-8B-8bit"
ADAPTER_TWOSTAGE = os.path.expanduser("~/nko-brain-scanner/adapters-combined")
ADAPTER_BPE = os.path.expanduser("~/nko-brain-scanner/adapters-bpe")
ENG_EVAL = os.path.expanduser("~/nko-brain-scanner/eval/english_eval.jsonl")
NKO_EVAL = os.path.expanduser("~/nko-brain-scanner/eval/nko_eval.jsonl")
OUTPUT_PATH = os.path.expanduser("~/nko-brain-scanner/results/profiler_corrected.json")
MAX_SEQ_LEN = 256
NKO_RANGE = range(0x07C0, 0x0800)


def load_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except:
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


def run_model(name, adapter_path, eng_examples, nko_examples):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if adapter_path:
        model, tokenizer = load(MODEL_ID, adapter_path=adapter_path)
    else:
        model, tokenizer = load(MODEL_ID)

    print("Evaluating on English...")
    eng_result = evaluate(model, tokenizer, eng_examples, f"{name}_eng")
    print(f"  Loss={eng_result['avg_loss']}, PPL={eng_result['perplexity']}, "
          f"Top1={eng_result['top1_accuracy']}")

    print("Evaluating on N'Ko...")
    nko_result = evaluate(model, tokenizer, nko_examples, f"{name}_nko")
    print(f"  Loss={nko_result['avg_loss']}, PPL={nko_result['perplexity']}, "
          f"Top1={nko_result['top1_accuracy']}, NKo={nko_result['nko_token_accuracy']}")

    del model
    mx.clear_cache()
    return {"english": eng_result, "nko": nko_result}


def main():
    print("Loading frozen eval sets...")
    eng_examples = load_examples(ENG_EVAL)
    nko_examples = load_examples(NKO_EVAL)
    print(f"  English: {len(eng_examples)} examples")
    print(f"  N'Ko: {len(nko_examples)} examples")

    # Run all 3 configurations
    base = run_model("Base (Qwen3-8B)", None, eng_examples, nko_examples)
    twostage = run_model("Two-Stage (CPT+SFT)", ADAPTER_TWOSTAGE, eng_examples, nko_examples)
    threestage = run_model("Three-Stage (CPT+SFT+BPE)", ADAPTER_BPE, eng_examples, nko_examples)

    # Summary table
    print("\n" + "=" * 80)
    print("CORRECTED COMPARISON (100 English + 100 N'Ko examples)")
    print("=" * 80)
    print(f"{'Metric':<28} {'Base':>10} {'2-Stage':>10} {'3-Stage':>10}")
    print("-" * 80)

    for metric in ["avg_loss", "perplexity", "top1_accuracy", "nko_token_accuracy"]:
        for lang in ["english", "nko"]:
            label = f"{lang.title()} {metric.replace('_', ' ').title()}"
            b = base[lang][metric]
            t = twostage[lang][metric]
            s = threestage[lang][metric]
            if isinstance(b, float):
                print(f"{label:<28} {b:>10.4f} {t:>10.4f} {s:>10.4f}")
            else:
                print(f"{label:<28} {b:>10} {t:>10} {s:>10}")

    # Translation tax
    base_tax = base["nko"]["perplexity"] / max(base["english"]["perplexity"], 0.01)
    ft_tax = threestage["nko"]["perplexity"] / max(threestage["english"]["perplexity"], 0.01)
    print(f"\n{'Translation Tax (NKo/Eng PPL)':<28} {base_tax:>10.2f} {'':>10} {ft_tax:>10.2f}")

    results = {
        "base": base,
        "two_stage": twostage,
        "three_stage_bpe": threestage,
        "translation_tax": {
            "base": round(base_tax, 4),
            "three_stage": round(ft_tax, 4),
            "reduction": round((base_tax - ft_tax) / base_tax * 100, 1),
        },
        "metadata": {
            "model": MODEL_ID,
            "english_examples": len(eng_examples),
            "nko_examples": len(nko_examples),
            "max_seq_len": MAX_SEQ_LEN,
            "eval_files": {
                "english": ENG_EVAL,
                "nko": NKO_EVAL,
            },
            "adapters": {
                "two_stage": ADAPTER_TWOSTAGE,
                "three_stage": ADAPTER_BPE,
            },
            "timestamp": datetime.now().isoformat(),
            "note": "Corrected eval with 100 examples per language (previous runs had 4 English examples)",
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
