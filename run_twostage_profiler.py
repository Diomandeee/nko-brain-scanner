#!/usr/bin/env python3
"""Brain scan profiler for two-stage (CPT+SFT) model."""

import json
import math
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL_ID = "mlx-community/Qwen3-8B-8bit"
ADAPTER_PATH = os.path.expanduser("~/nko-brain-scanner/adapters-combined")
VALID_PATH = os.path.expanduser("~/nko-brain-scanner/training/valid.jsonl")
OUTPUT_PATH = os.path.expanduser("~/nko-brain-scanner/results/profiler_twostage.json")
MAX_EXAMPLES = 30
MAX_SEQ_LEN = 256
NKO_RANGE = range(0x07C0, 0x0800)


def evaluate(model, tokenizer, examples, label):
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_nko_correct = 0
    total_nko_tokens = 0

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
    nko_acc = total_nko_correct / max(total_nko_tokens, 1)
    return {
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "top1_accuracy": round(acc, 4),
        "nko_token_accuracy": round(nko_acc, 4),
        "nko_tokens_evaluated": total_nko_tokens,
        "total_tokens": total_tokens,
        "num_examples": len(examples),
    }


def main():
    # Load examples
    nko_examples = []
    eng_examples = []
    with open(VALID_PATH) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            if len(msgs) < 2:
                continue
            text = msgs[0]["content"] + msgs[1]["content"]
            has_nko = any(ord(c) in NKO_RANGE for c in text)
            if has_nko and len(nko_examples) < MAX_EXAMPLES:
                nko_examples.append(ex)
            elif not has_nko and len(eng_examples) < MAX_EXAMPLES:
                eng_examples.append(ex)
            if len(nko_examples) >= MAX_EXAMPLES and len(eng_examples) >= MAX_EXAMPLES:
                break

    print(f"Loaded {len(nko_examples)} NKo, {len(eng_examples)} English examples")

    # Base model
    print("Loading base model...")
    model, tokenizer = load(MODEL_ID)
    print("Evaluating base on NKo...")
    base_nko = evaluate(model, tokenizer, nko_examples, "base_nko")
    print(f"  Loss={base_nko['avg_loss']}, PPL={base_nko['perplexity']}, Acc={base_nko['top1_accuracy']}")
    print("Evaluating base on English...")
    base_eng = evaluate(model, tokenizer, eng_examples, "base_eng")
    print(f"  Loss={base_eng['avg_loss']}, PPL={base_eng['perplexity']}, Acc={base_eng['top1_accuracy']}")

    del model
    mx.metal.clear_cache()

    # Two-stage model
    print("Loading two-stage (CPT+SFT) model...")
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)
    print("Evaluating two-stage on NKo...")
    ft_nko = evaluate(model, tokenizer, nko_examples, "ft_nko")
    print(f"  Loss={ft_nko['avg_loss']}, PPL={ft_nko['perplexity']}, Acc={ft_nko['top1_accuracy']}")
    print("Evaluating two-stage on English...")
    ft_eng = evaluate(model, tokenizer, eng_examples, "ft_eng")
    print(f"  Loss={ft_eng['avg_loss']}, PPL={ft_eng['perplexity']}, Acc={ft_eng['top1_accuracy']}")

    results = {
        "base": {"nko": base_nko, "english": base_eng},
        "finetuned": {"nko": ft_nko, "english": ft_eng},
        "deltas": {
            "nko": {
                "avg_loss_delta": round(ft_nko["avg_loss"] - base_nko["avg_loss"], 4),
                "perplexity_delta": round(ft_nko["perplexity"] - base_nko["perplexity"], 2),
                "top1_accuracy_delta": round(ft_nko["top1_accuracy"] - base_nko["top1_accuracy"], 4),
                "nko_token_accuracy_delta": round(ft_nko["nko_token_accuracy"] - base_nko["nko_token_accuracy"], 4),
            },
            "english": {
                "avg_loss_delta": round(ft_eng["avg_loss"] - base_eng["avg_loss"], 4),
                "perplexity_delta": round(ft_eng["perplexity"] - base_eng["perplexity"], 2),
                "top1_accuracy_delta": round(ft_eng["top1_accuracy"] - base_eng["top1_accuracy"], 4),
                "nko_token_accuracy_delta": round(ft_eng["nko_token_accuracy"] - base_eng["nko_token_accuracy"], 4),
            },
        },
        "metadata": {
            "model": MODEL_ID,
            "adapter_path": ADAPTER_PATH,
            "training": "two-stage: CPT (2000 iters on Wikipedia) + SFT (1000 iters on combined)",
            "max_seq_len": MAX_SEQ_LEN,
            "nko_examples": len(nko_examples),
            "english_examples": len(eng_examples),
            "timestamp": datetime.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
