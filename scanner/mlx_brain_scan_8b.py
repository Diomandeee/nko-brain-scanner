#!/usr/bin/env python3
"""8B Brain Scan: Before/After Fine-Tuning Activation Profiling.

Uses wrapper modules to capture hidden states during the model's own
forward pass. This avoids mask dtype issues by letting the model handle
its own attention mechanism.

Metrics per layer:
  - L2 norm (mean activation magnitude per token)
  - Sparsity (fraction of near-zero activations)
  - Max activation (peak magnitude)
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
ADAPTER_BPE = os.path.expanduser("~/nko-brain-scanner/adapters-bpe")
NKO_EVAL = os.path.expanduser("~/nko-brain-scanner/eval/nko_eval.jsonl")
ENG_EVAL = os.path.expanduser("~/nko-brain-scanner/eval/english_eval.jsonl")
OUTPUT_PATH = os.path.expanduser("~/nko-brain-scanner/results/brain_scan_8b.json")
MAX_SEQ_LEN = 128
NUM_EXAMPLES = 30

# Global capture list
_captured = []


class LayerCapture(nn.Module):
    """Wrapper that captures layer output stats during forward pass."""

    def __init__(self, layer, idx):
        super().__init__()
        self._inner = layer
        self._idx = idx

    def __call__(self, *args, **kwargs):
        result = self._inner(*args, **kwargs)
        # Extract hidden state from result
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result
        # Compute and store stats
        stats = _compute_stats(h)
        _captured.append((self._idx, stats))
        return result


def _compute_stats(h):
    """Compute activation stats from hidden state tensor."""
    # h: (batch, seq, hidden) or (seq, hidden)
    if h.ndim == 3:
        h = h[0]

    l2_per_token = mx.sqrt(mx.sum(h * h, axis=-1))
    l2_mean = mx.mean(l2_per_token).item()

    h_flat = h.reshape(-1)
    near_zero = mx.sum(mx.abs(h_flat) < 0.01).item()
    sparsity = near_zero / max(h_flat.size, 1)

    max_act = mx.max(mx.abs(h_flat)).item()
    mean_abs = mx.mean(mx.abs(h_flat)).item()

    return {
        "l2_norm": round(l2_mean, 4),
        "sparsity": round(sparsity, 6),
        "max_act": round(max_act, 4),
        "mean_abs": round(mean_abs, 4),
    }


def load_examples(path, n=NUM_EXAMPLES):
    examples = []
    with open(path) as f:
        for line in f:
            try:
                ex = json.loads(line)
                if len(ex.get("messages", [])) >= 2:
                    examples.append(ex)
            except:
                continue
            if len(examples) >= n:
                break
    return examples


def wrap_layers(model):
    """Replace each transformer layer with a capturing wrapper."""
    qwen = model.model if hasattr(model, 'model') else model
    wrapped = []
    for i, layer in enumerate(qwen.layers):
        wrapped.append(LayerCapture(layer, i))
    qwen.layers = wrapped
    return len(wrapped)


def run_forward(model, tokenizer, text):
    """Run normal forward pass, return captured layer stats."""
    global _captured
    _captured = []

    tokens = tokenizer.encode(text)[:MAX_SEQ_LEN]
    if len(tokens) < 4:
        return None

    x = mx.array(tokens[:-1])[None]
    logits = model(x)
    mx.eval(logits)

    result = {}
    for idx, stats in _captured:
        result[idx] = stats
    return result


def profile(model, tokenizer, examples, label):
    """Profile model on examples."""
    print(f"\n  Profiling {label} ({len(examples)} examples)...")

    accum = {}
    valid = 0

    for i, ex in enumerate(examples):
        msgs = ex["messages"]
        text = msgs[0]["content"] + " " + msgs[1]["content"]

        try:
            stats = run_forward(model, tokenizer, text)
        except Exception as e:
            print(f"    Skip {i}: {e}")
            continue

        if stats is None:
            continue

        valid += 1
        for layer_idx, s in stats.items():
            if layer_idx not in accum:
                accum[layer_idx] = []
            accum[layer_idx].append(s)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(examples)}")

    # Average across examples
    averaged = {}
    for idx in sorted(accum.keys()):
        entries = accum[idx]
        n = len(entries)
        averaged[str(idx)] = {
            "l2_norm": round(sum(e["l2_norm"] for e in entries) / n, 4),
            "sparsity": round(sum(e["sparsity"] for e in entries) / n, 6),
            "max_act": round(sum(e["max_act"] for e in entries) / n, 4),
            "mean_abs": round(sum(e["mean_abs"] for e in entries) / n, 4),
        }

    print(f"    {valid} valid, {len(averaged)} layers captured")
    return {"layer_stats": averaged, "valid_examples": valid}


def scan(name, adapter_path, eng, nko):
    """Full brain scan for one model configuration."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if adapter_path:
        model, tokenizer = load(MODEL_ID, adapter_path=adapter_path)
    else:
        model, tokenizer = load(MODEL_ID)

    num_layers = wrap_layers(model)
    print(f"  Wrapped {num_layers} layers for capture")

    eng_profile = profile(model, tokenizer, eng, f"{name} Eng")
    nko_profile = profile(model, tokenizer, nko, f"{name} NKo")

    del model
    mx.clear_cache()

    return {"english": eng_profile, "nko": nko_profile}


def main():
    print("8B Brain Scan: Activation Profiling")
    print("=" * 60)

    eng = load_examples(ENG_EVAL)
    nko = load_examples(NKO_EVAL)
    print(f"  English: {len(eng)}, N'Ko: {len(nko)}")

    base = scan("Base (Qwen3-8B)", None, eng, nko)
    ft = scan("Fine-Tuned (3-Stage)", ADAPTER_BPE, eng, nko)

    # Deltas for N'Ko
    nl = max(len(base["nko"]["layer_stats"]), len(ft["nko"]["layer_stats"]))
    deltas = {}
    for i in range(nl):
        k = str(i)
        b = base["nko"]["layer_stats"].get(k)
        f = ft["nko"]["layer_stats"].get(k)
        if b and f:
            deltas[k] = {
                "l2_delta": round(f["l2_norm"] - b["l2_norm"], 4),
                "sparsity_delta": round(f["sparsity"] - b["sparsity"], 6),
                "max_delta": round(f["max_act"] - b["max_act"], 4),
                "mean_delta": round(f["mean_abs"] - b["mean_abs"], 4),
            }

    results = {
        "base": base,
        "fine_tuned": ft,
        "nko_deltas": deltas,
        "metadata": {
            "model": MODEL_ID,
            "adapter": ADAPTER_BPE,
            "examples_per_lang": NUM_EXAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "num_layers": nl,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Print summary
    print("\n" + "=" * 90)
    print("N'Ko Activations: Base vs Fine-Tuned")
    print("=" * 90)
    print(f"{'Lyr':<5} {'Base L2':>9} {'FT L2':>9} {'ΔL2':>9} "
          f"{'Base Sp':>9} {'FT Sp':>9} {'Base MaxA':>9} {'FT MaxA':>9}")
    print("-" * 90)
    for i in range(nl):
        k = str(i)
        b = base["nko"]["layer_stats"].get(k, {})
        f = ft["nko"]["layer_stats"].get(k, {})
        d = deltas.get(k, {})
        print(f"{i:<5} {b.get('l2_norm',0):>9.2f} {f.get('l2_norm',0):>9.2f} "
              f"{d.get('l2_delta',0):>+9.2f} "
              f"{b.get('sparsity',0):>9.4f} {f.get('sparsity',0):>9.4f} "
              f"{b.get('max_act',0):>9.2f} {f.get('max_act',0):>9.2f}")

    if deltas:
        print("\n  Top 5 layers by |ΔL2|:")
        top = sorted(deltas.items(), key=lambda x: abs(x[1]["l2_delta"]), reverse=True)[:5]
        for k, d in top:
            print(f"    Layer {k}: ΔL2={d['l2_delta']:+.4f}, ΔSpar={d['sparsity_delta']:+.6f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
