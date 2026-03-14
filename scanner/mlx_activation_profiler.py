#!/usr/bin/env python3
"""
MLX-based activation profiler for comparing base vs fine-tuned models.

Extracts per-layer hidden states from MLX models and computes the same 4 metrics
as the original HuggingFace-based profiler: L2 norm, Shannon entropy, sparsity, kurtosis.

Usage:
  # Profile base model
  python3 scanner/mlx_activation_profiler.py \
    --model mlx-community/Qwen3-8B-8bit \
    --data training/valid.jsonl \
    --output results/mlx_base_profile.json

  # Profile fine-tuned model (with adapter)
  python3 scanner/mlx_activation_profiler.py \
    --model mlx-community/Qwen3-8B-8bit \
    --adapter adapters \
    --data training/valid.jsonl \
    --output results/mlx_finetuned_profile.json
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
    from mlx_lm.utils import generate_step
except ImportError:
    print("ERROR: mlx and mlx_lm required. pip install mlx mlx-lm")
    sys.exit(1)


def compute_metrics(hidden_state):
    """Compute L2 norm, entropy, sparsity, kurtosis from a hidden state tensor.

    Args:
        hidden_state: MLX array of shape [seq_len, hidden_dim] or [1, seq_len, hidden_dim]

    Returns:
        dict with l2_norm, entropy, sparsity, kurtosis
    """
    # Flatten to 1D for global stats
    if hidden_state.ndim == 3:
        hidden_state = hidden_state[0]  # Remove batch dim

    flat = hidden_state.reshape(-1).astype(mx.float32)

    # L2 Norm (mean across sequence positions)
    l2_per_pos = mx.sqrt(mx.sum(hidden_state.astype(mx.float32) ** 2, axis=-1))
    l2_norm = float(mx.mean(l2_per_pos))

    # Sparsity: fraction of values near zero (|x| < 0.01 * std)
    std_val = float(mx.std(flat))
    if std_val > 0:
        threshold = 0.01 * std_val
        near_zero = mx.sum(mx.abs(flat) < threshold)
        sparsity = float(near_zero) / flat.size
    else:
        sparsity = 1.0

    # Shannon Entropy on absolute value distribution (binned)
    abs_vals = mx.abs(flat)
    max_val = float(mx.max(abs_vals))
    if max_val > 0:
        # Normalize to [0, 1] and bin into 8192 bins
        num_bins = min(8192, flat.size)
        normalized = abs_vals / max_val
        bin_indices = mx.minimum(
            (normalized * num_bins).astype(mx.int32),
            mx.array(num_bins - 1)
        )
        # Count manually using scatter-add approach
        counts = mx.zeros(num_bins)
        for i in range(flat.size):
            idx = int(bin_indices[i])
            counts = counts.at[idx].add(1.0)

        # Actually, let's use a simpler histogram approach
        # Convert to numpy-like binning via sorting
        probs = counts / float(mx.sum(counts))
        # Filter non-zero for log
        mask = probs > 0
        log_probs = mx.where(mask, mx.log2(mx.maximum(probs, mx.array(1e-30))), mx.array(0.0))
        entropy = -float(mx.sum(probs * log_probs))
    else:
        entropy = 0.0

    # Kurtosis (excess kurtosis)
    mean_val = float(mx.mean(flat))
    if std_val > 0:
        centered = flat - mean_val
        m4 = float(mx.mean(centered ** 4))
        kurtosis = m4 / (std_val ** 4) - 3.0
    else:
        kurtosis = 0.0

    return {
        "l2_norm": l2_norm,
        "entropy": entropy,
        "sparsity": sparsity,
        "kurtosis": kurtosis,
        "mean": mean_val,
        "std": std_val,
    }


def extract_hidden_states(model, tokenizer, text, max_tokens=128):
    """Extract hidden states from all layers for a given text input.

    Returns list of dicts with per-layer metrics.
    """
    # Tokenize
    inputs = tokenizer.encode(text)
    if len(inputs) > max_tokens:
        inputs = inputs[:max_tokens]

    input_ids = mx.array([inputs])

    # We need to hook into the model to extract hidden states.
    # MLX models don't have output_hidden_states by default,
    # so we monkey-patch the forward pass.

    hidden_states = []

    # Get the transformer block list
    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model

    # Get embedding
    if hasattr(inner, 'embed_tokens'):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, 'wte'):
        h = inner.wte(input_ids)
    else:
        raise ValueError("Cannot find embedding layer")

    hidden_states.append(h)

    # Get layers
    if hasattr(inner, 'layers'):
        layers = inner.layers
    elif hasattr(inner, 'h'):
        layers = inner.h
    else:
        raise ValueError("Cannot find transformer layers")

    # Build mask and cache if needed
    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    # Forward through each layer
    for i, layer in enumerate(layers):
        try:
            h = layer(h, mask=mask)
            if isinstance(h, tuple):
                h = h[0]  # Some layers return (hidden, cache)
        except TypeError:
            # Try without mask
            try:
                h = layer(h)
                if isinstance(h, tuple):
                    h = h[0]
            except Exception as e:
                print(f"  Layer {i} failed: {e}")
                break
        hidden_states.append(h)
        mx.eval(h)  # Force evaluation to free memory

    # Compute metrics for each layer
    results = []
    for layer_idx, hs in enumerate(hidden_states):
        mx.eval(hs)
        metrics = compute_metrics(hs)
        metrics["layer"] = layer_idx
        results.append(metrics)

    return results


def profile_dataset(model, tokenizer, data_path, num_samples=50, max_tokens=128):
    """Profile activation patterns across a dataset.

    Returns per-layer averaged metrics.
    """
    # Load data
    examples = []
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            msgs = entry.get("messages", [])
            if len(msgs) >= 1:
                text = msgs[0]["content"]  # User message
                # Check if it contains N'Ko
                has_nko = any(0x07C0 <= ord(ch) <= 0x07FF for ch in text)
                examples.append({"text": text, "has_nko": has_nko})

    # Split into NKO and English examples
    nko_examples = [e for e in examples if e["has_nko"]]
    en_examples = [e for e in examples if not e["has_nko"]]

    print(f"Dataset: {len(examples)} total, {len(nko_examples)} N'Ko, {len(en_examples)} English")

    # Sample
    import random
    random.seed(42)
    nko_sample = random.sample(nko_examples, min(num_samples, len(nko_examples)))
    en_sample = random.sample(en_examples, min(num_samples, len(en_examples)))

    results = {}

    for label, sample in [("nko", nko_sample), ("english", en_sample)]:
        print(f"\nProfiling {label} ({len(sample)} examples)...")
        all_layer_metrics = {}

        for idx, example in enumerate(sample):
            if idx % 10 == 0:
                print(f"  {idx}/{len(sample)}...")

            try:
                layer_metrics = extract_hidden_states(
                    model, tokenizer, example["text"], max_tokens=max_tokens
                )

                for lm in layer_metrics:
                    layer = lm["layer"]
                    if layer not in all_layer_metrics:
                        all_layer_metrics[layer] = {
                            "l2_norm": [], "entropy": [], "sparsity": [],
                            "kurtosis": [], "mean": [], "std": []
                        }
                    for key in ["l2_norm", "entropy", "sparsity", "kurtosis", "mean", "std"]:
                        all_layer_metrics[layer][key].append(lm[key])
            except Exception as e:
                print(f"  Example {idx} failed: {e}")
                continue

        # Average
        averaged = []
        for layer in sorted(all_layer_metrics.keys()):
            metrics = all_layer_metrics[layer]
            avg = {"layer": layer}
            for key in ["l2_norm", "entropy", "sparsity", "kurtosis", "mean", "std"]:
                vals = metrics[key]
                avg[key] = sum(vals) / len(vals) if vals else 0.0
            averaged.append(avg)

        results[label] = averaged
        print(f"  {label}: {len(averaged)} layers profiled")

    return results


def main():
    parser = argparse.ArgumentParser(description="MLX Activation Profiler")
    parser.add_argument("--model", required=True, help="Model path or HF repo")
    parser.add_argument("--adapter", default=None, help="Adapter path for fine-tuned model")
    parser.add_argument("--data", required=True, help="JSONL data file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per language")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max input tokens")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.adapter:
        print(f"With adapter: {args.adapter}")
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter,
            tokenizer_config={"trust_remote_code": True}
        )
    else:
        model, tokenizer = load(
            args.model,
            tokenizer_config={"trust_remote_code": True}
        )

    # Print model structure for debugging
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        print(f"Model has {num_layers} layers")

    results = profile_dataset(
        model, tokenizer, args.data,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    for label in ["english", "nko"]:
        if label in results:
            data = results[label]
            if data:
                mid = len(data) // 2
                print(f"\n{label.upper()} summary:")
                print(f"  Layer 0: L2={data[0]['l2_norm']:.1f}, entropy={data[0]['entropy']:.2f}, sparsity={data[0]['sparsity']:.4f}")
                print(f"  Layer {mid}: L2={data[mid]['l2_norm']:.1f}, entropy={data[mid]['entropy']:.2f}, kurtosis={data[mid]['kurtosis']:.1f}")
                print(f"  Layer {len(data)-1}: L2={data[-1]['l2_norm']:.1f}, entropy={data[-1]['entropy']:.2f}")


if __name__ == "__main__":
    main()
