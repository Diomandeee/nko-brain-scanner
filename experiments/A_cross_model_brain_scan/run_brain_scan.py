#!/usr/bin/env python3
"""
Cross-Model Brain Scan: Activation profiling across multiple architectures.

Runs the N'Ko activation profiling pipeline on any HuggingFace model.
Adapted from scanner/activation_profiler.py and scanner/mlx_activation_profiler.py.

Usage:
    # Local MLX model (Apple Silicon)
    python3 run_brain_scan.py \
        --model mlx-community/Qwen3-8B-8bit \
        --backend mlx \
        --output results/qwen3_8b.json

    # HuggingFace model with BnB 4-bit (GPU)
    python3 run_brain_scan.py \
        --model Qwen/Qwen2-72B-Instruct \
        --backend hf \
        --quantize 4bit \
        --output results/qwen2_72b.json

    # HuggingFace model FP16
    python3 run_brain_scan.py \
        --model meta-llama/Llama-3.1-8B \
        --backend hf \
        --output results/llama_8b.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
EVAL_PAIRS_PATH = Path(__file__).parent / "data" / "eval_pairs.jsonl"


def load_eval_pairs(path=None, limit=100):
    """Load parallel English/N'Ko evaluation pairs."""
    path = path or EVAL_PAIRS_PATH
    pairs = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("nko") and entry.get("english"):
                pairs.append(entry)
            if len(pairs) >= limit:
                break
    print(f"Loaded {len(pairs)} parallel pairs from {path}")
    return pairs


# ---------------------------------------------------------------
# HuggingFace backend (GPU, BnB quantization)
# ---------------------------------------------------------------

def scan_hf(model_name, pairs, quantize="4bit"):
    """Run brain scan using HuggingFace transformers + torch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} via HuggingFace (quantize={quantize})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {param_count:.1f}B params")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    device = next(model.parameters()).device

    def profile_text(text):
        """Extract per-layer metrics for a single text input."""
        inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        layer_metrics = []
        for layer_idx, hs in enumerate(outputs.hidden_states):
            h = hs[0].float().cpu().numpy()  # [seq_len, hidden_dim]
            h_flat = h.mean(axis=0)  # [hidden_dim]

            metrics = {"layer": layer_idx}
            metrics["l2_norm"] = float(np.linalg.norm(h_flat))
            metrics["mean"] = float(np.mean(h_flat))
            metrics["std"] = float(np.std(h_flat))

            # Shannon entropy
            abs_h = np.abs(h_flat)
            abs_sum = abs_h.sum()
            if abs_sum > 0:
                p = abs_h / abs_sum
                p = p[p > 0]
                metrics["entropy"] = float(-np.sum(p * np.log2(p)))
            else:
                metrics["entropy"] = 0.0

            # Sparsity
            metrics["sparsity"] = float(np.mean(np.abs(h_flat) < 1e-3))

            # Kurtosis
            if metrics["std"] > 0:
                centered = (h_flat - metrics["mean"]) / metrics["std"]
                metrics["kurtosis"] = float(np.mean(centered ** 4) - 3.0)
            else:
                metrics["kurtosis"] = 0.0

            layer_metrics.append(metrics)
        return layer_metrics

    return _run_profiles(pairs, profile_text)


# ---------------------------------------------------------------
# MLX backend (Apple Silicon)
# ---------------------------------------------------------------

def scan_mlx(model_name, pairs, adapter_path=None):
    """Run brain scan using MLX (Apple Silicon)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    print(f"Loading {model_name} via MLX...")
    if adapter_path:
        model, tokenizer = load(model_name, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_name)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        print(f"Model has {num_layers} layers")

    def profile_text(text):
        """Extract per-layer metrics via manual layer-by-layer forward pass."""
        inputs = tokenizer.encode(text)
        if len(inputs) > 256:
            inputs = inputs[:256]
        input_ids = mx.array([inputs])

        # Get inner model
        inner = model.model if hasattr(model, "model") else model

        # Embedding
        if hasattr(inner, "embed_tokens"):
            h = inner.embed_tokens(input_ids)
        elif hasattr(inner, "wte"):
            h = inner.wte(input_ids)
        else:
            raise ValueError("Cannot find embedding layer")

        hidden_states = [h]

        # Layers
        layers = inner.layers if hasattr(inner, "layers") else inner.h
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for i, layer in enumerate(layers):
            try:
                h = layer(h, mask=mask)
                if isinstance(h, tuple):
                    h = h[0]
            except TypeError:
                try:
                    h = layer(h)
                    if isinstance(h, tuple):
                        h = h[0]
                except Exception as e:
                    print(f"  Layer {i} failed: {e}")
                    break
            hidden_states.append(h)
            mx.eval(h)

        layer_metrics = []
        for layer_idx, hs in enumerate(hidden_states):
            mx.eval(hs)
            if hs.ndim == 3:
                hs_2d = hs[0]
            else:
                hs_2d = hs

            flat = hs_2d.reshape(-1).astype(mx.float32)

            # L2 norm
            l2_per_pos = mx.sqrt(mx.sum(hs_2d.astype(mx.float32) ** 2, axis=-1))
            l2_norm = float(mx.mean(l2_per_pos))

            # Mean / std
            mean_val = float(mx.mean(flat))
            std_val = float(mx.std(flat))

            # Sparsity
            if std_val > 0:
                threshold = 0.01 * std_val
                near_zero = mx.sum(mx.abs(flat) < threshold)
                sparsity = float(near_zero) / flat.size
            else:
                sparsity = 1.0

            # Entropy (simplified binned)
            abs_vals = mx.abs(flat)
            max_val = float(mx.max(abs_vals))
            if max_val > 0 and flat.size > 0:
                # Use numpy for histogram (MLX histogram is cumbersome)
                flat_np = np.array(flat.tolist())
                counts, _ = np.histogram(np.abs(flat_np), bins=min(1024, flat.size))
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log2(probs)))
            else:
                entropy = 0.0

            # Kurtosis
            if std_val > 0:
                centered_np = (np.array(flat.tolist()) - mean_val) / std_val
                kurtosis = float(np.mean(centered_np ** 4) - 3.0)
            else:
                kurtosis = 0.0

            layer_metrics.append({
                "layer": layer_idx,
                "l2_norm": l2_norm,
                "entropy": entropy,
                "sparsity": sparsity,
                "kurtosis": kurtosis,
                "mean": mean_val,
                "std": std_val,
            })

        return layer_metrics

    return _run_profiles(pairs, profile_text)


# ---------------------------------------------------------------
# Shared profiling loop
# ---------------------------------------------------------------

def _run_profiles(pairs, profile_fn):
    """Profile English and N'Ko inputs, average across pairs."""
    results = {}

    for label, key in [("english", "english"), ("nko", "nko")]:
        print(f"\nProfiling {label} ({len(pairs)} inputs)...")
        all_layer_metrics = {}

        for idx, pair in enumerate(pairs):
            text = pair[key]
            if not text or not text.strip():
                continue

            if (idx + 1) % 20 == 0:
                print(f"  [{idx + 1}/{len(pairs)}]")

            try:
                layer_metrics = profile_fn(text)
                for lm in layer_metrics:
                    layer = lm["layer"]
                    if layer not in all_layer_metrics:
                        all_layer_metrics[layer] = {
                            "l2_norm": [], "entropy": [], "sparsity": [],
                            "kurtosis": [], "mean": [], "std": [],
                        }
                    for k in ["l2_norm", "entropy", "sparsity", "kurtosis", "mean", "std"]:
                        all_layer_metrics[layer][k].append(lm[k])
            except Exception as e:
                print(f"  Input {idx} failed: {e}")
                continue

        # Average across all inputs
        averaged = []
        for layer in sorted(all_layer_metrics.keys()):
            metrics = all_layer_metrics[layer]
            avg = {"layer": layer}
            for k in ["l2_norm", "entropy", "sparsity", "kurtosis", "mean", "std"]:
                vals = metrics[k]
                avg[k] = sum(vals) / len(vals) if vals else 0.0
            averaged.append(avg)

        results[label] = averaged
        print(f"  {label}: {len(averaged)} layers profiled")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Model Brain Scan: activation profiling for N'Ko vs English"
    )
    parser.add_argument("--model", required=True, help="Model name or HuggingFace repo ID")
    parser.add_argument("--backend", choices=["hf", "mlx"], default="mlx",
                        help="Backend: 'hf' for HuggingFace/torch, 'mlx' for Apple Silicon")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "fp16"], default="8bit",
                        help="Quantization level (hf backend only)")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (mlx backend only)")
    parser.add_argument("--pairs", default=str(EVAL_PAIRS_PATH),
                        help="Path to evaluation pairs JSONL")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max number of parallel pairs to profile")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    pairs = load_eval_pairs(args.pairs, limit=args.limit)
    if not pairs:
        print("ERROR: No evaluation pairs found. Check --pairs path.")
        sys.exit(1)

    t0 = time.time()

    if args.backend == "mlx":
        results = scan_mlx(args.model, pairs, adapter_path=args.adapter)
    else:
        results = scan_hf(args.model, pairs, quantize=args.quantize)

    elapsed = time.time() - t0

    # Add metadata
    results["metadata"] = {
        "model": args.model,
        "backend": args.backend,
        "quantize": args.quantize if args.backend == "hf" else "mlx-native",
        "num_pairs": len(pairs),
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")

    # Print summary
    for label in ["english", "nko"]:
        if label in results and results[label]:
            data = results[label]
            mid = len(data) // 2
            print(f"\n{label.upper()} summary ({len(data)} layers):")
            print(f"  Layer 0:   L2={data[0]['l2_norm']:.1f}, entropy={data[0]['entropy']:.2f}, sparsity={data[0]['sparsity']:.4f}")
            print(f"  Layer {mid}: L2={data[mid]['l2_norm']:.1f}, entropy={data[mid]['entropy']:.2f}, kurtosis={data[mid]['kurtosis']:.1f}")
            print(f"  Layer {len(data)-1}: L2={data[-1]['l2_norm']:.1f}, entropy={data[-1]['entropy']:.2f}")


if __name__ == "__main__":
    main()
