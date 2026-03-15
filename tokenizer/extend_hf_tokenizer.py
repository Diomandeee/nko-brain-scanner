#!/usr/bin/env python3
"""Extend Qwen3-8B vocabulary with top N'Ko BPE merge tokens.

Performs manual embedding surgery on quantized model:
1. Dequantize embeddings + lm_head
2. Add new rows (constituent-mean initialization)
3. Re-quantize
4. Save extended model

Usage:
  python3 tokenizer/extend_hf_tokenizer.py [--num-merges 250]
"""

import argparse
import json
import os
import shutil
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


# V2: Extend from BASE model (not fused) to avoid tokenizer/model vocab mismatch.
# The V1 bug: extending from fused model caused tokenizer to report 151,643 while
# model config said 152,192, leading to degenerate generation after LoRA fusion.
# mlx_lm.load() resolves HF model IDs from cache automatically
BASE_MODEL = "mlx-community/Qwen3-8B-8bit"
BPE_VOCAB = os.path.expanduser("~/nko-brain-scanner/bpe_vocab.json")
OUTPUT_DIR = os.path.expanduser("~/nko-brain-scanner/extended-nko-qwen3-v2")


def pad_to_multiple(n, multiple):
    """Pad n up to nearest multiple."""
    remainder = n % multiple
    if remainder == 0:
        return n
    return n + (multiple - remainder)


def find_token_id(tokenizer, char_str):
    """Find the token ID for a single character or short string."""
    ids = tokenizer.encode(char_str, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # If tokenizer splits it into multiple tokens, return all of them
    return ids


def compute_constituent_embedding(full_weight, tokenizer, merge_parts):
    """Compute mean embedding from merge constituents.

    merge_parts: list of 2 strings, e.g. ['ߟ', 'ߊ߫']
    Returns: mean embedding vector (hidden_dim,)
    """
    all_ids = []
    for part in merge_parts:
        ids = tokenizer.encode(part, add_special_tokens=False)
        all_ids.extend(ids)

    if not all_ids:
        # Fallback: random init
        return None

    # Gather embeddings for all constituent token IDs
    embeddings = []
    for tid in all_ids:
        if tid < full_weight.shape[0]:
            embeddings.append(full_weight[tid])

    if not embeddings:
        return None

    # Stack and mean
    stacked = mx.stack(embeddings, axis=0)
    return mx.mean(stacked, axis=0)


def dequantize_layer_weight(layer):
    """Dequantize a QuantizedEmbedding or QuantizedLinear weight."""
    biases = layer.biases if hasattr(layer, "biases") and layer.biases is not None else None

    full = mx.dequantize(
        layer.weight,
        layer.scales,
        biases,
        group_size=layer.group_size,
        bits=layer.bits,
    )
    mx.eval(full)
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-merges", type=int, default=250,
                        help="Number of top BPE merges to add (default: 250)")
    parser.add_argument("--model-path", default=BASE_MODEL)
    parser.add_argument("--output-path", default=OUTPUT_DIR,
                        help="Output dir (default: extended-nko-qwen3-v2)")
    args = parser.parse_args()

    print("=" * 60)
    print("N'Ko BPE Vocabulary Extension for Qwen3-8B")
    print("=" * 60)

    # Load BPE vocab
    print(f"\nLoading BPE vocab from {BPE_VOCAB}...")
    with open(BPE_VOCAB) as f:
        bpe = json.load(f)

    merges = bpe["merges"][:args.num_merges]
    token_to_id = bpe["token_to_id"]
    id_to_token = {v: k for k, v in token_to_id.items()}

    # Build merge token strings (concatenate the two parts)
    new_tokens = []
    merge_parts_list = []
    for parts in merges:
        token_str = "".join(parts)
        new_tokens.append(token_str)
        merge_parts_list.append(parts)

    print(f"Will add {len(new_tokens)} BPE merge tokens")
    print(f"First 10: {[repr(t) for t in new_tokens[:10]]}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, tokenizer = load(args.model_path)

    old_vocab_size = model.model.embed_tokens.weight.shape[0]
    hidden_size = 4096  # Known from config
    group_size = model.model.embed_tokens.group_size
    bits = model.model.embed_tokens.bits

    print(f"Current vocab size: {old_vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")

    # Step 1: Dequantize embedding layer
    print("\nDequantizing embedding layer...")
    embed_full = dequantize_layer_weight(model.model.embed_tokens)
    print(f"  Dequantized shape: {embed_full.shape}, dtype: {embed_full.dtype}")

    # Step 2: Dequantize lm_head
    print("Dequantizing lm_head...")
    lm_head_full = dequantize_layer_weight(model.lm_head)
    print(f"  Dequantized shape: {lm_head_full.shape}, dtype: {lm_head_full.dtype}")

    # Step 3: Compute constituent-mean embeddings for new tokens
    print(f"\nComputing constituent-mean embeddings for {len(new_tokens)} tokens...")
    new_embed_rows = []
    new_lm_head_rows = []
    fallback_count = 0

    for i, (token_str, parts) in enumerate(zip(new_tokens, merge_parts_list)):
        # Embedding: constituent mean
        embed_init = compute_constituent_embedding(embed_full, tokenizer, parts)
        if embed_init is None:
            embed_init = mx.mean(embed_full, axis=0)
            fallback_count += 1
        # Add small noise to break symmetry
        noise = mx.random.normal(shape=(hidden_size,)) * 0.01
        new_embed_rows.append(embed_init + noise)

        # lm_head: constituent mean (same logic)
        lm_init = compute_constituent_embedding(lm_head_full, tokenizer, parts)
        if lm_init is None:
            lm_init = mx.mean(lm_head_full, axis=0)
        noise2 = mx.random.normal(shape=(hidden_size,)) * 0.01
        new_lm_head_rows.append(lm_init + noise2)

    print(f"  Constituent-mean: {len(new_tokens) - fallback_count}, Fallback: {fallback_count}")

    # Step 4: Concatenate new rows
    new_embed_matrix = mx.stack(new_embed_rows, axis=0)
    extended_embed = mx.concatenate([embed_full, new_embed_matrix], axis=0)

    new_lm_matrix = mx.stack(new_lm_head_rows, axis=0)
    extended_lm = mx.concatenate([lm_head_full, new_lm_matrix], axis=0)

    # Pad to nearest multiple of group_size
    new_vocab_raw = old_vocab_size + len(new_tokens)
    new_vocab_padded = pad_to_multiple(new_vocab_raw, group_size)
    pad_count = new_vocab_padded - new_vocab_raw

    if pad_count > 0:
        print(f"\nPadding vocab from {new_vocab_raw} to {new_vocab_padded} (+{pad_count} padding tokens)")
        pad_embed = mx.zeros((pad_count, hidden_size))
        extended_embed = mx.concatenate([extended_embed, pad_embed], axis=0)
        pad_lm = mx.zeros((pad_count, hidden_size))
        extended_lm = mx.concatenate([extended_lm, pad_lm], axis=0)

    mx.eval(extended_embed)
    mx.eval(extended_lm)
    print(f"Extended embed shape: {extended_embed.shape}")
    print(f"Extended lm_head shape: {extended_lm.shape}")

    # Step 5: Re-quantize
    print("\nRe-quantizing embedding layer...")
    q_weight, q_scales, q_biases = mx.quantize(extended_embed, group_size=group_size, bits=bits)
    mx.eval(q_weight, q_scales, q_biases)
    print(f"  Quantized weight: {q_weight.shape} {q_weight.dtype}")
    print(f"  Scales: {q_scales.shape}")

    # Create new QuantizedEmbedding
    new_embed = nn.QuantizedEmbedding(
        num_embeddings=new_vocab_padded,
        dims=hidden_size,
        group_size=group_size,
        bits=bits,
    )
    new_embed.weight = q_weight
    new_embed.scales = q_scales
    new_embed.biases = q_biases

    print("Re-quantizing lm_head...")
    lm_q_weight, lm_q_scales, lm_q_biases = mx.quantize(extended_lm, group_size=group_size, bits=bits)
    mx.eval(lm_q_weight, lm_q_scales, lm_q_biases)
    print(f"  Quantized weight: {lm_q_weight.shape} {lm_q_weight.dtype}")

    # Create new QuantizedLinear
    new_lm_head = nn.QuantizedLinear(
        input_dims=hidden_size,
        output_dims=new_vocab_padded,
        bias=False,
        group_size=group_size,
        bits=bits,
    )
    new_lm_head.weight = lm_q_weight
    new_lm_head.scales = lm_q_scales
    new_lm_head.biases = lm_q_biases

    # Step 6: Replace layers in model
    print("\nReplacing model layers...")
    model.model.embed_tokens = new_embed
    model.lm_head = new_lm_head

    # Step 7: Add tokens to HF tokenizer (load directly, not via mlx wrapper)
    from transformers import AutoTokenizer
    print(f"\nLoading HF tokenizer from {args.model_path}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    old_tok_size = len(hf_tokenizer)
    num_added = hf_tokenizer.add_tokens(new_tokens)
    print(f"  Added {num_added} tokens (tokenizer: {old_tok_size} → {len(hf_tokenizer)})")

    # V2 CRITICAL CHECK: tokenizer vocab MUST match model embedding size
    tok_vocab = len(hf_tokenizer)
    if tok_vocab != new_vocab_padded:
        # Pad tokenizer to match model embedding dimension
        pad_tokens_needed = new_vocab_padded - tok_vocab
        if pad_tokens_needed > 0:
            pad_token_strs = [f"<|pad_{i}|>" for i in range(pad_tokens_needed)]
            hf_tokenizer.add_tokens(pad_token_strs)
            print(f"  ALIGNMENT: Added {pad_tokens_needed} padding tokens to tokenizer")
        print(f"  Tokenizer vocab: {len(hf_tokenizer)}, Model vocab: {new_vocab_padded}")
    assert len(hf_tokenizer) == new_vocab_padded, \
        f"FATAL: Tokenizer ({len(hf_tokenizer)}) != Model ({new_vocab_padded}). Cannot proceed."
    print(f"  VERIFIED: Tokenizer and model vocab aligned at {new_vocab_padded}")

    # Step 8: Save extended model
    print(f"\nSaving extended model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)

    # Save tokenizer
    hf_tokenizer.save_pretrained(args.output_path)
    print(f"  Saved tokenizer")

    # Update config — resolve from HF cache if model_path is a HF model ID
    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.exists(config_path):
        # Model loaded from HF cache — find the cached config
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(args.model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["vocab_size"] = new_vocab_padded
    config["extended_vocab"] = {
        "num_bpe_tokens_added": len(new_tokens),
        "pad_tokens": pad_count,
        "init_strategy": "constituent_mean",
    }
    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save model weights (handle nested dicts + lists)
    weights = dict(model.parameters())
    flat_weights = {}
    def flatten(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    flatten(v, key)
                elif isinstance(v, mx.array):
                    flat_weights[key] = v
        elif isinstance(d, list):
            for i, item in enumerate(d):
                key = f"{prefix}.{i}"
                if isinstance(item, (dict, list)):
                    flatten(item, key)
                elif isinstance(item, mx.array):
                    flat_weights[key] = item
    flatten(weights)

    # Evaluate all arrays before saving
    arrays = list(flat_weights.values())
    # Eval in batches to avoid memory pressure
    batch_size = 100
    for i in range(0, len(arrays), batch_size):
        mx.eval(*arrays[i:i+batch_size])
    print(f"  Collected {len(flat_weights)} weight tensors")

    # Check total size
    total_bytes = sum(v.nbytes for v in flat_weights.values())
    print(f"  Total weight size: {total_bytes / 1e9:.2f} GB")

    # Save as safetensors (split into 2 shards if > 5GB)
    if total_bytes > 5e9:
        keys = sorted(flat_weights.keys())
        mid = len(keys) // 2
        shard1 = {k: flat_weights[k] for k in keys[:mid]}
        shard2 = {k: flat_weights[k] for k in keys[mid:]}

        path1 = os.path.join(args.output_path, "model-00001-of-00002.safetensors")
        path2 = os.path.join(args.output_path, "model-00002-of-00002.safetensors")
        mx.save_safetensors(path1, shard1)
        mx.save_safetensors(path2, shard2)

        # Create index file
        weight_map = {}
        for k in shard1:
            weight_map[k] = "model-00001-of-00002.safetensors"
        for k in shard2:
            weight_map[k] = "model-00002-of-00002.safetensors"
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        with open(os.path.join(args.output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

        print(f"  Saved shard 1: {os.path.getsize(path1)/1e9:.2f} GB")
        print(f"  Saved shard 2: {os.path.getsize(path2)/1e9:.2f} GB")
    else:
        save_path = os.path.join(args.output_path, "model.safetensors")
        mx.save_safetensors(save_path, flat_weights)
        print(f"  Saved weights to {save_path}")
        print(f"  Size: {os.path.getsize(save_path) / 1e9:.2f} GB")

    # Copy over any other config files needed
    for fname in ["merges.txt", "model.safetensors.index.json"]:
        src = os.path.join(args.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_path, fname))
        elif not os.path.isdir(args.model_path):
            # HF model ID — try downloading
            try:
                from huggingface_hub import hf_hub_download
                cached = hf_hub_download(args.model_path, fname)
                shutil.copy2(cached, os.path.join(args.output_path, fname))
            except Exception:
                pass  # File may not exist in the repo

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Test encoding with the extended tokenizer
    test_token = new_tokens[0]
    test_ids = hf_tokenizer.encode(test_token, add_special_tokens=False)
    print(f"Token {repr(test_token)} encodes to IDs: {test_ids}")
    print(f"  (should be single ID >= {old_vocab_size})")

    # Test that model still produces valid logits
    test_text = "Hello world"
    test_ids_full = hf_tokenizer.encode(test_text)
    x = mx.array(test_ids_full)[None]
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits finite: {bool(mx.all(mx.isfinite(logits)).item())}")

    # Test with N'Ko text using new tokens
    nko_test = new_tokens[0] + new_tokens[1] + new_tokens[2]
    nko_ids = hf_tokenizer.encode(nko_test, add_special_tokens=False)
    print(f"\nN'Ko test: {repr(nko_test)}")
    print(f"  Token IDs: {nko_ids}")
    x_nko = mx.array(nko_ids)[None]
    nko_logits = model(x_nko)
    print(f"  Logits shape: {nko_logits.shape}")
    print(f"  Logits finite: {bool(mx.all(mx.isfinite(nko_logits)).item())}")

    print(f"\nDone! Extended model saved to {args.output_path}")
    print(f"  Old vocab: {old_vocab_size}")
    print(f"  New vocab: {new_vocab_padded} (+{len(new_tokens)} BPE + {pad_count} padding)")


if __name__ == "__main__":
    main()
