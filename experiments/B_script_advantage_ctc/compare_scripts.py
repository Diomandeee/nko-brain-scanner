#!/usr/bin/env python3
"""
Compare N'Ko vs Latin CTC ASR performance.

Loads both trained models, runs inference on an aligned test set,
and computes CER (Character Error Rate) and WER (Word Error Rate) for each.

Usage:
    python3 compare_scripts.py \
        --nko-checkpoint checkpoints/nko/best.pt \
        --latin-checkpoint checkpoints/latin/best.pt \
        --test-pairs test_pairs_aligned.jsonl \
        --features-dir /path/to/features/ \
        --output results/comparison.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def edit_distance(ref, hyp):
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def compute_cer(reference, hypothesis):
    """Character Error Rate."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def compute_wer(reference, hypothesis):
    """Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def ctc_greedy_decode(log_probs, idx_to_char, blank_idx=0):
    """Greedy CTC decoding: take argmax, collapse repeats, remove blanks."""
    indices = log_probs.argmax(dim=-1).tolist()

    decoded = []
    prev = None
    for idx in indices:
        if idx != blank_idx and idx != prev:
            if idx in idx_to_char:
                decoded.append(idx_to_char[idx])
        prev = idx

    return "".join(decoded)


def evaluate_model(model, features_dir, test_pairs, char_vocab, text_key, device):
    """Run inference and compute CER/WER on test set."""
    idx_to_char = {v: k for k, v in char_vocab.items() if k != "<blank>"}

    model.eval()
    results = []

    with torch.no_grad():
        for pair in test_pairs:
            feat_path = Path(features_dir) / f"{pair['feat_id']}.pt"
            if not feat_path.exists():
                continue

            features = torch.load(feat_path, weights_only=True).float()
            if features.shape[0] > 375:
                features = features[:375]

            features = features.unsqueeze(0).to(device)
            audio_len = torch.tensor([features.shape[1]], dtype=torch.long).to(device)

            log_probs, output_lens = model(features, audio_len)
            log_probs = log_probs.squeeze(1)  # (time, classes)

            hypothesis = ctc_greedy_decode(log_probs.cpu(), idx_to_char)
            reference = pair.get(text_key, "")

            cer = compute_cer(reference, hypothesis)
            wer = compute_wer(reference, hypothesis)

            results.append({
                "feat_id": pair["feat_id"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": cer,
                "wer": wer,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare N'Ko vs Latin CTC ASR")
    parser.add_argument("--nko-checkpoint", required=True)
    parser.add_argument("--latin-checkpoint", required=True)
    parser.add_argument("--test-pairs", required=True, help="JSONL with aligned test pairs (feat_id, nko, latin)")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output", default="results/comparison.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load test pairs
    test_pairs = []
    with open(args.test_pairs) as f:
        for line in f:
            test_pairs.append(json.loads(line.strip()))
    print(f"Test pairs: {len(test_pairs)}")

    # Import model classes
    from train_nko_ctc import CharASR_NKo, build_nko_char_vocab
    from train_latin_ctc import CharASR_Latin, build_latin_char_vocab

    # N'Ko model
    nko_vocab, nko_classes = build_nko_char_vocab()
    nko_model = CharASR_NKo(num_classes=nko_classes).to(device)
    nko_model.load_state_dict(torch.load(args.nko_checkpoint, map_location=device, weights_only=True))
    print(f"Loaded N'Ko model: {nko_classes} classes")

    nko_results = evaluate_model(nko_model, args.features_dir, test_pairs, nko_vocab, "nko", device)

    # Latin model
    latin_vocab, latin_classes = build_latin_char_vocab()
    latin_model = CharASR_Latin(num_classes=latin_classes).to(device)
    latin_model.load_state_dict(torch.load(args.latin_checkpoint, map_location=device, weights_only=True))
    print(f"Loaded Latin model: {latin_classes} classes")

    latin_results = evaluate_model(latin_model, args.features_dir, test_pairs, latin_vocab, "latin", device)

    # Aggregate
    nko_cer = np.mean([r["cer"] for r in nko_results]) if nko_results else 1.0
    nko_wer = np.mean([r["wer"] for r in nko_results]) if nko_results else 1.0
    latin_cer = np.mean([r["cer"] for r in latin_results]) if latin_results else 1.0
    latin_wer = np.mean([r["wer"] for r in latin_results]) if latin_results else 1.0

    print("\n" + "=" * 60)
    print("RESULTS: N'Ko vs Latin CTC ASR")
    print("=" * 60)
    print(f"\n  N'Ko  CTC:  CER = {nko_cer:.4f} ({nko_cer*100:.1f}%)  |  WER = {nko_wer:.4f} ({nko_wer*100:.1f}%)")
    print(f"  Latin CTC:  CER = {latin_cer:.4f} ({latin_cer*100:.1f}%)  |  WER = {latin_wer:.4f} ({latin_wer*100:.1f}%)")

    if latin_cer > 0:
        advantage = (latin_cer - nko_cer) / latin_cer * 100
        print(f"\n  Script advantage: {advantage:+.1f}% CER improvement for N'Ko")
    if nko_cer < latin_cer:
        print("  VERDICT: N'Ko's phonetic transparency provides measurable ASR advantage")
    elif nko_cer > latin_cer:
        print("  VERDICT: Latin's lower class count outweighs phonetic ambiguity")
    else:
        print("  VERDICT: No significant difference between scripts")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "nko": {
            "cer": float(nko_cer),
            "wer": float(nko_wer),
            "num_classes": nko_classes,
            "num_samples": len(nko_results),
            "per_sample": nko_results[:10],  # first 10 for inspection
        },
        "latin": {
            "cer": float(latin_cer),
            "wer": float(latin_wer),
            "num_classes": latin_classes,
            "num_samples": len(latin_results),
            "per_sample": latin_results[:10],
        },
        "advantage_pct": float((latin_cer - nko_cer) / latin_cer * 100) if latin_cer > 0 else 0,
    }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
