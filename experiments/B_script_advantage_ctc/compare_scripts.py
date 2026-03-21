#!/usr/bin/env python3
"""
Compare N'Ko vs Latin CTC ASR performance.
==========================================

Loads both trained models, runs inference on an aligned test set,
and computes CER (Character Error Rate) and WER (Word Error Rate) for each.

Handles both checkpoint formats:
  - Full checkpoint (from hardened training): loads model_state from dict
  - Raw state_dict (from simple torch.save): loads directly

Usage:
    python3 compare_scripts.py \\
        --nko-checkpoint checkpoints/nko/best.pt \\
        --latin-checkpoint checkpoints/latin/best.pt \\
        --test-pairs test_pairs_aligned.jsonl \\
        --features-dir /path/to/features/ \\
        --output results/comparison.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


def log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def load_model_weights(model, checkpoint_path, device):
    """Load model weights from either full checkpoint or raw state_dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # Full checkpoint from hardened training
        model.load_state_dict(ckpt["model_state"])
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", "?")
        reason = ckpt.get("reason", "?")
        log(f"  Loaded from full checkpoint (epoch={epoch}, val_loss={val_loss}, reason={reason})")
    elif isinstance(ckpt, dict) and any(k.startswith("downsample") or k.startswith("output") for k in ckpt.keys()):
        # Raw state_dict
        model.load_state_dict(ckpt)
        log(f"  Loaded from raw state_dict")
    else:
        # Try loading as state_dict anyway
        model.load_state_dict(ckpt)
        log(f"  Loaded checkpoint (unknown format)")


def evaluate_model(model, features_dir, test_pairs, char_vocab, text_key, device):
    """Run inference and compute CER/WER on test set."""
    idx_to_char = {v: k for k, v in char_vocab.items() if k != "<blank>"}

    model.eval()
    results = []
    skipped = 0

    with torch.no_grad():
        for i, pair in enumerate(test_pairs):
            feat_path = Path(features_dir) / f"{pair['feat_id']}.pt"
            if not feat_path.exists():
                skipped += 1
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

            if (i + 1) % 100 == 0:
                log(f"  Evaluated {i+1}/{len(test_pairs)} samples...")

    if skipped > 0:
        log(f"  Skipped {skipped} samples (missing features)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare N'Ko vs Latin CTC ASR")
    parser.add_argument("--nko-checkpoint", required=True)
    parser.add_argument("--latin-checkpoint", required=True)
    parser.add_argument("--test-pairs", required=True,
                        help="JSONL with aligned test pairs (feat_id, nko, latin)")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output", default="results/comparison.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    log("=" * 70)
    log("Experiment B: N'Ko vs Latin CTC Script Comparison")
    log("=" * 70)

    # Load test pairs
    test_pairs = []
    with open(args.test_pairs) as f:
        for line in f:
            line = line.strip()
            if line:
                test_pairs.append(json.loads(line))
    log(f"Test pairs: {len(test_pairs)}")

    # Import model classes
    from train_nko_ctc import CharASR_NKo, build_nko_char_vocab
    from train_latin_ctc import CharASR_Latin, build_latin_char_vocab

    # ── N'Ko model ──
    log("\nEvaluating N'Ko CTC model...")
    nko_vocab, nko_classes = build_nko_char_vocab()
    nko_model = CharASR_NKo(num_classes=nko_classes).to(device)
    load_model_weights(nko_model, args.nko_checkpoint, device)
    log(f"  N'Ko model: {nko_classes} classes")

    t0 = time.time()
    nko_results = evaluate_model(nko_model, args.features_dir, test_pairs, nko_vocab, "nko", device)
    nko_time = time.time() - t0
    log(f"  N'Ko evaluation: {len(nko_results)} samples in {nko_time:.1f}s")

    # Free GPU memory
    del nko_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Latin model ──
    log("\nEvaluating Latin CTC model...")
    latin_vocab, latin_classes = build_latin_char_vocab()
    latin_model = CharASR_Latin(num_classes=latin_classes).to(device)
    load_model_weights(latin_model, args.latin_checkpoint, device)
    log(f"  Latin model: {latin_classes} classes")

    t0 = time.time()
    latin_results = evaluate_model(latin_model, args.features_dir, test_pairs, latin_vocab, "latin", device)
    latin_time = time.time() - t0
    log(f"  Latin evaluation: {len(latin_results)} samples in {latin_time:.1f}s")

    # ── Aggregate results ──
    nko_cer = np.mean([r["cer"] for r in nko_results]) if nko_results else 1.0
    nko_wer = np.mean([r["wer"] for r in nko_results]) if nko_results else 1.0
    nko_cer_std = np.std([r["cer"] for r in nko_results]) if nko_results else 0.0
    nko_cer_median = np.median([r["cer"] for r in nko_results]) if nko_results else 1.0

    latin_cer = np.mean([r["cer"] for r in latin_results]) if latin_results else 1.0
    latin_wer = np.mean([r["wer"] for r in latin_results]) if latin_results else 1.0
    latin_cer_std = np.std([r["cer"] for r in latin_results]) if latin_results else 0.0
    latin_cer_median = np.median([r["cer"] for r in latin_results]) if latin_results else 1.0

    print("\n" + "=" * 70)
    print("RESULTS: N'Ko vs Latin CTC ASR — Experiment B")
    print("=" * 70)
    print(f"\n  N'Ko  CTC ({nko_classes} classes):")
    print(f"    CER = {nko_cer:.4f} ({nko_cer*100:.1f}%) +/- {nko_cer_std:.4f}  median={nko_cer_median:.4f}")
    print(f"    WER = {nko_wer:.4f} ({nko_wer*100:.1f}%)")
    print(f"    Samples: {len(nko_results)}")
    print(f"\n  Latin CTC ({latin_classes} classes):")
    print(f"    CER = {latin_cer:.4f} ({latin_cer*100:.1f}%) +/- {latin_cer_std:.4f}  median={latin_cer_median:.4f}")
    print(f"    WER = {latin_wer:.4f} ({latin_wer*100:.1f}%)")
    print(f"    Samples: {len(latin_results)}")

    if latin_cer > 0:
        advantage = (latin_cer - nko_cer) / latin_cer * 100
        print(f"\n  Script advantage: {advantage:+.1f}% CER improvement for N'Ko")
    else:
        advantage = 0.0

    print()
    if nko_cer < latin_cer * 0.95:
        verdict = "N'Ko's phonetic transparency provides measurable ASR advantage"
    elif nko_cer > latin_cer * 1.05:
        verdict = "Latin's lower class count outweighs phonetic ambiguity"
    else:
        verdict = "No significant difference between scripts (within 5% margin)"
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "experiment": "B_script_advantage_ctc",
        "timestamp": datetime.utcnow().isoformat(),
        "verdict": verdict,
        "nko": {
            "cer_mean": float(nko_cer),
            "cer_std": float(nko_cer_std),
            "cer_median": float(nko_cer_median),
            "wer_mean": float(nko_wer),
            "num_classes": nko_classes,
            "num_samples": len(nko_results),
            "eval_time_s": nko_time,
            "per_sample": nko_results[:20],  # first 20 for inspection
        },
        "latin": {
            "cer_mean": float(latin_cer),
            "cer_std": float(latin_cer_std),
            "cer_median": float(latin_cer_median),
            "wer_mean": float(latin_wer),
            "num_classes": latin_classes,
            "num_samples": len(latin_results),
            "eval_time_s": latin_time,
            "per_sample": latin_results[:20],
        },
        "advantage_pct": float(advantage),
    }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
