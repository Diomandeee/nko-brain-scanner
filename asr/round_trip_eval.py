#!/usr/bin/env python3
"""
Round-trip evaluation for the syllable retriever.

Pipeline: codebook embeddings -> k-NN retrieval -> FSM beam search -> N'Ko text
Compares against ground truth from synthetic eval pairs.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import mlx.core as mx
from asr.syllable_retriever import SyllableRetriever
from nko_core import transliterate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def char_accuracy(predicted, reference):
    """Character-level accuracy using longest common subsequence ratio."""
    if not reference:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0

    # LCS length
    m, n = len(predicted), len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return lcs / max(m, n)


def run_eval(eval_path, beam_width=5):
    """Run round-trip evaluation on synthetic pairs."""
    retriever = SyllableRetriever()

    with open(eval_path) as f:
        pairs = json.load(f)

    results = {
        "total": len(pairs),
        "exact_match": 0,
        "char_accuracies": [],
        "retrieval_at_1": 0,
        "retrieval_at_5": 0,
        "retrieval_at_10": 0,
        "transliteration_results": [],
    }

    for i, pair in enumerate(pairs):
        indices = pair["indices"]
        expected_text = pair["nko_text"]

        if not indices:
            continue

        # Use codebook embeddings as "perfect" audio embeddings
        audio_embeddings = [retriever.embeddings[idx] for idx in indices]

        # Beam search
        predicted_text, score = retriever.fsm_beam_search(audio_embeddings, beam_width=beam_width)

        # Exact match
        is_exact = predicted_text == expected_text
        if is_exact:
            results["exact_match"] += 1

        # Character accuracy
        ca = char_accuracy(predicted_text, expected_text)
        results["char_accuracies"].append(ca)

        # Retrieval accuracy (per frame)
        for idx, emb in zip(indices, audio_embeddings):
            top_indices, _ = retriever.retrieve_top_k(emb, k=10)
            top_list = top_indices.tolist()
            if top_list[0] == idx:
                results["retrieval_at_1"] += 1
            if idx in top_list[:5]:
                results["retrieval_at_5"] += 1
            if idx in top_list:
                results["retrieval_at_10"] += 1

        # Transliteration round-trip
        if predicted_text:
            pred_latin = transliterate(predicted_text)
            expected_latin = transliterate(expected_text)
            results["transliteration_results"].append({
                "expected_nko": expected_text,
                "predicted_nko": predicted_text,
                "expected_latin": expected_latin,
                "predicted_latin": pred_latin,
                "match": pred_latin == expected_latin,
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(pairs)}] exact={results['exact_match']}")

    # Compute aggregates
    total_frames = sum(len(p["indices"]) for p in pairs if p["indices"])
    avg_char_acc = sum(results["char_accuracies"]) / len(results["char_accuracies"]) if results["char_accuracies"] else 0

    summary = {
        "total_pairs": results["total"],
        "exact_match": results["exact_match"],
        "exact_match_rate": results["exact_match"] / results["total"] if results["total"] else 0,
        "avg_char_accuracy": avg_char_acc,
        "retrieval_at_1": results["retrieval_at_1"] / total_frames if total_frames else 0,
        "retrieval_at_5": results["retrieval_at_5"] / total_frames if total_frames else 0,
        "retrieval_at_10": results["retrieval_at_10"] / total_frames if total_frames else 0,
        "total_frames": total_frames,
        "beam_width": beam_width,
        "transliteration_match_rate": (
            sum(1 for r in results["transliteration_results"] if r["match"])
            / len(results["transliteration_results"])
            if results["transliteration_results"] else 0
        ),
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-trip eval for syllable retriever")
    parser.add_argument("--eval-data", default=os.path.join(PROJECT_ROOT, "data", "synthetic_eval_pairs.json"))
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "results", "round_trip_eval.json"))
    args = parser.parse_args()

    print("=== Round-Trip Evaluation ===")
    print(f"Eval data: {args.eval_data}")
    print(f"Beam width: {args.beam_width}")

    summary = run_eval(args.eval_data, beam_width=args.beam_width)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Results ===")
    print(f"  Exact match rate: {summary['exact_match_rate']:.3f} ({summary['exact_match']}/{summary['total_pairs']})")
    print(f"  Avg char accuracy: {summary['avg_char_accuracy']:.3f}")
    print(f"  Retrieval @1:  {summary['retrieval_at_1']:.3f}")
    print(f"  Retrieval @5:  {summary['retrieval_at_5']:.3f}")
    print(f"  Retrieval @10: {summary['retrieval_at_10']:.3f}")
    print(f"  Transliteration match: {summary['transliteration_match_rate']:.3f}")
    print(f"\nSaved to {args.output}")
