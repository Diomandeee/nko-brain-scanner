#!/usr/bin/env python3
"""
Measure and visualize compression ratios from sigil encoding.

Reads the output of sigil_encoder.py and produces:
1. Compression ratio statistics (BPE vs N'Ko vs Sigil)
2. Distribution plots
3. Per-sigil frequency analysis
4. Information retention analysis

Usage:
    python3 measure_compression.py \
        --input results/sigil_encoded.jsonl \
        --output results/compression_stats.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SIGIL_NAMES = {
    "ߛ": "stabilization",
    "ߜ": "dispersion",
    "ߕ": "transition",
    "ߙ": "return",
    "ߡ": "dwell",
    "ߚ": "oscillation",
    "ߞ": "recovery",
    "ߣ": "novelty",
    "ߠ": "place_shift",
    "ߥ": "echo",
}


def load_encoded(path):
    """Load sigil-encoded data."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_stats(data):
    """Compute compression statistics."""
    bpe_tokens = [d["bpe_tokens"] for d in data]
    nko_chars = [d["nko_chars"] for d in data]
    sigil_counts = [d["sigil_count"] for d in data]

    # Compression ratios per turn
    bpe_to_sigil = [b / s if s > 0 else 0 for b, s in zip(bpe_tokens, sigil_counts)]
    nko_to_sigil = [n / s if s > 0 and n > 0 else 0 for n, s in zip(nko_chars, sigil_counts)]
    bpe_to_nko = [b / n if n > 0 else 0 for b, n in zip(bpe_tokens, nko_chars)]

    # Sigil frequency
    all_sigils = []
    for d in data:
        all_sigils.extend(list(d["sigil_sequence"]))
    sigil_freq = Counter(all_sigils)

    stats = {
        "num_turns": len(data),
        "totals": {
            "bpe_tokens": sum(bpe_tokens),
            "nko_chars": sum(nko_chars),
            "sigils": sum(sigil_counts),
        },
        "overall_compression": {
            "bpe_to_sigil": sum(bpe_tokens) / sum(sigil_counts) if sum(sigil_counts) > 0 else 0,
            "nko_to_sigil": sum(nko_chars) / sum(sigil_counts) if sum(sigil_counts) > 0 else 0,
            "bpe_to_nko": sum(bpe_tokens) / sum(nko_chars) if sum(nko_chars) > 0 else 0,
        },
        "per_turn_compression": {
            "bpe_to_sigil": {
                "mean": float(np.mean(bpe_to_sigil)),
                "median": float(np.median(bpe_to_sigil)),
                "std": float(np.std(bpe_to_sigil)),
                "min": float(np.min(bpe_to_sigil)),
                "max": float(np.max(bpe_to_sigil)),
            },
            "nko_to_sigil": {
                "mean": float(np.mean([r for r in nko_to_sigil if r > 0])) if any(r > 0 for r in nko_to_sigil) else 0,
                "median": float(np.median([r for r in nko_to_sigil if r > 0])) if any(r > 0 for r in nko_to_sigil) else 0,
            },
        },
        "sigil_frequency": {
            SIGIL_NAMES.get(char, char): count
            for char, count in sigil_freq.most_common()
        },
        "token_stats": {
            "avg_bpe_per_turn": float(np.mean(bpe_tokens)),
            "avg_nko_per_turn": float(np.mean(nko_chars)),
            "avg_sigils_per_turn": float(np.mean(sigil_counts)),
        },
    }

    return stats


def print_table(stats):
    """Print formatted compression table."""
    print("\n" + "=" * 70)
    print("COMPRESSION ANALYSIS")
    print("=" * 70)

    t = stats["totals"]
    c = stats["overall_compression"]
    print(f"\n  Turns analyzed: {stats['num_turns']}")
    print(f"\n  Layer         | Total Count | Avg/Turn | Compression vs Sigil")
    print(f"  --------------|-------------|----------|--------------------")
    print(f"  English BPE   | {t['bpe_tokens']:>11,} | {stats['token_stats']['avg_bpe_per_turn']:>8.1f} | {c['bpe_to_sigil']:.1f}x")
    print(f"  N'Ko chars    | {t['nko_chars']:>11,} | {stats['token_stats']['avg_nko_per_turn']:>8.1f} | {c['nko_to_sigil']:.1f}x")
    print(f"  Sigils        | {t['sigils']:>11,} | {stats['token_stats']['avg_sigils_per_turn']:>8.1f} | 1.0x (baseline)")

    print(f"\n  Sigil Frequency:")
    for name, count in stats["sigil_frequency"].items():
        pct = count / t["sigils"] * 100 if t["sigils"] > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {name:<15} {count:>5} ({pct:>5.1f}%) {bar}")


def plot_compression(data, output_dir):
    """Generate compression visualization."""
    if plt is None:
        print("matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpe_tokens = [d["bpe_tokens"] for d in data]
    sigil_counts = [d["sigil_count"] for d in data]
    compression = [b / s if s > 0 else 0 for b, s in zip(bpe_tokens, sigil_counts)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Compression ratio distribution
    axes[0].hist(compression, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(np.mean(compression), color="#E91E63", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(compression):.1f}x")
    axes[0].set_xlabel("BPE-to-Sigil Compression Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Compression Ratio Distribution")
    axes[0].legend()

    # BPE tokens vs Sigils scatter
    axes[1].scatter(bpe_tokens, sigil_counts, alpha=0.3, s=10, color="#4CAF50")
    axes[1].set_xlabel("BPE Tokens (original)")
    axes[1].set_ylabel("Sigil Count (compressed)")
    axes[1].set_title("Token Count vs Sigil Count")

    # Sigil frequency bar chart
    all_sigils = []
    for d in data:
        all_sigils.extend(list(d["sigil_sequence"]))
    freq = Counter(all_sigils)
    names = [SIGIL_NAMES.get(c, c) for c in freq.keys()]
    counts = list(freq.values())
    colors = ["#FF9800", "#2196F3", "#4CAF50", "#E91E63", "#9C27B0",
              "#00BCD4", "#FF5722", "#795548", "#607D8B", "#CDDC39"]
    axes[2].barh(names, counts, color=colors[:len(names)])
    axes[2].set_xlabel("Frequency")
    axes[2].set_title("Sigil Usage Distribution")

    plt.tight_layout()
    fig_path = output_dir / "compression_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Measure sigil compression")
    parser.add_argument("--input", required=True, help="Sigil-encoded JSONL from sigil_encoder.py")
    parser.add_argument("--output", default="results/compression_stats.json")
    args = parser.parse_args()

    data = load_encoded(args.input)
    print(f"Loaded {len(data)} encoded turns")

    stats = compute_stats(data)
    print_table(stats)

    # Plot
    figures_dir = Path(args.output).parent / "figures"
    plot_compression(data, figures_dir)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats saved to {output_path}")


if __name__ == "__main__":
    main()
