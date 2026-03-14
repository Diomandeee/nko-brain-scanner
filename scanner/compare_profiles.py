#!/usr/bin/env python3
"""
Compare activation profiles between base and fine-tuned models.

Generates publication-quality before/after comparison figures showing
how LoRA fine-tuning changes N'Ko processing across all layers.

Usage:
  python3 scanner/compare_profiles.py \
    --base results/mlx_base_profile.json \
    --finetuned results/mlx_finetuned_profile.json \
    --output results/figures/finetuning_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required")
    sys.exit(1)


def load_profile(path):
    with open(path) as f:
        return json.load(f)


def extract_metric(profile_data, metric):
    """Extract a metric across all layers for a language."""
    layers = [d["layer"] for d in profile_data]
    values = [d[metric] for d in profile_data]
    return layers, values


def plot_comparison(base, finetuned, output_path):
    """Generate 4-panel comparison figure."""
    metrics = [
        ("l2_norm", "L2 Norm (Activation Magnitude)"),
        ("entropy", "Shannon Entropy (bits)"),
        ("sparsity", "Sparsity (Near-Zero Fraction)"),
        ("kurtosis", "Excess Kurtosis (Specialization)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "N'Ko Activation Profiles: Base vs Fine-Tuned\n"
        "(Qwen3-8B, LoRA fine-tuned on 4,312 N'Ko SFT examples)",
        fontsize=14, fontweight="bold", y=0.98
    )

    colors = {
        "base_en": "#2196F3",       # Blue
        "base_nko": "#FF9800",      # Orange
        "ft_en": "#4CAF50",         # Green
        "ft_nko": "#E91E63",        # Pink
    }

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]

        # Base model
        if "english" in base:
            layers, vals = extract_metric(base["english"], metric)
            ax.plot(layers, vals, color=colors["base_en"], linewidth=1.5,
                    alpha=0.5, linestyle="--", label="Base English")
        if "nko" in base:
            layers, vals = extract_metric(base["nko"], metric)
            ax.plot(layers, vals, color=colors["base_nko"], linewidth=1.5,
                    alpha=0.5, linestyle="--", label="Base N'Ko")

        # Fine-tuned model
        if "english" in finetuned:
            layers, vals = extract_metric(finetuned["english"], metric)
            ax.plot(layers, vals, color=colors["ft_en"], linewidth=2,
                    label="Fine-Tuned English")
        if "nko" in finetuned:
            layers, vals = extract_metric(finetuned["nko"], metric)
            ax.plot(layers, vals, color=colors["ft_nko"], linewidth=2,
                    label="Fine-Tuned N'Ko")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison figure to {output_path}")
    plt.close()


def plot_nko_improvement(base, finetuned, output_path):
    """Generate N'Ko-focused improvement chart."""
    metrics = [
        ("l2_norm", "L2 Norm"),
        ("entropy", "Entropy"),
        ("kurtosis", "Kurtosis"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "N'Ko Processing Improvement After Fine-Tuning",
        fontsize=14, fontweight="bold"
    )

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        if "nko" in base and "nko" in finetuned:
            base_layers, base_vals = extract_metric(base["nko"], metric)
            ft_layers, ft_vals = extract_metric(finetuned["nko"], metric)

            # Align lengths
            min_len = min(len(base_vals), len(ft_vals))
            base_vals = base_vals[:min_len]
            ft_vals = ft_vals[:min_len]
            layers = base_layers[:min_len]

            ax.fill_between(layers, base_vals, ft_vals, alpha=0.3,
                            color="#4CAF50" if metric != "entropy" else "#FF5722",
                            label="Improvement" if metric != "entropy" else "Reduction")
            ax.plot(layers, base_vals, color="#FF9800", linewidth=2,
                    linestyle="--", label="Before (Base)")
            ax.plot(layers, ft_vals, color="#E91E63", linewidth=2,
                    label="After (Fine-Tuned)")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved improvement figure to {output_path}")
    plt.close()


def print_summary(base, finetuned):
    """Print key comparison metrics."""
    print("\n" + "=" * 60)
    print("FINE-TUNING IMPACT SUMMARY")
    print("=" * 60)

    for metric, label in [("l2_norm", "L2 Norm"), ("entropy", "Entropy"),
                           ("sparsity", "Sparsity"), ("kurtosis", "Kurtosis")]:
        print(f"\n{label}:")

        for lang in ["english", "nko"]:
            if lang in base and lang in finetuned:
                base_vals = [d[metric] for d in base[lang]]
                ft_vals = [d[metric] for d in finetuned[lang]]

                min_len = min(len(base_vals), len(ft_vals))
                base_mid = base_vals[min_len // 2]
                ft_mid = ft_vals[min_len // 2]

                if base_mid != 0:
                    change = ((ft_mid - base_mid) / abs(base_mid)) * 100
                    print(f"  {lang:>8} mid-layer: {base_mid:.1f} -> {ft_mid:.1f} ({change:+.1f}%)")
                else:
                    print(f"  {lang:>8} mid-layer: {base_mid:.1f} -> {ft_mid:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned profiles")
    parser.add_argument("--base", required=True, help="Base model profile JSON")
    parser.add_argument("--finetuned", required=True, help="Fine-tuned model profile JSON")
    parser.add_argument("--output", default="results/figures/finetuning_comparison.png",
                        help="Output figure path")
    args = parser.parse_args()

    base = load_profile(args.base)
    finetuned = load_profile(args.finetuned)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_comparison(base, finetuned, str(output_path))

    # Also generate improvement-focused chart
    imp_path = str(output_path).replace(".png", "_improvement.png")
    plot_nko_improvement(base, finetuned, imp_path)

    print_summary(base, finetuned)


if __name__ == "__main__":
    main()
