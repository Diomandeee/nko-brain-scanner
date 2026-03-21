#!/usr/bin/env python3
"""
Compare brain scan results across multiple models.

Loads JSON results from run_brain_scan.py for all 4 models and produces:
1. Cross-model comparison table (printed + saved as JSON)
2. Figure descriptions for publication
3. Statistical tests for universality of script invisibility

Usage:
    python3 compare_models.py --results-dir results/ --output results/comparison.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


MODEL_ORDER = [
    "qwen3_8b",
    "llama_8b",
    "gemma_12b",
    "qwen2_72b",
]

MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama_8b": "Llama-3.1-8B",
    "gemma_12b": "Gemma-3-12B",
    "qwen2_72b": "Qwen2-72B",
}

METRICS = ["l2_norm", "entropy", "sparsity", "kurtosis"]


def load_results(results_dir):
    """Load all JSON result files from a directory."""
    results = {}
    results_path = Path(results_dir)

    for json_file in sorted(results_path.glob("*.json")):
        if json_file.name == "comparison.json":
            continue
        model_key = json_file.stem
        with open(json_file) as f:
            data = json.load(f)
        results[model_key] = data
        print(f"Loaded: {json_file.name} ({len(data.get('english', []))} layers)")

    return results


def compute_summary(results):
    """Compute per-model summary statistics."""
    summary = {}

    for model_key, data in results.items():
        model_summary = {"model": model_key}

        for lang in ["english", "nko"]:
            if lang not in data or not data[lang]:
                continue

            layers = data[lang]
            num_layers = len(layers)

            for metric in METRICS:
                values = [l[metric] for l in layers]
                model_summary[f"{lang}_{metric}_mean"] = np.mean(values)
                model_summary[f"{lang}_{metric}_std"] = np.std(values)

                # Mid-layer value (where reasoning happens)
                mid = num_layers // 2
                model_summary[f"{lang}_{metric}_mid"] = values[mid]

        # Translation tax: ratio of NKo to English metrics
        for metric in METRICS:
            en_key = f"english_{metric}_mean"
            nko_key = f"nko_{metric}_mean"
            if en_key in model_summary and nko_key in model_summary:
                en_val = model_summary[en_key]
                nko_val = model_summary[nko_key]
                if en_val != 0:
                    model_summary[f"tax_{metric}"] = nko_val / en_val
                else:
                    model_summary[f"tax_{metric}"] = float("inf")

        # Overall invisibility score: average normalized difference across all metrics
        diffs = []
        for metric in METRICS:
            en_key = f"english_{metric}_mean"
            nko_key = f"nko_{metric}_mean"
            if en_key in model_summary and nko_key in model_summary:
                en_val = model_summary[en_key]
                nko_val = model_summary[nko_key]
                if en_val != 0:
                    diffs.append(abs(nko_val - en_val) / abs(en_val))
        model_summary["invisibility_score"] = np.mean(diffs) if diffs else 0.0

        summary[model_key] = model_summary

    return summary


def print_comparison_table(summary):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-MODEL COMPARISON: N'Ko vs English Activation Patterns")
    print("=" * 100)

    header = f"{'Model':<20}"
    for metric in METRICS:
        header += f" {'EN ' + metric:<12} {'NKo ' + metric:<12} {'Tax':<8}"
    header += f" {'Invis.':<8}"
    print(header)
    print("-" * 100)

    for model_key in MODEL_ORDER:
        if model_key not in summary:
            continue
        s = summary[model_key]
        label = MODEL_LABELS.get(model_key, model_key)
        row = f"{label:<20}"
        for metric in METRICS:
            en = s.get(f"english_{metric}_mean", 0)
            nko = s.get(f"nko_{metric}_mean", 0)
            tax = s.get(f"tax_{metric}", 0)
            row += f" {en:>10.3f}  {nko:>10.3f}  {tax:>6.2f}x"
        row += f" {s.get('invisibility_score', 0):>6.3f}"
        print(row)


def plot_comparison(results, output_dir):
    """Generate cross-model comparison figures."""
    if plt is None:
        print("matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4-panel figure: one per metric, all models overlaid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Script Invisibility Across Architectures\nN'Ko vs English Activation Patterns",
                 fontsize=14, fontweight="bold", y=0.98)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    linestyles_en = ["-", "-", "-", "-"]
    linestyles_nko = ["--", "--", "--", "--"]

    for metric_idx, metric in enumerate(METRICS):
        ax = axes[metric_idx // 2][metric_idx % 2]

        for model_idx, model_key in enumerate(MODEL_ORDER):
            if model_key not in results:
                continue
            data = results[model_key]
            label = MODEL_LABELS.get(model_key, model_key)
            color = colors[model_idx % len(colors)]

            for lang, ls, alpha in [("english", "-", 0.5), ("nko", "--", 1.0)]:
                if lang in data and data[lang]:
                    layers_data = data[lang]
                    x = [l["layer"] for l in layers_data]
                    y = [l[metric] for l in layers_data]
                    ax.plot(x, y, color=color, linestyle=ls, alpha=alpha,
                            linewidth=2, label=f"{label} {lang.upper()}")

        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        if metric_idx == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = output_dir / "cross_model_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")
    plt.close()

    # Invisibility score bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = []
    scores = []
    for model_key in MODEL_ORDER:
        if model_key not in results:
            continue
        model_names.append(MODEL_LABELS.get(model_key, model_key))
        # Recompute from summary
        data = results[model_key]
        diffs = []
        for metric in METRICS:
            en_vals = [l[metric] for l in data.get("english", [])]
            nko_vals = [l[metric] for l in data.get("nko", [])]
            if en_vals and nko_vals:
                en_mean = np.mean(en_vals)
                nko_mean = np.mean(nko_vals)
                if en_mean != 0:
                    diffs.append(abs(nko_mean - en_mean) / abs(en_mean))
        scores.append(np.mean(diffs) if diffs else 0.0)

    bars = ax.bar(model_names, scores, color=colors[:len(model_names)])
    ax.set_ylabel("Invisibility Score\n(avg normalized EN-NKo difference)")
    ax.set_title("Script Invisibility Score by Model", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", fontsize=11)

    fig_path = output_dir / "invisibility_scores.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare brain scans across models")
    parser.add_argument("--results-dir", required=True, help="Directory with per-model JSON results")
    parser.add_argument("--output", default="results/comparison.json", help="Output comparison JSON")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print("ERROR: No result files found in", args.results_dir)
        sys.exit(1)

    summary = compute_summary(results)
    print_comparison_table(summary)

    # Plot
    figures_dir = Path(args.results_dir) / "figures"
    plot_comparison(results, figures_dir)

    # Universality test
    print("\n" + "=" * 60)
    print("UNIVERSALITY TEST")
    print("=" * 60)
    all_scores = [s.get("invisibility_score", 0) for s in summary.values()]
    if all_scores:
        print(f"  Models scanned: {len(all_scores)}")
        print(f"  Invisibility scores: {[round(s, 3) for s in all_scores]}")
        print(f"  Mean: {np.mean(all_scores):.3f}")
        print(f"  Std:  {np.std(all_scores):.3f}")
        if all(s > 0.05 for s in all_scores):
            print("  RESULT: Script invisibility appears UNIVERSAL across tested architectures")
        else:
            print("  RESULT: Script invisibility varies by architecture")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "models": {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                       for kk, vv in v.items()}
                   for k, v in summary.items()},
        "universality": {
            "all_invisible": all(s > 0.05 for s in all_scores) if all_scores else False,
            "mean_score": float(np.mean(all_scores)) if all_scores else 0,
            "std_score": float(np.std(all_scores)) if all_scores else 0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=float)
    print(f"\nComparison saved to {output_path}")


if __name__ == "__main__":
    main()
