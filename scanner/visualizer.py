"""
Publication-quality visualization for activation profiles and heatmaps.

Generates:
- Activation profile comparison charts (English vs NKO, 4 metrics)
- Side-by-side (i,j) heatmaps (the hero figure)
- Optimal config comparison tables
- Token compression ratio charts

Usage::

    from scanner.visualizer import Visualizer

    viz = Visualizer()
    viz.plot_activation_comparison(english_profile, nko_profile, "figures/activation_comparison.png")
    viz.plot_heatmap_comparison(english_heatmap, nko_heatmap, "figures/heatmap_hero.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None  # type: ignore
    sns = None  # type: ignore

from .activation_profiler import LayerMetrics


class Visualizer:
    """Publication-quality figure generator for the brain scanner project."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: Tuple[int, int] = (14, 8)):
        if plt is not None:
            try:
                plt.style.use(style)
            except OSError:
                plt.style.use("seaborn-v0_8")
        self.figsize = figsize
        self.english_color = "#2196F3"  # Blue
        self.nko_color = "#FF9800"      # Orange
        self.improvement_cmap = "RdBu_r"  # Red = improvement, Blue = degradation

    def plot_activation_comparison(
        self,
        english_metrics: List[LayerMetrics],
        nko_metrics: List[LayerMetrics],
        output_path: str,
        title: str = "Per-Layer Activation Profiles: English vs N'Ko",
    ) -> None:
        """
        Plot 4-panel activation profile comparison.

        Panels: L2 Norm, Shannon Entropy, Sparsity, Kurtosis
        X-axis: Layer index, Y-axis: Metric value
        Two lines: English (blue) and NKO (orange)
        """
        if plt is None:
            raise ImportError("matplotlib required")

        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        metrics = [
            ("L2 Norm", "l2_norm", "Activation Magnitude"),
            ("Shannon Entropy", "entropy", "Distribution Spread (bits)"),
            ("Sparsity", "sparsity", "Fraction Near-Zero"),
            ("Excess Kurtosis", "kurtosis", "Peakedness"),
        ]

        layers_en = [m.layer_idx for m in english_metrics]
        layers_nko = [m.layer_idx for m in nko_metrics]

        for ax, (name, attr, ylabel) in zip(axes.flat, metrics):
            en_vals = [getattr(m, attr) for m in english_metrics]
            nko_vals = [getattr(m, attr) for m in nko_metrics]

            ax.plot(layers_en, en_vals, color=self.english_color, linewidth=1.5,
                    label="English", alpha=0.8)
            ax.plot(layers_nko, nko_vals, color=self.nko_color, linewidth=1.5,
                    label="N'Ko", alpha=0.8)

            # Shade the difference
            ax.fill_between(
                layers_en, en_vals, nko_vals,
                alpha=0.15, color="gray", label="_nolegend_"
            )

            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.set_title(name, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved activation comparison to {output_path}")

    def plot_heatmap_comparison(
        self,
        english_heatmap: np.ndarray,
        nko_heatmap: np.ndarray,
        output_path: str,
        baseline_score: float = 0.0,
        title: str = "Circuit Duplication Heatmaps: English vs N'Ko",
    ) -> None:
        """
        Plot side-by-side (i,j) heatmaps — the hero figure.

        Left: English, Right: NKO
        Color: Red = improvement over baseline, Blue = degradation
        """
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn required")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(self.figsize[0] * 1.5, self.figsize[1]),
                                             gridspec_kw={"width_ratios": [1, 1, 1]})
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        # Normalize relative to baseline
        en_relative = english_heatmap - baseline_score
        nko_relative = nko_heatmap - baseline_score

        vmax = max(np.nanmax(np.abs(en_relative)), np.nanmax(np.abs(nko_relative)))
        vmin = -vmax

        # English heatmap
        sns.heatmap(en_relative, ax=ax1, cmap=self.improvement_cmap,
                    vmin=vmin, vmax=vmax, center=0,
                    xticklabels=10, yticklabels=10,
                    cbar_kws={"label": "Score Change"})
        ax1.set_title("English", fontweight="bold", fontsize=14)
        ax1.set_xlabel("j (end layer)")
        ax1.set_ylabel("i (start layer)")

        # NKO heatmap
        sns.heatmap(nko_relative, ax=ax2, cmap=self.improvement_cmap,
                    vmin=vmin, vmax=vmax, center=0,
                    xticklabels=10, yticklabels=10,
                    cbar_kws={"label": "Score Change"})
        ax2.set_title("N'Ko", fontweight="bold", fontsize=14)
        ax2.set_xlabel("j (end layer)")
        ax2.set_ylabel("i (start layer)")

        # Difference heatmap (NKO - English)
        diff = nko_relative - en_relative
        diff_vmax = np.nanmax(np.abs(diff))
        sns.heatmap(diff, ax=ax3, cmap="PiYG",
                    vmin=-diff_vmax, vmax=diff_vmax, center=0,
                    xticklabels=10, yticklabels=10,
                    cbar_kws={"label": "N'Ko Advantage"})
        ax3.set_title("Difference (N'Ko - English)", fontweight="bold", fontsize=14)
        ax3.set_xlabel("j (end layer)")
        ax3.set_ylabel("i (start layer)")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap comparison to {output_path}")

    def plot_token_compression(
        self,
        comparisons: List[dict],
        output_path: str,
        title: str = "Token Compression: N'Ko Morphemes vs BPE",
    ) -> None:
        """
        Bar chart comparing token counts: NKO morpheme tokenizer vs HF BPE.

        Parameters
        ----------
        comparisons : List[dict]
            Output from NkoTokenizer.compare_with_hf() for multiple texts.
        """
        if plt is None:
            raise ImportError("matplotlib required")

        texts = [c["text"][:30] + "..." if len(c["text"]) > 30 else c["text"] for c in comparisons]
        nko_counts = [c["nko_token_count"] for c in comparisons]
        hf_counts = [c["hf_token_count"] for c in comparisons]

        x = np.arange(len(texts))
        width = 0.35

        fig, ax = plt.subplots(figsize=self.figsize)
        bars1 = ax.bar(x - width/2, nko_counts, width, label="N'Ko Morphemes",
                       color=self.nko_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, hf_counts, width, label="HF BPE",
                       color=self.english_color, alpha=0.8)

        ax.set_xlabel("Input Text")
        ax.set_ylabel("Token Count")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(texts, rotation=45, ha="right", fontsize=8)
        ax.legend()

        # Add compression ratio annotations
        for i, c in enumerate(comparisons):
            ratio = c["compression_ratio"]
            ax.annotate(f"{ratio:.1f}x", xy=(x[i], max(nko_counts[i], hf_counts[i]) + 1),
                       ha="center", fontsize=7, color="gray")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved token compression chart to {output_path}")
