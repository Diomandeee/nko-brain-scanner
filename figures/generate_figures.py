#!/usr/bin/env python3
"""Generate brain scan figures for the paper."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent


def load_brain_scan():
    with open(RESULTS_DIR / "brain_scan_8b.json") as f:
        return json.load(f)


def fig1_l2_comparison(data):
    """Figure 1: Per-layer L2 norms — base vs fine-tuned, English vs N'Ko."""
    layers = list(range(36))

    base_eng = [data["base"]["english"]["layer_stats"][str(i)]["l2_norm"] for i in layers]
    base_nko = [data["base"]["nko"]["layer_stats"][str(i)]["l2_norm"] for i in layers]
    ft_eng = [data["fine_tuned"]["english"]["layer_stats"][str(i)]["l2_norm"] for i in layers]
    ft_nko = [data["fine_tuned"]["nko"]["layer_stats"][str(i)]["l2_norm"] for i in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Panel A: Base model
    ax1.plot(layers, base_eng, 'o-', color='#2196F3', markersize=3, linewidth=1.5, label='English')
    ax1.plot(layers, base_nko, 's-', color='#FF5722', markersize=3, linewidth=1.5, label="N'Ko")
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('L2 Norm')
    ax1.set_title('(A) Base Qwen3-8B')
    ax1.legend(fontsize=8)
    ax1.axvspan(27.5, 35.5, alpha=0.08, color='gray')
    ax1.set_xlim(-0.5, 35.5)

    # Panel B: Fine-tuned model
    ax2.plot(layers, ft_eng, 'o-', color='#2196F3', markersize=3, linewidth=1.5, label='English')
    ax2.plot(layers, ft_nko, 's-', color='#FF5722', markersize=3, linewidth=1.5, label="N'Ko")
    ax2.set_xlabel('Layer')
    ax2.set_title('(B) Three-Stage Fine-Tuned')
    ax2.legend(fontsize=8)
    ax2.axvspan(27.5, 35.5, alpha=0.08, color='gray')
    ax2.annotate('LoRA\nadaptation\nzone', xy=(31.5, 200), fontsize=7,
                 ha='center', color='gray', style='italic')
    ax2.set_xlim(-0.5, 35.5)

    plt.tight_layout()
    out = FIGURES_DIR / "brain_scan_l2_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig2_delta(data):
    """Figure 2: Per-layer L2 delta (fine-tuned minus base) for N'Ko."""
    deltas = data["nko_deltas"]
    layers = list(range(36))
    l2_deltas = [deltas[str(i)]["l2_delta"] for i in layers]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    colors = ['#4CAF50' if d > 0 else '#F44336' if d < 0 else '#BDBDBD' for d in l2_deltas]
    bars = ax.bar(layers, l2_deltas, color=colors, width=0.7, edgecolor='none')

    # Annotate the big output layer spike
    ax.annotate('+573', xy=(35, 572.67), fontsize=9, ha='center', va='bottom',
                fontweight='bold', color='#2E7D32')

    # Annotate adaptation zone
    ax.axvspan(27.5, 34.5, alpha=0.06, color='red')
    ax.annotate('Adaptation zone\n(reduced L2)', xy=(31, -120), fontsize=7,
                ha='center', color='#C62828', style='italic')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm Change (Fine-tuned - Base)')
    ax.set_title("N'Ko Activation Delta After Three-Stage Fine-Tuning")
    ax.set_xlim(-0.5, 35.5)

    plt.tight_layout()
    out = FIGURES_DIR / "brain_scan_delta.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


def fig3_training_loss():
    """Figure 3: V3 training loss curve (bonus figure)."""
    log_path = RESULTS_DIR / "training-v3.log"
    if not log_path.exists():
        print(f"Skipping training loss figure ({log_path} not found)")
        return

    iters_train, losses_train = [], []
    iters_val, losses_val = [], []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Iter") and "Train loss" in line:
                parts = line.split(",")
                it = int(parts[0].split()[1].rstrip(":"))
                loss = float(parts[0].split("Train loss ")[-1].split(",")[0])
                iters_train.append(it)
                losses_train.append(loss)
            elif line.startswith("Iter") and "Val loss" in line:
                parts = line.split(",")
                it = int(parts[0].split()[1].rstrip(":"))
                loss = float(parts[0].split("Val loss ")[-1].split(",")[0])
                iters_val.append(it)
                losses_val.append(loss)

    if not iters_train:
        print("No training data found in log")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters_train, losses_train, '-', color='#90CAF9', linewidth=0.8, alpha=0.7, label='Train loss')
    ax.plot(iters_val, losses_val, 'o-', color='#E53935', markersize=4, linewidth=1.5, label='Val loss')

    # Mark best checkpoint
    if losses_val:
        best_idx = np.argmin(losses_val)
        ax.annotate(f'Best: {losses_val[best_idx]:.3f}\n(iter {iters_val[best_idx]})',
                    xy=(iters_val[best_idx], losses_val[best_idx]),
                    xytext=(iters_val[best_idx] + 200, losses_val[best_idx] - 0.15),
                    arrowprops=dict(arrowstyle='->', color='#1B5E20'),
                    fontsize=8, color='#1B5E20', fontweight='bold')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('V3 Training: 92K Examples (nicolingua + All Sources)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "v3_training_loss.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_brain_scan()
    fig1_l2_comparison(data)
    fig2_delta(data)
    fig3_training_loss()
    print("\nAll figures generated.")
