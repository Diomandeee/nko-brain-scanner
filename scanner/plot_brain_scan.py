#!/usr/bin/env python3
"""Generate brain scan comparison figures from 8B scan results."""

import json
import os

# Try matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, generating text-only report")

RESULTS = os.path.expanduser("~/Desktop/nko-brain-scanner/results/brain_scan_8b.json")
OUTPUT_DIR = os.path.expanduser("~/Desktop/nko-brain-scanner/figures")


def load_data():
    with open(RESULTS) as f:
        return json.load(f)


def extract_series(data, config, lang, metric):
    """Extract a metric series across layers."""
    stats = data[config][lang]["layer_stats"]
    layers = sorted(stats.keys(), key=int)
    return [int(k) for k in layers], [stats[k][metric] for k in layers]


def text_report(data):
    """Generate ASCII text comparison."""
    num_layers = data["metadata"]["num_layers"]

    print("\n" + "=" * 100)
    print("8B BRAIN SCAN: Base vs Fine-Tuned (N'Ko)")
    print("=" * 100)

    print(f"\n{'Layer':<6} {'Base L2':>10} {'FT L2':>10} {'ΔL2':>10} "
          f"{'Base Spar':>10} {'FT Spar':>10} {'ΔSpar':>10}")
    print("-" * 76)

    for i in range(num_layers):
        k = str(i)
        b = data["base"]["nko"]["layer_stats"].get(k, {})
        f = data["fine_tuned"]["nko"]["layer_stats"].get(k, {})
        d = data["nko_deltas"].get(k, {})
        marker = " ◄" if abs(d.get("l2_delta", 0)) > 10 else ""
        print(f"{i:<6} {b.get('l2_norm',0):>10.2f} {f.get('l2_norm',0):>10.2f} "
              f"{d.get('l2_delta',0):>+10.2f} "
              f"{b.get('sparsity',0):>10.4f} {f.get('sparsity',0):>10.4f} "
              f"{d.get('sparsity_delta',0):>+10.4f}{marker}")

    # English comparison
    print(f"\n{'Layer':<6} {'Eng Base L2':>12} {'Eng FT L2':>12} {'NKo Base L2':>12} {'NKo FT L2':>12}")
    print("-" * 60)
    for i in range(num_layers):
        k = str(i)
        eb = data["base"]["english"]["layer_stats"].get(k, {})
        ef = data["fine_tuned"]["english"]["layer_stats"].get(k, {})
        nb = data["base"]["nko"]["layer_stats"].get(k, {})
        nf = data["fine_tuned"]["nko"]["layer_stats"].get(k, {})
        print(f"{i:<6} {eb.get('l2_norm',0):>12.2f} {ef.get('l2_norm',0):>12.2f} "
              f"{nb.get('l2_norm',0):>12.2f} {nf.get('l2_norm',0):>12.2f}")

    # Adaptation zone analysis
    print("\n\nADAPTATION ZONE ANALYSIS")
    print("=" * 60)
    nko_deltas = data["nko_deltas"]
    adapted_layers = [(int(k), v["l2_delta"]) for k, v in nko_deltas.items()
                      if abs(v["l2_delta"]) > 1.0]
    adapted_layers.sort(key=lambda x: x[0])

    if adapted_layers:
        first = adapted_layers[0][0]
        last = adapted_layers[-1][0]
        print(f"  Adaptation zone: layers {first}-{last}")
        print(f"  Frozen layers: 0-{first-1}")
        print(f"  Total adapted layers: {len(adapted_layers)}")
        print(f"  Max reduction: Layer {min(adapted_layers, key=lambda x: x[1])[0]} "
              f"(ΔL2={min(adapted_layers, key=lambda x: x[1])[1]:+.2f})")
        print(f"  Max increase: Layer {max(adapted_layers, key=lambda x: x[1])[0]} "
              f"(ΔL2={max(adapted_layers, key=lambda x: x[1])[1]:+.2f})")


def plot_figures(data):
    """Generate matplotlib comparison figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_layers = data["metadata"]["num_layers"]

    # Color scheme
    base_color = '#2196F3'     # Blue
    ft_color = '#FF5722'       # Orange-red
    eng_color = '#4CAF50'      # Green
    nko_color = '#9C27B0'      # Purple

    # Figure 1: L2 Norm comparison (hero figure)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel A: N'Ko activations before/after
    layers_b, l2_b = extract_series(data, "base", "nko", "l2_norm")
    layers_f, l2_f = extract_series(data, "fine_tuned", "nko", "l2_norm")

    axes[0].plot(layers_b, l2_b, '-o', color=base_color, label='Base Model', markersize=4, linewidth=2)
    axes[0].plot(layers_f, l2_f, '-s', color=ft_color, label='Fine-Tuned (3-Stage)', markersize=4, linewidth=2)

    # Highlight adaptation zone
    axes[0].axvspan(28, 35, alpha=0.1, color='red', label='LoRA Adaptation Zone')
    axes[0].set_ylabel('L2 Norm (activation magnitude)', fontsize=11)
    axes[0].set_title("A. N'Ko Activation Profiles: Before vs After Fine-Tuning", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Panel B: English vs N'Ko on fine-tuned model
    _, eng_ft = extract_series(data, "fine_tuned", "english", "l2_norm")
    _, nko_ft = extract_series(data, "fine_tuned", "nko", "l2_norm")

    axes[1].plot(layers_f, eng_ft, '-^', color=eng_color, label='English (Fine-Tuned)', markersize=4, linewidth=2)
    axes[1].plot(layers_f, nko_ft, '-v', color=nko_color, label="N'Ko (Fine-Tuned)", markersize=4, linewidth=2)
    axes[1].axvspan(28, 35, alpha=0.1, color='red')
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('L2 Norm', fontsize=11)
    axes[1].set_title("B. English vs N'Ko Activation Profiles (Fine-Tuned Model)", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'brain_scan_l2_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: brain_scan_l2_comparison.png")
    plt.close()

    # Figure 2: Delta plot (change per layer)
    fig, ax = plt.subplots(figsize=(12, 5))

    deltas = data["nko_deltas"]
    layers_d = sorted([int(k) for k in deltas.keys()])
    delta_vals = [deltas[str(k)]["l2_delta"] for k in layers_d]

    colors = ['#FF5722' if d > 0 else '#2196F3' for d in delta_vals]
    ax.bar(layers_d, delta_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('ΔL2 Norm (Fine-Tuned - Base)', fontsize=11)
    ax.set_title("N'Ko Activation Change per Layer After Fine-Tuning", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate adaptation zone
    ax.axvspan(27.5, 35.5, alpha=0.08, color='red')
    ax.text(31.5, max(delta_vals) * 0.9, 'LoRA\nZone', ha='center', fontsize=9, color='red', alpha=0.7)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'brain_scan_delta.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: brain_scan_delta.png")
    plt.close()

    # Figure 3: Sparsity comparison
    fig, ax = plt.subplots(figsize=(12, 5))

    _, spar_b = extract_series(data, "base", "nko", "sparsity")
    _, spar_f = extract_series(data, "fine_tuned", "nko", "sparsity")
    _, spar_eng = extract_series(data, "fine_tuned", "english", "sparsity")

    ax.plot(layers_b, spar_b, '-o', color=base_color, label="N'Ko Base", markersize=3, linewidth=2)
    ax.plot(layers_f, spar_f, '-s', color=ft_color, label="N'Ko Fine-Tuned", markersize=3, linewidth=2)
    ax.plot(layers_f, spar_eng, '-^', color=eng_color, label='English Fine-Tuned', markersize=3, linewidth=2)
    ax.axvspan(28, 35, alpha=0.1, color='red')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Sparsity (fraction near-zero)', fontsize=11)
    ax.set_title("Activation Sparsity Across Layers", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'brain_scan_sparsity.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: brain_scan_sparsity.png")
    plt.close()


def main():
    data = load_data()
    text_report(data)

    if HAS_MPL:
        print("\nGenerating figures...")
        plot_figures(data)
        print(f"\nAll figures saved to {OUTPUT_DIR}")
    else:
        print("\nInstall matplotlib for figures: pip3 install matplotlib")


if __name__ == "__main__":
    main()
