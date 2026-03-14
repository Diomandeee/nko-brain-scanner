#!/usr/bin/env python3
"""
Main experiment orchestrator for the NKO Brain Scanner.

Runs both experiments:
  1. Activation Profiling — per-layer hidden state comparison (English vs NKO)
  2. Circuit Duplication Heatmaps — (i,j) sweep for both scripts

Usage on Vast.ai::

    # Experiment 1: Activation profiling (~2 hours)
    python -m scanner.run_experiment --experiment activation --corpus data/parallel_corpus.jsonl

    # Experiment 2: Coarse heatmap scan (~6 hours)
    python -m scanner.run_experiment --experiment heatmap --mode coarse

    # Experiment 2: Full heatmap sweep (~40 hours)
    python -m scanner.run_experiment --experiment heatmap --mode full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None  # type: ignore

from .activation_profiler import ActivationProfiler
from .layer_duplicator import LayerDuplicator
from .heatmap_generator import HeatmapGenerator
from .visualizer import Visualizer


RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_model(model_name: str = "Qwen/Qwen2-72B-Instruct", quantize_4bit: bool = True):
    """Load model with output_hidden_states=True. Supports BnB 4-bit and FP16."""
    print(f"Loading {model_name} (4-bit={quantize_4bit})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    return model, tokenizer


def load_corpus(path: str, limit: int = 100) -> List[dict]:
    """Load parallel corpus JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("nko") and entry.get("english"):
                pairs.append(entry)
            if len(pairs) >= limit:
                break
    print(f"Loaded {len(pairs)} parallel pairs from {path}")
    return pairs


def run_activation_profiling(model, tokenizer, corpus_path: str):
    """
    Experiment 1: Per-layer activation profiling.

    Compare English vs NKO activation patterns across all layers.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Activation Profiling")
    print("=" * 60)

    corpus = load_corpus(corpus_path, limit=100)
    profiler = ActivationProfiler(model, tokenizer)

    # Profile English inputs
    print("\nProfiling English inputs...")
    english_profiles = []
    for i, pair in enumerate(corpus):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(corpus)}]")
        profile = profiler.profile(pair["english"])
        english_profiles.append(profile)

    # Profile NKO inputs
    print("\nProfiling N'Ko inputs...")
    nko_profiles = []
    for i, pair in enumerate(corpus):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(corpus)}]")
        profile = profiler.profile(pair["nko"])
        nko_profiles.append(profile)

    # Average across all inputs
    english_avg = profiler.average_profiles(english_profiles)
    nko_avg = profiler.average_profiles(nko_profiles)

    # Save raw metrics
    out_dir = RESULTS_DIR / "activation_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "english_profile.json", "w") as f:
        json.dump([m.to_dict() for m in english_avg], f, indent=2)
    with open(out_dir / "nko_profile.json", "w") as f:
        json.dump([m.to_dict() for m in nko_avg], f, indent=2)

    # Generate visualization
    viz = Visualizer()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    viz.plot_activation_comparison(
        english_avg, nko_avg,
        str(FIGURES_DIR / "activation_comparison.png"),
    )

    # Print summary
    print("\n--- Summary ---")
    print(f"Layers profiled: {len(english_avg)}")
    print(f"Inputs per script: {len(corpus)}")

    # Find layers with biggest differences
    for metric_name in ["l2_norm", "entropy", "sparsity", "kurtosis"]:
        diffs = []
        for en, nk in zip(english_avg, nko_avg):
            en_val = getattr(en, metric_name)
            nk_val = getattr(nk, metric_name)
            diffs.append((en.layer_idx, abs(en_val - nk_val), en_val, nk_val))
        diffs.sort(key=lambda x: x[1], reverse=True)
        top = diffs[:3]
        print(f"\n{metric_name} — biggest differences:")
        for layer, diff, en_val, nk_val in top:
            print(f"  Layer {layer}: EN={en_val:.4f} NKO={nk_val:.4f} (diff={diff:.4f})")


def run_heatmap_sweep(model, tokenizer, mode: str = "coarse"):
    """
    Experiment 2: Circuit duplication heatmap sweep.

    Run the (i,j) sweep for both English and NKO inputs.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: Heatmap Sweep ({mode})")
    print("=" * 60)

    # Load probes
    probes_dir = Path(__file__).parent.parent / "probes"
    with open(probes_dir / "math_probes.json") as f:
        math_probes = json.load(f)["probes"]
    with open(probes_dir / "semantic_probes.json") as f:
        semantic_probes = json.load(f)["probes"]

    gen = HeatmapGenerator(model, tokenizer, math_probes, semantic_probes)

    # Get configurations — use step=8 for coarse (fewer configs, faster)
    if mode == "coarse":
        configs = gen.duplicator.coarse_configs(step=8)
        max_math = 4   # 4 of 16 math probes
        max_sem = 4    # 4 of 16 semantic probes
    else:
        configs = gen.duplicator.valid_configs()
        max_math = 0   # all probes
        max_sem = 0

    probes_per_config = (max_math or len(math_probes)) + (max_sem or len(semantic_probes))
    est_seconds = len(configs) * probes_per_config * 15  # ~15s per probe (50 tokens on 4-bit 72B)
    print(f"Configurations to evaluate: {len(configs)}", flush=True)
    print(f"Probes per config: {probes_per_config} ({max_math or 'all'} math + {max_sem or 'all'} semantic)", flush=True)
    print(f"Estimated time: {est_seconds / 3600:.1f} hours per script", flush=True)

    # Run for English
    print("\n--- English Sweep ---", flush=True)
    checkpoint_dir = str(RESULTS_DIR / "heatmaps")
    en_results = gen.run_sweep(configs, script="english",
                                checkpoint_dir=checkpoint_dir, checkpoint_every=20,
                                max_math_probes=max_math, max_sem_probes=max_sem)
    gen.save_results(en_results, f"{checkpoint_dir}/english_{mode}.npz")

    # Run for NKO
    print("\n--- N'Ko Sweep ---", flush=True)
    nko_results = gen.run_sweep(configs, script="nko",
                                 checkpoint_dir=checkpoint_dir, checkpoint_every=20,
                                 max_math_probes=max_math, max_sem_probes=max_sem)
    gen.save_results(nko_results, f"{checkpoint_dir}/nko_{mode}.npz")

    # Generate heatmaps
    en_heatmap = gen.results_to_heatmap(en_results)
    nko_heatmap = gen.results_to_heatmap(nko_results)

    viz = Visualizer()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    viz.plot_heatmap_comparison(
        en_heatmap, nko_heatmap,
        str(FIGURES_DIR / f"heatmap_{mode}.png"),
    )

    # Find optimal configs
    print("\n--- Optimal Configurations ---")
    en_best = max(en_results, key=lambda r: r.combined_score)
    nko_best = max(nko_results, key=lambda r: r.combined_score)
    print(f"English: ({en_best.i}, {en_best.j}) score={en_best.combined_score:.4f}")
    print(f"N'Ko:    ({nko_best.i}, {nko_best.j}) score={nko_best.combined_score:.4f}")

    # Improvement zones
    en_improved = sum(1 for r in en_results if r.combined_score > 0)
    nko_improved = sum(1 for r in nko_results if r.combined_score > 0)
    print(f"\nImprovement zones: English={en_improved}/{len(configs)}, "
          f"N'Ko={nko_improved}/{len(configs)}")


def main():
    parser = argparse.ArgumentParser(description="NKO Brain Scanner Experiment Runner")
    parser.add_argument("--experiment", choices=["activation", "heatmap", "both"],
                       default="both", help="Which experiment to run")
    parser.add_argument("--mode", choices=["coarse", "full"], default="coarse",
                       help="Heatmap sweep mode (coarse=every 4th layer)")
    parser.add_argument("--corpus", default="data/parallel_corpus.jsonl",
                       help="Path to parallel corpus")
    parser.add_argument("--model", default="Qwen/Qwen2-72B-Instruct",
                       help="Model name or path")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Load in FP16 instead of BnB 4-bit")
    args = parser.parse_args()

    if torch is None:
        print("ERROR: torch is required. Run on Vast.ai with GPU.")
        return

    model, tokenizer = load_model(args.model, quantize_4bit=not args.no_quantize)

    if args.experiment in ("activation", "both"):
        run_activation_profiling(model, tokenizer, args.corpus)

    if args.experiment in ("heatmap", "both"):
        run_heatmap_sweep(model, tokenizer, args.mode)

    print("\nAll experiments complete. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
