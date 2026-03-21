#!/usr/bin/env python3
"""
Log brain scan results to Weights & Biases.
Creates comparison charts across models for the paper.

Usage:
    python3 log_to_wandb.py --results-dir results/
    python3 log_to_wandb.py --results-dir results/ --project nko-brain-scanner
"""

import argparse
import json
import os
from pathlib import Path

try:
    import wandb
except ImportError:
    print("pip install wandb")
    exit(1)


def load_results(results_dir):
    """Load all JSON result files from the results directory."""
    models = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        model_name = f.stem.replace("_", "-")
        models[model_name] = data
    return models


def log_single_model(model_name, data, run):
    """Log a single model's activation profiles to W&B."""
    for lang in ["english", "nko"]:
        if lang not in data:
            continue
        layers = data[lang] if isinstance(data[lang], list) else data[lang].get("layers", [])
        for layer in layers:
            idx = layer.get("layer_idx", layer.get("layer", 0))
            run.log({
                f"{lang}/l2_norm": layer.get("l2_norm", 0),
                f"{lang}/entropy": layer.get("entropy", 0),
                f"{lang}/kurtosis": layer.get("kurtosis", 0),
                f"{lang}/sparsity": layer.get("sparsity", 0),
                "layer": idx,
            })

    # Log per-layer ratios
    en_raw = data.get("english", [])
    nko_raw = data.get("nko", [])
    en_list = en_raw if isinstance(en_raw, list) else en_raw.get("layers", [])
    nko_list = nko_raw if isinstance(nko_raw, list) else nko_raw.get("layers", [])
    en_layers = {l.get("layer_idx", l.get("layer", i)): l for i, l in enumerate(en_list)}
    nko_layers = {l.get("layer_idx", l.get("layer", i)): l for i, l in enumerate(nko_list)}

    for idx in sorted(en_layers.keys()):
        if idx in nko_layers:
            en = en_layers[idx]
            nko = nko_layers[idx]
            ratio = en.get("l2_norm", 1) / max(nko.get("l2_norm", 1), 0.001)
            run.log({
                "ratio/l2_norm": ratio,
                "ratio/entropy_delta": nko.get("entropy", 0) - en.get("entropy", 0),
                "ratio/sparsity_delta": nko.get("sparsity", 0) - en.get("sparsity", 0),
                "layer": idx,
            })


def log_comparison(models, project):
    """Log cross-model comparison to W&B."""
    run = wandb.init(
        project=project,
        name="cross-model-comparison",
        config={"models": list(models.keys()), "num_models": len(models)},
    )

    # Summary table: translation tax per model
    table_data = []
    for model_name, data in models.items():
        en_raw = data.get("english", [])
        nko_raw = data.get("nko", [])
        en_layers = en_raw if isinstance(en_raw, list) else en_raw.get("layers", [])
        nko_layers = nko_raw if isinstance(nko_raw, list) else nko_raw.get("layers", [])

        if not en_layers or not nko_layers:
            continue

        # Compute average translation tax
        ratios = []
        for en, nko in zip(en_layers, nko_layers):
            if nko.get("l2_norm", 0) > 0.001:
                ratios.append(en.get("l2_norm", 0) / nko["l2_norm"])

        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        embed_ratio = ratios[0] if ratios else 0
        mid_idx = len(ratios) // 2
        mid_ratio = ratios[mid_idx] if ratios else 0
        output_ratio = ratios[-1] if ratios else 0

        # Sparsity at embedding
        en_sparsity = en_layers[0].get("sparsity", 0)
        nko_sparsity = nko_layers[0].get("sparsity", 0)

        # Kurtosis at output
        en_kurtosis = en_layers[-1].get("kurtosis", 0)
        nko_kurtosis = nko_layers[-1].get("kurtosis", 0)
        kurtosis_deficit = (1 - nko_kurtosis / max(en_kurtosis, 0.001)) * 100

        num_layers = len(en_layers)

        table_data.append([
            model_name, num_layers, f"{avg_ratio:.2f}x",
            f"{embed_ratio:.2f}x", f"{mid_ratio:.2f}x", f"{output_ratio:.2f}x",
            f"{en_sparsity:.1%}", f"{nko_sparsity:.1%}",
            f"{kurtosis_deficit:.1f}%",
        ])

        # Log per-model summary
        run.summary[f"{model_name}/avg_tax"] = avg_ratio
        run.summary[f"{model_name}/embed_tax"] = embed_ratio
        run.summary[f"{model_name}/output_tax"] = output_ratio
        run.summary[f"{model_name}/kurtosis_deficit"] = kurtosis_deficit
        run.summary[f"{model_name}/num_layers"] = num_layers

    # Create comparison table
    table = wandb.Table(
        columns=["Model", "Layers", "Avg Tax", "Embed Tax", "Mid Tax", "Output Tax",
                 "EN Sparsity", "NKo Sparsity", "Kurtosis Deficit"],
        data=table_data,
    )
    run.log({"comparison_table": table})

    # Create per-layer line charts for each metric
    for metric in ["l2_norm", "entropy", "kurtosis", "sparsity"]:
        chart_data = []
        for model_name, data in models.items():
            for lang in ["english", "nko"]:
                layers = data.get(lang, {}).get("layers", [])
                for layer in layers:
                    chart_data.append([
                        model_name, lang, layer["layer_idx"],
                        layer.get(metric, 0),
                    ])

        if chart_data:
            table = wandb.Table(
                columns=["model", "language", "layer", metric],
                data=chart_data,
            )
            run.log({f"{metric}_by_layer": table})

    run.finish()
    print(f"W&B comparison logged to project: {project}")


def main():
    parser = argparse.ArgumentParser(description="Log brain scan results to W&B")
    parser.add_argument("--results-dir", required=True, help="Directory with JSON results")
    parser.add_argument("--project", default="nko-brain-scanner", help="W&B project name")
    parser.add_argument("--per-model", action="store_true", help="Also log individual model runs")
    args = parser.parse_args()

    models = load_results(args.results_dir)
    print(f"Loaded {len(models)} model results: {list(models.keys())}")

    if not models:
        print("No results found")
        return

    if args.per_model:
        for model_name, data in models.items():
            run = wandb.init(
                project=args.project,
                name=f"brain-scan-{model_name}",
                config={"model": model_name},
                reinit=True,
            )
            log_single_model(model_name, data, run)
            run.finish()
            print(f"  Logged: {model_name}")

    log_comparison(models, args.project)


if __name__ == "__main__":
    main()
