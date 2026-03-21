#!/usr/bin/env python3
"""
Train a LoRA adapter on N'Ko-transliterated SFT data.

Takes the translated N'Ko SFT JSONL (from translate_sft_to_nko.py) and trains
a LoRA adapter using MLX, matching the same hyperparameters as the English
cognitive twin training (KARL adapter pipeline).

This script:
  1. Prepares data in MLX LoRA format (train.jsonl / valid.jsonl / test.jsonl)
  2. Runs mlx_lm LoRA training
  3. Saves the adapter to adapters-nko/

Designed to run on Mac5 (Apple Silicon with MLX) or any machine with mlx-lm.

Hyperparameters (matched to KARL English twin training):
  - LoRA rank: 8 (MLX default)
  - Num layers: 4
  - Batch size: 1
  - Learning rate: 1e-5
  - Max seq length: 256
  - Iterations: 500 (adjustable)

Usage:
    python3 train_nko_adapter.py \
        --input data/nko_sft.jsonl \
        --base-model mlx-community/gemma-3-1b-it-4bit \
        --output-adapter adapters-nko \
        --iters 500

    # Quick test run (50 iters)
    python3 train_nko_adapter.py \
        --input data/nko_sft.jsonl \
        --base-model mlx-community/gemma-3-1b-it-4bit \
        --output-adapter adapters-nko \
        --iters 50 \
        --dry-run
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def prepare_data(input_path: str, data_dir: str, val_split: float = 0.1) -> dict:
    """
    Convert a flat SFT JSONL into MLX LoRA train/valid/test splits.

    MLX LoRA expects:
      - train.jsonl: one JSON object per line with "messages" key
      - valid.jsonl: same format
      - test.jsonl: same format (can be copy of valid)

    Each line: {"messages": [{"role": "...", "content": "..."}, ...]}

    Returns stats dict.
    """
    examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                # Validate structure
                if "messages" in ex and len(ex["messages"]) >= 2:
                    examples.append(ex)
            except json.JSONDecodeError:
                continue

    if not examples:
        print(f"ERROR: No valid examples found in {input_path}")
        sys.exit(1)

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(examples)

    # Split
    n_val = max(1, int(len(examples) * val_split))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    # Ensure minimum sizes
    if len(train_examples) < 2:
        print(f"WARNING: Only {len(train_examples)} training examples. Results may be poor.")

    os.makedirs(data_dir, exist_ok=True)

    # Write splits
    for split_name, split_data in [
        ("train", train_examples),
        ("valid", val_examples),
        ("test", val_examples),  # MLX requires test.jsonl, mirror valid
    ]:
        path = os.path.join(data_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "total": len(examples),
        "train": len(train_examples),
        "valid": len(val_examples),
        "test": len(val_examples),
        "input_path": input_path,
        "data_dir": data_dir,
    }
    print(f"Data prepared: {stats['train']} train, {stats['valid']} valid, {stats['test']} test")
    return stats


def run_training(
    base_model: str,
    data_dir: str,
    adapter_path: str,
    iters: int = 500,
    batch_size: int = 1,
    num_layers: int = 4,
    max_seq_length: int = 256,
    learning_rate: float = 1e-5,
    dry_run: bool = False,
) -> dict:
    """
    Run MLX LoRA training via subprocess.

    Uses the same command format as KARL trainer:
        python3 -m mlx_lm lora --model X --data Y --adapter-path Z --train ...
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--train",
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--max-seq-length", str(max_seq_length),
        "--learning-rate", str(learning_rate),
    ]

    print(f"\nTraining command:")
    print(f"  {' '.join(cmd)}")
    print()

    if dry_run:
        print("[DRY RUN] Would execute the above command. Skipping.")
        return {"status": "dry_run", "command": cmd}

    # Check mlx_lm is available
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("ERROR: mlx_lm not installed. Run: pip install mlx-lm")
        print("This script must run on Apple Silicon (Mac5) with MLX.")
        sys.exit(1)

    os.makedirs(adapter_path, exist_ok=True)

    print(f"Starting training: {iters} iterations, batch_size={batch_size}, "
          f"lr={learning_rate}, num_layers={num_layers}")
    print(f"Adapter output: {adapter_path}")
    print("-" * 60)

    result = subprocess.run(
        cmd,
        capture_output=False,  # Stream output live
        text=True,
    )

    if result.returncode != 0:
        print(f"\nERROR: Training failed with exit code {result.returncode}")
        return {"status": "failed", "exit_code": result.returncode}

    # Verify adapter files were created
    adapter_files = list(Path(adapter_path).glob("*.safetensors"))
    config_files = list(Path(adapter_path).glob("*.json"))

    print(f"\nTraining complete.")
    print(f"  Adapter files: {len(adapter_files)} safetensors, {len(config_files)} configs")
    print(f"  Location: {adapter_path}")

    return {
        "status": "success",
        "exit_code": 0,
        "adapter_path": adapter_path,
        "adapter_files": [str(f) for f in adapter_files],
        "config_files": [str(f) for f in config_files],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train N'Ko LoRA adapter for cognitive twin experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training run
  python3 train_nko_adapter.py \\
      --input data/nko_sft.jsonl \\
      --base-model mlx-community/gemma-3-1b-it-4bit \\
      --output-adapter adapters-nko \\
      --iters 500

  # Quick test (50 iters, dry run)
  python3 train_nko_adapter.py \\
      --input data/nko_sft.jsonl \\
      --base-model mlx-community/gemma-3-1b-it-4bit \\
      --output-adapter adapters-nko \\
      --iters 50 --dry-run

  # Match English twin model exactly
  python3 train_nko_adapter.py \\
      --input data/nko_sft.jsonl \\
      --base-model mlx-community/Qwen3-8B-8bit \\
      --output-adapter adapters-nko \\
      --iters 1000 --learning-rate 5e-6
        """,
    )

    parser.add_argument("--input", required=True,
                        help="N'Ko SFT JSONL (output of translate_sft_to_nko.py)")
    parser.add_argument("--base-model", required=True,
                        help="HuggingFace model ID (e.g., mlx-community/gemma-3-1b-it-4bit)")
    parser.add_argument("--output-adapter", default="adapters-nko",
                        help="Directory to save the trained LoRA adapter (default: adapters-nko)")
    parser.add_argument("--iters", type=int, default=500,
                        help="Number of training iterations (default: 500, KARL default)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size (default: 1, KARL default)")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of LoRA layers (default: 4, KARL default)")
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Max sequence length (default: 256, KARL default)")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5, KARL default)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory for prepared train/valid/test splits "
                             "(default: <output-adapter>_data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare data and print command, but don't train")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data preparation (use existing train/valid/test in data-dir)")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    adapter_path = Path(args.output_adapter)
    if not adapter_path.is_absolute():
        adapter_path = script_dir / adapter_path

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = str(adapter_path) + "_data"
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = script_dir / data_dir_path
    data_dir = str(data_dir_path)

    print("=" * 60)
    print("N'Ko Cognitive Twin - LoRA Adapter Training")
    print("=" * 60)
    print(f"  Input:          {input_path}")
    print(f"  Base model:     {args.base_model}")
    print(f"  Output adapter: {adapter_path}")
    print(f"  Data dir:       {data_dir}")
    print(f"  Iterations:     {args.iters}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Num layers:     {args.num_layers}")
    print(f"  Max seq len:    {args.max_seq_length}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Val split:      {args.val_split}")
    print(f"  Dry run:        {args.dry_run}")
    print()

    # Step 1: Prepare data
    if not args.skip_data_prep:
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            print("Run translate_sft_to_nko.py first to generate N'Ko SFT data.")
            sys.exit(1)

        data_stats = prepare_data(str(input_path), data_dir, args.val_split)
    else:
        # Verify data files exist
        for split in ["train", "valid", "test"]:
            p = os.path.join(data_dir, f"{split}.jsonl")
            if not os.path.exists(p):
                print(f"ERROR: --skip-data-prep but {p} does not exist")
                sys.exit(1)
        data_stats = {"status": "skipped"}
        print("Data preparation skipped (--skip-data-prep)")

    # Step 2: Train
    result = run_training(
        base_model=args.base_model,
        data_dir=data_dir,
        adapter_path=str(adapter_path),
        iters=args.iters,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        dry_run=args.dry_run,
    )

    # Step 3: Save metadata
    if result.get("status") == "success":
        meta = {
            "experiment": "C_nko_cognitive_twin",
            "base_model": args.base_model,
            "adapter_path": str(adapter_path),
            "hyperparameters": {
                "iters": args.iters,
                "batch_size": args.batch_size,
                "num_layers": args.num_layers,
                "max_seq_length": args.max_seq_length,
                "learning_rate": args.learning_rate,
                "lora_rank": 8,  # MLX default
            },
            "data": data_stats,
            "matched_to": "KARL English twin adapter (karl_trainer.py)",
        }
        meta_path = os.path.join(str(adapter_path), "nko_training_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nMetadata saved to {meta_path}")

    # Print next steps
    print("\n" + "=" * 60)
    if result.get("status") == "dry_run":
        print("DRY RUN COMPLETE. To train for real, remove --dry-run flag.")
    elif result.get("status") == "success":
        print("TRAINING COMPLETE. Next steps:")
        print(f"  1. Run comparison:")
        print(f"     python3 compare_twins.py \\")
        print(f"         --model {args.base_model} \\")
        print(f"         --english-adapter ~/adapters/latest/karl-v4 \\")
        print(f"         --nko-adapter {adapter_path} \\")
        print(f"         --prompts eval_prompts.jsonl \\")
        print(f"         --output results/twin_comparison.json")
    else:
        print("TRAINING FAILED. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
