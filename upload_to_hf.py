#!/usr/bin/env python3
"""Upload NKo-Qwen3-8B-V3 model and dataset to HuggingFace.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  # or set HF_TOKEN env var

Usage:
    # From Mac5 (where the fused model lives):
    python3 upload_to_hf.py

    # Or from Mac1, after scp'ing the model:
    python3 upload_to_hf.py --model-path ~/nko-brain-scanner/fused-v3-nko-qwen3
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "Diomande/nko-qwen3-8b-v3"
MODEL_DIR = os.path.expanduser("~/nko-brain-scanner/fused-v3-nko-qwen3")
SCANNER_DIR = os.path.expanduser("~/nko-brain-scanner")


def upload_model(api, model_path):
    """Upload fused model files."""
    print(f"Uploading model from {model_path}...")
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # Upload all model files
    api.upload_folder(
        folder_path=model_path,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Model uploaded to https://huggingface.co/{REPO_ID}")


def upload_model_card(api):
    """Upload model card."""
    card_path = os.path.join(SCANNER_DIR, "model_card.md")
    if os.path.exists(card_path):
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("Model card uploaded as README.md")


def upload_tokenizer(api):
    """Upload N'Ko BPE tokenizer files."""
    tokenizer_files = [
        ("tokenizer/bpe_vocab.json", "tokenizer/nko_bpe_vocab.json"),
        ("tokenizer/morpheme_bpe_vocab.json", "tokenizer/morpheme_bpe_vocab.json"),
        ("data/syllable_codebook.json", "tokenizer/syllable_codebook.json"),
    ]
    for local_path, repo_path in tokenizer_files:
        full_path = os.path.join(SCANNER_DIR, local_path)
        if os.path.exists(full_path):
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded {local_path} -> {repo_path}")
        else:
            print(f"Skipped {local_path} (not found)")


def upload_results(api):
    """Upload evaluation results."""
    result_files = [
        "results/profiler_v3.json",
        "results/v3_generation.json",
        "results/admissibility_comparison.json",
        "results/tokenizer_comparison.json",
        "results/round_trip_eval.json",
        "results/profiler_corrected.json",
    ]
    for f in result_files:
        full_path = os.path.join(SCANNER_DIR, f)
        if os.path.exists(full_path):
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=f,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_DIR)
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip uploading model weights (just card + results)")
    args = parser.parse_args()

    api = HfApi()
    whoami = api.whoami()
    print(f"Authenticated as: {whoami['name']}")

    if not args.skip_model:
        upload_model(api, args.model_path)

    upload_model_card(api)
    upload_tokenizer(api)
    upload_results(api)

    print(f"\nDone! Model available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
