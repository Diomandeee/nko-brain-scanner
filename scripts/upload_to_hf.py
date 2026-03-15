#!/usr/bin/env python3
"""Upload fused NKo model and dataset to HuggingFace Hub.

Prerequisites:
  huggingface-cli login  (on Mac5)

Usage:
  python3 scripts/upload_to_hf.py --model   # Upload fused model
  python3 scripts/upload_to_hf.py --dataset  # Upload eval + training data
  python3 scripts/upload_to_hf.py --all      # Upload everything
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo

HF_USER = "Diomande"
MODEL_REPO = f"{HF_USER}/nko-qwen3-8b-v2"
DATASET_REPO = f"{HF_USER}/nko-parallel-corpus"

FUSED_MODEL_DIR = os.path.expanduser("~/nko-brain-scanner/fused-extended-nko-qwen3")
ADAPTER_DIR = os.path.expanduser("~/nko-brain-scanner/adapters-extended")
EVAL_DIR = os.path.expanduser("~/nko-brain-scanner/eval")
RESULTS_DIR = os.path.expanduser("~/nko-brain-scanner/results")
MODEL_CARD = os.path.expanduser("~/nko-brain-scanner/model_card.md")


def upload_model(api):
    """Upload fused model to HF Hub."""
    print(f"Creating repo: {MODEL_REPO}")
    create_repo(MODEL_REPO, repo_type="model", exist_ok=True)

    # Upload model card
    if os.path.exists(MODEL_CARD):
        api.upload_file(
            path_or_fileobj=MODEL_CARD,
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
        )
        print("  Uploaded README.md (model card)")

    # Upload fused model files
    print(f"  Uploading fused model from {FUSED_MODEL_DIR}...")
    api.upload_folder(
        folder_path=FUSED_MODEL_DIR,
        repo_id=MODEL_REPO,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
    )
    print("  Fused model uploaded")

    # Upload adapter separately
    if os.path.exists(ADAPTER_DIR):
        print(f"  Uploading adapter from {ADAPTER_DIR}...")
        api.upload_folder(
            folder_path=ADAPTER_DIR,
            repo_id=MODEL_REPO,
            path_in_repo="adapter",
            allow_patterns=["*.safetensors", "*.json"],
        )
        print("  Adapter uploaded")

    print(f"\nModel available at: https://huggingface.co/{MODEL_REPO}")


def upload_dataset(api):
    """Upload eval sets and results as a dataset."""
    print(f"Creating dataset repo: {DATASET_REPO}")
    create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

    # Upload eval data
    for fname in ["english_eval.jsonl", "nko_eval.jsonl"]:
        fpath = os.path.join(EVAL_DIR, fname)
        if os.path.exists(fpath):
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"eval/{fname}",
                repo_id=DATASET_REPO,
                repo_type="dataset",
            )
            print(f"  Uploaded eval/{fname}")

    # Upload results
    for fname in ["profiler_corrected.json", "brain_scan_8b.json",
                   "benchmark_comparison.json", "tokenizer_comparison.json",
                   "admissibility_comparison.json", "fsm_validation.json",
                   "round_trip_eval.json", "v2_20_prompts.json"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"results/{fname}",
                repo_id=DATASET_REPO,
                repo_type="dataset",
            )
            print(f"  Uploaded results/{fname}")

    print(f"\nDataset available at: https://huggingface.co/datasets/{DATASET_REPO}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.model, args.dataset, args.all]):
        parser.print_help()
        return

    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")

    if args.model or args.all:
        upload_model(api)
    if args.dataset or args.all:
        upload_dataset(api)


if __name__ == "__main__":
    main()
