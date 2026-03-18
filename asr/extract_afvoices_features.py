#!/usr/bin/env python3
"""
Pre-extract Whisper Features from AfVoices for Fast N'Ko ASR Training
======================================================================
Loads RobotsMali/afvoices from HuggingFace cache, extracts Whisper large-v3
encoder features, downsamples 4x, and saves as float16 .pt files.

This mirrors the pattern in extract_human_features.py but targets the
afvoices dataset (human-corrected subset, ~253K train + ~6.7K test).

Features are saved as:
    /workspace/afvoices_features/train_NNNNNN.pt   (float16, shape [375, 1280] max)
    /workspace/afvoices_features/test_NNNNNN.pt

Each .pt file contains the 4x-downsampled Whisper encoder output:
    - Original: [1500, 1280] (30s padded)
    - Saved: [375, 1280] (every 4th frame, float16)

Usage (on Vast.ai GPU):
    # Extract all features
    python3 asr/extract_afvoices_features.py

    # Extract only first 10K for testing
    python3 asr/extract_afvoices_features.py --limit 10000

    # Use a different Whisper model
    python3 asr/extract_afvoices_features.py --whisper-model large-v2

    # Resume (skips already-extracted files)
    python3 asr/extract_afvoices_features.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import whisper


def resample_audio(audio_tensor: torch.Tensor, source_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Resample audio tensor from source_sr to target_sr using linear interpolation.

    This avoids depending on torchaudio/librosa. For ASR feature extraction,
    linear interpolation is sufficient (Whisper's mel spectrogram is robust
    to minor resampling artifacts).
    """
    if source_sr == target_sr:
        return audio_tensor
    ratio = target_sr / source_sr
    new_len = int(len(audio_tensor) * ratio)
    return torch.nn.functional.interpolate(
        audio_tensor.unsqueeze(0).unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze()


def extract_features(
    output_dir: str = "/workspace/afvoices_features",
    cache_dir: str = "/workspace/hf_data",
    whisper_model_name: str = "large-v3",
    limit: int = 0,
    log_interval: int = 1000,
):
    """Extract Whisper encoder features from afvoices dataset."""

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not found. Install with: pip install datasets>=3.6.0")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load Whisper ────────────────────────────────────────
    print(f"Loading Whisper {whisper_model_name}...")
    model = whisper.load_model(whisper_model_name, device=device)
    model.eval()
    print("Whisper loaded.")

    # ── 2. Load dataset ────────────────────────────────────────
    print("Loading RobotsMali/afvoices (human-corrected)...")
    ds = load_dataset(
        "RobotsMali/afvoices",
        "human-corrected",
        cache_dir=cache_dir,
    )
    train_total = len(ds["train"])
    test_total = len(ds["test"]) if "test" in ds else 0
    print(f"Train: {train_total:,}, Test: {test_total:,}")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 3. Extract train features ──────────────────────────────
    train_limit = min(limit, train_total) if limit > 0 else train_total
    print(f"\nExtracting train features ({train_limit:,} samples)...")
    n_train = _extract_split(
        ds["train"], "train", train_limit, model, device, outdir, log_interval
    )
    print(f"Train: {n_train:,} features extracted to {outdir}")

    # ── 4. Extract test features ───────────────────────────────
    if test_total > 0:
        test_limit = min(limit, test_total) if limit > 0 else test_total
        print(f"\nExtracting test features ({test_limit:,} samples)...")
        n_test = _extract_split(
            ds["test"], "test", test_limit, model, device, outdir, log_interval
        )
        print(f"Test: {n_test:,} features extracted")
    else:
        n_test = 0

    print(f"\nDone. Total: {n_train + n_test:,} features in {outdir}")

    # Write a metadata file for downstream scripts
    meta_path = outdir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"dataset: RobotsMali/afvoices (human-corrected)\n")
        f.write(f"whisper_model: {whisper_model_name}\n")
        f.write(f"downsample: 4x\n")
        f.write(f"dtype: float16\n")
        f.write(f"train_features: {n_train}\n")
        f.write(f"test_features: {n_test}\n")
        f.write(f"feature_dim: 1280\n")
        f.write(f"max_frames: 375\n")
    print(f"Metadata: {meta_path}")


def _extract_split(
    split_data,
    prefix: str,
    total: int,
    model,
    device: str,
    outdir: Path,
    log_interval: int,
) -> int:
    """Extract Whisper encoder features for one split.

    Saves each sample as {prefix}_{idx:06d}.pt containing a float16 tensor
    of shape [T//4, 1280] where T is the Whisper encoder sequence length (1500).

    Skips samples where the .pt file already exists (resume-safe).
    """
    t0 = time.time()
    extracted = 0
    skipped_existing = 0
    errors = 0

    for i in range(total):
        feat_path = outdir / f"{prefix}_{i:06d}.pt"

        # Resume: skip if already extracted
        if feat_path.exists():
            skipped_existing += 1
            extracted += 1
            if (i + 1) % log_interval == 0:
                _log_extraction_progress(i + 1, total, extracted, errors, skipped_existing, t0, prefix)
            continue

        try:
            sample = split_data[i]

            # Handle audio: may be dict with 'array' and 'sampling_rate',
            # or a path string (depending on HF datasets version/config).
            audio_data = sample.get("audio")
            if audio_data is None:
                errors += 1
                if i < 5:
                    print(f"  WARNING [{prefix}][{i}]: No audio data")
                continue

            if isinstance(audio_data, dict):
                audio_array = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
            else:
                # If it's already a numpy array or tensor
                audio_array = audio_data
                sr = 16000

            audio_t = torch.tensor(audio_array, dtype=torch.float32)

            # Resample to 16kHz if needed
            if sr != 16000:
                audio_t = resample_audio(audio_t, sr, 16000)

            # Pad or trim to 30s (Whisper's expected input length)
            audio_padded = whisper.pad_or_trim(audio_t)

            # Compute log-mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128).to(device)

            # Run Whisper encoder
            with torch.no_grad():
                features = model.encoder(mel.unsqueeze(0)).squeeze(0).cpu()

            # Downsample 4x and save as float16 (same as bam-asr-early features)
            # [1500, 1280] -> [375, 1280], float32 -> float16
            torch.save(features[::4].half(), feat_path)
            extracted += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERROR [{prefix}][{i}]: {e}")

        if (i + 1) % log_interval == 0:
            _log_extraction_progress(i + 1, total, extracted, errors, skipped_existing, t0, prefix)

    elapsed = time.time() - t0
    print(
        f"  [{prefix}] COMPLETE: {extracted:,} extracted, "
        f"{skipped_existing:,} resumed, {errors:,} errors, "
        f"{elapsed:.1f}s ({elapsed/60:.1f}m)",
        flush=True,
    )
    return extracted


def _log_extraction_progress(
    current: int, total: int, extracted: int, errors: int,
    skipped: int, t0: float, prefix: str,
):
    """Log feature extraction progress."""
    elapsed = time.time() - t0
    rate = current / max(elapsed, 0.01)
    eta = (total - current) / max(rate, 0.01) / 60
    print(
        f"  [{prefix}] {current:,}/{total:,} "
        f"({rate:.1f}/s, ETA {eta:.1f}m) "
        f"extracted={extracted:,} resumed={skipped:,} errors={errors:,}",
        flush=True,
    )


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Whisper features from RobotsMali/afvoices"
    )
    parser.add_argument(
        "--output-dir", default="/workspace/afvoices_features",
        help="Output directory for .pt feature files",
    )
    parser.add_argument(
        "--cache-dir", default="/workspace/hf_data",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--whisper-model", default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max samples per split to process (0 = all)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1000,
        help="Log progress every N samples (default: 1000)",
    )
    args = parser.parse_args()

    extract_features(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        whisper_model_name=args.whisper_model,
        limit=args.limit,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
