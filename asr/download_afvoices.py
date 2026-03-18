#!/usr/bin/env python3
"""
Download and Prepare AfVoices Bambara Dataset for N'Ko ASR Training
====================================================================
Downloads RobotsMali/afvoices (human-corrected subset, 159h, ~260K segments,
CC-BY-4.0), bridges Latin Bambara transcriptions to N'Ko script, and outputs
a JSONL manifest for downstream feature extraction and training.

The human-corrected subset has columns:
  - audio:    AudioObject (16kHz WAV via HF datasets)
  - text:     Human-validated Bambara transcription (Latin script)
  - duration: Float (seconds)
  - label-v1: Automatic label from soloni-v0 (not used)
  - label-v2: Automatic label from soloni-v2 (not used)

Output format (one JSON object per line):
  {"id": N, "bam": "latin text", "nko": "N'Ko text", "feat_id": "afvoices_NNNNNN", "duration": float}

Usage:
    # Full download + bridge (on Vast.ai)
    python3 asr/download_afvoices.py

    # Limit to first 10K for testing
    python3 asr/download_afvoices.py --limit 10000

    # Use a different cache directory
    python3 asr/download_afvoices.py --cache-dir /data/hf_cache

    # Include model-annotated subset (label-v2 as text, no human-corrected text)
    python3 asr/download_afvoices.py --include-model-annotated
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add repo root so nko.transliterate is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from nko.transliterate import transliterate, detect_script


# ── Helpers ─────────────────────────────────────────────────────

def has_nko(text: str) -> bool:
    """Check if text contains N'Ko characters (U+07C0-U+07FF)."""
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)


def bridge_latin_to_nko(latin_text: str) -> str | None:
    """Convert Latin Bambara text to N'Ko script.

    Returns None if the bridge produces no N'Ko characters (e.g. input is
    all punctuation or digits with no Bambara content).
    """
    if not latin_text or not latin_text.strip():
        return None
    try:
        nko = transliterate(latin_text.strip(), source="latin", target="nko")
        if nko and has_nko(nko):
            return nko
        return None
    except Exception:
        return None


def compute_audio_duration_from_array(audio_dict: dict) -> float:
    """Compute duration from an audio dict with 'array' and 'sampling_rate'."""
    arr = audio_dict.get("array")
    sr = audio_dict.get("sampling_rate", 16000)
    if arr is not None and sr > 0:
        return len(arr) / sr
    return 0.0


# ── Main Pipeline ──────────────────────────────────────────────

def download_and_prepare(
    output_path: str = "/workspace/afvoices_nko.jsonl",
    cache_dir: str = "/workspace/hf_data",
    limit: int = 0,
    include_model_annotated: bool = False,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    min_text_chars: int = 2,
):
    """Download afvoices, bridge transcriptions to N'Ko, write JSONL manifest."""

    # Lazy import so the script fails fast with a clear message if datasets is missing
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not found. Install with: pip install datasets>=3.6.0")
        sys.exit(1)

    t0 = time.time()

    # ── 1. Load dataset ────────────────────────────────────────
    # The actual HuggingFace repo is "RobotsMali/afvoices".
    # The user may refer to it as "bam-afvoices" but that is not the HF name.
    dataset_name = "RobotsMali/afvoices"

    print(f"Loading {dataset_name} (human-corrected subset)...")
    print(f"  Cache dir: {cache_dir}")
    try:
        ds_human = load_dataset(
            dataset_name,
            "human-corrected",
            cache_dir=cache_dir,
        )
    except Exception as e:
        # Fallback: try the name the user provided in case HF resolves it
        print(f"  Failed with 'human-corrected' subset: {e}")
        print("  Trying RobotsMali/bam-afvoices as fallback...")
        try:
            ds_human = load_dataset(
                "RobotsMali/bam-afvoices",
                cache_dir=cache_dir,
            )
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            print("  Trying RobotsMali/afvoices without subset name...")
            ds_human = load_dataset(dataset_name, cache_dir=cache_dir)

    # Use train split (253K samples, 155h)
    train_data = ds_human["train"]
    total = len(train_data)
    print(f"  Loaded {total:,} training samples")

    if "test" in ds_human:
        test_data = ds_human["test"]
        print(f"  Test split: {len(test_data):,} samples")

    # Optionally load model-annotated subset
    model_data = None
    if include_model_annotated:
        print("Loading model-annotated subset...")
        try:
            ds_model = load_dataset(
                dataset_name,
                "model-annotated",
                cache_dir=cache_dir,
            )
            model_data = ds_model["train"]
            print(f"  Loaded {len(model_data):,} model-annotated samples")
        except Exception as e:
            print(f"  Could not load model-annotated subset: {e}")

    # ── 2. Process and bridge ──────────────────────────────────
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_processed": 0,
        "bridged": 0,
        "skipped_empty_text": 0,
        "skipped_short_text": 0,
        "skipped_bridge_fail": 0,
        "skipped_duration": 0,
    }

    process_limit = limit if limit > 0 else total
    process_limit = min(process_limit, total)

    print(f"\nProcessing {process_limit:,} samples...")
    print(f"  Output: {output_path}")
    print(f"  Duration filter: {min_duration}s - {max_duration}s")
    print(f"  Min text chars: {min_text_chars}")

    sample_pairs = []  # Store first 5 for display

    with open(output_file, "w", encoding="utf-8") as fout:
        for i in range(process_limit):
            sample = train_data[i]

            # Get transcription text. The human-corrected subset has 'text'.
            # Fall back to label-v2 if text is missing.
            bam_text = (sample.get("text") or "").strip()
            if not bam_text:
                bam_text = (sample.get("label-v2") or "").strip()

            if not bam_text:
                stats["skipped_empty_text"] += 1
                stats["total_processed"] += 1
                if (i + 1) % 1000 == 0:
                    _log_progress(i + 1, process_limit, stats, t0)
                continue

            if len(bam_text) < min_text_chars:
                stats["skipped_short_text"] += 1
                stats["total_processed"] += 1
                if (i + 1) % 1000 == 0:
                    _log_progress(i + 1, process_limit, stats, t0)
                continue

            # Get duration. The dataset has a 'duration' column, but we also
            # compute from audio array as a fallback.
            duration = sample.get("duration")
            if duration is None or duration <= 0:
                if "audio" in sample and sample["audio"] is not None:
                    duration = compute_audio_duration_from_array(sample["audio"])
                else:
                    duration = 0.0

            if duration < min_duration or duration > max_duration:
                stats["skipped_duration"] += 1
                stats["total_processed"] += 1
                if (i + 1) % 1000 == 0:
                    _log_progress(i + 1, process_limit, stats, t0)
                continue

            # Bridge Latin Bambara to N'Ko
            nko_text = bridge_latin_to_nko(bam_text)
            if nko_text is None:
                stats["skipped_bridge_fail"] += 1
                stats["total_processed"] += 1
                if (i + 1) % 1000 == 0:
                    _log_progress(i + 1, process_limit, stats, t0)
                continue

            # Write entry
            entry = {
                "id": stats["bridged"],
                "bam": bam_text,
                "nko": nko_text,
                "feat_id": f"afvoices_{stats['bridged']:06d}",
                "duration": round(duration, 3),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stats["bridged"] += 1
            stats["total_processed"] += 1

            # Collect samples for display
            if len(sample_pairs) < 5:
                sample_pairs.append(entry)

            if (i + 1) % 1000 == 0:
                _log_progress(i + 1, process_limit, stats, t0)

    # ── 3. Process model-annotated subset (if requested) ───────
    if model_data is not None:
        model_limit = min(len(model_data), limit if limit > 0 else len(model_data))
        print(f"\nProcessing {model_limit:,} model-annotated samples...")

        with open(output_file, "a", encoding="utf-8") as fout:
            for i in range(model_limit):
                sample = model_data[i]

                # Model-annotated uses label-v2 as text (no human 'text' field)
                bam_text = (sample.get("label-v2") or sample.get("text") or "").strip()
                if not bam_text or len(bam_text) < min_text_chars:
                    stats["skipped_empty_text"] += 1
                    stats["total_processed"] += 1
                    continue

                duration = sample.get("duration") or 0.0
                if duration < min_duration or duration > max_duration:
                    stats["skipped_duration"] += 1
                    stats["total_processed"] += 1
                    continue

                nko_text = bridge_latin_to_nko(bam_text)
                if nko_text is None:
                    stats["skipped_bridge_fail"] += 1
                    stats["total_processed"] += 1
                    continue

                entry = {
                    "id": stats["bridged"],
                    "bam": bam_text,
                    "nko": nko_text,
                    "feat_id": f"afvoices_{stats['bridged']:06d}",
                    "duration": round(duration, 3),
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["bridged"] += 1
                stats["total_processed"] += 1

                if (i + 1) % 1000 == 0:
                    _log_progress(i + 1, model_limit, stats, t0, prefix="model-annotated")

    # ── 4. Summary ─────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"AfVoices Download + Bridge Complete")
    print(f"{'='*60}")
    print(f"  Total processed:     {stats['total_processed']:,}")
    print(f"  Successfully bridged: {stats['bridged']:,}")
    print(f"  Skipped (empty text): {stats['skipped_empty_text']:,}")
    print(f"  Skipped (short text): {stats['skipped_short_text']:,}")
    print(f"  Skipped (bridge fail):{stats['skipped_bridge_fail']:,}")
    print(f"  Skipped (duration):   {stats['skipped_duration']:,}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Output: {output_path}")

    if sample_pairs:
        print(f"\nSample pairs:")
        for s in sample_pairs:
            print(f"  [{s['feat_id']}] dur={s['duration']:.1f}s")
            print(f"    BAM: {s['bam'][:80]}")
            print(f"    NKO: {s['nko'][:80]}")

    # Write stats alongside the output
    stats_path = str(output_file.with_suffix(".stats.json"))
    with open(stats_path, "w") as f:
        json.dump({**stats, "elapsed_seconds": round(elapsed, 1)}, f, indent=2)
    print(f"  Stats: {stats_path}")

    return stats


def _log_progress(current: int, total: int, stats: dict, t0: float, prefix: str = "human-corrected"):
    """Log progress every N samples."""
    elapsed = time.time() - t0
    rate = current / max(elapsed, 0.01)
    eta = (total - current) / max(rate, 0.01) / 60
    bridge_rate = stats["bridged"] / max(current, 1) * 100
    print(
        f"  [{prefix}] {current:,}/{total:,} "
        f"({rate:.0f}/s, ETA {eta:.1f}m) "
        f"bridged={stats['bridged']:,} ({bridge_rate:.1f}%)",
        flush=True,
    )


# ── Test split processing ──────────────────────────────────────

def process_test_split(
    output_path: str = "/workspace/afvoices_nko_test.jsonl",
    cache_dir: str = "/workspace/hf_data",
    min_text_chars: int = 2,
):
    """Process the test split separately for evaluation."""
    from datasets import load_dataset

    print("Loading afvoices test split...")
    ds = load_dataset("RobotsMali/afvoices", "human-corrected", cache_dir=cache_dir)
    if "test" not in ds:
        print("No test split found.")
        return

    test_data = ds["test"]
    total = len(test_data)
    print(f"  Test: {total:,} samples")

    bridged = 0
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for i in range(total):
            sample = test_data[i]
            bam_text = (sample.get("text") or "").strip()
            if not bam_text or len(bam_text) < min_text_chars:
                continue

            duration = sample.get("duration") or 0.0
            nko_text = bridge_latin_to_nko(bam_text)
            if nko_text is None:
                continue

            entry = {
                "id": bridged,
                "bam": bam_text,
                "nko": nko_text,
                "feat_id": f"afvoices_test_{bridged:06d}",
                "duration": round(duration, 3),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            bridged += 1

    print(f"  Test bridged: {bridged:,}/{total:,}")
    print(f"  Output: {output_path}")


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download RobotsMali/afvoices and bridge Bambara to N'Ko"
    )
    parser.add_argument(
        "--output", default="/workspace/afvoices_nko.jsonl",
        help="Output JSONL path (default: /workspace/afvoices_nko.jsonl)",
    )
    parser.add_argument(
        "--cache-dir", default="/workspace/hf_data",
        help="HuggingFace cache directory (default: /workspace/hf_data)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max samples to process (0 = all, default: 0)",
    )
    parser.add_argument(
        "--include-model-annotated", action="store_true",
        help="Also process the model-annotated subset (label-v2 text)",
    )
    parser.add_argument(
        "--include-test", action="store_true",
        help="Also process the test split into a separate file",
    )
    parser.add_argument(
        "--test-output", default="/workspace/afvoices_nko_test.jsonl",
        help="Output path for test split JSONL",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.3,
        help="Minimum audio duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--min-text-chars", type=int, default=2,
        help="Minimum transcription length in characters (default: 2)",
    )
    args = parser.parse_args()

    stats = download_and_prepare(
        output_path=args.output,
        cache_dir=args.cache_dir,
        limit=args.limit,
        include_model_annotated=args.include_model_annotated,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_text_chars=args.min_text_chars,
    )

    if args.include_test:
        process_test_split(
            output_path=args.test_output,
            cache_dir=args.cache_dir,
            min_text_chars=args.min_text_chars,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
