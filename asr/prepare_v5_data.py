#!/usr/bin/env python3
"""
V5 Data Preparation Pipeline — All Available Bambara/Manding Audio
====================================================================
Downloads, bridges, feature-extracts, and splits ALL available datasets
into a unified training manifest for the V5 full-scale ASR model.

Datasets:
  1. RobotsMali/afvoices (human-corrected, ~612h, ~253K segments)
  2. RobotsMali/bam-asr-early (~37h, ~13K train + ~3.5K test)
  3. MALIBA-AI/bambara-asr (if available)
  4. sudoping01/nko-asr (if access granted)
  5. mozilla-foundation/common_voice_17_0 bam subset (~5h)

Pipeline:
  1. Download each dataset via HuggingFace
  2. Bridge Latin Bambara transcriptions to N'Ko via nko.transliterate
  3. Extract Whisper large-v3 encoder features (frozen, 4x downsample, float16)
  4. Create unified train/val/test splits (90/5/5)
  5. Write manifest files with feature paths + N'Ko labels

Output:
  /workspace/v5_data/
    manifests/
      train.jsonl      — 90% of all data
      val.jsonl         — 5% of all data
      test.jsonl        — 5% of all data (+ dedicated test splits)
      stats.json        — corpus statistics
    features/
      afvoices_NNNNNN.pt
      bamasr_NNNNNN.pt
      cv_NNNNNN.pt
      ...

Usage:
    # Full pipeline (GPU required for feature extraction)
    python3 asr/prepare_v5_data.py

    # Download + bridge only (no GPU needed)
    python3 asr/prepare_v5_data.py --skip-features

    # Resume (skips existing features)
    python3 asr/prepare_v5_data.py

    # Limit for testing
    python3 asr/prepare_v5_data.py --limit 1000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add repo root for nko.transliterate
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nko.transliterate import transliterate
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("WARNING: nko.transliterate not available. Install from ~/Desktop/NKo")


# ── Helpers ─────────────────────────────────────────────────────

def has_nko(text: str) -> bool:
    """Check if text contains N'Ko characters (U+07C0-U+07FF)."""
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)


def bridge_latin_to_nko(latin_text: str) -> str | None:
    """Convert Latin Bambara text to N'Ko script."""
    if not BRIDGE_AVAILABLE or not latin_text or not latin_text.strip():
        return None
    try:
        nko = transliterate(latin_text.strip(), source="latin", target="nko")
        if nko and has_nko(nko):
            return nko
        return None
    except Exception:
        return None


def resample_audio(audio_tensor, source_sr: int, target_sr: int = 16000):
    """Resample audio tensor from source_sr to target_sr."""
    import torch
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


# ── Dataset Loaders ─────────────────────────────────────────────

def load_afvoices(cache_dir: str, limit: int = 0) -> list:
    """Load RobotsMali/afvoices (human-corrected subset)."""
    from datasets import load_dataset

    print("\n[1/5] Loading RobotsMali/afvoices (human-corrected)...")
    try:
        ds = load_dataset("RobotsMali/afvoices", "human-corrected", cache_dir=cache_dir)
    except Exception as e:
        print(f"  Failed with subset 'human-corrected': {e}")
        try:
            ds = load_dataset("RobotsMali/afvoices", cache_dir=cache_dir)
        except Exception as e2:
            print(f"  Failed entirely: {e2}")
            return []

    entries = []
    train_data = ds["train"]
    total = min(len(train_data), limit) if limit > 0 else len(train_data)
    print(f"  Processing {total:,} train samples...")

    bridged = 0
    skipped = 0
    t0 = time.time()

    for i in range(total):
        sample = train_data[i]
        bam_text = (sample.get("text") or "").strip()
        if not bam_text:
            bam_text = (sample.get("label-v2") or "").strip()

        if not bam_text or len(bam_text) < 2:
            skipped += 1
            continue

        duration = sample.get("duration") or 0.0
        if duration < 0.3 or duration > 30.0:
            skipped += 1
            continue

        nko = bridge_latin_to_nko(bam_text)
        if nko is None:
            skipped += 1
            continue

        entries.append({
            "source": "afvoices",
            "source_idx": i,
            "bam": bam_text,
            "nko": nko,
            "feat_id": f"afvoices_{bridged:06d}",
            "duration": round(duration, 3),
            "split": "train",
        })
        bridged += 1

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1:,}/{total:,} bridged={bridged:,} ({rate:.0f}/s)", flush=True)

    # Also process test split if available
    if "test" in ds:
        test_data = ds["test"]
        test_total = min(len(test_data), limit // 10) if limit > 0 else len(test_data)
        print(f"  Processing {test_total:,} test samples...")

        for i in range(test_total):
            sample = test_data[i]
            bam_text = (sample.get("text") or "").strip()
            if not bam_text or len(bam_text) < 2:
                continue
            duration = sample.get("duration") or 0.0
            if duration < 0.3 or duration > 30.0:
                continue
            nko = bridge_latin_to_nko(bam_text)
            if nko is None:
                continue

            entries.append({
                "source": "afvoices_test",
                "source_idx": i,
                "bam": bam_text,
                "nko": nko,
                "feat_id": f"afvoices_test_{len([e for e in entries if e['source'] == 'afvoices_test']):06d}",
                "duration": round(duration, 3),
                "split": "test",
            })

    elapsed = time.time() - t0
    print(f"  AfVoices: {len(entries):,} entries ({elapsed:.0f}s)")
    return entries


def load_bam_asr_early(cache_dir: str, limit: int = 0) -> list:
    """Load RobotsMali/bam-asr-early."""
    from datasets import load_dataset

    print("\n[2/5] Loading RobotsMali/bam-asr-early...")
    try:
        ds = load_dataset("RobotsMali/bam-asr-early", cache_dir=cache_dir)
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    entries = []
    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        split_data = ds[split_name]
        total = min(len(split_data), limit) if limit > 0 else len(split_data)
        print(f"  Processing {split_name}: {total:,} samples...")

        bridged = 0
        for i in range(total):
            sample = split_data[i]
            bam_text = (sample.get("sentence") or sample.get("text") or "").strip()
            if not bam_text or len(bam_text) < 2:
                continue

            # Compute duration from audio
            audio = sample.get("audio")
            duration = 0.0
            if audio and isinstance(audio, dict):
                arr = audio.get("array")
                sr = audio.get("sampling_rate", 16000)
                if arr is not None:
                    duration = len(arr) / sr

            if duration < 0.3 or duration > 30.0:
                continue

            nko = bridge_latin_to_nko(bam_text)
            if nko is None:
                continue

            entries.append({
                "source": f"bamasr_{split_name}",
                "source_idx": i,
                "bam": bam_text,
                "nko": nko,
                "feat_id": f"bamasr_{split_name}_{bridged:06d}",
                "duration": round(duration, 3),
                "split": "test" if split_name == "test" else "train",
            })
            bridged += 1

    print(f"  bam-asr-early: {len(entries):,} entries")
    return entries


def load_common_voice_bam(cache_dir: str, limit: int = 0) -> list:
    """Load Common Voice Bambara subset."""
    from datasets import load_dataset

    print("\n[3/5] Loading Common Voice Bambara...")
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "bm",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed (may need HF token or access grant): {e}")
        return []

    entries = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        split_data = ds[split_name]
        total = min(len(split_data), limit) if limit > 0 else len(split_data)
        print(f"  Processing {split_name}: {total:,} samples...")

        bridged = 0
        for i in range(total):
            sample = split_data[i]
            bam_text = (sample.get("sentence") or "").strip()
            if not bam_text or len(bam_text) < 2:
                continue

            audio = sample.get("audio")
            duration = 0.0
            if audio and isinstance(audio, dict):
                arr = audio.get("array")
                sr = audio.get("sampling_rate", 16000)
                if arr is not None:
                    duration = len(arr) / sr

            if duration < 0.3 or duration > 30.0:
                continue

            nko = bridge_latin_to_nko(bam_text)
            if nko is None:
                continue

            target_split = "test" if split_name in ("test", "validation") else "train"
            entries.append({
                "source": f"cv_{split_name}",
                "source_idx": i,
                "bam": bam_text,
                "nko": nko,
                "feat_id": f"cv_{split_name}_{bridged:06d}",
                "duration": round(duration, 3),
                "split": target_split,
            })
            bridged += 1

    print(f"  Common Voice: {len(entries):,} entries")
    return entries


def load_maliba_bambara(cache_dir: str, limit: int = 0) -> list:
    """Try to load MALIBA-AI Bambara ASR dataset."""
    from datasets import load_dataset

    print("\n[4/5] Trying MALIBA-AI/bambara-asr...")
    candidates = [
        "MALIBA-AI/bambara-asr",
        "MALIBA-AI/bam-asr",
        "MALIBA-AI/bambara_speech",
    ]
    for name in candidates:
        try:
            ds = load_dataset(name, cache_dir=cache_dir)
            print(f"  Loaded {name}")
            entries = []
            for split_name in ds.keys():
                split_data = ds[split_name]
                total = min(len(split_data), limit) if limit > 0 else len(split_data)
                bridged = 0
                for i in range(total):
                    sample = split_data[i]
                    bam_text = (sample.get("sentence") or sample.get("text") or "").strip()
                    if not bam_text or len(bam_text) < 2:
                        continue
                    audio = sample.get("audio")
                    duration = 0.0
                    if audio and isinstance(audio, dict):
                        arr = audio.get("array")
                        sr = audio.get("sampling_rate", 16000)
                        if arr is not None:
                            duration = len(arr) / sr
                    if duration < 0.3 or duration > 30.0:
                        continue
                    nko = bridge_latin_to_nko(bam_text)
                    if nko is None:
                        continue
                    entries.append({
                        "source": f"maliba_{split_name}",
                        "source_idx": i,
                        "bam": bam_text,
                        "nko": nko,
                        "feat_id": f"maliba_{split_name}_{bridged:06d}",
                        "duration": round(duration, 3),
                        "split": "test" if split_name == "test" else "train",
                    })
                    bridged += 1
            print(f"  MALIBA: {len(entries):,} entries")
            return entries
        except Exception as e:
            print(f"  {name}: {e}")
            continue

    print("  No MALIBA dataset found (expected)")
    return []


def load_nko_asr(cache_dir: str, limit: int = 0) -> list:
    """Try to load sudoping01/nko-asr (gated, needs access approval)."""
    from datasets import load_dataset

    print("\n[5/5] Trying sudoping01/nko-asr (gated)...")
    try:
        ds = load_dataset("sudoping01/nko-asr", cache_dir=cache_dir)
        entries = []
        for split_name in ds.keys():
            split_data = ds[split_name]
            total = min(len(split_data), limit) if limit > 0 else len(split_data)
            bridged = 0
            for i in range(total):
                sample = split_data[i]
                # This dataset may already have N'Ko text
                nko_text = (sample.get("sentence") or sample.get("text") or "").strip()
                bam_text = ""

                # Check if text is already N'Ko
                if has_nko(nko_text):
                    nko = nko_text
                else:
                    bam_text = nko_text
                    nko = bridge_latin_to_nko(bam_text)
                    if nko is None:
                        continue

                audio = sample.get("audio")
                duration = 0.0
                if audio and isinstance(audio, dict):
                    arr = audio.get("array")
                    sr = audio.get("sampling_rate", 16000)
                    if arr is not None:
                        duration = len(arr) / sr

                if duration < 0.3 or duration > 30.0:
                    continue

                entries.append({
                    "source": f"nkoasr_{split_name}",
                    "source_idx": i,
                    "bam": bam_text,
                    "nko": nko,
                    "feat_id": f"nkoasr_{split_name}_{bridged:06d}",
                    "duration": round(duration, 3),
                    "split": "test" if split_name == "test" else "train",
                })
                bridged += 1

        print(f"  nko-asr: {len(entries):,} entries")
        return entries
    except Exception as e:
        print(f"  Not available (gated or missing): {e}")
        return []


# ── Feature Extraction ─────────────────────────────────────────

def extract_features_for_entries(
    entries: list,
    datasets_cache: dict,
    output_dir: Path,
    log_interval: int = 1000,
):
    """Extract Whisper large-v3 encoder features for all entries.

    Args:
        entries: list of entry dicts with 'source', 'source_idx', 'feat_id'
        datasets_cache: dict mapping source prefix to loaded HF dataset split
        output_dir: directory to save .pt feature files
    """
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nExtracting Whisper features on {device}...")

    print("Loading Whisper large-v3...")
    model = whisper.load_model("large-v3", device=device)
    model.eval()
    print("Whisper loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    extracted = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, entry in enumerate(entries):
        feat_path = output_dir / f"{entry['feat_id']}.pt"

        # Resume: skip existing
        if feat_path.exists():
            skipped += 1
            extracted += 1
            if (i + 1) % log_interval == 0:
                _log_feat_progress(i + 1, total, extracted, skipped, errors, t0)
            continue

        source = entry["source"]
        source_idx = entry["source_idx"]

        try:
            # Look up the right dataset split
            ds_split = datasets_cache.get(source)
            if ds_split is None:
                errors += 1
                continue

            sample = ds_split[source_idx]
            audio_data = sample.get("audio")
            if audio_data is None:
                errors += 1
                continue

            if isinstance(audio_data, dict):
                audio_array = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
            else:
                audio_array = audio_data
                sr = 16000

            audio_t = torch.tensor(audio_array, dtype=torch.float32)
            if sr != 16000:
                audio_t = resample_audio(audio_t, sr, 16000)

            audio_padded = whisper.pad_or_trim(audio_t)
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128).to(device)

            with torch.no_grad():
                features = model.encoder(mel.unsqueeze(0)).squeeze(0).cpu()

            # 4x downsample + float16
            torch.save(features[::4].half(), feat_path)
            extracted += 1

        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"  ERROR [{entry['feat_id']}]: {e}")

        if (i + 1) % log_interval == 0:
            _log_feat_progress(i + 1, total, extracted, skipped, errors, t0)

    elapsed = time.time() - t0
    print(f"\nFeature extraction complete:")
    print(f"  Extracted: {extracted:,} (resumed: {skipped:,})")
    print(f"  Errors: {errors:,}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    return extracted, errors


def _log_feat_progress(current, total, extracted, skipped, errors, t0):
    elapsed = time.time() - t0
    rate = current / max(elapsed, 0.01)
    eta = (total - current) / max(rate, 0.01) / 60
    print(
        f"  {current:,}/{total:,} ({rate:.1f}/s, ETA {eta:.1f}m) "
        f"extracted={extracted:,} resumed={skipped:,} errors={errors:,}",
        flush=True,
    )


# ── Mel Spectrogram Extraction (for LoRA training) ────────────

def extract_mels_for_entries(
    entries: list,
    datasets_cache: dict,
    output_dir: Path,
    log_interval: int = 1000,
):
    """Extract and cache mel spectrograms for all entries.

    Unlike extract_features_for_entries (which runs the Whisper encoder and
    saves encoder output), this saves the RAW mel spectrograms. This is needed
    for V5 LoRA training because the encoder is being fine-tuned and must
    process the mel at each forward pass.

    Mel specs are much smaller than encoder features (128x3000 float16 = 768KB
    vs 1500x1280 float32 = 7.7MB per sample), and the extraction is GPU-free.

    Saves: {feat_id}.mel.pt  containing float16 tensor [128, 3000]
    """
    import torch
    import whisper

    print(f"\nExtracting mel spectrograms (no GPU needed)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    extracted = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, entry in enumerate(entries):
        mel_path = output_dir / f"{entry['feat_id']}.mel.pt"

        # Resume: skip existing
        if mel_path.exists():
            skipped += 1
            extracted += 1
            if (i + 1) % log_interval == 0:
                _log_feat_progress(i + 1, total, extracted, skipped, errors, t0)
            continue

        source = entry["source"]
        source_idx = entry["source_idx"]

        try:
            ds_split = datasets_cache.get(source)
            if ds_split is None:
                errors += 1
                continue

            sample = ds_split[source_idx]
            audio_data = sample.get("audio")
            if audio_data is None:
                errors += 1
                continue

            if isinstance(audio_data, dict):
                audio_array = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
            else:
                audio_array = audio_data
                sr = 16000

            audio_t = torch.tensor(audio_array, dtype=torch.float32)
            if sr != 16000:
                audio_t = resample_audio(audio_t, sr, 16000)

            audio_padded = whisper.pad_or_trim(audio_t)
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128)

            # Save as float16 (128x3000 = 768KB vs 7.7MB for encoder features)
            torch.save(mel.half(), mel_path)
            extracted += 1

        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"  ERROR [{entry['feat_id']}]: {e}")

        if (i + 1) % log_interval == 0:
            _log_feat_progress(i + 1, total, extracted, skipped, errors, t0)

    elapsed = time.time() - t0
    print(f"\nMel extraction complete:")
    print(f"  Extracted: {extracted:,} (resumed: {skipped:,})")
    print(f"  Errors: {errors:,}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    return extracted, errors


# ── Splitting + Manifest ──────────────────────────────────────

def create_splits(entries: list, seed: int = 42):
    """Split entries into train/val/test (90/5/5).

    Entries that came from dedicated test splits stay in test.
    Everything else is split 90/5/5.
    """
    # Separate dedicated test entries
    test_entries = [e for e in entries if e.get("split") == "test"]
    pool_entries = [e for e in entries if e.get("split") != "test"]

    # Shuffle pool
    rng = random.Random(seed)
    rng.shuffle(pool_entries)

    n = len(pool_entries)
    val_size = max(int(n * 0.05), 1)
    test_from_pool = max(int(n * 0.05), 1)

    val_entries = pool_entries[:val_size]
    test_from_pool_entries = pool_entries[val_size:val_size + test_from_pool]
    train_entries = pool_entries[val_size + test_from_pool:]

    # Combine test splits
    all_test = test_entries + test_from_pool_entries

    return train_entries, val_entries, all_test


def write_manifests(train, val, test, output_dir: Path):
    """Write train/val/test manifest JSONL files."""
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for name, entries in [("train", train), ("val", val), ("test", test)]:
        path = manifest_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(entries):,} entries -> {path}")


def compute_stats(train, val, test, all_entries):
    """Compute corpus statistics."""
    total_duration = sum(e.get("duration", 0) for e in all_entries)
    train_duration = sum(e.get("duration", 0) for e in train)
    val_duration = sum(e.get("duration", 0) for e in val)
    test_duration = sum(e.get("duration", 0) for e in test)

    # Vocabulary coverage
    all_nko_chars = set()
    for e in all_entries:
        for c in e.get("nko", ""):
            if 0x07C0 <= ord(c) <= 0x07FF:
                all_nko_chars.add(c)

    # Source distribution
    source_counts = defaultdict(int)
    source_durations = defaultdict(float)
    for e in all_entries:
        src = e["source"].split("_")[0]  # Group by prefix
        source_counts[src] += 1
        source_durations[src] += e.get("duration", 0)

    stats = {
        "total_samples": len(all_entries),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "total_hours": round(total_duration / 3600, 1),
        "train_hours": round(train_duration / 3600, 1),
        "val_hours": round(val_duration / 3600, 1),
        "test_hours": round(test_duration / 3600, 1),
        "nko_vocab_size": len(all_nko_chars),
        "nko_vocab_chars": sorted(all_nko_chars),
        "sources": {
            src: {
                "samples": source_counts[src],
                "hours": round(source_durations[src] / 3600, 1),
            }
            for src in sorted(source_counts.keys())
        },
    }
    return stats


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V5 Data Preparation: All Bambara/Manding Audio"
    )
    parser.add_argument(
        "--output-dir", default="/workspace/v5_data",
        help="Output directory for features and manifests",
    )
    parser.add_argument(
        "--cache-dir", default="/workspace/hf_data",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max samples per dataset (0 = all)",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature extraction (manifests only)",
    )
    parser.add_argument(
        "--extract-mel", action="store_true",
        help="Extract mel spectrograms instead of encoder features (for LoRA training)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--skip-common-voice", action="store_true",
        help="Skip Common Voice (may need auth)",
    )
    args = parser.parse_args()

    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download + Bridge all datasets ──────────────────────
    print("=" * 60)
    print("V5 Data Preparation Pipeline")
    print("=" * 60)

    all_entries = []
    datasets_cache = {}

    # Load afvoices
    afvoices_entries = load_afvoices(args.cache_dir, args.limit)
    all_entries.extend(afvoices_entries)

    # Load bam-asr-early
    bamasr_entries = load_bam_asr_early(args.cache_dir, args.limit)
    all_entries.extend(bamasr_entries)

    # Load Common Voice
    if not args.skip_common_voice:
        cv_entries = load_common_voice_bam(args.cache_dir, args.limit)
        all_entries.extend(cv_entries)

    # Load MALIBA
    maliba_entries = load_maliba_bambara(args.cache_dir, args.limit)
    all_entries.extend(maliba_entries)

    # Load nko-asr
    nkoasr_entries = load_nko_asr(args.cache_dir, args.limit)
    all_entries.extend(nkoasr_entries)

    if not all_entries:
        print("\nERROR: No data loaded. Check network/cache.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Total entries: {len(all_entries):,}")
    total_hours = sum(e.get("duration", 0) for e in all_entries) / 3600
    print(f"Total duration: {total_hours:.1f} hours")

    # ── 2. Create splits ───────────────────────────────────────
    print("\nCreating train/val/test splits...")
    train, val, test = create_splits(all_entries, args.seed)

    # ── 3. Write manifests ─────────────────────────────────────
    print("\nWriting manifests...")
    write_manifests(train, val, test, output_dir)

    # ── 4. Compute and save stats ──────────────────────────────
    stats = compute_stats(train, val, test, all_entries)
    stats_path = output_dir / "manifests" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nStats saved to {stats_path}")

    print(f"\nCorpus Summary:")
    print(f"  Total: {stats['total_samples']:,} samples, {stats['total_hours']} hours")
    print(f"  Train: {stats['train_samples']:,} samples, {stats['train_hours']} hours")
    print(f"  Val:   {stats['val_samples']:,} samples, {stats['val_hours']} hours")
    print(f"  Test:  {stats['test_samples']:,} samples, {stats['test_hours']} hours")
    print(f"  N'Ko vocab: {stats['nko_vocab_size']} characters")
    print(f"\n  Sources:")
    for src, info in stats["sources"].items():
        print(f"    {src}: {info['samples']:,} samples, {info['hours']} hours")

    # ── 5. Extract features ────────────────────────────────────
    if not args.skip_features:
        # Reload datasets for feature extraction (need audio arrays)
        print("\nReloading datasets for feature extraction...")
        from datasets import load_dataset

        # Build datasets_cache mapping source -> HF split object
        try:
            ds_afv = load_dataset("RobotsMali/afvoices", "human-corrected", cache_dir=args.cache_dir)
            datasets_cache["afvoices"] = ds_afv["train"]
            if "test" in ds_afv:
                datasets_cache["afvoices_test"] = ds_afv["test"]
        except Exception as e:
            print(f"  Could not reload afvoices: {e}")

        try:
            ds_bam = load_dataset("RobotsMali/bam-asr-early", cache_dir=args.cache_dir)
            datasets_cache["bamasr_train"] = ds_bam["train"]
            if "test" in ds_bam:
                datasets_cache["bamasr_test"] = ds_bam["test"]
        except Exception as e:
            print(f"  Could not reload bam-asr-early: {e}")

        if not args.skip_common_voice:
            try:
                ds_cv = load_dataset(
                    "mozilla-foundation/common_voice_17_0", "bm",
                    cache_dir=args.cache_dir, trust_remote_code=True,
                )
                for split_name in ds_cv.keys():
                    datasets_cache[f"cv_{split_name}"] = ds_cv[split_name]
            except Exception:
                pass

        if args.extract_mel:
            # V5 LoRA mode: save mel spectrograms (no GPU needed for extraction)
            mel_dir = output_dir / "mels"
            extracted, errors = extract_mels_for_entries(
                all_entries, datasets_cache, mel_dir,
            )
            print(f"\nMels: {extracted:,} extracted, {errors:,} errors")

            # Verify mel coverage
            print("\nVerifying mel coverage...")
            missing = 0
            for entry in all_entries:
                mel_path = mel_dir / f"{entry['feat_id']}.mel.pt"
                if not mel_path.exists():
                    missing += 1
            print(f"  Missing mels: {missing:,}/{len(all_entries):,}")

            if missing > 0:
                print("  Filtering manifests to exclude missing mels...")
                mel_set = set(p.stem.replace(".mel", "") for p in mel_dir.glob("*.mel.pt"))
                train = [e for e in train if e["feat_id"] in mel_set]
                val = [e for e in val if e["feat_id"] in mel_set]
                test = [e for e in test if e["feat_id"] in mel_set]
                write_manifests(train, val, test, output_dir)

                all_with_mels = train + val + test
                stats = compute_stats(train, val, test, all_with_mels)
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        else:
            # Legacy mode: save encoder features (for CTC-head-only training)
            features_dir = output_dir / "features"
            extracted, errors = extract_features_for_entries(
                all_entries, datasets_cache, features_dir,
            )
            print(f"\nFeatures: {extracted:,} extracted, {errors:,} errors")

            # Verify features exist for manifest entries
            print("\nVerifying feature coverage...")
            missing = 0
            for entry in all_entries:
                feat_path = features_dir / f"{entry['feat_id']}.pt"
                if not feat_path.exists():
                    missing += 1
            print(f"  Missing features: {missing:,}/{len(all_entries):,}")

            if missing > 0:
                # Rewrite manifests excluding entries with missing features
                print("  Filtering manifests to exclude missing features...")
                feat_set = set(p.stem for p in features_dir.glob("*.pt"))
                train = [e for e in train if e["feat_id"] in feat_set]
                val = [e for e in val if e["feat_id"] in feat_set]
                test = [e for e in test if e["feat_id"] in feat_set]
                write_manifests(train, val, test, output_dir)

                # Update stats
                all_with_feats = train + val + test
                stats = compute_stats(train, val, test, all_with_feats)
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    # ── 6. Final Summary ───────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V5 Data Preparation Complete")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Train manifest: {output_dir}/manifests/train.jsonl")
    print(f"  Val manifest:   {output_dir}/manifests/val.jsonl")
    print(f"  Test manifest:  {output_dir}/manifests/test.jsonl")
    if not args.skip_features:
        if args.extract_mel:
            print(f"  Mels dir:       {output_dir}/mels/")
        else:
            print(f"  Features dir:   {output_dir}/features/")


if __name__ == "__main__":
    main()
