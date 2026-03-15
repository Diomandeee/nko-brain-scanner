#!/usr/bin/env python3
"""
Bambara ASR Transcription Pipeline
===================================
Transcribes audio segments using MALIBA-AI/bambara-asr-v3 (Whisper fine-tune).
Supports batch processing with checkpointing and resume.

Usage:
    # Transcribe from a directory of WAV files
    python3 asr/transcribe_bambara.py --input-dir djoko_audio/ --output results/transcriptions.jsonl

    # Transcribe from the LearnNKo CSV catalog
    python3 asr/transcribe_bambara.py --from-csv ~/projects/LearnNKo/ml/data/bambara_dataset.csv

    # Resume from checkpoint
    python3 asr/transcribe_bambara.py --input-dir djoko_audio/ --resume

    # Use local Whisper instead of MALIBA-AI
    python3 asr/transcribe_bambara.py --input-dir djoko_audio/ --model openai/whisper-large-v3
"""

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Try to import the cross-script bridge
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from nko.transliterate import transliterate as nko_transliterate
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

# ASR model config
DEFAULT_MODEL = "MALIBA-AI/whisper-small-bambara-cv17"
FALLBACK_MODELS = [
    "MALIBA-AI/whisper-small-bambara-cv17",
    "openai/whisper-large-v3",
]

CHECKPOINT_FILE = "transcription_checkpoint.json"


def load_whisper_pipeline(model_id: str, device: str = "auto"):
    """Load Whisper ASR pipeline."""
    try:
        from transformers import pipeline
        print(f"Loading ASR model: {model_id}")
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            chunk_length_s=30,
            return_timestamps=True,
        )
        print(f"Model loaded on {asr.device}")
        return asr
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None


def transcribe_file(asr_pipeline, audio_path: str, language: str = "bm") -> Dict[str, Any]:
    """Transcribe a single audio file."""
    try:
        result = asr_pipeline(
            audio_path,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        # Attempt N'Ko transliteration if bridge available
        nko_text = None
        if BRIDGE_AVAILABLE and text:
            try:
                nko_text = nko_transliterate(text, source="latin", target="nko")
            except Exception:
                pass

        return {
            "audio_path": audio_path,
            "transcription_latin": text,
            "transcription_nko": nko_text,
            "chunks": chunks,
            "language": language,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "audio_path": audio_path,
            "transcription_latin": None,
            "transcription_nko": None,
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat(),
        }


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load processing checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"processed": [], "errors": [], "count": 0, "started_at": datetime.now().isoformat()}


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict):
    """Save processing checkpoint."""
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def collect_audio_files(input_dir: str, extensions: tuple = (".wav", ".mp3", ".ogg", ".flac")) -> List[str]:
    """Collect all audio files from a directory."""
    files = []
    for ext in extensions:
        files.extend(sorted(Path(input_dir).rglob(f"*{ext}")))
    return [str(f) for f in files]


def collect_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """Collect audio file references from LearnNKo CSV catalog."""
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "pending":
                entries.append({
                    "file_id": row.get("file_id", ""),
                    "audio_path": row.get("audio_path", ""),
                    "original_file": row.get("original_file", ""),
                    "duration_seconds": float(row.get("duration_seconds", 0)),
                })
    return entries


def run_batch_transcription(
    asr_pipeline,
    audio_files: List[str],
    output_path: str,
    checkpoint_dir: str = ".",
    resume: bool = False,
    nko_bridge: bool = True,
):
    """Run batch transcription with checkpointing."""
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_FILE
    checkpoint = load_checkpoint(checkpoint_path) if resume else {
        "processed": [], "errors": [], "count": 0, "started_at": datetime.now().isoformat()
    }

    already_done = set(checkpoint["processed"])
    remaining = [f for f in audio_files if f not in already_done]

    print(f"\nTotal files: {len(audio_files)}")
    print(f"Already processed: {len(already_done)}")
    print(f"Remaining: {len(remaining)}")
    print(f"N'Ko bridge: {'available' if BRIDGE_AVAILABLE else 'unavailable'}")
    print(f"Output: {output_path}\n")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if resume and output_file.exists() else "w"
    start_time = time.time()

    with open(output_file, mode) as out:
        for i, audio_path in enumerate(remaining):
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 1) * 3600  # files per hour

            print(f"  [{i+1}/{len(remaining)}] {Path(audio_path).name} ", end="", flush=True)

            result = transcribe_file(asr_pipeline, audio_path)

            if result["status"] == "completed":
                print(f"-> {result['transcription_latin'][:60]}...")
                checkpoint["processed"].append(audio_path)
                checkpoint["count"] += 1
            else:
                print(f"ERROR: {result.get('error', 'unknown')}")
                checkpoint["errors"].append(audio_path)

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            # Save checkpoint every 50 files
            if (i + 1) % 50 == 0:
                save_checkpoint(checkpoint_path, checkpoint)
                print(f"  [checkpoint saved, {rate:.0f} files/hr]")

    save_checkpoint(checkpoint_path, checkpoint)

    total_time = time.time() - start_time
    print(f"\nDone: {checkpoint['count']} transcribed, {len(checkpoint['errors'])} errors")
    print(f"Time: {total_time/60:.1f} minutes ({total_time/max(checkpoint['count'],1):.1f}s per file)")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Bambara ASR Transcription Pipeline")
    parser.add_argument("--input-dir", help="Directory of audio files")
    parser.add_argument("--from-csv", help="LearnNKo CSV catalog path")
    parser.add_argument("--output", default="results/transcriptions.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ASR model ID")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/mps/cuda)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files")
    parser.add_argument("--dry-run", action="store_true", help="List files without processing")
    args = parser.parse_args()

    if not args.input_dir and not args.from_csv:
        parser.error("Provide --input-dir or --from-csv")

    # Collect files
    if args.input_dir:
        audio_files = collect_audio_files(args.input_dir)
    else:
        entries = collect_from_csv(args.from_csv)
        audio_files = [e["audio_path"] for e in entries]

    if args.limit > 0:
        audio_files = audio_files[:args.limit]

    if args.dry_run:
        print(f"Found {len(audio_files)} audio files:")
        for f in audio_files[:20]:
            print(f"  {f}")
        if len(audio_files) > 20:
            print(f"  ... and {len(audio_files)-20} more")
        return

    # Load ASR model
    asr = load_whisper_pipeline(args.model, args.device)
    if asr is None:
        print("Failed to load ASR model. Exiting.")
        sys.exit(1)

    # Run
    run_batch_transcription(
        asr,
        audio_files,
        args.output,
        checkpoint_dir=str(Path(args.output).parent),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
