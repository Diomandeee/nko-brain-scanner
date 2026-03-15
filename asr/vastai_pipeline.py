#!/usr/bin/env python3
"""
Vast.ai GPU Pipeline — Batch ASR Transcription
================================================
Runs on a Vast.ai RTX 3090 instance to transcribe Bambara audio
segments using FarmRadio Whisper at GPU speed (~2-3s per segment
vs 60s on CPU).

Setup (run on Vast.ai instance):
    pip install transformers torch torchaudio soundfile
    # Segments should be SCP'd to /workspace/segments/

Usage:
    python3 vastai_pipeline.py --input /workspace/segments/ --output /workspace/results/
    python3 vastai_pipeline.py --input /workspace/segments/ --output /workspace/results/ --resume
"""

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path
from datetime import datetime

# Config
DEFAULT_MODEL = "FarmRadioInternational/bambara-whisper-asr"
CHECKPOINT_FILE = "vastai_checkpoint.json"
BATCH_REPORT_INTERVAL = 50


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"processed": [], "count": 0, "errors": 0}


def save_checkpoint(path, cp):
    with open(path, "w") as f:
        json.dump(cp, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with WAV segments")
    parser.add_argument("--output", default="/workspace/results", help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all WAV files
    wavs = sorted(glob.glob(os.path.join(args.input, "**", "*.wav"), recursive=True))
    print(f"Found {len(wavs)} WAV segments")

    # Resume support
    cp_path = str(out_dir / CHECKPOINT_FILE)
    cp = load_checkpoint(cp_path) if args.resume else {"processed": [], "count": 0, "errors": 0}
    done_set = set(cp["processed"])
    remaining = [w for w in wavs if w not in done_set]
    print(f"Already done: {len(done_set)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to process.")
        return

    # Check GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, running on CPU (slow)")

    # Load model
    from transformers import pipeline
    print(f"Loading ASR model: {args.model}")
    kwargs = {"device": device, "return_timestamps": True}
    if args.hf_token:
        kwargs["token"] = args.hf_token
    asr = pipeline("automatic-speech-recognition", model=args.model, **kwargs)
    print("Model loaded.")

    # Transcribe
    output_file = out_dir / "transcriptions.jsonl"
    mode = "a" if args.resume and output_file.exists() else "w"
    t_start = time.time()

    with open(output_file, mode) as out:
        for i, wav in enumerate(remaining):
            t0 = time.time()
            try:
                result = asr(wav)
                text = result["text"].strip()
                elapsed = time.time() - t0

                entry = {
                    "file": wav,
                    "text": text,
                    "time": round(elapsed, 2),
                    "status": "ok",
                }
                cp["processed"].append(wav)
                cp["count"] += 1
            except Exception as e:
                elapsed = time.time() - t0
                entry = {
                    "file": wav,
                    "text": "",
                    "time": round(elapsed, 2),
                    "status": "error",
                    "error": str(e),
                }
                cp["errors"] += 1

            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % BATCH_REPORT_INTERVAL == 0 or i == 0:
                total_elapsed = time.time() - t_start
                rate = (i + 1) / total_elapsed
                eta_min = (len(remaining) - i - 1) / rate / 60
                preview = entry.get("text", "")[:60]
                print(
                    f"  [{i+1}/{len(remaining)}] {elapsed:.1f}s | "
                    f"{rate:.1f} seg/s | ETA {eta_min:.0f}m | "
                    f"{preview}"
                )
                save_checkpoint(cp_path, cp)

    save_checkpoint(cp_path, cp)
    total = time.time() - t_start
    print(f"\nDone: {cp['count']} transcribed, {cp['errors']} errors")
    print(f"Time: {total/60:.1f} min ({total/max(cp['count'],1):.1f}s avg)")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
