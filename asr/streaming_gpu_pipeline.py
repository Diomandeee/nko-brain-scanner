#!/usr/bin/env python3
"""
Streaming GPU Pipeline — Download + Transcribe + Bridge in Real-Time
=====================================================================
Runs entirely on Vast.ai GPU instance. No intermediate storage on Mac1.

1. Downloads audio from YouTube via yt-dlp (with cookies)
2. Segments into 30s WAV chunks
3. Transcribes each segment immediately with Whisper
4. Bridges Latin → N'Ko
5. Deletes audio after transcription (minimal disk usage)

Usage (on Vast.ai):
    python3 streaming_gpu_pipeline.py --channel djoko --limit 100
    python3 streaming_gpu_pipeline.py --channel babamamadidiane --limit 532
    python3 streaming_gpu_pipeline.py --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
import glob
from pathlib import Path
from datetime import datetime

# Channels
CHANNELS = {
    "djoko": "https://www.youtube.com/channel/UCXiIHk2N-qWE-ZFPfPVJ06w/videos",
    "babamamadidiane": "https://www.youtube.com/@babamamadidiane/videos",
}

CHECKPOINT_FILE = "/workspace/streaming_checkpoint.json"
OUTPUT_FILE = "/workspace/results/streaming_transcriptions.jsonl"
NKO_OUTPUT_FILE = "/workspace/results/streaming_nko_pairs.jsonl"
TEMP_DIR = "/workspace/temp_audio"
COOKIES_FILE = "/workspace/cookies.txt"
SEGMENT_DURATION = 30


def list_videos(channel_url, limit=0):
    """List video IDs from channel."""
    cmd = ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", channel_url]
    if os.path.exists(COOKIES_FILE):
        cmd.extend(["--cookies", COOKIES_FILE])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    videos = []
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            vid_id, title = line.split("\t", 1)
            videos.append({"id": vid_id, "title": title})
    if limit > 0:
        videos = videos[:limit]
    return videos


def download_audio(video_id):
    """Download audio as WAV."""
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    output = f"{TEMP_DIR}/{video_id}.%(ext)s"
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
        "-o", output, "--no-playlist", "--force-ipv4", "--socket-timeout", "60",
    ]
    if os.path.exists(COOKIES_FILE):
        cmd.extend(["--cookies", COOKIES_FILE])
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wav = f"{TEMP_DIR}/{video_id}.wav"
    return wav if os.path.exists(wav) else None


def segment_audio(audio_path):
    """Segment into 30s chunks at 16kHz mono."""
    base = Path(audio_path).stem
    seg_dir = f"{TEMP_DIR}/segments/{base}"
    Path(seg_dir).mkdir(parents=True, exist_ok=True)
    pattern = f"{seg_dir}/{base}_%04d.wav"
    subprocess.run([
        "ffmpeg", "-i", audio_path, "-f", "segment",
        "-segment_time", str(SEGMENT_DURATION),
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        pattern, "-y", "-loglevel", "error",
    ], capture_output=True, timeout=120)
    segments = sorted(glob.glob(f"{seg_dir}/{base}_*.wav"))
    # Delete full audio immediately
    os.remove(audio_path)
    return segments


def load_asr_model():
    """Load Whisper ASR model on GPU."""
    from transformers import pipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ASR on {device}...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="FarmRadioInternational/bambara-whisper-asr",
        device=device,
        return_timestamps=False,
    )
    print("ASR loaded.")
    return asr


def transcribe_segments(asr, segments):
    """Transcribe all segments for one video."""
    results = []
    for seg in segments:
        t0 = time.time()
        try:
            result = asr(seg)
            text = result["text"].strip()
            elapsed = time.time() - t0
            results.append({"file": seg, "text": text, "time": round(elapsed, 2), "status": "ok"})
        except Exception as e:
            results.append({"file": seg, "text": "", "time": round(time.time() - t0, 2),
                            "status": "error", "error": str(e)[:100]})
        # Delete segment after transcription
        os.remove(seg)
    return results


def bridge_to_nko(text):
    """Convert Latin Bambara to N'Ko (simplified bridge on GPU instance)."""
    # Basic Latin → N'Ko character mapping (subset of full bridge)
    # Full bridge requires the nko package, but for streaming we use a lookup table
    LATIN_TO_NKO = {
        'a': 'ߊ', 'b': 'ߓ', 'c': 'ߗ', 'd': 'ߘ', 'e': 'ߍ',
        'f': 'ߝ', 'g': 'ߜ', 'h': 'ߤ', 'i': 'ߌ', 'j': 'ߖ',
        'k': 'ߞ', 'l': 'ߟ', 'm': 'ߡ', 'n': 'ߣ', 'o': 'ߏ',
        'p': 'ߔ', 'r': 'ߙ', 's': 'ߛ', 't': 'ߕ', 'u': 'ߎ',
        'v': 'ߝ', 'w': 'ߥ', 'y': 'ߦ', 'z': 'ߖ',
        'ɔ': 'ߏ', 'ɛ': 'ߐ', 'ɲ': 'ߢ', 'ŋ': 'ߒ',
    }
    nko = []
    for char in text.lower():
        if char in LATIN_TO_NKO:
            nko.append(LATIN_TO_NKO[char])
        elif char == ' ':
            nko.append(' ')
    return ''.join(nko)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed_videos": [], "total_segments": 0, "total_transcribed": 0, "total_nko_pairs": 0}


def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="djoko", choices=list(CHANNELS.keys()))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-offset", type=int, default=0, help="Skip first N videos (for multi-GPU split)")
    args = parser.parse_args()

    Path("/workspace/results").mkdir(exist_ok=True)

    # Load model first (takes 30s, do it once)
    asr = load_asr_model()

    # List videos
    print(f"Listing {args.channel} videos...")
    videos = list_videos(CHANNELS[args.channel], args.limit)
    print(f"Found {len(videos)} videos")

    if args.start_offset > 0:
        videos = videos[args.start_offset:]
        print(f"Starting from offset {args.start_offset}, {len(videos)} remaining")

    # Resume
    cp = load_checkpoint() if args.resume else {
        "processed_videos": [], "total_segments": 0, "total_transcribed": 0, "total_nko_pairs": 0
    }
    done = set(cp["processed_videos"])
    remaining = [v for v in videos if v["id"] not in done]
    print(f"Already done: {len(done)}, remaining: {len(remaining)}")

    t_start = time.time()

    for i, video in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {video['title'][:50]}")

        # Download
        t0 = time.time()
        audio_path = download_audio(video["id"])
        if not audio_path:
            print(f"  Download failed, skipping")
            continue
        dl_time = time.time() - t0

        # Segment
        segments = segment_audio(audio_path)
        print(f"  Downloaded ({dl_time:.0f}s) -> {len(segments)} segments")

        # Transcribe immediately
        t0 = time.time()
        transcriptions = transcribe_segments(asr, segments)
        tr_time = time.time() - t0
        ok_count = sum(1 for t in transcriptions if t["status"] == "ok" and t["text"])

        # Bridge to N'Ko
        nko_pairs = []
        for tr in transcriptions:
            if tr["status"] == "ok" and tr["text"]:
                nko = bridge_to_nko(tr["text"])
                if nko.strip():
                    nko_pairs.append({
                        "video_id": video["id"],
                        "latin": tr["text"],
                        "nko": nko,
                        "segment_file": os.path.basename(tr["file"]),
                    })

        print(f"  Transcribed ({tr_time:.0f}s): {ok_count}/{len(segments)} ok, {len(nko_pairs)} N'Ko pairs")

        # Write results
        with open(OUTPUT_FILE, "a") as f:
            for tr in transcriptions:
                tr["video_id"] = video["id"]
                f.write(json.dumps(tr, ensure_ascii=False) + "\n")

        with open(NKO_OUTPUT_FILE, "a") as f:
            for pair in nko_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # Update checkpoint
        cp["processed_videos"].append(video["id"])
        cp["total_segments"] += len(segments)
        cp["total_transcribed"] += ok_count
        cp["total_nko_pairs"] += len(nko_pairs)
        save_checkpoint(cp)

        # Clean up segment dir
        seg_dir = f"{TEMP_DIR}/segments/{video['id']}"
        if os.path.exists(seg_dir):
            subprocess.run(["rm", "-rf", seg_dir], capture_output=True)

        # Progress
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed * 3600
        print(f"  Rate: {rate:.0f} vids/hr | Total: {cp['total_nko_pairs']} N'Ko pairs")

    total = time.time() - t_start
    print(f"\nDone: {cp['total_transcribed']} transcribed, {cp['total_nko_pairs']} N'Ko pairs")
    print(f"Time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
