#!/usr/bin/env python3
"""
GCS Video OCR Pipeline — Extract N'Ko text from babamamadidiane teaching videos
================================================================================
Downloads videos from GCS, extracts keyframes, runs Gemini 3 Flash OCR,
extracts audio, produces aligned (audio_segment, nko_text) pairs.

Usage:
    python3 asr/gcs_ocr_pipeline.py --limit 5        # Test on 5 videos
    python3 asr/gcs_ocr_pipeline.py --all             # Process all 189
    python3 asr/gcs_ocr_pipeline.py --resume          # Resume from checkpoint
"""

import argparse
import asyncio
import aiohttp
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GCS_BUCKET = "gs://learnnko-videos/videos/"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ocr_extractions"
CHECKPOINT_FILE = OUTPUT_DIR / "ocr_checkpoint.json"
FRAMES_PER_VIDEO = 20
GEMINI_MODEL = "gemini-3-flash-preview"

OCR_PROMPT = """Look at this image carefully. It shows a frame from an N'Ko teaching video.

Extract ALL N'Ko script (ߒߞߏ) text visible in the image. N'Ko is written right-to-left.

Return the text in this format:
NKO: [the N'Ko text exactly as shown]
LATIN: [any Latin/French text visible, if any]
CONTEXT: [brief description of what's shown - e.g. "whiteboard with N'Ko alphabet", "slide showing vocabulary"]

If no N'Ko text is visible (e.g. just a person talking with no text on screen), respond with:
NO_TEXT_VISIBLE"""


def has_nko(text):
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)


def count_nko(text):
    return sum(1 for c in text if 0x07C0 <= ord(c) <= 0x07FF)


def clean_ocr_text(raw_text: str) -> str:
    """Strip OCR output artifacts: NKO: prefix, model thinking leaks, markdown."""
    import re
    text = raw_text.strip()
    # Remove common prefixes
    for prefix in ["NKO:", "**NKO:**", "**NKO**:", "NKO :", "nko:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove model thinking leaks
    if any(leak in text.lower() for leak in ["thought", "wait,", "let's", "let me", "(wait"]):
        # Extract just the N'Ko characters
        text = "".join(c for c in text if 0x07C0 <= ord(c) <= 0x07FF or c in " ߸߹")
    # Remove markdown artifacts
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'`+', '', text)
    # Remove Latin text after N'Ko (keep only N'Ko + spaces + N'Ko punctuation)
    # But allow N'Ko digits and tone marks
    cleaned = []
    for char in text:
        if 0x07C0 <= ord(char) <= 0x07FF or char in " \n":
            cleaned.append(char)
    return "".join(cleaned).strip()


MIN_NKO_CHARS = 3  # Minimum N'Ko characters for a valid extraction


def list_gcs_videos():
    """List all MP4s in the GCS bucket."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", GCS_BUCKET],
        capture_output=True, text=True, timeout=30,
    )
    return [l.strip() for l in result.stdout.strip().split("\n") if l.strip().endswith(".mp4")]


def download_gcs_video(gcs_url, local_path):
    """Download a video from GCS."""
    subprocess.run(
        ["gcloud", "storage", "cp", gcs_url, local_path],
        capture_output=True, timeout=120,
    )
    return os.path.exists(local_path)


def extract_frames(video_path, output_dir, num_frames=20):
    """Extract evenly-spaced frames from a video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration
    dur_cmd = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=15,
    )
    try:
        duration = float(dur_cmd.stdout.strip())
    except ValueError:
        duration = 120

    frames = []
    for i in range(num_frames):
        ts = duration * (i + 0.5) / num_frames
        frame_path = output_dir / f"frame_{i:03d}.png"
        subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", str(video_path),
             "-vframes", "1", "-y", "-loglevel", "error", str(frame_path)],
            capture_output=True, timeout=15,
        )
        if frame_path.exists():
            frames.append({"path": str(frame_path), "timestamp": ts, "index": i})

    return frames


def extract_audio_segments(video_path, output_dir, segment_duration=30):
    """Extract audio and segment into chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Path(video_path).stem
    pattern = str(output_dir / f"{base}_%04d.wav")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path),
         "-f", "segment", "-segment_time", str(segment_duration),
         "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
         pattern, "-y", "-loglevel", "error"],
        capture_output=True, timeout=120,
    )
    return sorted(output_dir.glob(f"{base}_*.wav"))


async def gemini_ocr(image_path, session):
    """Run Gemini 3 Flash OCR on a single frame."""
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [
            {"text": OCR_PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": img_data}},
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }

    t0 = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            elapsed = time.time() - t0
            if resp.status == 200:
                data = await resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return {"text": text, "nko_chars": count_nko(text), "has_nko": has_nko(text), "time": elapsed}
            else:
                err = await resp.text()
                return {"text": "", "nko_chars": 0, "has_nko": False, "time": elapsed, "error": f"{resp.status}: {err[:100]}"}
    except Exception as e:
        return {"text": "", "nko_chars": 0, "has_nko": False, "time": time.time() - t0, "error": str(e)[:100]}


async def process_video(gcs_url, session, temp_dir="/tmp/ocr_pipeline"):
    """Full pipeline for one video: download, extract frames + audio, OCR, align."""
    video_id = Path(gcs_url).stem.split("_")[0]
    temp = Path(temp_dir) / video_id
    temp.mkdir(parents=True, exist_ok=True)
    video_path = temp / f"{video_id}.mp4"

    # Download
    if not download_gcs_video(gcs_url, str(video_path)):
        return {"video_id": video_id, "error": "download failed", "frames": []}

    # Extract frames
    frames_dir = temp / "frames"
    frames = extract_frames(str(video_path), frames_dir, FRAMES_PER_VIDEO)

    # Extract audio segments
    audio_dir = temp / "audio"
    audio_segments = extract_audio_segments(str(video_path), audio_dir)

    # OCR each frame
    ocr_results = []
    for frame in frames:
        result = await gemini_ocr(frame["path"], session)
        result["timestamp"] = frame["timestamp"]
        result["frame_index"] = frame["index"]
        ocr_results.append(result)

    # Align: for each OCR result with N'Ko text, find the closest audio segment
    nko_frames = [r for r in ocr_results if r["has_nko"]]
    aligned_pairs = []
    for nko_frame in nko_frames:
        # Clean the OCR text
        cleaned = clean_ocr_text(nko_frame["text"])
        nko_char_count = count_nko(cleaned)

        # Skip if too few N'Ko characters after cleaning
        if nko_char_count < MIN_NKO_CHARS:
            continue

        ts = nko_frame["timestamp"]
        seg_idx = int(ts // 30)
        if seg_idx < len(audio_segments):
            aligned_pairs.append({
                "audio_path": str(audio_segments[seg_idx]),
                "nko_text": cleaned,
                "nko_chars": nko_char_count,
                "timestamp": ts,
                "frame_index": nko_frame["frame_index"],
            })

    # Cleanup video (keep audio segments)
    video_path.unlink(missing_ok=True)
    for f in frames:
        Path(f["path"]).unlink(missing_ok=True)

    return {
        "video_id": video_id,
        "gcs_url": gcs_url,
        "total_frames": len(frames),
        "nko_frames": len(nko_frames),
        "aligned_pairs": len(aligned_pairs),
        "audio_segments": len(audio_segments),
        "ocr_results": ocr_results,
        "pairs": aligned_pairs,
    }


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed": [], "total_pairs": 0}


def save_checkpoint(cp):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = list_gcs_videos()
    print(f"Found {len(videos)} GCS videos")

    if args.limit > 0:
        videos = videos[:args.limit]
    elif not args.all:
        videos = videos[:5]
        print("(use --all for all videos, --limit N for specific count)")

    cp = load_checkpoint() if args.resume else {"processed": [], "total_pairs": 0}
    done = set(cp["processed"])
    remaining = [v for v in videos if Path(v).stem.split("_")[0] not in done]
    print(f"Already done: {len(done)}, remaining: {len(remaining)}")

    pairs_file = OUTPUT_DIR / "ocr_pairs.jsonl"
    results_file = OUTPUT_DIR / "ocr_results.jsonl"
    t_start = time.time()

    async with aiohttp.ClientSession() as session:
        for i, gcs_url in enumerate(remaining):
            vid_name = Path(gcs_url).stem[:40]
            print(f"\n[{i+1}/{len(remaining)}] {vid_name}")

            result = await process_video(gcs_url, session)

            if result.get("error"):
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Frames: {result['total_frames']}, N'Ko: {result['nko_frames']}, Pairs: {result['aligned_pairs']}")

            # Save results
            with open(results_file, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Save aligned pairs
            for pair in result.get("pairs", []):
                with open(pairs_file, "a") as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            cp["processed"].append(result["video_id"])
            cp["total_pairs"] += result.get("aligned_pairs", 0)
            save_checkpoint(cp)

    total = time.time() - t_start
    print(f"\nDone: {len(cp['processed'])} videos, {cp['total_pairs']} aligned pairs")
    print(f"Time: {total/60:.1f} min")
    print(f"Output: {pairs_file}")


if __name__ == "__main__":
    asyncio.run(main())
