#!/usr/bin/env python3
"""
Dynamic Multi-Pass OCR Pipeline for N'Ko Teaching Videos
=========================================================
Instead of static 20-frame sampling, this pipeline adapts to the
structure of each teaching video:

Pass 1: Scene Detection
  - FFmpeg scene change detection finds slide transitions
  - Each unique slide gets one frame (captures every text display)
  - Dynamic: 5 slides = 5 frames, 40 slides = 40 frames

Pass 2: Text Window Mapping
  - For each scene with N'Ko text, define the temporal window
  - Window = [scene_start, next_scene_start] = the time the text is displayed
  - The teacher's voice during this window is explaining THAT text

Pass 3: Audio-Text Alignment
  - Extract the audio for each text window (not fixed 30s chunks)
  - Each (audio_window, nko_text) pair is a ground-truth alignment
  - The teacher literally reads/explains the text shown on screen

This produces much higher quality training pairs than static sampling
because each audio segment is aligned to the specific N'Ko text
the teacher is discussing at that moment.

Usage:
    python3 asr/dynamic_ocr_pipeline.py --limit 5
    python3 asr/dynamic_ocr_pipeline.py --all --resume
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
from typing import List, Dict, Optional, Tuple

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GCS_BUCKET = "gs://learnnko-videos/videos/"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "dynamic_ocr"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
GEMINI_MODEL = "gemini-3-flash-preview"

# Scene detection config
SCENE_THRESHOLD = 0.25  # Lower = more sensitive to slide changes
MIN_SCENE_DURATION = 3.0  # Minimum seconds between scenes
MIN_NKO_CHARS = 3

OCR_PROMPT = """This is a frame from an N'Ko (ߒߞߏ) language teaching video showing a digital slide.
Parse the slide structure and return a JSON object with this exact schema:
{
  "slide_title": "the lesson title in N'Ko if visible, else null",
  "sections": [
    {
      "type": "vocabulary|example|definition|exercise|alphabet|numbers",
      "nko_text": "the N'Ko text in this section",
      "latin_text": "any Latin/French text alongside it, else null"
    }
  ],
  "reading_order": [0, 1, 2],
  "has_nko": true
}

Rules:
- Return ONLY valid JSON. No markdown fences, no explanation.
- If no N'Ko text is visible set has_nko to false and sections to [].
- reading_order lists section indices in the order a student would read them.
- Choose the best type for each section from the allowed values."""


def has_nko(text):
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)

def count_nko(text):
    return sum(1 for c in text if 0x07C0 <= ord(c) <= 0x07FF)

def clean_nko(text):
    """Extract only N'Ko characters and spaces."""
    import re
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'`+', '', text)
    for prefix in ["NKO:", "**NKO:**", "NKO :", "nko:"]:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):]
    return "".join(c for c in text if 0x07C0 <= ord(c) <= 0x07FF or c in " \n").strip()


def parse_structured_ocr(raw: str) -> Dict:
    """Parse Gemini JSON slide response. Falls back to flat text extraction on failure.

    Returns a dict with keys:
      structured   bool   – True if JSON parsed successfully
      slide_title  str|None
      sections     list   – [{type, nko_text, latin_text}, ...]
      reading_order list  – section indices
      has_nko      bool
      cleaned      str    – flat N'Ko text (all sections joined, for backward compat)
      nko_chars    int
    """
    import re

    # Strip markdown code fences Gemini sometimes adds despite instructions
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        data = json.loads(text)
        sections = data.get("sections") or []
        # Validate and normalise sections
        clean_sections = []
        for sec in sections:
            nko_text = clean_nko(sec.get("nko_text") or "")
            if not nko_text:
                continue
            clean_sections.append({
                "type": sec.get("type", "vocabulary"),
                "nko_text": nko_text,
                "latin_text": (sec.get("latin_text") or "").strip() or None,
            })

        # reading_order: keep valid indices only
        raw_order = data.get("reading_order") or list(range(len(clean_sections)))
        reading_order = [i for i in raw_order if isinstance(i, int) and i < len(clean_sections)]
        if not reading_order:
            reading_order = list(range(len(clean_sections)))

        all_nko = " ".join(s["nko_text"] for s in clean_sections)
        nko_chars = count_nko(all_nko)

        return {
            "structured": True,
            "slide_title": (data.get("slide_title") or "").strip() or None,
            "sections": clean_sections,
            "reading_order": reading_order,
            "has_nko": bool(data.get("has_nko")) and nko_chars >= MIN_NKO_CHARS,
            "cleaned": all_nko,
            "nko_chars": nko_chars,
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: treat raw response as flat N'Ko text (old behaviour)
        cleaned = clean_nko(raw)
        nko_chars = count_nko(cleaned)
        return {
            "structured": False,
            "slide_title": None,
            "sections": [{"type": "vocabulary", "nko_text": cleaned, "latin_text": None}] if cleaned else [],
            "reading_order": [0] if cleaned else [],
            "has_nko": nko_chars >= MIN_NKO_CHARS,
            "cleaned": cleaned,
            "nko_chars": nko_chars,
        }


# ── Pass 1: Scene Detection ──────────────────────────────────

def detect_scenes(video_path: str) -> List[float]:
    """Detect scene changes using FFmpeg. Returns timestamps."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", video_path,
        "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        timestamps = [0.0]  # Always include start
        for line in result.stderr.split('\n'):
            if 'pts_time:' in line:
                try:
                    start = line.find('pts_time:') + len('pts_time:')
                    end = line.find(' ', start)
                    if end == -1:
                        end = len(line)
                    ts = float(line[start:end])
                    if not timestamps or (ts - timestamps[-1]) >= MIN_SCENE_DURATION:
                        timestamps.append(ts)
                except ValueError:
                    continue
        return timestamps
    except Exception as e:
        print(f"  Scene detection failed: {e}")
        return [0.0]


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(result.stdout.strip())
    except (ValueError, Exception):
        return 120.0


def extract_frame_at(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame at a specific timestamp."""
    subprocess.run(
        ["ffmpeg", "-ss", str(timestamp), "-i", video_path,
         "-vframes", "1", "-y", "-loglevel", "error", output_path],
        capture_output=True, timeout=15,
    )
    return os.path.exists(output_path)


# ── Pass 2: OCR + Text Window Mapping ────────────────────────

async def ocr_frame(image_path: str, session: aiohttp.ClientSession) -> Dict:
    """Run Gemini OCR on a frame. Returns structured slide data via parse_structured_ocr."""
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [
            {"text": OCR_PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": img_data}},
        ]}],
        # Higher token budget for JSON output; temp 0.1 keeps structure deterministic
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }

    t0 = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            elapsed = time.time() - t0
            if resp.status == 200:
                data = await resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = parse_structured_ocr(raw)
                parsed["raw"] = raw
                parsed["time"] = elapsed
                return parsed
            return {
                "raw": "", "cleaned": "", "nko_chars": 0, "has_nko": False,
                "structured": False, "slide_title": None, "sections": [],
                "reading_order": [], "time": elapsed, "error": f"API {resp.status}",
            }
    except Exception as e:
        return {
            "raw": "", "cleaned": "", "nko_chars": 0, "has_nko": False,
            "structured": False, "slide_title": None, "sections": [], "reading_order": [],
            "time": time.time() - t0, "error": str(e)[:100],
        }


# ── Pass 3: Audio-Text Alignment ─────────────────────────────

def extract_audio_window(video_path: str, start: float, end: float, output_path: str) -> bool:
    """Extract audio for a specific time window."""
    duration = end - start
    if duration < 1.0:
        return False
    cmd = [
        "ffmpeg", "-ss", str(start), "-i", video_path,
        "-t", str(duration), "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", "-loglevel", "error", output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(output_path)
    except Exception:
        return False


# ── Full Pipeline ─────────────────────────────────────────────

async def process_video(gcs_url: str, session: aiohttp.ClientSession,
                        temp_dir: str = "/tmp/dynamic_ocr") -> Dict:
    """Full dynamic multi-pass pipeline for one video."""
    video_id = Path(gcs_url).stem.split("_")[0]
    temp = Path(temp_dir) / video_id
    temp.mkdir(parents=True, exist_ok=True)
    video_path = str(temp / f"{video_id}.mp4")

    # Download from GCS
    subprocess.run(["gcloud", "storage", "cp", gcs_url, video_path],
                   capture_output=True, timeout=120)
    if not os.path.exists(video_path):
        return {"video_id": video_id, "error": "download failed", "pairs": []}

    duration = get_video_duration(video_path)

    # ── Pass 1: Scene Detection ──
    scenes = detect_scenes(video_path)
    # Add end timestamp
    scenes.append(duration)
    num_scenes = len(scenes) - 1
    print(f"  Pass 1: {num_scenes} scenes detected ({duration:.0f}s video)")

    # ── Pass 2: OCR each scene's first frame ──
    scene_results = []
    frames_dir = temp / "frames"
    frames_dir.mkdir(exist_ok=True)

    nko_scenes = 0
    for i in range(num_scenes):
        scene_start = scenes[i]
        scene_end = scenes[i + 1]
        # Take frame 0.5s into the scene (skip transition)
        frame_ts = scene_start + 0.5
        if frame_ts >= scene_end:
            frame_ts = scene_start

        frame_path = str(frames_dir / f"scene_{i:04d}.png")
        if not extract_frame_at(video_path, frame_ts, frame_path):
            continue

        ocr = await ocr_frame(frame_path, session)
        ocr["scene_index"] = i
        ocr["scene_start"] = scene_start
        ocr["scene_end"] = scene_end
        ocr["scene_duration"] = scene_end - scene_start
        scene_results.append(ocr)

        if ocr["has_nko"]:
            nko_scenes += 1

        # Clean up frame
        Path(frame_path).unlink(missing_ok=True)

    print(f"  Pass 2: {nko_scenes}/{num_scenes} scenes have N'Ko text")

    # ── Pass 3: Extract aligned audio for N'Ko scenes ──
    audio_dir = temp / "audio"
    audio_dir.mkdir(exist_ok=True)
    pairs = []

    for scene in scene_results:
        if not scene["has_nko"]:
            continue

        # Audio window = full scene duration (teacher explains this text)
        audio_path = str(audio_dir / f"{video_id}_scene{scene['scene_index']:04d}.wav")
        if not extract_audio_window(video_path, scene["scene_start"], scene["scene_end"], audio_path):
            continue

        sections = scene.get("sections") or []
        # Walk sections in teaching (reading) order
        order = scene.get("reading_order") or list(range(len(sections)))

        if sections:
            # Structured path: one training pair per section
            for sec_idx in order:
                if sec_idx >= len(sections):
                    continue
                sec = sections[sec_idx]
                if not sec.get("nko_text"):
                    continue
                pairs.append({
                    "audio_path": audio_path,
                    "nko_text": sec["nko_text"],
                    "nko_chars": count_nko(sec["nko_text"]),
                    "section_type": sec.get("type", "vocabulary"),
                    "latin_text": sec.get("latin_text"),
                    "slide_title": scene.get("slide_title"),
                    "section_index": sec_idx,
                    "scene_start": scene["scene_start"],
                    "scene_end": scene["scene_end"],
                    "scene_duration": scene["scene_duration"],
                    "scene_index": scene["scene_index"],
                    "structured": True,
                })
        else:
            # Flat fallback: one pair for the whole scene
            pairs.append({
                "audio_path": audio_path,
                "nko_text": scene["cleaned"],
                "nko_chars": scene["nko_chars"],
                "section_type": "vocabulary",
                "latin_text": None,
                "slide_title": None,
                "section_index": 0,
                "scene_start": scene["scene_start"],
                "scene_end": scene["scene_end"],
                "scene_duration": scene["scene_duration"],
                "scene_index": scene["scene_index"],
                "structured": False,
            })

    print(f"  Pass 3: {len(pairs)} aligned (audio_window, nko_text) pairs")

    # Clean up video (keep audio segments)
    Path(video_path).unlink(missing_ok=True)

    return {
        "video_id": video_id,
        "gcs_url": gcs_url,
        "duration": duration,
        "total_scenes": num_scenes,
        "nko_scenes": nko_scenes,
        "aligned_pairs": len(pairs),
        "pairs": pairs,
    }


def list_gcs_videos():
    result = subprocess.run(["gcloud", "storage", "ls", GCS_BUCKET],
                            capture_output=True, text=True, timeout=30)
    return [l.strip() for l in result.stdout.strip().split("\n") if l.strip().endswith(".mp4")]


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed": [], "total_pairs": 0, "total_scenes": 0}


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
        videos = videos[:3]

    cp = load_checkpoint() if args.resume else {"processed": [], "total_pairs": 0, "total_scenes": 0}
    done = set(cp["processed"])
    remaining = [v for v in videos if Path(v).stem.split("_")[0] not in done]
    print(f"Already done: {len(done)}, remaining: {len(remaining)}")

    pairs_file = OUTPUT_DIR / "dynamic_pairs.jsonl"
    results_file = OUTPUT_DIR / "dynamic_results.jsonl"

    async with aiohttp.ClientSession() as session:
        for i, gcs_url in enumerate(remaining):
            vid_name = Path(gcs_url).stem[:40]
            print(f"\n[{i+1}/{len(remaining)}] {vid_name}")

            result = await process_video(gcs_url, session)

            with open(results_file, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            for pair in result.get("pairs", []):
                with open(pairs_file, "a") as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            cp["processed"].append(result["video_id"])
            cp["total_pairs"] += result.get("aligned_pairs", 0)
            cp["total_scenes"] += result.get("total_scenes", 0)
            save_checkpoint(cp)

            if result.get("error"):
                print(f"  ERROR: {result['error']}")

    print(f"\nDone: {len(cp['processed'])} videos")
    print(f"Total scenes: {cp['total_scenes']}")
    print(f"Total aligned pairs: {cp['total_pairs']}")
    print(f"Output: {pairs_file}")


if __name__ == "__main__":
    asyncio.run(main())
