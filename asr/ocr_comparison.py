#!/usr/bin/env python3
"""
OCR Model Comparison for N'Ko Frame Text Extraction
=====================================================
Tests multiple VLM/OCR models on babamamadidiane teaching video frames
to find the best N'Ko text extractor.

Models tested:
1. Gemini 3 Flash (Preview) - latest Google VLM
2. Gemini 2.5 Flash - stable Google VLM
3. Qwen2-VL-7B - open source, runs on GPU (Vast.ai)
4. GPT-4o-mini - OpenAI vision

Usage:
    # Extract sample frames from GCS videos first
    python3 asr/ocr_comparison.py --extract-frames 5

    # Run comparison on extracted frames
    python3 asr/ocr_comparison.py --compare

    # Run single model
    python3 asr/ocr_comparison.py --model gemini-3-flash --frames-dir /tmp/ocr_test_frames/
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# N'Ko detection
def has_nko(text: str) -> bool:
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)

def count_nko(text: str) -> int:
    return sum(1 for c in text if 0x07C0 <= ord(c) <= 0x07FF)

# Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

GEMINI_PROMPT = """Look at this image carefully. It contains N'Ko script (ߒߞߏ), a West African writing system written right-to-left.

Extract ALL N'Ko text visible in the image. Return ONLY the N'Ko text, nothing else. If there is also Latin/French text, include it on a separate line prefixed with "Latin: ".

If no N'Ko text is visible, respond with "NO_NKO_TEXT"."""


async def gemini_ocr(image_path: str, model: str = "gemini-2.5-flash") -> dict:
    """Run Gemini OCR on a single frame."""
    import aiohttp

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT},
                {"inline_data": {"mime_type": "image/png", "data": img_data}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}?key={GOOGLE_API_KEY}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            elapsed = time.time() - t0
            if resp.status != 200:
                err = await resp.text()
                return {"model": model, "text": "", "nko_chars": 0, "time": elapsed, "error": err[:200]}
            data = await resp.json()

    text = ""
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        pass

    return {
        "model": model,
        "text": text,
        "nko_chars": count_nko(text),
        "has_nko": has_nko(text),
        "time": round(elapsed, 2),
    }


async def openai_ocr(image_path: str, model: str = "gpt-4.1") -> dict:
    """Run OpenAI vision OCR on a single frame."""
    import openai

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": GEMINI_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_data}",
                    }},
                ],
            }],
            max_tokens=500,
            temperature=0.1,
        )
        elapsed = time.time() - t0
        text = response.choices[0].message.content.strip()
    except Exception as e:
        elapsed = time.time() - t0
        return {"model": model, "text": "", "nko_chars": 0, "time": elapsed, "error": str(e)[:200]}

    return {
        "model": model,
        "text": text,
        "nko_chars": count_nko(text),
        "has_nko": has_nko(text),
        "time": round(elapsed, 2),
    }


def extract_sample_frames(num_videos: int = 5, frames_per_video: int = 3):
    """Download sample videos from GCS and extract frames."""
    frames_dir = Path("/tmp/ocr_test_frames")
    frames_dir.mkdir(exist_ok=True)

    # List GCS videos
    result = subprocess.run(
        ["gcloud", "storage", "ls", "gs://learnnko-videos/videos/"],
        capture_output=True, text=True, timeout=30,
    )
    videos = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    videos = videos[:num_videos]

    print(f"Extracting frames from {len(videos)} GCS videos...")

    for vid_url in videos:
        vid_name = vid_url.split("/")[-1].replace(".mp4", "")
        local_path = f"/tmp/ocr_test_{vid_name}.mp4"

        # Download from GCS
        subprocess.run(
            ["gcloud", "storage", "cp", vid_url, local_path],
            capture_output=True, timeout=60,
        )

        if not os.path.exists(local_path):
            print(f"  Skip {vid_name} (download failed)")
            continue

        # Extract frames at 25%, 50%, 75% of duration
        duration_cmd = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", local_path],
            capture_output=True, text=True, timeout=15,
        )
        try:
            duration = float(duration_cmd.stdout.strip())
        except ValueError:
            duration = 120

        for pct in [0.25, 0.50, 0.75]:
            ts = duration * pct
            frame_path = frames_dir / f"{vid_name}_f{int(pct*100)}.png"
            subprocess.run(
                ["ffmpeg", "-ss", str(ts), "-i", local_path,
                 "-vframes", "1", "-y", str(frame_path)],
                capture_output=True, timeout=15,
            )

        # Clean up video
        os.remove(local_path)
        frames = list(frames_dir.glob(f"{vid_name}_*.png"))
        print(f"  {vid_name}: {len(frames)} frames")

    total = list(frames_dir.glob("*.png"))
    print(f"Total frames: {len(total)} in {frames_dir}")
    return frames_dir


async def run_comparison(frames_dir: str):
    """Run all models on the test frames and compare."""
    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Testing {len(frames)} frames across models...")

    models = [
        ("gemini-3-flash-preview", "Gemini 3 Flash"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite"),
    ]

    # Also test OpenAI models
    openai_models = [
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-4.1-nano", "GPT-4.1 Nano"),
    ]

    results = {}
    for model_id, model_name in models:
        print(f"\n--- {model_name} ({model_id}) ---")
        model_results = []

        for frame in frames:
            r = await gemini_ocr(str(frame), model=model_id)
            r["frame"] = frame.name
            model_results.append(r)

            status = "NKO" if r.get("has_nko") else "---"
            print(f"  [{status}] {frame.name}: {r.get('text', '')[:60]} ({r['time']:.1f}s)")

        # Stats
        nko_count = sum(1 for r in model_results if r.get("has_nko"))
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        avg_chars = sum(r.get("nko_chars", 0) for r in model_results) / len(model_results)

        results[model_id] = {
            "name": model_name,
            "frames_tested": len(model_results),
            "nko_detected": nko_count,
            "avg_time_s": round(avg_time, 2),
            "avg_nko_chars": round(avg_chars, 1),
            "details": model_results,
        }

        print(f"  Summary: {nko_count}/{len(frames)} detected N'Ko, "
              f"avg {avg_chars:.1f} chars, {avg_time:.2f}s/frame")

    # Test OpenAI models
    for model_id, model_name in openai_models:
        print(f"\n--- {model_name} ({model_id}) ---")
        model_results = []

        for frame in frames:
            r = await openai_ocr(str(frame), model=model_id)
            r["frame"] = frame.name
            model_results.append(r)

            status = "NKO" if r.get("has_nko") else "---"
            print(f"  [{status}] {frame.name}: {r.get('text', '')[:60]} ({r['time']:.1f}s)")

        nko_count = sum(1 for r in model_results if r.get("has_nko"))
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        avg_chars = sum(r.get("nko_chars", 0) for r in model_results) / len(model_results)

        results[model_id] = {
            "name": model_name,
            "frames_tested": len(model_results),
            "nko_detected": nko_count,
            "avg_time_s": round(avg_time, 2),
            "avg_nko_chars": round(avg_chars, 1),
            "details": model_results,
        }

        print(f"  Summary: {nko_count}/{len(frames)} detected N'Ko, "
              f"avg {avg_chars:.1f} chars, {avg_time:.2f}s/frame")

    # Final comparison table
    print("\n" + "=" * 70)
    print("OCR MODEL COMPARISON — N'Ko Text Extraction")
    print("=" * 70)
    print(f"{'Model':<25} {'NKo Detect':<14} {'Avg Chars':<12} {'Avg Time':<10}")
    print("-" * 70)
    for model_id, stats in results.items():
        print(f"{stats['name']:<25} "
              f"{stats['nko_detected']}/{stats['frames_tested']:<11} "
              f"{stats['avg_nko_chars']:<12.1f} "
              f"{stats['avg_time_s']:<10.2f}s")

    # Save results
    out_path = Path(frames_dir) / "ocr_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-frames", type=int, metavar="N",
                        help="Extract frames from N GCS videos")
    parser.add_argument("--compare", action="store_true",
                        help="Run model comparison on extracted frames")
    parser.add_argument("--frames-dir", default="/tmp/ocr_test_frames")
    parser.add_argument("--model", help="Run single model only")
    args = parser.parse_args()

    if args.extract_frames:
        extract_sample_frames(num_videos=args.extract_frames)

    if args.compare or args.model:
        asyncio.run(run_comparison(args.frames_dir))


if __name__ == "__main__":
    main()
