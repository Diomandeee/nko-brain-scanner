#!/usr/bin/env python3
"""
Djoko Series Audio Downloader
==============================
Downloads audio from Djoko YouTube episodes and segments into chunks.
Supports streaming mode (download → segment → delete original).

Usage:
    # Download first 50 episodes (audio only)
    python3 asr/download_djoko.py --limit 50

    # Download all, streaming mode (delete after segmenting)
    python3 asr/download_djoko.py --streaming

    # Resume from checkpoint
    python3 asr/download_djoko.py --resume

    # Download babamamadidiane instead of Djoko
    python3 asr/download_djoko.py --channel babamamadidiane
"""

import subprocess
import json
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

CHANNELS = {
    "djoko": {
        "id": "UCXiIHk2N-qWE-ZFPfPVJ06w",
        "name": "Koman Diabate - Film Djoko",
        "url": "https://www.youtube.com/channel/UCXiIHk2N-qWE-ZFPfPVJ06w/videos",
    },
    "babamamadidiane": {
        "id": None,
        "name": "babamamadidiane",
        "url": "https://www.youtube.com/@babamamadidiane/videos",
    },
}

DEFAULT_OUTPUT = Path(__file__).parent.parent / "djoko_audio"
SEGMENT_DURATION = 30  # seconds
CHECKPOINT_FILE = "download_checkpoint.json"


def list_channel_videos(channel_url: str, limit: int = 0) -> List[dict]:
    """List all video IDs and titles from a YouTube channel."""
    cmd = ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", channel_url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        videos = []
        for line in result.stdout.strip().split("\n"):
            if "\t" in line:
                vid_id, title = line.split("\t", 1)
                videos.append({"id": vid_id, "title": title})
        if limit > 0:
            videos = videos[:limit]
        return videos
    except Exception as e:
        print(f"Error listing channel: {e}")
        return []


def download_audio(video_id: str, output_dir: Path, cookies: Optional[str] = None) -> Optional[str]:
    """Download audio from a YouTube video as WAV."""
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        "--force-ipv4",
        "--socket-timeout", "60",
    ]
    if cookies:
        cmd.extend(["--cookies", cookies])
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wav_path = output_dir / f"{video_id}.wav"
        if wav_path.exists():
            return str(wav_path)
        # yt-dlp might have used a different name
        for f in output_dir.glob(f"{video_id}.*"):
            if f.suffix in (".wav", ".mp3", ".m4a", ".opus"):
                return str(f)
    except subprocess.TimeoutExpired:
        print(f"  Timeout downloading {video_id}")
    except Exception as e:
        print(f"  Error downloading {video_id}: {e}")
    return None


def segment_audio(audio_path: str, output_dir: Path, segment_duration: int = 30) -> List[str]:
    """Segment an audio file into fixed-length chunks using ffmpeg."""
    base = Path(audio_path).stem
    segment_dir = output_dir / "segments" / base
    segment_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(segment_dir / f"{base}_%04d.wav")
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c:a", "pcm_s16le",
        "-ar", "16000",  # 16kHz for Whisper
        "-ac", "1",      # mono
        pattern,
        "-y", "-loglevel", "error",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        segments = sorted(segment_dir.glob(f"{base}_*.wav"))
        return [str(s) for s in segments]
    except Exception as e:
        print(f"  Segmentation failed: {e}")
        return []


def load_checkpoint(output_dir: Path) -> dict:
    cp = output_dir / CHECKPOINT_FILE
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return {"downloaded": [], "segmented": [], "errors": [], "total_segments": 0}


def save_checkpoint(output_dir: Path, checkpoint: dict):
    with open(output_dir / CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download Djoko/babamamadidiane audio")
    parser.add_argument("--channel", default="djoko", choices=list(CHANNELS.keys()))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to download")
    parser.add_argument("--segment-duration", type=int, default=SEGMENT_DURATION)
    parser.add_argument("--streaming", action="store_true",
                        help="Delete full audio after segmenting (save disk)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cookies", help="Cookies file for yt-dlp")
    args = parser.parse_args()

    channel = CHANNELS[args.channel]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Channel: {channel['name']}")
    print(f"Output: {output_dir}")

    # List videos
    print("Listing channel videos...")
    videos = list_channel_videos(channel["url"], limit=args.limit)
    print(f"Found {len(videos)} videos")

    if args.dry_run:
        for v in videos[:20]:
            print(f"  {v['id']}: {v['title']}")
        if len(videos) > 20:
            print(f"  ... and {len(videos)-20} more")
        return

    # Load checkpoint
    checkpoint = load_checkpoint(output_dir) if args.resume else {
        "downloaded": [], "segmented": [], "errors": [], "total_segments": 0
    }
    already_done = set(checkpoint["downloaded"])

    remaining = [v for v in videos if v["id"] not in already_done]
    print(f"Already done: {len(already_done)}, remaining: {len(remaining)}")

    start_time = time.time()

    for i, video in enumerate(remaining):
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 1) * 3600

        print(f"\n[{i+1}/{len(remaining)}] {video['title'][:60]}")

        # Download
        audio_path = download_audio(video["id"], output_dir, args.cookies)
        if not audio_path:
            checkpoint["errors"].append(video["id"])
            save_checkpoint(output_dir, checkpoint)
            continue

        # Segment
        segments = segment_audio(audio_path, output_dir, args.segment_duration)
        print(f"  -> {len(segments)} segments")

        checkpoint["downloaded"].append(video["id"])
        checkpoint["segmented"].extend(segments)
        checkpoint["total_segments"] += len(segments)

        # Streaming mode: delete full audio after segmenting
        if args.streaming and segments:
            os.remove(audio_path)
            print(f"  -> deleted {Path(audio_path).name} (streaming mode)")

        # Checkpoint every 10 videos
        if (i + 1) % 10 == 0:
            save_checkpoint(output_dir, checkpoint)
            print(f"  [checkpoint: {checkpoint['total_segments']} total segments, {rate:.0f} vids/hr]")

    save_checkpoint(output_dir, checkpoint)
    total_time = time.time() - start_time
    print(f"\nDone: {len(checkpoint['downloaded'])} downloaded, "
          f"{checkpoint['total_segments']} segments, "
          f"{len(checkpoint['errors'])} errors, "
          f"{total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
