#!/usr/bin/env python3
"""
Djoko Audio Extraction Pipeline
================================
Downloads Bambara dialogue episodes from the Djoko YouTube series,
applies Voice Activity Detection (VAD) to segment speech regions,
and produces a manifest for downstream ASR training.

Usage:
    # Download + VAD for episodes 1-5
    python3 audio_pipeline.py --download --vad --limit 5

    # VAD only on already-downloaded files
    python3 audio_pipeline.py --vad

    # Download only (no segmentation)
    python3 audio_pipeline.py --download --limit 50

    # Full pipeline: download 50 + segment
    python3 audio_pipeline.py --download --vad --limit 50
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Force unbuffered output for SSH compatibility
print = functools.partial(print, flush=True)

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHANNEL_URL = "https://www.youtube.com/channel/UCXiIHk2N-qWE-ZFPfPVJ06w/videos"
BASE_DIR = Path.home() / "nko-brain-scanner"
AUDIO_DIR = BASE_DIR / "data" / "djoko-audio"
SEGMENTS_DIR = AUDIO_DIR / "segments"
ARCHIVE_FILE = AUDIO_DIR / "downloaded.txt"
MANIFEST_FILE = AUDIO_DIR / "manifest.jsonl"
EPISODES_FILE = AUDIO_DIR / "episodes.json"
LISTING_CACHE = AUDIO_DIR / "channel_listing.json"

# VAD parameters
VAD_SAMPLE_RATE = 16000  # silero-vad requires 16kHz
VAD_THRESHOLD = 0.5      # speech probability threshold
MIN_SPEECH_SEC = 2.0     # minimum segment duration
MAX_SPEECH_SEC = 15.0    # maximum segment duration
PAD_SEC = 0.3            # padding around speech boundaries

# ---------------------------------------------------------------------------
# Step 1: Channel listing
# ---------------------------------------------------------------------------

def fetch_channel_listing(limit: int = 0) -> list[dict]:
    """Fetch video IDs and titles from the channel, filtering Djoko episodes only."""
    print(f"[listing] Fetching channel videos...")

    cmd = [
        "yt-dlp", "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(duration)s",
        CHANNEL_URL,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[listing] WARNING: yt-dlp stderr: {result.stderr[:500]}")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        vid_id, title = parts[0], parts[1]
        duration = parts[2] if len(parts) > 2 else "0"

        # Filter: only Djoko episodes
        title_lower = title.lower()
        if "djoko" not in title_lower:
            continue
        if "episode" not in title_lower and "épisode" not in title_lower:
            continue

        # Extract episode number
        ep_num = None
        for word in title.split():
            try:
                # Handle cases like "953-" with trailing dash
                cleaned = word.strip("-").strip()
                ep_num = int(cleaned)
                if ep_num > 0 and ep_num < 2000:
                    break
                ep_num = None
            except ValueError:
                continue

        videos.append({
            "id": vid_id,
            "title": title,
            "episode": ep_num,
            "duration": float(duration) if duration and duration != "NA" else 0,
        })

    # Sort by episode number (None last)
    videos.sort(key=lambda v: (v["episode"] is None, v["episode"] or 9999))

    # Deduplicate by episode number (keep first occurrence)
    seen_eps = set()
    deduped = []
    for v in videos:
        if v["episode"] is not None and v["episode"] in seen_eps:
            continue
        if v["episode"] is not None:
            seen_eps.add(v["episode"])
        deduped.append(v)

    videos = deduped

    # Cache the listing
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with open(LISTING_CACHE, "w") as f:
        json.dump(videos, f, indent=2)

    print(f"[listing] Found {len(videos)} Djoko episodes")

    if limit > 0:
        videos = videos[:limit]
        print(f"[listing] Limited to first {limit} episodes")

    return videos


def load_cached_listing(limit: int = 0) -> list[dict] | None:
    """Load cached listing if fresh (< 24 hours old)."""
    if not LISTING_CACHE.exists():
        return None
    age = time.time() - LISTING_CACHE.stat().st_mtime
    if age > 86400:  # 24 hours
        return None
    with open(LISTING_CACHE) as f:
        videos = json.load(f)
    if limit > 0:
        videos = videos[:limit]
    return videos


# ---------------------------------------------------------------------------
# Step 2: Download
# ---------------------------------------------------------------------------

def download_episodes(videos: list[dict]) -> list[Path]:
    """Download episodes as opus audio files with resume support."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-downloaded IDs
    downloaded = set()
    if ARCHIVE_FILE.exists():
        with open(ARCHIVE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    # yt-dlp archive format: "youtube VIDEO_ID"
                    parts = line.split()
                    if len(parts) >= 2:
                        downloaded.add(parts[1])
                    else:
                        downloaded.add(parts[0])

    to_download = [v for v in videos if v["id"] not in downloaded]
    print(f"[download] {len(to_download)} new episodes to download ({len(downloaded)} already done)")

    paths = []
    for i, video in enumerate(to_download):
        vid_id = video["id"]
        ep_str = f"Ep{video['episode']}" if video["episode"] else vid_id
        print(f"[download] [{i+1}/{len(to_download)}] {ep_str} ({vid_id})")

        # Check disk space before each download
        stat = os.statvfs(str(AUDIO_DIR))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb < 5:
            print(f"[download] STOPPING: Only {free_gb:.1f}GB free (need 5GB minimum)")
            break

        url = f"https://www.youtube.com/watch?v={vid_id}"
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "opus",
            "-o", str(AUDIO_DIR / "%(id)s.%(ext)s"),
            "--download-archive", str(ARCHIVE_FILE),
            "--no-playlist",
            "--retries", "3",
            "--socket-timeout", "30",
            url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                opus_path = AUDIO_DIR / f"{vid_id}.opus"
                if opus_path.exists():
                    paths.append(opus_path)
                    print(f"  -> {opus_path.name} ({opus_path.stat().st_size / 1024 / 1024:.1f}MB)")
                else:
                    # yt-dlp may save as .webm if opus extraction fails
                    webm_path = AUDIO_DIR / f"{vid_id}.webm"
                    if webm_path.exists():
                        paths.append(webm_path)
                        print(f"  -> {webm_path.name} (webm fallback)")
            else:
                print(f"  -> FAILED: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  -> TIMEOUT (skipping)")
        except Exception as e:
            print(f"  -> ERROR: {e}")

        # Brief delay to be respectful
        time.sleep(1)

    # Collect all opus files
    all_paths = sorted(AUDIO_DIR.glob("*.opus"))
    print(f"[download] Total audio files on disk: {len(all_paths)}")
    return all_paths


# ---------------------------------------------------------------------------
# Step 3: VAD Segmentation
# ---------------------------------------------------------------------------

def load_vad_model():
    """Load Silero VAD model."""
    print("[vad] Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model, utils


def load_audio_16k(audio_path: Path) -> torch.Tensor:
    """Load audio file and resample to 16kHz mono using ffmpeg (codec-agnostic)."""
    # Use ffmpeg to decode any format to raw 16kHz mono float32 PCM
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ar", str(VAD_SAMPLE_RATE),
        "-ac", "1",
        "-f", "f32le",  # raw float32 little-endian
        "-acodec", "pcm_f32le",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()[:200]}")

    # Convert raw bytes to numpy then torch
    audio_np = np.frombuffer(result.stdout, dtype=np.float32)
    return torch.from_numpy(audio_np.copy())


def merge_speech_segments(timestamps: list[dict]) -> list[dict]:
    """Merge speech timestamps, enforce min/max duration, add padding."""
    if not timestamps:
        return []

    merged = []
    current = {
        "start": max(0, timestamps[0]["start"] - PAD_SEC),
        "end": timestamps[0]["end"] + PAD_SEC,
    }

    for ts in timestamps[1:]:
        gap = ts["start"] - current["end"]
        projected_duration = ts["end"] + PAD_SEC - current["start"]

        # Merge if gap < 0.5s and combined duration < max
        if gap < 0.5 and projected_duration <= MAX_SPEECH_SEC:
            current["end"] = ts["end"] + PAD_SEC
        else:
            # Finalize current segment
            duration = current["end"] - current["start"]
            if duration >= MIN_SPEECH_SEC:
                merged.append(current)
            # Start new segment
            current = {
                "start": max(0, ts["start"] - PAD_SEC),
                "end": ts["end"] + PAD_SEC,
            }

    # Don't forget last segment
    duration = current["end"] - current["start"]
    if duration >= MIN_SPEECH_SEC:
        merged.append(current)

    # Split segments that exceed max duration
    final = []
    for seg in merged:
        duration = seg["end"] - seg["start"]
        if duration <= MAX_SPEECH_SEC:
            final.append(seg)
        else:
            # Split into roughly equal chunks
            n_chunks = int(duration / MAX_SPEECH_SEC) + 1
            chunk_dur = duration / n_chunks
            for i in range(n_chunks):
                chunk_start = seg["start"] + i * chunk_dur
                chunk_end = seg["start"] + (i + 1) * chunk_dur
                if chunk_end - chunk_start >= MIN_SPEECH_SEC:
                    final.append({"start": chunk_start, "end": chunk_end})

    return final


def segment_episode(
    audio_path: Path,
    vad_model,
    vad_utils,
    episode_id: str,
) -> list[dict]:
    """Run VAD on a single episode and save speech segments."""
    get_speech_timestamps = vad_utils[0]

    # Load and resample
    waveform = load_audio_16k(audio_path)
    total_duration = len(waveform) / VAD_SAMPLE_RATE

    # Run VAD
    speech_timestamps = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
    )

    # Convert sample indices to seconds
    timestamps_sec = [
        {"start": ts["start"] / VAD_SAMPLE_RATE, "end": ts["end"] / VAD_SAMPLE_RATE}
        for ts in speech_timestamps
    ]

    # Merge and filter segments
    segments = merge_speech_segments(timestamps_sec)

    if not segments:
        print(f"  -> No speech segments found in {audio_path.name}")
        return []

    # Create episode segment directory
    ep_dir = SEGMENTS_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Export segments as opus files using ffmpeg
    manifest_entries = []
    for i, seg in enumerate(segments):
        seg_name = f"seg_{i:04d}.opus"
        seg_path = ep_dir / seg_name
        duration = seg["end"] - seg["start"]

        # Use ffmpeg to extract segment as 16kHz mono opus
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", f"{seg['start']:.3f}",
            "-t", f"{duration:.3f}",
            "-ar", "16000", "-ac", "1",
            "-c:a", "libopus", "-b:a", "32k",
            "-loglevel", "error",
            str(seg_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except subprocess.CalledProcessError as e:
            print(f"  -> ffmpeg error on {seg_name}: {e.stderr[:100] if e.stderr else 'unknown'}")
            continue
        except subprocess.TimeoutExpired:
            print(f"  -> ffmpeg timeout on {seg_name}")
            continue

        entry = {
            "episode": episode_id,
            "segment": seg_name,
            "path": str(seg_path.relative_to(BASE_DIR)),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "duration": round(duration, 3),
        }
        manifest_entries.append(entry)

    speech_ratio = sum(s["duration"] for s in manifest_entries) / total_duration if total_duration > 0 else 0
    print(
        f"  -> {len(manifest_entries)} segments, "
        f"{sum(s['duration'] for s in manifest_entries):.0f}s speech / "
        f"{total_duration:.0f}s total ({speech_ratio:.0%} speech)"
    )

    return manifest_entries


def run_vad(audio_files: list[Path], video_lookup: dict[str, dict]):
    """Run VAD segmentation on all downloaded audio files."""
    vad_model, vad_utils = load_vad_model()

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to support incremental runs
    existing_episodes = set()
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing_episodes.add(entry["episode"])
                except (json.JSONDecodeError, KeyError):
                    continue

    all_entries = []
    to_process = []
    for path in audio_files:
        episode_id = path.stem
        if episode_id in existing_episodes:
            continue
        to_process.append(path)

    print(f"[vad] {len(to_process)} episodes to segment ({len(existing_episodes)} already done)")

    for i, path in enumerate(to_process):
        episode_id = path.stem
        info = video_lookup.get(episode_id, {})
        ep_num = info.get("episode", "?")
        print(f"[vad] [{i+1}/{len(to_process)}] Ep{ep_num} ({episode_id})")

        try:
            entries = segment_episode(path, vad_model, vad_utils, episode_id)
            all_entries.extend(entries)
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

        # Reset VAD model state between episodes
        vad_model.reset_states()

    # Append to manifest
    if all_entries:
        with open(MANIFEST_FILE, "a") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"[vad] Wrote {len(all_entries)} new entries to manifest")

    # Report totals
    total_entries = len(existing_episodes) + len(all_entries)
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            all_segs = [json.loads(l) for l in f if l.strip()]
        total_hrs = sum(s["duration"] for s in all_segs) / 3600
        total_eps = len(set(s["episode"] for s in all_segs))
        print(f"[vad] TOTAL: {total_eps} episodes, {len(all_segs)} segments, {total_hrs:.1f} hours")


# ---------------------------------------------------------------------------
# Step 4: Episode metadata
# ---------------------------------------------------------------------------

def build_episodes_json(audio_files: list[Path], video_lookup: dict[str, dict]):
    """Build episodes.json with metadata for all downloaded episodes."""
    episodes = []
    for path in audio_files:
        vid_id = path.stem
        info = video_lookup.get(vid_id, {})

        # Get actual file duration via ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            file_duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        except Exception:
            file_duration = 0

        episodes.append({
            "id": vid_id,
            "title": info.get("title", ""),
            "episode": info.get("episode"),
            "file": str(path.relative_to(BASE_DIR)),
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "duration_sec": round(file_duration, 1),
        })

    episodes.sort(key=lambda e: (e["episode"] is None, e["episode"] or 9999))

    with open(EPISODES_FILE, "w") as f:
        json.dump({"count": len(episodes), "episodes": episodes}, f, indent=2)

    print(f"[metadata] Wrote {len(episodes)} episodes to {EPISODES_FILE.name}")
    total_size = sum(e["file_size_mb"] for e in episodes)
    total_dur = sum(e["duration_sec"] for e in episodes)
    print(f"[metadata] Total size: {total_size:.1f}MB, Total duration: {total_dur/3600:.1f}h")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Djoko Audio Extraction Pipeline")
    parser.add_argument("--download", action="store_true", help="Download episodes")
    parser.add_argument("--vad", action="store_true", help="Run VAD segmentation")
    parser.add_argument("--metadata", action="store_true", help="Build episodes.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes (0 = all)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (download + vad + metadata)")
    args = parser.parse_args()

    if args.all:
        args.download = True
        args.vad = True
        args.metadata = True

    if not (args.download or args.vad or args.metadata):
        parser.print_help()
        sys.exit(1)

    # Ensure directories exist
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Get video listing
    videos = load_cached_listing(args.limit) or fetch_channel_listing(args.limit)
    video_lookup = {v["id"]: v for v in videos}

    # Step 1: Download
    if args.download:
        audio_files = download_episodes(videos)
    else:
        audio_files = sorted(AUDIO_DIR.glob("*.opus"))
        print(f"[skip-download] Found {len(audio_files)} existing audio files")

    if not audio_files:
        print("[pipeline] No audio files available. Run with --download first.")
        sys.exit(1)

    # Step 2: VAD
    if args.vad:
        run_vad(audio_files, video_lookup)

    # Step 3: Metadata
    if args.metadata or args.download:
        build_episodes_json(audio_files, video_lookup)

    print("[pipeline] Done.")


if __name__ == "__main__":
    main()
