#!/usr/bin/env python3
"""
Pass 1: Video Extraction + OCR + Audio Segmentation

Downloads videos, extracts frames with scene detection, runs OCR,
and saves audio segments for future ASR transcription.
Stores raw detections in Supabase for Pass 2 consolidation.

Features:
- Robust resume/checkpoint mechanism
- Graceful shutdown on Ctrl+C (SIGINT)
- Multiple download retry strategies
- Automatic cleanup of partial downloads

Usage:
    python run_extraction.py                    # Process all videos
    python run_extraction.py --limit 10         # Process first 10 videos
    python run_extraction.py --resume           # Resume from checkpoint
    python run_extraction.py --dry-run          # List videos without processing
    python run_extraction.py --no-audio         # Skip audio extraction
    python run_extraction.py --retry-failed     # Retry previously failed videos
    python run_extraction.py --from-manifest --resume  # Use existing manifest + resume
"""

import asyncio
import argparse
import json
import os
import sys
import shutil
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nko_analyzer import NkoAnalyzer, load_config
from audio_extractor import AudioExtractor, VideoManifest
from frame_filter import SceneChangeDetector
from supabase_reporter import get_reporter, SupabaseReporter
import scrapetube

# Checkpoint file for resumability
CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "extraction_checkpoint.json"
PROGRESS_FILE = Path(__file__).parent.parent / "data" / "extraction_progress.json"
LOCK_FILE = Path(__file__).parent.parent / "data" / ".extraction.lock"

# Global flag for graceful shutdown
_shutdown_requested = False
_current_video_id = None


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\n⚠ Force quit requested. Exiting immediately...")
        sys.exit(1)
    
    _shutdown_requested = True
    print(f"\n\n⚠ Shutdown requested (signal {signum})")
    print(f"  Finishing current video: {_current_video_id}")
    print(f"  Checkpoint will be saved. Use --resume to continue.")
    print(f"  Press Ctrl+C again to force quit.\n")


def _cleanup_lock():
    """Remove lock file on exit."""
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except:
            pass


def _acquire_lock() -> bool:
    """
    Acquire exclusive lock to prevent concurrent runs.
    
    Returns True if lock acquired, False if another instance is running.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if LOCK_FILE.exists():
        # Check if the process is still running
        try:
            with open(LOCK_FILE) as f:
                lock_data = json.load(f)
            pid = lock_data.get("pid")
            if pid:
                # Check if process exists
                try:
                    os.kill(pid, 0)
                    print(f"⚠ Another extraction is running (PID {pid})")
                    print(f"  Started: {lock_data.get('started')}")
                    print(f"  Use --force to override (not recommended)")
                    return False
                except OSError:
                    # Process doesn't exist, stale lock
                    print(f"  Removing stale lock file (PID {pid} not running)")
        except:
            pass
    
    # Create lock
    with open(LOCK_FILE, "w") as f:
        json.dump({
            "pid": os.getpid(),
            "started": datetime.now().isoformat(),
        }, f)
    
    # Register cleanup
    atexit.register(_cleanup_lock)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    return True


def get_channel_videos(channel_url: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Get all video URLs from a YouTube channel."""
    print(f"Fetching videos from {channel_url}...")
    
    # Extract channel handle - scrapetube needs just the handle without @
    if "@" in channel_url:
        # Extract handle like "babamamadidiane" from "@babamamadidiane"
        channel_id = channel_url.split("@")[-1].split("/")[0].split("?")[0]
    elif "channel/" in channel_url:
        # Handle channel ID format
        channel_id = channel_url.split("channel/")[-1].split("/")[0].split("?")[0]
    else:
        channel_id = channel_url
    
    print(f"  Channel ID: {channel_id}")
    
    videos = []
    try:
        for video in scrapetube.get_channel(channel_id):
            try:
                video_id = video["videoId"]
                title = video["title"]["runs"][0]["text"]
                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                })
                
                if limit and len(videos) >= limit:
                    break
            except (KeyError, IndexError) as e:
                print(f"  Warning: Could not parse video: {e}")
                continue
    except Exception as e:
        print(f"  Error fetching channel: {e}")
        # Fall back to yt-dlp if scrapetube fails
        print("  Trying yt-dlp fallback...")
        try:
            import subprocess
            cmd = [
                "yt-dlp",
                "--flat-playlist",
                "--print", "%(id)s|%(title)s",
                f"https://www.youtube.com/@{channel_id}/videos",
            ]
            if limit:
                cmd.extend(["--playlist-end", str(limit)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "|" in line:
                        video_id, title = line.split("|", 1)
                        videos.append({
                            "video_id": video_id,
                            "title": title,
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                        })
        except Exception as e2:
            print(f"  yt-dlp fallback also failed: {e2}")
    
    print(f"Found {len(videos)} videos")
    return videos


def load_videos_from_manifest(manifest_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load videos from an existing manifest file instead of fetching from YouTube.

    Args:
        manifest_path: Path to manifest file, or None to auto-detect
        limit: Maximum number of videos to return

    Returns:
        List of video dicts with video_id, title, url
    """
    # Auto-detect manifest path
    if manifest_path is None or manifest_path == 'auto':
        manifest_path = Path(__file__).parent.parent / "data" / "video_manifest.json"
    else:
        manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        print(f"Manifest file not found: {manifest_path}")
        return []

    print(f"Loading videos from manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    videos = []
    for v in manifest.get("videos", []):
        # Handle both 'id' and 'video_id' formats
        video_id = v.get("video_id") or v.get("id")
        if not video_id:
            continue

        videos.append({
            "video_id": video_id,
            "title": v.get("title", f"Video {video_id}"),
            "url": v.get("url", f"https://www.youtube.com/watch?v={video_id}"),
        })

        if limit and len(videos) >= limit:
            break

    channel = manifest.get("channel", "unknown")
    print(f"Loaded {len(videos)} videos from {channel} (manifest: {manifest.get('total_videos', '?')} total)")

    return videos


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint for resumability."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_videos": [], "failed_videos": [], "last_updated": None}


def save_checkpoint(checkpoint: Dict[str, Any]):
    """Save checkpoint for resumability."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def save_progress(progress: Dict[str, Any]):
    """Save progress statistics."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


async def run_extraction(
    videos: List[Dict[str, str]],
    config: Dict[str, Any],
    resume: bool = False,
    dry_run: bool = False,
    extract_audio: bool = True,
    channel_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Pass 1: Extraction + OCR + Audio Segmentation on all videos.
    
    Args:
        videos: List of video dicts with video_id, title, url
        config: Configuration dictionary
        resume: Resume from checkpoint
        dry_run: List videos without processing
        extract_audio: Whether to extract and segment audio
        channel_name: Optional channel name for tracking
    
    Returns:
        Progress statistics
    """
    checkpoint = load_checkpoint() if resume else {"completed_videos": [], "failed_videos": []}
    
    # Filter out already completed videos
    if resume:
        completed_ids = set(checkpoint["completed_videos"])
        videos = [v for v in videos if v["video_id"] not in completed_ids]
        print(f"Resuming: {len(completed_ids)} already completed, {len(videos)} remaining")
    
    if dry_run:
        print("\n=== DRY RUN: Videos to process ===")
        for i, v in enumerate(videos, 1):
            print(f"  {i}. [{v['video_id']}] {v['title'][:50]}...")
        return {"dry_run": True, "total_videos": len(videos)}
    
    # Progress tracking
    progress = {
        "start_time": datetime.now().isoformat(),
        "total_videos": len(videos),
        "completed": 0,
        "failed": 0,
        "total_frames": 0,
        "total_detections": 0,
        "total_audio_segments": 0,
        "estimated_cost": 0.0,
    }
    
    # Initialize Supabase reporter for dashboard observability
    reporter = get_reporter()
    run_id = reporter.start_run(
        run_type="extraction",
        channel_name=channel_name,
        videos_total=len(videos),
        metadata={"resume": resume, "extract_audio": extract_audio}
    )
    
    # Get config values
    extraction_config = config.get("extraction", {})
    storage_config = config.get("storage", {}).get("local", {})
    audio_config = config.get("audio", {})
    
    target_frames = extraction_config.get("target_frames", 100)
    keep_frames = storage_config.get("keep_frames", True)
    keep_audio = storage_config.get("keep_audio", True) and extract_audio
    base_dir = storage_config.get("base_dir", "./data/videos")
    
    # Initialize audio extractor if needed
    audio_extractor = None
    scene_detector = None
    if keep_audio:
        audio_extractor = AudioExtractor(
            audio_format=audio_config.get("format", "m4a"),
            audio_bitrate=audio_config.get("bitrate", "128k"),
        )
        scene_detector = SceneChangeDetector(
            threshold=extraction_config.get("scene_threshold", 0.3),
            min_scene_duration=extraction_config.get("min_scene_duration", 2.0),
        )
    
    print(f"\n{'='*60}")
    print(f"PASS 1: EXTRACTION + OCR + AUDIO")
    print(f"{'='*60}")
    print(f"Videos to process: {len(videos)}")
    print(f"Target frames per video: {target_frames}")
    print(f"Keep frames: {keep_frames}")
    print(f"Extract audio: {keep_audio}")
    print(f"Estimated OCR cost: ${len(videos) * 55 * 0.002:.2f}")
    print(f"{'='*60}\n")
    
    # Process videos
    global _current_video_id
    
    async with NkoAnalyzer(
        api_key=os.getenv("GEMINI_API_KEY"),
        generate_worlds=False,  # Pass 1: No world generation
        store_supabase=True,
        config=config,
    ) as analyzer:
        for i, video in enumerate(videos, 1):
            # Check for shutdown request
            if _shutdown_requested:
                print("\n⚠ Shutdown requested. Saving checkpoint and exiting...")
                break
            
            video_id = video["video_id"]
            title = video["title"]
            _current_video_id = video_id
            
            print(f"\n[{i}/{len(videos)}] Processing: {title[:50]}...")
            
            # Report video start to dashboard
            reporter.video_start(video_id, title)
            video_start_time = datetime.now()
            
            # Setup output directory
            video_output_dir = os.path.join(base_dir, video_id)
            temp_dir = str(Path(__file__).parent.parent / "data" / "temp")
            
            try:
                # Analyze video (frames + OCR)
                result = await analyzer.analyze_video(
                    video_id=video_id,
                    title=title,
                    youtube_url=video["url"],
                    temp_dir=temp_dir,
                    target_frames=target_frames,
                    use_smart_extraction=True,
                    use_scene_detection=extraction_config.get("use_scene_detection", True),
                    channel_name=config.get("channel", {}).get("name", "babamamadidiane"),
                )
                
                if result.status == "completed":
                    # Post-processing: Save frames and extract audio
                    audio_segments = 0
                    
                    if keep_frames or keep_audio:
                        # Find downloaded video
                        video_path = None
                        for ext in [".mp4", ".webm", ".mkv"]:
                            candidate = os.path.join(temp_dir, f"{video_id}{ext}")
                            if os.path.exists(candidate):
                                video_path = candidate
                                break
                        
                        if video_path:
                            # Create output directory
                            os.makedirs(video_output_dir, exist_ok=True)
                            
                            # Copy frames to permanent location
                            if keep_frames and result.frames:
                                frames_dir = os.path.join(video_output_dir, "frames")
                                os.makedirs(frames_dir, exist_ok=True)
                                
                                for frame in result.frames:
                                    if frame.frame_path and os.path.exists(frame.frame_path):
                                        dest = os.path.join(frames_dir, os.path.basename(frame.frame_path))
                                        shutil.copy2(frame.frame_path, dest)
                                
                                print(f"  Saved {len(result.frames)} frames to {frames_dir}")
                            
                            # Extract and segment audio
                            if keep_audio and audio_extractor and scene_detector:
                                print(f"  Extracting audio segments...")
                                
                                # Detect scenes for audio segmentation
                                scene_timestamps = scene_detector.detect_scenes(video_path)
                                
                                # Add frame timestamps as scene markers too
                                frame_timestamps = [f.timestamp for f in result.frames if f.has_nko]
                                all_timestamps = sorted(set(scene_timestamps + frame_timestamps))
                                
                                # Create manifest with audio
                                manifest = audio_extractor.process_video(
                                    video_path=video_path,
                                    output_dir=video_output_dir,
                                    video_id=video_id,
                                    title=title,
                                    youtube_url=video["url"],
                                    scene_timestamps=all_timestamps,
                                    channel_name=config.get("channel", {}).get("name"),
                                )
                                
                                # Link frames to scenes
                                frame_infos = [
                                    {
                                        "path": f.frame_path,
                                        "timestamp": f.timestamp,
                                        "has_nko": f.has_nko,
                                        "nko_text": f.nko_text,
                                        "latin_text": f.latin_transliteration,
                                        "english_text": f.english_translation,
                                        "confidence": f.confidence,
                                    }
                                    for f in result.frames
                                ]
                                manifest = audio_extractor.link_frames_to_scenes(manifest, frame_infos)
                                manifest.save(os.path.join(video_output_dir, "manifest.json"))
                                
                                audio_segments = manifest.total_audio_segments
                                print(f"  Created {audio_segments} audio segments")
                        else:
                            print(f"  Warning: Video file not found for post-processing")
                    
                    checkpoint["completed_videos"].append(video_id)
                    progress["completed"] += 1
                    progress["total_frames"] += result.frames_analyzed
                    progress["total_detections"] += result.frames_with_nko
                    progress["total_audio_segments"] += audio_segments
                    progress["estimated_cost"] += result.frames_analyzed * 0.002
                    
                    # Report to dashboard
                    duration_ms = int((datetime.now() - video_start_time).total_seconds() * 1000)
                    reporter.video_complete(
                        video_id=video_id,
                        frames=result.frames_analyzed,
                        detections=result.frames_with_nko,
                        audio_segments=audio_segments,
                        duration_ms=duration_ms
                    )
                    
                    # Log individual detections for live feed
                    for frame in (result.frames or []):
                        if frame.has_nko and frame.nko_text:
                            reporter.detection_found(
                                video_id=video_id,
                                nko_text=frame.nko_text,
                                latin_text=frame.latin_transliteration,
                                confidence=frame.confidence or 0.0
                            )
                    
                    print(f"  ✓ {result.frames_analyzed} frames, {result.frames_with_nko} detections, {audio_segments} audio segments")
                else:
                    checkpoint["failed_videos"].append({"id": video_id, "error": result.error})
                    progress["failed"] += 1
                    reporter.video_failed(video_id, result.error or "Unknown error")
                    print(f"  ✗ Failed: {result.error}")
                
            except Exception as e:
                checkpoint["failed_videos"].append({"id": video_id, "error": str(e)})
                progress["failed"] += 1
                reporter.video_failed(video_id, str(e))
                print(f"  ✗ Exception: {e}")
            
            # Save checkpoint after each video
            save_checkpoint(checkpoint)
            save_progress(progress)
            
            # Rate limiting between videos
            await asyncio.sleep(1)
    
    progress["end_time"] = datetime.now().isoformat()
    save_progress(progress)
    
    # Complete or stop the pipeline run in Supabase
    if _shutdown_requested:
        reporter.stop_run("Graceful shutdown requested by user")
    else:
        reporter.complete_run()
    
    print(f"\n{'='*60}")
    print(f"PASS 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {progress['completed']}/{progress['total_videos']}")
    print(f"Failed: {progress['failed']}")
    print(f"Total frames: {progress['total_frames']}")
    print(f"Total detections: {progress['total_detections']}")
    print(f"Total audio segments: {progress['total_audio_segments']}")
    print(f"Estimated cost: ${progress['estimated_cost']:.2f}")
    print(f"{'='*60}\n")
    
    return progress


def main():
    parser = argparse.ArgumentParser(
        description="Pass 1: Video Extraction + OCR + Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_extraction.py --limit 10        # Process first 10 videos
  python run_extraction.py --resume          # Resume from last checkpoint
  python run_extraction.py --retry-failed    # Retry previously failed videos
  python run_extraction.py --status          # Show current progress
  python run_extraction.py --from-manifest   # Use existing video_manifest.json
        """
    )
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed videos")
    parser.add_argument("--from-manifest", type=str, nargs='?', const='auto',
                        help="Load videos from manifest file instead of YouTube. Use 'auto' or provide path.")
    parser.add_argument("--dry-run", action="store_true", help="List videos without processing")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio extraction")
    parser.add_argument("--force", action="store_true", help="Force run even if another instance running")
    parser.add_argument("--status", action="store_true", help="Show current progress and exit")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--channel", type=str, help="Channel name to process (e.g., 'babamamadidiane', 'ankataa')")
    parser.add_argument("--all-channels", action="store_true", help="Process all configured channels in priority order")
    parser.add_argument("--list-channels", action="store_true", help="List available channels from config")
    args = parser.parse_args()
    
    # Load config first (needed for list-channels and channel selection)
    config = load_config(args.config)
    
    # List available channels
    if args.list_channels:
        print("\n=== Available Channels ===\n")
        channels = config.get("channels", [])
        if not channels:
            # Fallback to single channel config
            channel = config.get("channel", {})
            if channel:
                print(f"  {channel.get('name', 'default')}: {channel.get('url', 'N/A')}")
        else:
            for ch in channels:
                priority = ch.get("priority", "-")
                lang = ch.get("language", "unknown")
                print(f"  [{priority}] {ch.get('name')}: {ch.get('url')}")
                print(f"      Language: {lang}, Description: {ch.get('description', 'N/A')}")
        return
    
    # Show status and exit
    if args.status:
        print("\n=== Extraction Status ===\n")
        
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)
            print(f"Start time: {progress.get('start_time', 'N/A')}")
            print(f"End time: {progress.get('end_time', 'Still running...')}")
            print(f"Videos: {progress.get('completed', 0)}/{progress.get('total_videos', '?')} completed")
            print(f"Failed: {progress.get('failed', 0)}")
            print(f"Frames: {progress.get('total_frames', 0)}")
            print(f"Detections: {progress.get('total_detections', 0)}")
            print(f"Audio segments: {progress.get('total_audio_segments', 0)}")
            print(f"Estimated cost: ${progress.get('estimated_cost', 0):.2f}")
        else:
            print("No extraction has been run yet.")
        
        if CHECKPOINT_FILE.exists():
            checkpoint = load_checkpoint()
            print(f"\nCheckpoint:")
            print(f"  Completed: {len(checkpoint.get('completed_videos', []))} videos")
            print(f"  Failed: {len(checkpoint.get('failed_videos', []))} videos")
            if checkpoint.get('failed_videos'):
                print(f"  Failed IDs:")
                for v in checkpoint.get('failed_videos', [])[:5]:
                    print(f"    - {v.get('id')}: {v.get('error', 'Unknown')[:50]}")
                if len(checkpoint.get('failed_videos', [])) > 5:
                    print(f"    ... and {len(checkpoint.get('failed_videos', [])) - 5} more")
        
        return
    
    # Acquire lock (unless dry-run or forced)
    if not args.dry_run and not args.force:
        if not _acquire_lock():
            print("\nUse --force to override or --status to check progress.")
            sys.exit(1)
    
    # Determine which channel to process
    channel_url = None
    channel_name = None
    
    if args.channel:
        # Look for specified channel in config
        channels = config.get("channels", [])
        for ch in channels:
            if ch.get("name") == args.channel:
                channel_url = ch.get("url")
                channel_name = ch.get("name")
                print(f"Processing channel: {channel_name}")
                break
        
        if not channel_url:
            # Try as a direct URL or handle
            if args.channel.startswith("http") or args.channel.startswith("@"):
                channel_url = args.channel
                channel_name = args.channel.split("@")[-1].split("/")[0]
            else:
                print(f"Channel '{args.channel}' not found in config.")
                print("Use --list-channels to see available channels.")
                sys.exit(1)
    else:
        # Default to first channel or legacy config
        channels = config.get("channels", [])
        if channels:
            # Use highest priority channel (lowest number)
            channels_sorted = sorted(channels, key=lambda x: x.get("priority", 999))
            channel_url = channels_sorted[0].get("url")
            channel_name = channels_sorted[0].get("name")
        else:
            channel_url = config.get("channel", {}).get("url", "https://www.youtube.com/@babamamadidiane")
            channel_name = config.get("channel", {}).get("name", "default")
    
    # Handle all-channels mode
    if args.all_channels:
        channels = config.get("channels", [])
        if not channels:
            channels = config.get("sources", {}).get("channels", [])
        
        if not channels:
            print("No channels found in config!")
            sys.exit(1)
        
        # Sort by priority
        channels_sorted = sorted(channels, key=lambda x: x.get("priority", 999))
        
        print(f"\n=== Processing {len(channels_sorted)} Channels ===\n")
        
        total_progress = {
            "total_videos": 0,
            "completed": 0,
            "failed": 0,
            "total_frames": 0,
            "total_detections": 0,
        }
        
        for ch in channels_sorted:
            ch_name = ch.get("name")
            ch_url = ch.get("url")
            ch_limit = ch.get("max_videos") if args.limit is None else args.limit
            
            print(f"\n--- Channel: {ch_name} ({ch_url}) ---\n")
            
            videos = get_channel_videos(ch_url, limit=ch_limit)
            if not videos:
                print(f"No videos found for {ch_name}, skipping...")
                continue
            
            try:
                progress = asyncio.run(run_extraction(
                    videos=videos,
                    config=config,
                    resume=args.resume,
                    dry_run=args.dry_run,
                    extract_audio=not args.no_audio,
                    channel_name=ch_name,  # Track channel in checkpoint
                ))
                
                # Aggregate progress
                for key in total_progress:
                    total_progress[key] += progress.get(key, 0)
                    
            except KeyboardInterrupt:
                print(f"\n⚠ Interrupted during {ch_name}!")
                break
        
        print(f"\n=== All Channels Complete ===")
        print(f"Total: {total_progress['completed']}/{total_progress['total_videos']} videos")
        print(f"Frames: {total_progress['total_frames']}")
        print(f"Detections: {total_progress['total_detections']}")
        print(f"Failed: {total_progress['failed']}")
        
        _cleanup_lock()
        return
    
    print(f"Channel: {channel_name} ({channel_url})")
    
    if args.retry_failed:
        # Load failed videos from checkpoint
        checkpoint = load_checkpoint()
        failed = checkpoint.get("failed_videos", [])
        if not failed:
            print("No failed videos to retry!")
            sys.exit(0)

        videos = [
            {
                "video_id": v["id"],
                "title": f"Retry: {v['id']}",
                "url": f"https://www.youtube.com/watch?v={v['id']}",
            }
            for v in failed
        ]
        print(f"Retrying {len(videos)} failed videos...")

        # Clear failed list
        checkpoint["failed_videos"] = []
        save_checkpoint(checkpoint)
    elif args.from_manifest:
        # Load videos from existing manifest file (bypasses YouTube fetch)
        manifest_path = None if args.from_manifest == 'auto' else args.from_manifest
        videos = load_videos_from_manifest(manifest_path, limit=args.limit)
    else:
        videos = get_channel_videos(channel_url, limit=args.limit)
    
    if not videos:
        print("No videos found!")
        sys.exit(1)
    
    # Run extraction
    try:
        progress = asyncio.run(run_extraction(
            videos=videos,
            config=config,
            resume=args.resume or args.retry_failed,  # Always resume when retrying
            dry_run=args.dry_run,
            extract_audio=not args.no_audio,
        ))
        
        print(f"\nProgress saved to: {PROGRESS_FILE}")
        print(f"Data saved to: {config.get('storage', {}).get('local', {}).get('base_dir', './data/videos')}")
        
        # Show summary if there were failures
        if progress.get("failed", 0) > 0:
            print(f"\n⚠ {progress['failed']} videos failed. Run with --retry-failed to retry them.")
        
        if _shutdown_requested:
            print(f"\n✓ Graceful shutdown complete. Run with --resume to continue.")
            
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Checkpoint saved. Use --resume to continue.")
    finally:
        _cleanup_lock()


if __name__ == "__main__":
    main()

