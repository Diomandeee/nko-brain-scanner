#!/usr/bin/env python3
"""
N'Ko Video Analysis Pipeline

Hybrid approach:
1. Resolve YouTube URLs to direct HLS/MP4 streams locally (yt-dlp works here)
2. Send direct stream URLs to Cloud Run backend for frame analysis
3. Collect results and store in Supabase

This bypasses the Deno/yt-dlp issues in Cloud Run by doing URL resolution locally.
"""

import asyncio
import aiohttp
import subprocess
import json
import os
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import scrapetube
except ImportError:
    print("Run: pip install scrapetube aiohttp")
    sys.exit(1)


# Configuration
CLOUD_RUN_URL = os.getenv(
    "ANALYZER_BACKEND_URL", 
    "https://cc-music-pipeline-owq2vk3wya-uc.a.run.app"
)
CHANNEL_URL = "https://www.youtube.com/@babamamadidiane"


@dataclass
class VideoInfo:
    video_id: str
    youtube_url: str
    title: str
    duration: str
    direct_url: Optional[str] = None
    analysis_session_id: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None


def resolve_youtube_url(youtube_url: str, cookies_file: Optional[str] = None) -> Optional[str]:
    """
    Use yt-dlp locally to resolve YouTube URL to direct stream URL.
    This is the key step that fails in Cloud Run but works locally.
    
    We specifically request video-only to ensure we get video frames.
    """
    cmd = [
        "yt-dlp",
        "-g",  # Get URL only
        # Request video format with height <= 720, prefer mp4
        "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        "--no-playlist",
        "--no-warnings",
        "--no-check-certificates",
        "--force-ipv4",
        "--socket-timeout", "30",
    ]
    
    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])
    
    cmd.append(youtube_url)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
        
        if result.returncode == 0:
            # Get first URL (video stream)
            urls = result.stdout.strip().split('\n')
            if urls and urls[0]:
                direct_url = urls[0]
                print(f"  ✓ Resolved to: {direct_url[:80]}...")
                return direct_url
        else:
            print(f"  ✗ yt-dlp error: {result.stderr[:100]}")
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout resolving URL")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return None


async def start_analysis(session: aiohttp.ClientSession, direct_url: str, max_frames: int = 50) -> Optional[str]:
    """Start analysis on Cloud Run backend using the resolved direct URL."""
    try:
        async with session.post(
            f"{CLOUD_RUN_URL}/api/analyze/start",
            json={"url": direct_url, "max_frames": max_frames},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status in (200, 202):
                data = await response.json()
                return data.get("session_id")
            else:
                text = await response.text()
                print(f"  ✗ API error: {response.status} - {text[:100]}")
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
    
    return None


async def get_analysis_status(session: aiohttp.ClientSession, session_id: str) -> Dict[str, Any]:
    """Get analysis session status from Cloud Run."""
    try:
        async with session.get(
            f"{CLOUD_RUN_URL}/api/analyze/status/{session_id}",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        print(f"  ✗ Status check failed: {e}")
    
    return {"status": "error", "error": str(e)}


async def wait_for_completion(
    session: aiohttp.ClientSession, 
    session_id: str, 
    max_wait: int = 300,
    poll_interval: int = 5,
) -> Dict[str, Any]:
    """Poll for analysis completion."""
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < max_wait:
        status = await get_analysis_status(session, session_id)
        
        current_status = status.get("status", "unknown")
        frames = status.get("frames_analyzed", 0)
        
        if current_status == "completed":
            return status
        elif current_status in ("failed", "error"):
            return status
        
        print(f"    Status: {current_status} | Frames: {frames}")
        await asyncio.sleep(poll_interval)
    
    return {"status": "timeout", "error": f"Timed out after {max_wait}s"}


def get_channel_videos(limit: Optional[int] = None) -> List[VideoInfo]:
    """Fetch video list from YouTube channel."""
    print(f"Fetching videos from {CHANNEL_URL}...")
    
    videos = []
    try:
        for i, video in enumerate(scrapetube.get_channel(channel_url=CHANNEL_URL)):
            if limit and i >= limit:
                break
            
            video_id = video.get("videoId", "")
            title_runs = video.get("title", {}).get("runs", [])
            title = title_runs[0].get("text", "Unknown") if title_runs else "Unknown"
            duration = video.get("lengthText", {}).get("simpleText", "Unknown")
            
            videos.append(VideoInfo(
                video_id=video_id,
                youtube_url=f"https://www.youtube.com/watch?v={video_id}",
                title=title,
                duration=duration,
            ))
            
    except Exception as e:
        print(f"Error fetching videos: {e}")
    
    print(f"Found {len(videos)} videos")
    return videos


async def process_video(
    session: aiohttp.ClientSession,
    video: VideoInfo,
    cookies_file: Optional[str] = None,
    max_frames: int = 50,
) -> VideoInfo:
    """Process a single video through the pipeline."""
    print(f"\n[{video.video_id}] {video.title[:50]}...")
    
    # Step 1: Resolve YouTube URL to direct stream (locally)
    print("  Resolving YouTube URL...")
    video.direct_url = resolve_youtube_url(video.youtube_url, cookies_file)
    
    if not video.direct_url:
        video.status = "resolution_failed"
        video.error = "Failed to resolve YouTube URL"
        return video
    
    # Step 2: Send direct URL to Cloud Run for analysis
    print("  Starting Cloud Run analysis...")
    video.analysis_session_id = await start_analysis(session, video.direct_url, max_frames)
    
    if not video.analysis_session_id:
        video.status = "api_failed"
        video.error = "Failed to start analysis"
        return video
    
    print(f"  Session: {video.analysis_session_id}")
    
    # Step 3: Wait for completion
    print("  Waiting for analysis...")
    result = await wait_for_completion(session, video.analysis_session_id)
    
    video.status = result.get("status", "unknown")
    if result.get("error"):
        video.error = result["error"]
    
    frames = result.get("frames_analyzed", 0)
    print(f"  ✓ Completed: {frames} frames analyzed")
    
    return video


async def run_pipeline(
    limit: Optional[int] = None,
    cookies_file: Optional[str] = None,
    max_frames: int = 50,
    output_file: str = "pipeline_results.json",
):
    """Run the full analysis pipeline."""
    print("=" * 60)
    print("N'Ko Video Analysis Pipeline")
    print(f"Backend: {CLOUD_RUN_URL}")
    print("=" * 60)
    
    # Get video list
    videos = get_channel_videos(limit)
    
    if not videos:
        print("No videos to process!")
        return
    
    # Process videos
    results = []
    async with aiohttp.ClientSession() as session:
        for video in videos:
            processed = await process_video(
                session, video, cookies_file, max_frames
            )
            results.append(processed)
    
    # Summary
    success = sum(1 for v in results if v.status == "completed")
    failed = sum(1 for v in results if v.status not in ("completed", "pending"))
    
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Completed: {success}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(results)}")
    
    # Save results
    output_data = {
        "run_time": datetime.now().isoformat(),
        "backend": CLOUD_RUN_URL,
        "results": [
            {
                "video_id": v.video_id,
                "title": v.title,
                "youtube_url": v.youtube_url,
                "direct_url": v.direct_url,
                "session_id": v.analysis_session_id,
                "status": v.status,
                "error": v.error,
            }
            for v in results
        ]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="N'Ko Video Analysis Pipeline")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of videos")
    parser.add_argument("--cookies", type=str, help="Path to cookies file")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames per video")
    parser.add_argument("--output", "-o", type=str, default="pipeline_results.json")
    parser.add_argument(
        "--test-url", type=str, 
        help="Test with a single YouTube URL instead of channel"
    )
    
    args = parser.parse_args()
    
    if args.test_url:
        # Test mode: single video
        print(f"Testing with: {args.test_url}")
        direct_url = resolve_youtube_url(args.test_url, args.cookies)
        if direct_url:
            print(f"\n✓ Direct URL:\n{direct_url}")
        else:
            print("\n✗ Failed to resolve URL")
        return
    
    # Run full pipeline
    asyncio.run(run_pipeline(
        limit=args.limit,
        cookies_file=args.cookies,
        max_frames=args.max_frames,
        output_file=args.output,
    ))


if __name__ == "__main__":
    main()

