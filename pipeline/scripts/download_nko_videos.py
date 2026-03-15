#!/usr/bin/env python3
"""
N'Ko Video Downloader

Downloads videos from the @babamamadidiane YouTube channel for N'Ko learning.
Uses scrapetube to get video URLs and yt-dlp to download.

Usage:
    python download_nko_videos.py --output ./downloads --limit 10
    python download_nko_videos.py --output ./downloads --all
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import subprocess
import argparse
import json
import re
import os

# Try to import scrapetube, provide instructions if missing
try:
    import scrapetube
except ImportError:
    print("scrapetube not installed. Run: pip install scrapetube")
    exit(1)


# Channel configuration
CHANNEL_URL = "https://www.youtube.com/@babamamadidiane"
CHANNEL_HANDLE = "@babamamadidiane"
# Note: scrapetube needs the channel URL, not the handle
CHANNEL_ID = "UCQq8Z3H2M8VqR3N7x8xwK-g"  # This may need to be updated


def extract_video_info(video_data: Dict) -> Dict[str, str]:
    """Extract video URL and metadata from scrapetube video data."""
    base_video_url = "https://www.youtube.com/watch?v="
    video_id = video_data.get("videoId", "")
    
    # Get title from nested structure
    title_runs = video_data.get("title", {}).get("runs", [])
    video_title = title_runs[0].get("text", "Unknown") if title_runs else "Unknown"
    
    # Get duration if available
    length_text = video_data.get("lengthText", {}).get("simpleText", "Unknown")
    
    # Get view count
    view_count = video_data.get("viewCountText", {}).get("simpleText", "0 views")
    
    return {
        "video_id": video_id,
        "url": f"{base_video_url}{video_id}",
        "title": video_title,
        "duration": length_text,
        "views": view_count,
    }


def get_channel_videos(channel_url: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Get all video URLs and metadata from a YouTube channel."""
    print(f"Fetching videos from channel: {channel_url}")
    
    try:
        # scrapetube can accept channel_url parameter
        videos = scrapetube.get_channel(channel_url=channel_url)
        video_list = []
        
        for i, video in enumerate(videos):
            if limit and i >= limit:
                break
            video_info = extract_video_info(video)
            video_list.append(video_info)
            print(f"  Found: {video_info['title'][:50]}...")
            
        print(f"\nTotal: {len(video_list)} videos found")
        return video_list
        
    except Exception as e:
        print(f"Error fetching channel videos: {e}")
        # Try alternative method
        print("Trying with channel handle...")
        try:
            # Extract handle from URL
            if "@" in channel_url:
                handle = channel_url.split("@")[-1].split("/")[0]
                videos = scrapetube.get_channel(channel_username=handle)
                video_list = []
                for i, video in enumerate(videos):
                    if limit and i >= limit:
                        break
                    video_info = extract_video_info(video)
                    video_list.append(video_info)
                print(f"Found {len(video_list)} videos")
                return video_list
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
        return []


def download_video(
    video_url: str,
    output_path: str,
    format_selector: str = "best[height<=720]",
    cookies_file: Optional[str] = None,
) -> Optional[str]:
    """
    Download a video using yt-dlp.
    
    Args:
        video_url: YouTube video URL
        output_path: Directory to save the video
        format_selector: yt-dlp format string
        cookies_file: Optional path to cookies file
        
    Returns:
        Path to downloaded file or None if failed
    """
    os.makedirs(output_path, exist_ok=True)
    
    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "-o", os.path.join(output_path, "%(title)s.%(ext)s"),
        "--no-playlist",
        "--no-warnings",
        "--no-check-certificates",
        "--force-ipv4",
        "--socket-timeout", "30",
        "--retries", "3",
    ]
    
    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])
    
    # Add the URL
    cmd.append(video_url)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per video
        )
        
        if result.returncode == 0:
            # Try to find the downloaded file
            # yt-dlp doesn't give us the exact filename, so we look for recent files
            return output_path
        else:
            print(f"yt-dlp error: {result.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Download timed out for {video_url}")
        return None
    except Exception as e:
        print(f"Download error: {e}")
        return None


def download_videos_batch(
    videos: List[Dict[str, str]],
    output_path: str,
    format_selector: str = "best[height<=720]",
    cookies_file: Optional[str] = None,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Download multiple videos with progress tracking.
    
    Returns:
        List of download results with status
    """
    results = []
    
    for video in tqdm(videos, desc="Downloading videos"):
        video_id = video["video_id"]
        title = video["title"]
        url = video["url"]
        
        # Check if already downloaded (simple check by video ID in filename)
        if skip_existing:
            existing_files = list(Path(output_path).glob(f"*{video_id}*"))
            if existing_files:
                print(f"Skipping (exists): {title[:50]}")
                results.append({
                    **video,
                    "status": "skipped",
                    "path": str(existing_files[0]),
                })
                continue
        
        print(f"\nDownloading: {title[:60]}...")
        
        result_path = download_video(
            url, 
            output_path, 
            format_selector,
            cookies_file,
        )
        
        results.append({
            **video,
            "status": "success" if result_path else "failed",
            "path": result_path,
        })
    
    return results


def save_video_index(videos: List[Dict], output_path: str) -> str:
    """Save video metadata to JSON index file."""
    index_path = os.path.join(output_path, "video_index.json")
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "channel": CHANNEL_URL,
            "video_count": len(videos),
            "videos": videos,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved video index to: {index_path}")
    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Download N'Ko learning videos from @babamamadidiane channel"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./nko_videos",
        help="Output directory for downloaded videos",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of videos to download (default: all)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="best[height<=720]",
        help="yt-dlp format selector (default: best[height<=720])",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to cookies.txt file for authentication",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only fetch video list and save index, don't download",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip videos that are already downloaded",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Fetch video list
    print(f"\n{'='*60}")
    print(f"N'Ko Video Downloader")
    print(f"Channel: {CHANNEL_URL}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    videos = get_channel_videos(CHANNEL_URL, limit=args.limit)
    
    if not videos:
        print("No videos found!")
        return
    
    # Save index
    save_video_index(videos, args.output)
    
    if args.index_only:
        print("\nIndex saved. Use without --index-only to download videos.")
        return
    
    # Download videos
    print(f"\nDownloading {len(videos)} videos...")
    results = download_videos_batch(
        videos,
        args.output,
        format_selector=args.format,
        cookies_file=args.cookies,
        skip_existing=args.skip_existing,
    )
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {len(results)}")
    print(f"{'='*60}\n")
    
    # Save results
    results_path = os.path.join(args.output, "download_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

