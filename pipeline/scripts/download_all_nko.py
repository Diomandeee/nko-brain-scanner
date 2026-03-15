#!/usr/bin/env python3
"""
Download All N'Ko Sources to GCS
================================
Downloads videos from multiple YouTube channels, playlists, and individual videos.
All videos are uploaded to Google Cloud Storage.

Sources:
- Multiple N'Ko education channels
- Specific playlists and individual videos
- Uses cookies for authenticated access (bypasses rate limiting)
"""

import subprocess
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
import re

# Configuration
GCS_BUCKET = "learnnko-videos"
TEMP_DIR = "/Users/mohameddiomande/projects/LearnNKo/pipeline/scripts/temp_gcs"
COOKIES_FILE = "/Users/mohameddiomande/projects/LearnNKo/pipeline/scripts/cookies.txt"
CHECKPOINT_FILE = "all_nko_checkpoint.json"

# ============================================================================
# N'KO VIDEO SOURCES
# ============================================================================

CHANNELS = [
    # Primary channel (already processing)
    {"name": "babamamadidiane", "url": "https://www.youtube.com/@babamamadidiane", "priority": 1},
    
    # High Quality Reading NKo
    {"name": "youbouba", "url": "https://www.youtube.com/@youbouba", "priority": 1},
    
    # Additional N'Ko education channels
    {"name": "projetnkopourtous1379", "url": "https://www.youtube.com/@projetnkopourtous1379", "priority": 2},
    {"name": "moussadiallo9872", "url": "https://www.youtube.com/@moussadiallo9872", "priority": 2},
    {"name": "lonytv", "url": "https://www.youtube.com/@lonytv", "priority": 3},
    {"name": "alieulawato3958", "url": "https://www.youtube.com/@alieulawato3958", "priority": 3},
    {"name": "mamadibabadiane1", "url": "https://www.youtube.com/@mamadibabadiane1", "priority": 2},
]

PLAYLISTS = [
    {"name": "nko_playlist_1", "url": "https://www.youtube.com/playlist?list=PLtMp7uW6QUU8PTS7B4seI16RaXUjin_dF"},
]

INDIVIDUAL_VIDEOS = [
    "LYP9Zrs_Yrw",
    "JYcD83VwiAo", 
    "FnDT2QobQgQ",
    "lJJVOPJaDe8",
    "PHnH5bitdiE",
    "bYSDgXMAue4",
    "Ks441MQYUmg",
    "oX6Ky30qDKI",
    "9Y6pjOQn81Q",
    "irK99wKpM8o",
]


@dataclass
class VideoTask:
    video_id: str
    title: str
    source: str  # channel name, playlist name, or "individual"
    status: str = "pending"
    size_bytes: int = 0
    error: Optional[str] = None


class MultiSourceDownloader:
    def __init__(
        self,
        bucket: str = GCS_BUCKET,
        temp_dir: str = TEMP_DIR,
        cookies_file: str = COOKIES_FILE,
        max_height: int = 720,
        delay: float = 15.0,
    ):
        self.bucket = bucket
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_height = max_height
        self.delay = delay
        
        # Cookies
        self.cookies_file = cookies_file if Path(cookies_file).exists() else None
        if self.cookies_file:
            print(f"🍪 Using cookies: {self.cookies_file}")
        else:
            print("⚠️ No cookies file - may hit rate limits")
        
        # Checkpoint
        self.checkpoint_path = self.temp_dir / CHECKPOINT_FILE
        self.checkpoint = self._load_checkpoint()
        
        # Stats
        self.session_stats = {
            "downloaded": 0,
            "uploaded": 0,
            "failed": 0,
            "bytes": 0,
            "start_time": time.time(),
        }
    
    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                data = json.load(f)
                # Convert old list format to dict format for failed
                if isinstance(data.get("failed"), list):
                    old_failed = data["failed"]
                    data["failed"] = {vid: "unknown_error" for vid in old_failed}
                    print(f"   ⚙️ Converted {len(old_failed)} failed entries to new format")
                return data
        return {
            "completed": [],  # Video IDs successfully in GCS
            "failed": {},     # video_id -> error
            "sources_scanned": [],  # Channels/playlists already scanned
            "total_bytes": 0,
        }
    
    def _save_checkpoint(self):
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _run_ytdlp(self, args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
        """Run yt-dlp with common options."""
        cmd = [sys.executable, "-m", "yt_dlp"]
        
        if self.cookies_file:
            cmd.extend(["--cookies", self.cookies_file])
        
        cmd.extend(args)
        
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    
    def get_channel_videos(self, channel_url: str) -> List[Dict]:
        """Get all video IDs from a channel."""
        print(f"   Scanning channel...")
        
        result = self._run_ytdlp([
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s",
            f"{channel_url}/videos"
        ], timeout=180)
        
        if result.returncode != 0:
            print(f"   ❌ Error: {result.stderr[:100]}")
            return []
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if '\t' in line:
                vid_id, title = line.split('\t', 1)
                if vid_id and len(vid_id) == 11:
                    videos.append({"id": vid_id, "title": title})
        
        return videos
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict]:
        """Get all video IDs from a playlist."""
        print(f"   Scanning playlist...")
        
        result = self._run_ytdlp([
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s",
            playlist_url
        ], timeout=180)
        
        if result.returncode != 0:
            print(f"   ❌ Error: {result.stderr[:100]}")
            return []
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if '\t' in line:
                vid_id, title = line.split('\t', 1)
                if vid_id and len(vid_id) == 11:
                    videos.append({"id": vid_id, "title": title})
        
        return videos
    
    def collect_all_videos(self) -> List[VideoTask]:
        """Collect videos from all sources."""
        all_videos: Dict[str, VideoTask] = {}  # Dedup by video ID
        
        print("\n" + "=" * 60)
        print("📋 Collecting videos from all sources")
        print("=" * 60)
        
        # 1. Individual videos (highest priority)
        print(f"\n🎬 Individual videos: {len(INDIVIDUAL_VIDEOS)}")
        for vid_id in INDIVIDUAL_VIDEOS:
            if vid_id not in self.checkpoint["completed"]:
                all_videos[vid_id] = VideoTask(
                    video_id=vid_id,
                    title=f"Individual video {vid_id}",
                    source="individual"
                )
        
        # 2. Playlists
        print(f"\n📑 Playlists: {len(PLAYLISTS)}")
        for playlist in PLAYLISTS:
            print(f"   • {playlist['name']}")
            videos = self.get_playlist_videos(playlist['url'])
            print(f"     Found {len(videos)} videos")
            
            for v in videos:
                if v['id'] not in self.checkpoint["completed"] and v['id'] not in all_videos:
                    all_videos[v['id']] = VideoTask(
                        video_id=v['id'],
                        title=v['title'],
                        source=playlist['name']
                    )
        
        # 3. Channels (sorted by priority)
        sorted_channels = sorted(CHANNELS, key=lambda x: x.get('priority', 99))
        print(f"\n📺 Channels: {len(CHANNELS)}")
        
        for channel in sorted_channels:
            print(f"   • @{channel['name']} (priority {channel.get('priority', 99)})")
            videos = self.get_channel_videos(channel['url'])
            print(f"     Found {len(videos)} videos")
            
            for v in videos:
                if v['id'] not in self.checkpoint["completed"] and v['id'] not in all_videos:
                    all_videos[v['id']] = VideoTask(
                        video_id=v['id'],
                        title=v['title'],
                        source=channel['name']
                    )
        
        tasks = list(all_videos.values())
        print(f"\n📊 Total unique videos to download: {len(tasks)}")
        print(f"   Already completed: {len(self.checkpoint['completed'])}")
        
        return tasks
    
    def download_video(self, task: VideoTask) -> bool:
        """Download a single video."""
        output_path = self.temp_dir / f"{task.video_id}.mp4"
        
        if output_path.exists():
            task.size_bytes = output_path.stat().st_size
            return True
        
        # Prefer HLS formats (m3u8) which bypass YouTube's 403 blocks on direct downloads
        # Format priority: HLS streams first, then fallback to direct
        format_str = f"best[height<={self.max_height}][protocol=m3u8_native]/best[height<={self.max_height}][protocol=m3u8]/best[height<={self.max_height}]/best"

        args = [
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-playlist",
            "--no-warnings",
            "--newline",
            "--no-check-certificates",
            "--hls-prefer-native",
            "--user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--sleep-interval", "2",
            "--max-sleep-interval", "5",
            f"https://www.youtube.com/watch?v={task.video_id}"
        ]

        # Use browser cookies directly (more reliable than cookies file)
        cookie_args = ["--cookies-from-browser", "chrome"] if not self.cookies_file else ["--cookies", self.cookies_file]

        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "yt_dlp"] +
                cookie_args +
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            last_line = ""
            for line in process.stdout:
                line = line.strip()
                if line:
                    last_line = line
                    if "%" in line and ("ETA" in line or "MiB" in line):
                        print(f"      {line[:70]}", end='\r', flush=True)
            
            process.wait(timeout=900)
            print()
            
            if process.returncode == 0 and output_path.exists():
                task.size_bytes = output_path.stat().st_size
                return True
            else:
                task.error = last_line[:100] if last_line else "Unknown error"
                return False
                
        except subprocess.TimeoutExpired:
            task.error = "Timeout"
            return False
        except Exception as e:
            task.error = str(e)[:100]
            return False
    
    def upload_to_gcs(self, task: VideoTask) -> bool:
        """Upload video to GCS."""
        local_path = self.temp_dir / f"{task.video_id}.mp4"
        if not local_path.exists():
            return False
        
        # Organize by source
        gcs_path = f"gs://{self.bucket}/videos/{task.source}/{task.video_id}.mp4"
        
        try:
            result = subprocess.run(
                ["gsutil", "-q", "cp", str(local_path), gcs_path],
                capture_output=True, text=True, timeout=600
            )
            return result.returncode == 0
        except Exception as e:
            task.error = f"Upload error: {e}"
            return False
    
    def cleanup(self, task: VideoTask):
        """Delete local file."""
        local_path = self.temp_dir / f"{task.video_id}.mp4"
        if local_path.exists():
            local_path.unlink()
    
    def process_video(self, task: VideoTask, index: int, total: int) -> bool:
        """Process a single video: download, upload, cleanup."""
        title_short = task.title[:50] if task.title else task.video_id
        print(f"\n[{index}/{total}] {task.video_id} ({task.source})")
        print(f"   📌 {title_short}")
        
        # Skip if already done
        if task.video_id in self.checkpoint["completed"]:
            print(f"   ⏭️ Already in GCS")
            return True
        
        # Download
        print(f"   📥 Downloading...")
        if not self.download_video(task):
            print(f"   ❌ Download failed: {task.error}")
            self.checkpoint["failed"][task.video_id] = task.error
            self._save_checkpoint()
            return False
        
        print(f"   ✅ Downloaded: {task.size_bytes / 1024 / 1024:.1f} MB")
        
        # Upload
        print(f"   ☁️ Uploading to GCS...")
        if not self.upload_to_gcs(task):
            print(f"   ❌ Upload failed: {task.error}")
            self.checkpoint["failed"][task.video_id] = task.error
            self._save_checkpoint()
            return False
        
        print(f"   ✅ Uploaded to GCS")
        
        # Cleanup and checkpoint
        self.cleanup(task)
        self.checkpoint["completed"].append(task.video_id)
        self.checkpoint["total_bytes"] += task.size_bytes
        self._save_checkpoint()
        
        # Update stats
        self.session_stats["uploaded"] += 1
        self.session_stats["bytes"] += task.size_bytes
        
        return True
    
    def categorize_failure(self, error: str) -> str:
        """Categorize failure type for retry decisions."""
        error_lower = error.lower() if error else ""
        if "private" in error_lower:
            return "private"
        elif "sign in" in error_lower or "age" in error_lower:
            return "auth_required"
        elif "unavailable" in error_lower or "removed" in error_lower:
            return "unavailable"
        elif "403" in error_lower or "rate" in error_lower:
            return "rate_limit"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "retryable"

    def get_retryable_failures(self) -> List[str]:
        """Get video IDs that are worth retrying (not private/unavailable)."""
        retryable = []
        non_retryable_types = {"private", "unavailable", "auth_required"}

        for vid_id, error in self.checkpoint.get("failed", {}).items():
            error_str = error if isinstance(error, str) else str(error)
            category = self.categorize_failure(error_str)
            if category not in non_retryable_types:
                retryable.append(vid_id)

        return retryable

    def retry_failed(self, limit: Optional[int] = None):
        """Retry failed downloads that are retryable."""
        retryable = self.get_retryable_failures()

        if not retryable:
            print("\n✅ No retryable failures!")
            failed = self.checkpoint.get("failed", {})
            if failed:
                print(f"\n📊 Non-retryable failures: {len(failed)}")
                categories = {}
                for vid_id, error in failed.items():
                    cat = self.categorize_failure(error if isinstance(error, str) else str(error))
                    categories[cat] = categories.get(cat, 0) + 1
                for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                    print(f"   • {cat}: {count}")
            return

        print(f"\n🔄 Retrying {len(retryable)} failed downloads...")

        if limit:
            retryable = retryable[:limit]

        for i, vid_id in enumerate(retryable, 1):
            task = VideoTask(video_id=vid_id, title=f"Retry {vid_id}", source="retry")

            print(f"\n[Retry {i}/{len(retryable)}] {vid_id}")

            # Remove from failed list before retry
            if vid_id in self.checkpoint["failed"]:
                del self.checkpoint["failed"][vid_id]
                self._save_checkpoint()

            success = self.process_video(task, i, len(retryable))

            if success:
                print(f"   ✅ Retry successful!")
            else:
                print(f"   ❌ Retry failed again")

            if i < len(retryable):
                print(f"   ⏳ Waiting {self.delay}s...")
                time.sleep(self.delay)

        # Summary
        remaining_failures = len(self.checkpoint.get("failed", {}))
        print(f"\n📊 Retry complete. Remaining failures: {remaining_failures}")

    def run(self, limit: Optional[int] = None, retry_failed: bool = False):
        """Run the multi-source downloader."""
        print("=" * 70)
        print("🚀 N'Ko Multi-Source Video Downloader")
        print("=" * 70)
        print(f"   GCS Bucket: gs://{self.bucket}/videos/")
        print(f"   Quality: {self.max_height}p")
        print(f"   Cookies: {'✅' if self.cookies_file else '❌'}")
        print(f"   Already completed: {len(self.checkpoint['completed'])}")
        print(f"   Total uploaded: {self.checkpoint['total_bytes'] / 1024 / 1024 / 1024:.2f} GB")

        # Handle retry mode
        if retry_failed:
            self.retry_failed(limit)
            return

        # Collect all videos
        tasks = self.collect_all_videos()

        if not tasks:
            print("\n✅ All videos already uploaded!")
            # Offer to retry failures
            retryable = self.get_retryable_failures()
            if retryable:
                print(f"\n💡 {len(retryable)} retryable failures. Run with --retry-failed to retry.")
            return

        if limit:
            tasks = tasks[:limit]
            print(f"\n⚠️ Limited to {limit} videos for this run")
        
        # Process videos
        print("\n" + "=" * 70)
        print("Starting downloads...")
        print("=" * 70)
        
        for i, task in enumerate(tasks, 1):
            success = self.process_video(task, i, len(tasks))
            
            # Progress
            elapsed = time.time() - self.session_stats["start_time"]
            rate_mbs = (self.session_stats["bytes"] / 1024 / 1024) / elapsed if elapsed > 0 else 0
            remaining = len(tasks) - i
            
            print(f"   📊 Session: {self.session_stats['uploaded']} uploaded | Rate: {rate_mbs:.1f} MB/s")
            print(f"   📊 Total: {len(self.checkpoint['completed'])} in GCS | {self.checkpoint['total_bytes']/1024/1024/1024:.2f} GB")
            
            # Delay to avoid rate limiting
            if i < len(tasks):
                print(f"   ⏳ Waiting {self.delay}s...")
                time.sleep(self.delay)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Run Complete!")
        print(f"   Videos in GCS: {len(self.checkpoint['completed'])}")
        print(f"   Total size: {self.checkpoint['total_bytes'] / 1024 / 1024 / 1024:.2f} GB")
        print(f"   Failed: {len(self.checkpoint['failed'])}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download N'Ko videos from multiple sources to GCS")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to download")
    parser.add_argument("--height", type=int, default=720, help="Max video height")
    parser.add_argument("--delay", type=float, default=15.0, help="Delay between downloads (seconds)")
    parser.add_argument("--cookies", type=str, default=COOKIES_FILE, help="Path to cookies.txt")
    parser.add_argument("--bucket", type=str, default=GCS_BUCKET, help="GCS bucket name")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed downloads (excludes private/unavailable)")
    parser.add_argument("--status", action="store_true", help="Show download status and failure breakdown")

    args = parser.parse_args()

    # Verify gsutil
    try:
        subprocess.run(["gsutil", "version"], capture_output=True, timeout=10)
    except:
        print("❌ gsutil not available. Install Google Cloud SDK.")
        return

    downloader = MultiSourceDownloader(
        bucket=args.bucket,
        cookies_file=args.cookies,
        max_height=args.height,
        delay=args.delay,
    )

    # Status mode
    if args.status:
        print("=" * 70)
        print("📊 Download Status")
        print("=" * 70)
        print(f"   Completed: {len(downloader.checkpoint.get('completed', []))}")
        print(f"   Total uploaded: {downloader.checkpoint.get('total_bytes', 0) / 1024 / 1024 / 1024:.2f} GB")

        failed = downloader.checkpoint.get("failed", {})
        print(f"\n   Failed: {len(failed)}")
        if failed:
            categories = {}
            for vid_id, error in failed.items():
                cat = downloader.categorize_failure(error if isinstance(error, str) else str(error))
                categories[cat] = categories.get(cat, 0) + 1
            print("   Breakdown:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"      • {cat}: {count}")

            retryable = downloader.get_retryable_failures()
            print(f"\n   Retryable: {len(retryable)}")
        return

    try:
        downloader.run(limit=args.limit, retry_failed=args.retry_failed)
    except KeyboardInterrupt:
        print("\n\n⏸️ Paused. Run again to resume.")


if __name__ == "__main__":
    main()

