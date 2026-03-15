#!/usr/bin/env python3
"""
Download to GCS Pipeline
========================
Downloads all videos from a YouTube channel and uploads to Google Cloud Storage.
Runs continuously until complete. Handles resume, retries, and cleanup.

Usage:
    python3 download_to_gcs.py --channel babamamadidiane --height 720

This will:
1. Get all video IDs from the channel
2. Download each video locally (720p by default)
3. Upload to GCS
4. Delete local file after successful upload
5. Continue until all videos are in GCS
"""

import subprocess
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import urllib.parse

# Configuration
GCS_BUCKET = "learnnko-videos"
TEMP_DIR = "/Users/mohameddiomande/projects/LearnNKo/pipeline/scripts/temp_gcs"
CHECKPOINT_FILE = "gcs_upload_checkpoint.json"


class GCSUploadPipeline:
    def __init__(
        self,
        channel: str = "babamamadidiane",
        bucket: str = GCS_BUCKET,
        temp_dir: str = TEMP_DIR,
        max_height: int = 720,
        max_local_gb: float = 5.0,  # Max local storage before waiting
        cookies_file: str = None,  # Path to cookies.txt
        delay_between: float = 10.0,  # Delay between downloads (seconds)
    ):
        self.channel = channel
        self.bucket = bucket
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_height = max_height
        self.max_local_bytes = max_local_gb * 1024 * 1024 * 1024
        self.delay_between = delay_between
        
        # Look for cookies file
        self.cookies_file = None
        if cookies_file and Path(cookies_file).exists():
            self.cookies_file = cookies_file
        else:
            # Try common locations
            for path in [
                Path(__file__).parent / "cookies.txt",
                Path.home() / "cookies.txt",
                Path.home() / ".config" / "yt-dlp" / "cookies.txt",
            ]:
                if path.exists():
                    self.cookies_file = str(path)
                    break
        
        if self.cookies_file:
            print(f"   🍪 Using cookies: {self.cookies_file}")
        
        self.checkpoint_file = self.temp_dir / CHECKPOINT_FILE
        self.checkpoint = self._load_checkpoint()
        
        # Stats
        self.session_uploaded = 0
        self.session_failed = 0
        self.session_bytes = 0
        self.start_time = time.time()
    
    def _load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "uploaded": [],  # Video IDs successfully uploaded
            "failed": [],    # Video IDs that failed (with error)
            "in_progress": None,  # Currently processing
            "total_bytes": 0,
            "started_at": datetime.now().isoformat(),
        }
    
    def _save_checkpoint(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_local_size(self) -> int:
        """Get total size of files in temp directory."""
        total = 0
        for f in self.temp_dir.glob("*.mp4"):
            total += f.stat().st_size
        for f in self.temp_dir.glob("*.part"):
            total += f.stat().st_size
        return total
    
    def get_all_videos(self) -> List[Dict]:
        """Get all video IDs from the channel."""
        channel_url = f"https://www.youtube.com/@{self.channel}/videos"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s\t%(duration)s",
            channel_url
        ]
        
        print(f"📋 Fetching video list from @{self.channel}...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                print(f"❌ Failed to get video list: {result.stderr[:200]}")
                return []
            
            videos = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    vid_id = parts[0].strip()
                    title = parts[1].strip()
                    duration = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                    
                    if vid_id and len(vid_id) == 11:  # Valid YouTube ID
                        videos.append({
                            'id': vid_id,
                            'title': title,
                            'duration': duration,
                        })
            
            print(f"✅ Found {len(videos)} total videos")
            return videos
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout getting video list")
            return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def video_exists_in_gcs(self, video_id: str) -> bool:
        """Check if video already exists in GCS."""
        # First check our checkpoint
        if video_id in self.checkpoint['uploaded']:
            return True
        
        # Then check GCS directly
        gcs_path = f"gs://{self.bucket}/videos/{video_id}.mp4"
        result = subprocess.run(
            ["gsutil", "-q", "stat", gcs_path],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    
    def download_video(self, video: Dict) -> Optional[Path]:
        """Download a video locally."""
        video_id = video['id']
        
        # Use simple filename (just video ID)
        output_path = self.temp_dir / f"{video_id}.mp4"
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"   ⏭️ Already downloaded locally")
            return output_path
        
        # Prefer HLS formats (m3u8) which bypass YouTube's 403 blocks on direct downloads
        # Format priority: HLS streams first, then fallback to direct
        if self.max_height == 0:
            format_str = "best[protocol=m3u8_native]/best[protocol=m3u8]/bestvideo+bestaudio/best"
        else:
            format_str = f"best[height<={self.max_height}][protocol=m3u8_native]/best[height<={self.max_height}][protocol=m3u8]/best[height<={self.max_height}]/best"

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-playlist",
            "--no-warnings",
            "--newline",  # Progress on new lines
            "--hls-prefer-native",  # Better HLS handling
            "--sleep-interval", "2",  # Sleep between requests
            "--max-sleep-interval", "5",
        ]

        # Use browser cookies directly (more reliable) or fallback to file
        if self.cookies_file:
            cmd.extend(["--cookies", self.cookies_file])
        else:
            cmd.extend(["--cookies-from-browser", "chrome"])
        
        cmd.append(f"https://www.youtube.com/watch?v={video_id}")
        
        try:
            # Run with output displayed
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read output line by line
            last_line = ""
            for line in process.stdout:
                line = line.strip()
                if line:
                    last_line = line
                    # Show download progress
                    if "%" in line and "ETA" in line:
                        print(f"   {line}", end='\r', flush=True)
            
            process.wait(timeout=900)
            print()  # New line after progress
            
            if process.returncode == 0 and output_path.exists():
                return output_path
            else:
                # Check for common errors
                if "Sign in to confirm" in last_line:
                    print(f"   ❌ Bot detection - may need cookies")
                elif "Video unavailable" in last_line:
                    print(f"   ❌ Video unavailable")
                else:
                    print(f"   ❌ Download failed: {last_line[:100]}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Download timeout (15 min)")
            # Clean up partial file
            for f in self.temp_dir.glob(f"{video_id}.*"):
                f.unlink()
            return None
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def upload_to_gcs(self, local_path: Path) -> bool:
        """Upload a file to GCS."""
        video_id = local_path.stem
        gcs_path = f"gs://{self.bucket}/videos/{video_id}.mp4"
        
        cmd = ["gsutil", "-q", "cp", str(local_path), gcs_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            if result.returncode == 0:
                return True
            else:
                print(f"   ❌ Upload failed: {result.stderr[:100]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Upload timeout")
            return False
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return False
    
    def delete_local(self, local_path: Path):
        """Delete local file after successful upload."""
        try:
            if local_path.exists():
                local_path.unlink()
            # Also clean up any temp files
            for f in self.temp_dir.glob(f"{local_path.stem}.*"):
                if f.suffix not in ['.json']:
                    f.unlink()
        except Exception as e:
            print(f"   ⚠️ Could not delete local file: {e}")
    
    def process_video(self, video: Dict) -> str:
        """Process a single video: download, upload, delete. Returns status."""
        video_id = video['id']
        title = video['title'][:50]
        
        # Check if already in GCS
        if self.video_exists_in_gcs(video_id):
            return "skipped"
        
        # Mark as in progress
        self.checkpoint['in_progress'] = video_id
        self._save_checkpoint()
        
        print(f"\n   📥 Downloading: {title}...")
        local_path = self.download_video(video)
        
        if not local_path or not local_path.exists():
            self.checkpoint['failed'].append({
                'id': video_id,
                'error': 'download_failed',
                'time': datetime.now().isoformat()
            })
            self.checkpoint['in_progress'] = None
            self._save_checkpoint()
            return "failed"
        
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"   ✅ Downloaded: {size_mb:.1f} MB")
        
        print(f"   ☁️ Uploading to GCS...")
        if self.upload_to_gcs(local_path):
            print(f"   ✅ Uploaded to GCS")
            
            # Update checkpoint
            self.checkpoint['uploaded'].append(video_id)
            self.checkpoint['total_bytes'] += local_path.stat().st_size
            self.session_uploaded += 1
            self.session_bytes += local_path.stat().st_size
            
            # Delete local
            self.delete_local(local_path)
            print(f"   🗑️ Deleted local file")
            
            self.checkpoint['in_progress'] = None
            self._save_checkpoint()
            return "success"
        else:
            self.checkpoint['failed'].append({
                'id': video_id,
                'error': 'upload_failed',
                'time': datetime.now().isoformat()
            })
            self.checkpoint['in_progress'] = None
            self._save_checkpoint()
            return "failed"
    
    def run(self):
        """Run the pipeline until all videos are uploaded."""
        print("=" * 70)
        print("🚀 Download to GCS Pipeline")
        print("=" * 70)
        print(f"   Channel: @{self.channel}")
        print(f"   Quality: {self.max_height}p" if self.max_height else "   Quality: Best")
        print(f"   GCS Bucket: gs://{self.bucket}/videos/")
        print(f"   Temp Dir: {self.temp_dir}")
        print(f"   Already uploaded: {len(self.checkpoint['uploaded'])}")
        print(f"   Total bytes uploaded: {self.checkpoint['total_bytes'] / 1024 / 1024 / 1024:.2f} GB")
        print()
        
        # Get all videos
        all_videos = self.get_all_videos()
        
        if not all_videos:
            print("❌ Could not get video list. Exiting.")
            return
        
        # Filter out already uploaded
        pending = [v for v in all_videos if v['id'] not in self.checkpoint['uploaded']]
        
        # Also filter out known failed (but allow retry after 10)
        failed_ids = set(f['id'] for f in self.checkpoint['failed'][-50:])  # Only block recent 50 failures
        pending = [v for v in pending if v['id'] not in failed_ids]
        
        print(f"\n📊 Status: {len(self.checkpoint['uploaded'])}/{len(all_videos)} uploaded, {len(pending)} pending")
        
        if not pending:
            print("✅ All videos uploaded!")
            return
        
        # Estimate total size
        avg_duration = sum(v['duration'] for v in pending if v['duration']) / max(1, len([v for v in pending if v['duration']]))
        est_size_per_video = (avg_duration / 60) * (15 if self.max_height <= 720 else 25)  # MB per video
        est_total_gb = (est_size_per_video * len(pending)) / 1024
        
        print(f"   Estimated remaining: ~{est_total_gb:.1f} GB")
        print()
        print("=" * 70)
        print("Starting download loop... (Ctrl+C to pause)")
        print("=" * 70)
        
        # Process videos one by one
        for i, video in enumerate(pending, 1):
            # Check local disk space
            local_size = self.get_local_size()
            if local_size > self.max_local_bytes:
                print(f"\n⚠️ Local storage full ({local_size/1024/1024/1024:.1f} GB). Waiting...")
                while self.get_local_size() > self.max_local_bytes * 0.5:
                    time.sleep(10)
            
            print(f"\n[{i}/{len(pending)}] {video['id']}: {video['title'][:60]}")
            
            status = self.process_video(video)
            
            # Progress stats
            elapsed = time.time() - self.start_time
            rate = self.session_bytes / elapsed if elapsed > 0 else 0
            eta_seconds = ((len(pending) - i) * est_size_per_video * 1024 * 1024) / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600
            
            print(f"   📊 Session: {self.session_uploaded} uploaded, {self.session_failed} failed")
            print(f"   📊 Rate: {rate/1024/1024:.1f} MB/s, ETA: {eta_hours:.1f} hours")
            
            # Delay between videos to avoid rate limiting
            if status != "skipped":
                print(f"   ⏳ Waiting {self.delay_between}s to avoid rate limits...")
                time.sleep(self.delay_between)
        
        print("\n" + "=" * 70)
        print("✅ Pipeline Complete!")
        print(f"   Total uploaded: {len(self.checkpoint['uploaded'])}")
        print(f"   Total size: {self.checkpoint['total_bytes'] / 1024 / 1024 / 1024:.2f} GB")
        print(f"   Failed: {len(self.checkpoint['failed'])}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download YouTube channel to GCS")
    parser.add_argument("--channel", type=str, default="babamamadidiane",
                        help="YouTube channel handle (without @)")
    parser.add_argument("--bucket", type=str, default=GCS_BUCKET,
                        help="GCS bucket name")
    parser.add_argument("--height", type=int, default=720,
                        help="Max video height (720, 1080, or 0 for best)")
    parser.add_argument("--temp-dir", type=str, default=TEMP_DIR,
                        help="Temporary download directory")
    parser.add_argument("--max-local-gb", type=float, default=5.0,
                        help="Max local storage to use (GB)")
    parser.add_argument("--cookies", type=str, default=None,
                        help="Path to cookies.txt for YouTube authentication")
    parser.add_argument("--delay", type=float, default=15.0,
                        help="Delay between downloads in seconds (default: 15)")
    
    args = parser.parse_args()
    
    # Check gsutil is available
    try:
        result = subprocess.run(["gsutil", "version"], capture_output=True, timeout=10)
        if result.returncode != 0:
            print("❌ gsutil not working. Run: gcloud auth login")
            return
    except FileNotFoundError:
        print("❌ gsutil not found. Install Google Cloud SDK")
        return
    
    pipeline = GCSUploadPipeline(
        channel=args.channel,
        bucket=args.bucket,
        temp_dir=args.temp_dir,
        max_height=args.height,
        max_local_gb=args.max_local_gb,
        cookies_file=args.cookies,
        delay_between=args.delay,
    )
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⏸️ Paused. Run again to resume from checkpoint.")
        print(f"   Checkpoint: {pipeline.checkpoint_file}")


if __name__ == "__main__":
    main()

