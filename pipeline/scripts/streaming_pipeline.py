#!/usr/bin/env python3
"""
Streaming Video Pipeline
========================
1. Download videos locally (bypasses bot detection)
2. Upload to GCP Cloud Storage
3. Delete local files after upload
4. DigitalOcean processes from GCS
5. Results stored in Supabase

This runs continuously, processing videos one at a time to minimize local storage.
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
import hashlib

# Configuration
DEFAULT_CONFIG = {
    "channel": "babamamadidiane",
    "gcs_bucket": "learnnko-videos",  # Your GCS bucket name
    "local_temp_dir": "/Users/mohameddiomande/Desktop/learnnko/training/temp_videos",
    "batch_size": 3,  # Videos to download before uploading
    "max_height": 720,  # 720p is good balance of quality/size
    "delete_after_upload": True,
    "checkpoint_file": "streaming_checkpoint.json",
}


class StreamingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.temp_dir = Path(config["local_temp_dir"])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.temp_dir / config["checkpoint_file"]
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> dict:
        """Load checkpoint to resume from last position."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "processed_ids": [],
            "uploaded_ids": [],
            "last_index": 0,
            "total_uploaded_gb": 0,
            "started_at": datetime.now().isoformat()
        }
    
    def _save_checkpoint(self):
        """Save checkpoint for resume capability."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_all_video_ids(self) -> List[Dict]:
        """Get all video IDs from channel."""
        channel_url = f"https://www.youtube.com/@{self.config['channel']}/videos"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s\t%(duration)s",
            channel_url
        ]
        
        print(f"📋 Fetching video list from @{self.config['channel']}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"❌ Failed to get video list: {result.stderr}")
            return []
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                vid_id = parts[0]
                title = parts[1]
                duration = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                videos.append({
                    'id': vid_id, 
                    'title': title, 
                    'duration': duration,
                    'estimated_size_mb': (duration / 60) * 15  # ~15 MB/min at 720p
                })
        
        print(f"✅ Found {len(videos)} total videos")
        return videos
    
    def download_video(self, video: dict) -> Optional[Path]:
        """Download a single video at specified quality."""
        video_id = video['id']
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Output with just video ID for cleaner filenames
        output_template = str(self.temp_dir / f"{video_id}.%(ext)s")
        
        max_height = self.config['max_height']
        if max_height == 0:
            format_str = "bestvideo+bestaudio/best"
        else:
            format_str = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--progress",
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Find the downloaded file
                for f in self.temp_dir.glob(f"{video_id}.*"):
                    if f.suffix in ['.mp4', '.webm', '.mkv']:
                        return f
            else:
                print(f"   ❌ Download failed: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Download timeout")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        return None
    
    def upload_to_gcs(self, local_file: Path) -> bool:
        """Upload file to Google Cloud Storage."""
        bucket = self.config['gcs_bucket']
        gcs_path = f"gs://{bucket}/videos/{local_file.name}"
        
        cmd = ["gsutil", "-q", "cp", str(local_file), gcs_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return False
    
    def delete_local(self, local_file: Path):
        """Delete local file after successful upload."""
        try:
            local_file.unlink()
            # Also delete any associated files
            for f in self.temp_dir.glob(f"{local_file.stem}.*"):
                f.unlink()
        except Exception as e:
            print(f"   ⚠️ Could not delete {local_file.name}: {e}")
    
    def get_local_storage_used(self) -> float:
        """Get total storage used in temp directory (MB)."""
        total = sum(f.stat().st_size for f in self.temp_dir.glob("*") if f.is_file())
        return total / 1024 / 1024
    
    def process_batch(self, videos: List[Dict]) -> dict:
        """Process a batch of videos: download, upload, delete."""
        results = {
            "downloaded": 0,
            "uploaded": 0,
            "failed": 0,
            "size_mb": 0
        }
        
        for video in videos:
            video_id = video['id']
            
            # Skip if already processed
            if video_id in self.checkpoint['processed_ids']:
                print(f"   ⏭️ Already processed: {video_id}")
                continue
            
            print(f"\n   📥 Downloading: {video['title'][:50]}...")
            local_file = self.download_video(video)
            
            if local_file and local_file.exists():
                size_mb = local_file.stat().st_size / 1024 / 1024
                results['downloaded'] += 1
                results['size_mb'] += size_mb
                print(f"   ✅ Downloaded: {size_mb:.1f} MB")
                
                # Upload to GCS
                print(f"   ☁️ Uploading to GCS...")
                if self.upload_to_gcs(local_file):
                    results['uploaded'] += 1
                    self.checkpoint['uploaded_ids'].append(video_id)
                    self.checkpoint['total_uploaded_gb'] += size_mb / 1024
                    print(f"   ✅ Uploaded to GCS")
                    
                    # Delete local file
                    if self.config['delete_after_upload']:
                        self.delete_local(local_file)
                        print(f"   🗑️ Deleted local file")
                else:
                    print(f"   ❌ Upload failed, keeping local file")
                    results['failed'] += 1
                
                self.checkpoint['processed_ids'].append(video_id)
                self._save_checkpoint()
            else:
                results['failed'] += 1
                self.checkpoint['processed_ids'].append(video_id)  # Mark as attempted
                self._save_checkpoint()
        
        return results
    
    def run(self, limit: int = None, continuous: bool = False):
        """Run the streaming pipeline."""
        print("=" * 60)
        print("🚀 Streaming Video Pipeline")
        print("=" * 60)
        print(f"   Channel: @{self.config['channel']}")
        print(f"   Quality: {self.config['max_height']}p")
        print(f"   GCS Bucket: {self.config['gcs_bucket']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Already processed: {len(self.checkpoint['processed_ids'])}")
        print()
        
        # Get all videos
        all_videos = self.get_all_video_ids()
        
        if not all_videos:
            print("❌ No videos found")
            return
        
        # Filter out already processed
        pending = [v for v in all_videos if v['id'] not in self.checkpoint['processed_ids']]
        
        if limit:
            pending = pending[:limit]
        
        print(f"\n📊 {len(pending)} videos pending (of {len(all_videos)} total)")
        
        # Estimate total size
        total_estimated_mb = sum(v.get('estimated_size_mb', 50) for v in pending)
        print(f"   Estimated total: {total_estimated_mb/1024:.1f} GB at {self.config['max_height']}p")
        
        # Process in batches
        batch_size = self.config['batch_size']
        total_uploaded = 0
        total_failed = 0
        
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(pending) + batch_size - 1) // batch_size
            
            print(f"\n{'='*60}")
            print(f"📦 Batch {batch_num}/{total_batches}")
            print(f"   Local storage used: {self.get_local_storage_used():.1f} MB")
            
            results = self.process_batch(batch)
            total_uploaded += results['uploaded']
            total_failed += results['failed']
            
            print(f"\n   Batch complete: {results['uploaded']} uploaded, {results['failed']} failed")
            print(f"   Total progress: {total_uploaded}/{len(pending)} uploaded")
            print(f"   Total uploaded: {self.checkpoint['total_uploaded_gb']:.2f} GB")
            
            if not continuous and i + batch_size >= len(pending):
                break
            
            # Small delay between batches
            time.sleep(2)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print(f"   Total uploaded: {total_uploaded}")
        print(f"   Total failed: {total_failed}")
        print(f"   Total size: {self.checkpoint['total_uploaded_gb']:.2f} GB")
        print("=" * 60)


def check_gcs_setup():
    """Check if GCS is configured."""
    try:
        result = subprocess.run(["gsutil", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ gsutil is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ gsutil not found. Install Google Cloud SDK:")
    print("   brew install google-cloud-sdk")
    print("   gcloud auth login")
    print("   gcloud config set project YOUR_PROJECT_ID")
    return False


def main():
    parser = argparse.ArgumentParser(description="Streaming Video Pipeline")
    parser.add_argument("--channel", type=str, default="babamamadidiane")
    parser.add_argument("--bucket", type=str, default="learnnko-videos",
                        help="GCS bucket name")
    parser.add_argument("--height", type=int, default=720,
                        help="Max video height (0 for best)")
    parser.add_argument("--batch", type=int, default=3,
                        help="Videos per batch")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total videos to process")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously")
    parser.add_argument("--check", action="store_true",
                        help="Just check GCS setup")
    
    args = parser.parse_args()
    
    if args.check:
        check_gcs_setup()
        return
    
    if not check_gcs_setup():
        print("\n⚠️ Set up GCS first, then run again")
        return
    
    config = {
        **DEFAULT_CONFIG,
        "channel": args.channel,
        "gcs_bucket": args.bucket,
        "max_height": args.height,
        "batch_size": args.batch,
    }
    
    pipeline = StreamingPipeline(config)
    pipeline.run(limit=args.limit, continuous=args.continuous)


if __name__ == "__main__":
    main()


