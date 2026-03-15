#!/usr/bin/env python3
"""
Hybrid Streaming Pipeline
=========================
Combines Python (YouTube download) with Rust backend (frame extraction + Gemini OCR).

Flow:
1. Python: Download video locally (bypasses bot detection)
2. Python: Upload to GCS with public URL
3. Rust: Process video from GCS URL via existing cc-stream + cc-gemini
4. Python: Delete local + optionally GCS after processing

This leverages the existing Rust infrastructure for async, parallel processing
while using Python only where necessary (yt-dlp).
"""

import subprocess
import os
import sys
import json
import asyncio
import aiohttp
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configuration
GCS_BUCKET = "learnnko-videos"
RUST_BACKEND_URL = os.environ.get(
    "ANALYZER_BACKEND_URL", 
    "https://cc-music-pipeline-684958912334.us-central1.run.app"
)
BATCH_SIZE = 5  # Process this many in parallel


@dataclass
class VideoTask:
    """A video to be processed."""
    video_id: str
    title: str
    youtube_url: str
    local_path: Optional[Path] = None
    gcs_url: Optional[str] = None
    status: str = "pending"  # pending, downloading, uploading, processing, done, failed
    frames_extracted: int = 0
    detections: int = 0
    error: Optional[str] = None


class HybridPipeline:
    """
    Hybrid pipeline using Python for download, Rust for processing.
    """
    
    def __init__(
        self,
        channel: str = "babamamadidiane",
        gcs_bucket: str = GCS_BUCKET,
        backend_url: str = RUST_BACKEND_URL,
        temp_dir: str = "./temp_videos",
        max_height: int = 720,
    ):
        self.channel = channel
        self.gcs_bucket = gcs_bucket
        self.backend_url = backend_url
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_height = max_height
        
        # Checkpoint for resume
        self.checkpoint_file = self.temp_dir / "hybrid_checkpoint.json"
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"processed": [], "failed": [], "stats": {"total_frames": 0, "total_detections": 0}}
    
    def _save_checkpoint(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_channel_videos(self, limit: Optional[int] = None) -> List[VideoTask]:
        """Get all videos from channel."""
        channel_url = f"https://www.youtube.com/@{self.channel}/videos"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s",
            channel_url
        ]
        
        if limit:
            cmd.extend(["-I", f"1:{limit}"])
        
        print(f"📋 Fetching videos from @{self.channel}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"❌ Failed: {result.stderr}")
            return []
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if '\t' in line:
                vid_id, title = line.split('\t', 1)
                if vid_id not in self.checkpoint['processed']:
                    videos.append(VideoTask(
                        video_id=vid_id,
                        title=title,
                        youtube_url=f"https://www.youtube.com/watch?v={vid_id}"
                    ))
        
        print(f"✅ Found {len(videos)} unprocessed videos")
        return videos
    
    def download_video(self, task: VideoTask) -> bool:
        """Download video locally using yt-dlp."""
        output_path = self.temp_dir / f"{task.video_id}.mp4"
        
        format_str = f"bestvideo[height<={self.max_height}]+bestaudio/best[height<={self.max_height}]/best"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-playlist",
            "--quiet",
            task.youtube_url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and output_path.exists():
                task.local_path = output_path
                task.status = "downloaded"
                return True
        except Exception as e:
            task.error = str(e)
        
        task.status = "failed"
        return False
    
    def upload_to_gcs(self, task: VideoTask) -> bool:
        """Upload video to GCS and get public URL."""
        if not task.local_path or not task.local_path.exists():
            return False
        
        gcs_path = f"gs://{self.gcs_bucket}/videos/{task.video_id}.mp4"
        
        # Upload with public read access
        cmd = [
            "gsutil", "-q",
            "-h", "Cache-Control:public,max-age=3600",
            "cp", str(task.local_path), gcs_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                # Make public
                subprocess.run(
                    ["gsutil", "-q", "acl", "ch", "-u", "AllUsers:R", gcs_path],
                    capture_output=True, timeout=30
                )
                
                # Public URL format
                task.gcs_url = f"https://storage.googleapis.com/{self.gcs_bucket}/videos/{task.video_id}.mp4"
                task.status = "uploaded"
                return True
        except Exception as e:
            task.error = str(e)
        
        return False
    
    async def process_with_rust_backend(self, task: VideoTask, session: aiohttp.ClientSession) -> bool:
        """Send video URL to Rust backend for processing."""
        if not task.gcs_url:
            return False
        
        # The Rust backend's /analyze endpoint accepts video URLs
        endpoint = f"{self.backend_url}/api/v1/analyze/video"
        
        payload = {
            "video_url": task.gcs_url,
            "max_frames": 100,
            "extract_audio": False,
            "run_ocr": True,
            "source_id": task.video_id,
        }
        
        try:
            async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    task.frames_extracted = result.get("frames_extracted", 0)
                    task.detections = result.get("detections", 0)
                    task.status = "done"
                    return True
                else:
                    error_text = await resp.text()
                    task.error = f"Backend error {resp.status}: {error_text[:100]}"
        except asyncio.TimeoutError:
            task.error = "Backend timeout"
        except Exception as e:
            task.error = str(e)
        
        task.status = "failed"
        return False
    
    def cleanup(self, task: VideoTask, delete_gcs: bool = False):
        """Clean up local and optionally GCS files."""
        # Delete local
        if task.local_path and task.local_path.exists():
            task.local_path.unlink()
        
        # Optionally delete from GCS
        if delete_gcs and task.gcs_url:
            gcs_path = f"gs://{self.gcs_bucket}/videos/{task.video_id}.mp4"
            subprocess.run(["gsutil", "-q", "rm", gcs_path], capture_output=True)
    
    async def process_batch(self, tasks: List[VideoTask], delete_after: bool = True):
        """Process a batch of videos in parallel."""
        
        # Step 1: Download all in parallel (using threads for subprocess)
        print(f"\n📥 Downloading {len(tasks)} videos...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            download_futures = {
                executor.submit(self.download_video, task): task 
                for task in tasks
            }
            for future in download_futures:
                future.result()
        
        downloaded = [t for t in tasks if t.status == "downloaded"]
        print(f"   ✅ Downloaded: {len(downloaded)}/{len(tasks)}")
        
        # Step 2: Upload all to GCS in parallel
        print(f"\n☁️ Uploading to GCS...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            upload_futures = {
                executor.submit(self.upload_to_gcs, task): task 
                for task in downloaded
            }
            for future in upload_futures:
                future.result()
        
        uploaded = [t for t in tasks if t.status == "uploaded"]
        print(f"   ✅ Uploaded: {len(uploaded)}/{len(downloaded)}")
        
        # Step 3: Process with Rust backend (async)
        print(f"\n🦀 Processing with Rust backend...")
        async with aiohttp.ClientSession() as session:
            process_tasks = [
                self.process_with_rust_backend(task, session)
                for task in uploaded
            ]
            await asyncio.gather(*process_tasks, return_exceptions=True)
        
        processed = [t for t in tasks if t.status == "done"]
        print(f"   ✅ Processed: {len(processed)}/{len(uploaded)}")
        
        # Step 4: Cleanup
        print(f"\n🗑️ Cleaning up...")
        for task in tasks:
            self.cleanup(task, delete_gcs=delete_after and task.status == "done")
            
            # Update checkpoint
            if task.status == "done":
                self.checkpoint['processed'].append(task.video_id)
                self.checkpoint['stats']['total_frames'] += task.frames_extracted
                self.checkpoint['stats']['total_detections'] += task.detections
            else:
                self.checkpoint['failed'].append({
                    'id': task.video_id,
                    'error': task.error
                })
        
        self._save_checkpoint()
        
        return processed
    
    async def run(self, limit: Optional[int] = None, batch_size: int = BATCH_SIZE):
        """Run the hybrid pipeline."""
        print("=" * 60)
        print("🚀 Hybrid Streaming Pipeline")
        print("   Python (download) → GCS → Rust (process)")
        print("=" * 60)
        print(f"   Channel: @{self.channel}")
        print(f"   Quality: {self.max_height}p")
        print(f"   Backend: {self.backend_url}")
        print(f"   GCS Bucket: {self.gcs_bucket}")
        print(f"   Already processed: {len(self.checkpoint['processed'])}")
        print()
        
        # Get videos
        videos = self.get_channel_videos(limit)
        
        if not videos:
            print("✅ All videos processed!")
            return
        
        # Process in batches
        total_processed = 0
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(videos) + batch_size - 1) // batch_size
            
            print(f"\n{'='*60}")
            print(f"📦 Batch {batch_num}/{total_batches}")
            
            processed = await self.process_batch(batch)
            total_processed += len(processed)
            
            print(f"\n📊 Progress: {total_processed}/{len(videos)}")
            print(f"   Total frames: {self.checkpoint['stats']['total_frames']}")
            print(f"   Total detections: {self.checkpoint['stats']['total_detections']}")
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print(f"   Processed: {total_processed}")
        print(f"   Total frames: {self.checkpoint['stats']['total_frames']}")
        print(f"   Total detections: {self.checkpoint['stats']['total_detections']}")
        print("=" * 60)


async def check_rust_backend(url: str) -> bool:
    """Check if Rust backend is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    print(f"✅ Rust backend healthy: {url}")
                    return True
    except Exception as e:
        print(f"❌ Rust backend not available: {e}")
    return False


def check_gcs_setup(bucket: str) -> bool:
    """Check GCS bucket exists and is accessible."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", f"gs://{bucket}"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(f"✅ GCS bucket accessible: {bucket}")
            return True
        else:
            # Try to create
            print(f"📦 Creating GCS bucket: {bucket}")
            create_result = subprocess.run(
                ["gsutil", "mb", f"gs://{bucket}"],
                capture_output=True, text=True, timeout=30
            )
            if create_result.returncode == 0:
                print(f"✅ Created GCS bucket: {bucket}")
                return True
    except Exception as e:
        print(f"❌ GCS error: {e}")
    return False


async def main():
    parser = argparse.ArgumentParser(description="Hybrid Streaming Pipeline")
    parser.add_argument("--channel", type=str, default="babamamadidiane")
    parser.add_argument("--bucket", type=str, default=GCS_BUCKET)
    parser.add_argument("--backend", type=str, default=RUST_BACKEND_URL)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--check", action="store_true", help="Just check setup")
    
    args = parser.parse_args()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...\n")
    
    gcs_ok = check_gcs_setup(args.bucket)
    backend_ok = await check_rust_backend(args.backend)
    
    if args.check:
        return
    
    if not gcs_ok:
        print("\n⚠️ Set up GCS first:")
        print("   gcloud auth login")
        print("   gsutil mb gs://learnnko-videos")
        return
    
    if not backend_ok:
        print("\n⚠️ Rust backend not available. Continue anyway? (processing will be skipped)")
        # Could still download and upload for later processing
    
    # Run pipeline
    pipeline = HybridPipeline(
        channel=args.channel,
        gcs_bucket=args.bucket,
        backend_url=args.backend,
        max_height=args.height,
    )
    
    await pipeline.run(limit=args.limit, batch_size=args.batch)


if __name__ == "__main__":
    asyncio.run(main())


