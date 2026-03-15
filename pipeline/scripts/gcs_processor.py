#!/usr/bin/env python3
"""
GCS Video Processor (Runs on DigitalOcean Droplet)
==================================================
1. Lists videos in GCS bucket
2. Streams/downloads one at a time
3. Extracts frames and runs OCR
4. Saves results to Supabase
5. Optionally deletes processed video from GCS

This processes videos that were uploaded by streaming_pipeline.py
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
import tempfile

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from supabase_reporter import SupabaseReporter
except ImportError:
    SupabaseReporter = None

try:
    from nko_analyzer import NKoAnalyzer
except ImportError:
    NKoAnalyzer = None


class GCSProcessor:
    def __init__(self, bucket: str, temp_dir: str = "/tmp/gcs_processing"):
        self.bucket = bucket
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Supabase reporter if available
        self.reporter = None
        if SupabaseReporter:
            try:
                self.reporter = SupabaseReporter()
            except Exception as e:
                print(f"⚠️ Supabase reporter not available: {e}")
        
        # Initialize analyzer if available
        self.analyzer = None
        if NKoAnalyzer:
            try:
                self.analyzer = NKoAnalyzer()
            except Exception as e:
                print(f"⚠️ NKo analyzer not available: {e}")
    
    def list_gcs_videos(self) -> List[str]:
        """List all video files in the GCS bucket."""
        gcs_path = f"gs://{self.bucket}/videos/"
        
        cmd = ["gsutil", "ls", gcs_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                videos = []
                for line in result.stdout.strip().split('\n'):
                    if line.endswith('.mp4') or line.endswith('.webm') or line.endswith('.mkv'):
                        videos.append(line)
                return videos
            else:
                print(f"❌ Failed to list GCS: {result.stderr}")
                return []
        except Exception as e:
            print(f"❌ Error listing GCS: {e}")
            return []
    
    def download_video(self, gcs_path: str) -> Optional[Path]:
        """Download a video from GCS to local temp."""
        filename = Path(gcs_path).name
        local_path = self.temp_dir / filename
        
        cmd = ["gsutil", "-q", "cp", gcs_path, str(local_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and local_path.exists():
                return local_path
            else:
                print(f"   ❌ Download failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"   ❌ Error downloading: {e}")
            return None
    
    def delete_from_gcs(self, gcs_path: str) -> bool:
        """Delete a video from GCS after processing."""
        cmd = ["gsutil", "-q", "rm", gcs_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            print(f"   ⚠️ Could not delete from GCS: {e}")
            return False
    
    def process_video(self, local_path: Path) -> Dict:
        """Process a video: extract frames, run OCR, save to Supabase."""
        video_id = local_path.stem
        results = {
            "video_id": video_id,
            "frames_extracted": 0,
            "detections": 0,
            "status": "pending"
        }
        
        try:
            if self.analyzer:
                # Use the existing analyzer
                analysis = self.analyzer.analyze_video(str(local_path))
                results["frames_extracted"] = analysis.get("frames", 0)
                results["detections"] = analysis.get("detections", 0)
                results["status"] = "success"
            else:
                # Fallback: just extract frames using ffmpeg
                frames_dir = self.temp_dir / f"{video_id}_frames"
                frames_dir.mkdir(exist_ok=True)
                
                # Extract 1 frame every 5 seconds
                cmd = [
                    "ffmpeg", "-i", str(local_path),
                    "-vf", "fps=0.2",  # 1 frame every 5 seconds
                    "-q:v", "2",  # High quality JPEG
                    str(frames_dir / "frame_%04d.jpg"),
                    "-y", "-hide_banner", "-loglevel", "error"
                ]
                
                subprocess.run(cmd, capture_output=True, timeout=300)
                
                frames = list(frames_dir.glob("*.jpg"))
                results["frames_extracted"] = len(frames)
                results["status"] = "frames_only"
                
                # Clean up frames (or keep them for manual processing)
                # for f in frames:
                #     f.unlink()
                # frames_dir.rmdir()
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def cleanup_local(self, local_path: Path):
        """Clean up local temp files."""
        try:
            if local_path.exists():
                local_path.unlink()
            
            # Clean up any associated frame directories
            frames_dir = self.temp_dir / f"{local_path.stem}_frames"
            if frames_dir.exists():
                for f in frames_dir.glob("*"):
                    f.unlink()
                frames_dir.rmdir()
        except Exception as e:
            print(f"   ⚠️ Cleanup error: {e}")
    
    def run(self, limit: int = None, delete_after: bool = False, continuous: bool = False):
        """Run the GCS processor."""
        print("=" * 60)
        print("🔄 GCS Video Processor")
        print("=" * 60)
        print(f"   Bucket: {self.bucket}")
        print(f"   Temp dir: {self.temp_dir}")
        print(f"   Delete after processing: {delete_after}")
        print()
        
        run_id = None
        if self.reporter:
            run_id = self.reporter.start_run("gcs_processing", channel_name=self.bucket)
        
        while True:
            # Get list of videos in GCS
            videos = self.list_gcs_videos()
            
            if not videos:
                if continuous:
                    print("⏳ No videos in queue, waiting 60s...")
                    time.sleep(60)
                    continue
                else:
                    print("✅ No videos to process")
                    break
            
            print(f"\n📊 Found {len(videos)} videos in GCS")
            
            if limit:
                videos = videos[:limit]
            
            processed = 0
            for i, gcs_path in enumerate(videos, 1):
                video_name = Path(gcs_path).name
                print(f"\n[{i}/{len(videos)}] Processing: {video_name}")
                
                # Download from GCS
                print(f"   ⬇️ Downloading from GCS...")
                local_path = self.download_video(gcs_path)
                
                if not local_path:
                    continue
                
                size_mb = local_path.stat().st_size / 1024 / 1024
                print(f"   ✅ Downloaded: {size_mb:.1f} MB")
                
                # Process the video
                print(f"   🎬 Processing video...")
                results = self.process_video(local_path)
                print(f"   ✅ Extracted {results['frames_extracted']} frames")
                
                if self.reporter and run_id:
                    self.reporter.log_event(
                        run_id,
                        "video_processed",
                        video_id=video_name,
                        message=f"Extracted {results['frames_extracted']} frames"
                    )
                
                # Cleanup local
                self.cleanup_local(local_path)
                print(f"   🗑️ Cleaned up local files")
                
                # Optionally delete from GCS
                if delete_after:
                    if self.delete_from_gcs(gcs_path):
                        print(f"   ☁️ Deleted from GCS")
                
                processed += 1
            
            if self.reporter and run_id:
                self.reporter.update_run(run_id, videos_completed=processed)
            
            if not continuous:
                break
            
            print("\n⏳ Batch complete, checking for more videos...")
            time.sleep(10)
        
        if self.reporter and run_id:
            self.reporter.end_run(run_id, "completed")
        
        print("\n" + "=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Process videos from GCS")
    parser.add_argument("--bucket", type=str, default="learnnko-videos",
                        help="GCS bucket name")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit videos to process")
    parser.add_argument("--delete", action="store_true",
                        help="Delete from GCS after processing")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously, checking for new videos")
    parser.add_argument("--temp-dir", type=str, default="/tmp/gcs_processing",
                        help="Local temp directory")
    
    args = parser.parse_args()
    
    processor = GCSProcessor(args.bucket, args.temp_dir)
    processor.run(limit=args.limit, delete_after=args.delete, continuous=args.continuous)


if __name__ == "__main__":
    main()


