#!/usr/bin/env python3
"""
Batch Local YouTube Download
Downloads videos from a YouTube channel locally, then you can upload to server.
"""

import subprocess
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_channel_videos(channel_handle: str, limit: int = None) -> list:
    """Get list of video IDs from a channel."""
    
    channel_url = f"https://www.youtube.com/@{channel_handle}/videos"
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s",
        channel_url
    ]
    
    if limit:
        cmd.extend(["-I", f"1:{limit}"])
    
    print(f"📋 Fetching video list from @{channel_handle}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"❌ Failed to get video list: {result.stderr}")
        return []
    
    videos = []
    for line in result.stdout.strip().split('\n'):
        if '\t' in line:
            vid_id, title = line.split('\t', 1)
            videos.append({'id': vid_id, 'title': title})
    
    print(f"✅ Found {len(videos)} videos")
    return videos


def download_video(video: dict, output_dir: str, max_height: int = 720) -> dict:
    """Download a single video."""
    
    video_id = video['id']
    title = video['title']
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Create safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_', 'ߋ', 'ߌ', 'ߍ', 'ߎ', 'ߏ', 'ߐ', 'ߑ', 'ߒ', 'ߓ', 'ߔ', 'ߕ', 'ߖ', 'ߗ', 'ߘ', 'ߙ', 'ߚ', 'ߛ', 'ߜ', 'ߝ', 'ߞ', 'ߟ', 'ߠ', 'ߡ', 'ߢ', 'ߣ', 'ߤ', 'ߥ', 'ߦ', 'ߧ', 'ߨ', 'ߩ', 'ߪ', '߫', '߬', '߭', '߮', '߯', '߰', '߱', '߲', '߳', 'ߴ', 'ߵ', '߶', '߷', '߸', '߹', '߀', '߁', '߂', '߃', '߄', '߅', '߆', '߇', '߈', '߉')).strip()[:100]
    
    output_template = f"{output_dir}/{video_id}_{safe_title}.%(ext)s"
    
    # Use best quality - merge best video + best audio
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
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Find the downloaded file
            for ext in ['mp4', 'webm', 'mkv']:
                pattern = f"{output_dir}/{video_id}*.{ext}"
                files = list(Path(output_dir).glob(f"{video_id}*.{ext}"))
                if files:
                    return {
                        'id': video_id,
                        'title': title,
                        'file': str(files[0]),
                        'size_mb': files[0].stat().st_size / 1024 / 1024,
                        'status': 'success'
                    }
            return {'id': video_id, 'title': title, 'status': 'success', 'file': 'unknown'}
        else:
            return {'id': video_id, 'title': title, 'status': 'failed', 'error': result.stderr[:200]}
            
    except subprocess.TimeoutExpired:
        return {'id': video_id, 'title': title, 'status': 'timeout'}
    except Exception as e:
        return {'id': video_id, 'title': title, 'status': 'error', 'error': str(e)}


def batch_download(channel_handle: str, output_dir: str, limit: int = None, 
                   max_height: int = 720, workers: int = 2):
    """Download multiple videos from a channel."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get video list
    videos = get_channel_videos(channel_handle, limit)
    
    if not videos:
        print("❌ No videos found")
        return
    
    print(f"\n📥 Downloading {len(videos)} videos to {output_dir}")
    print(f"   Max resolution: {max_height}p")
    print(f"   Workers: {workers}")
    print()
    
    results = []
    success = 0
    failed = 0
    total_size = 0
    
    # Download with progress
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video['title'][:60]}...")
        
        result = download_video(video, output_dir, max_height)
        results.append(result)
        
        if result['status'] == 'success':
            success += 1
            size = result.get('size_mb', 0)
            total_size += size
            print(f"   ✅ Downloaded ({size:.1f} MB)")
        else:
            failed += 1
            print(f"   ❌ Failed: {result.get('error', result['status'])[:50]}")
    
    # Save results
    results_file = f"{output_dir}/download_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'channel': channel_handle,
            'downloaded_at': datetime.now().isoformat(),
            'total': len(videos),
            'success': success,
            'failed': failed,
            'total_size_mb': total_size,
            'results': results
        }, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"📊 Download Summary")
    print(f"   ✅ Success: {success}")
    print(f"   ❌ Failed: {failed}")
    print(f"   💾 Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"   📄 Results saved to: {results_file}")
    print("=" * 60)
    
    if success > 0:
        print()
        print("🚀 Next Steps:")
        print(f"   1. Upload to Droplet:")
        print(f"      scp -r {output_dir}/ root@YOUR_DROPLET_IP:~/learnnko/training/data/videos/")
        print()
        print(f"   2. Or use rsync for resume capability:")
        print(f"      rsync -avz --progress {output_dir}/ root@YOUR_DROPLET_IP:~/learnnko/training/data/videos/")


def main():
    parser = argparse.ArgumentParser(description="Batch download YouTube videos locally")
    parser.add_argument("--channel", type=str, default="babamamadidiane",
                        help="YouTube channel handle (without @)")
    parser.add_argument("--output", type=str, 
                        default="/Users/mohameddiomande/Desktop/learnnko/training/data/local_videos",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of videos to download")
    parser.add_argument("--height", type=int, default=720,
                        help="Max video height (default: 720)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel downloads (default: 1)")
    
    args = parser.parse_args()
    
    batch_download(
        channel_handle=args.channel,
        output_dir=args.output,
        limit=args.limit,
        max_height=args.height,
        workers=args.workers
    )


if __name__ == "__main__":
    main()

