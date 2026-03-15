#!/usr/bin/env python3
"""
Local YouTube Download Test
Tests if yt-dlp can download videos from your local machine (bypassing bot detection).
"""

import subprocess
import os
import sys
from pathlib import Path

def test_download(video_id: str = "babamamadidiane", output_dir: str = "./test_download"):
    """Test downloading a single video from the channel."""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get one video URL from channel
    channel_url = f"https://www.youtube.com/@{video_id}/videos"
    
    print(f"🔍 Testing download from: {channel_url}")
    print(f"📁 Output directory: {output_dir}")
    
    # First, let's just try to get the video list using yt-dlp
    print("\n📋 Getting video list...")
    list_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--print", "%(id)s %(title)s",
        "-I", "1:3",  # Just first 3 videos
        channel_url
    ]
    
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Video list retrieved:")
            lines = result.stdout.strip().split('\n')
            for line in lines[:3]:
                print(f"   {line}")
            
            if lines:
                # Get first video ID
                first_video_id = lines[0].split()[0]
                video_url = f"https://www.youtube.com/watch?v={first_video_id}"
                
                print(f"\n📥 Testing download of: {video_url}")
                
                # Try to download first video
                download_cmd = [
                    sys.executable, "-m", "yt_dlp",
                    "-f", "best[height<=720]",  # 720p max to keep size manageable
                    "-o", f"{output_dir}/%(title)s.%(ext)s",
                    "--no-playlist",
                    video_url
                ]
                
                result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ Download successful!")
                    # List downloaded files
                    files = list(Path(output_dir).glob("*"))
                    for f in files:
                        print(f"   📄 {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
                    return True
                else:
                    print("❌ Download failed:")
                    print(result.stderr)
                    return False
        else:
            print("❌ Failed to get video list:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout - command took too long")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_with_cookies(video_url: str, cookies_path: str = "./cookies.txt", output_dir: str = "./test_download"):
    """Test download with cookies file."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🍪 Testing with cookies from: {cookies_path}")
    
    if not Path(cookies_path).exists():
        print(f"❌ Cookies file not found: {cookies_path}")
        print("   Export cookies from your browser using a browser extension like 'Get cookies.txt'")
        return False
    
    download_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--cookies", cookies_path,
        "-f", "best[height<=720]",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        "--no-playlist",
        video_url
    ]
    
    try:
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Download with cookies successful!")
            return True
        else:
            print("❌ Download failed:")
            print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    output_dir = "/Users/mohameddiomande/Desktop/learnnko/training/test_download"
    
    print("=" * 60)
    print("YouTube Local Download Test")
    print("=" * 60)
    
    # Test 1: Basic download (no cookies)
    print("\n🧪 Test 1: Basic download without cookies")
    success = test_download("babamamadidiane", output_dir)
    
    if not success:
        # Test 2: Try with cookies if available
        cookies_path = "/Users/mohameddiomande/Desktop/learnnko/training/scripts/cookies.txt"
        if Path(cookies_path).exists():
            print("\n🧪 Test 2: Trying with cookies...")
            # Use a known video from the channel
            success = test_with_cookies(
                "https://www.youtube.com/watch?v=babamamadidiane",
                cookies_path,
                output_dir
            )
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Local download works! You can download videos locally and upload to server.")
        print("\nNext steps:")
        print("1. Download videos locally: python local_download_test.py")
        print("2. Upload to Droplet: scp -r test_download/ root@YOUR_DROPLET_IP:~/learnnko/training/data/videos/")
        print("3. Run extraction on server pointing to uploaded videos")
    else:
        print("❌ Local download also failed. Try:")
        print("1. Update yt-dlp: pip install -U yt-dlp")
        print("2. Export fresh cookies from browser")
        print("3. Check if videos are public/unlisted")

