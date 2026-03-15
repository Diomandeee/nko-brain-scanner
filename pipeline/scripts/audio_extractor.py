#!/usr/bin/env python3
"""
Audio Extractor for N'Ko Video Pipeline

Extracts and segments audio from videos to match scene changes,
enabling future ASR transcription for curriculum building.

Features:
- Full audio extraction from video
- Scene-based audio segmentation
- Manifest generation with timestamps
- Ready for future Whisper ASR integration
"""

import json
import subprocess
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SceneInfo:
    """Information about a single scene."""
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    frame_path: Optional[str] = None
    audio_path: Optional[str] = None
    has_nko: bool = False
    nko_text: Optional[str] = None
    latin_text: Optional[str] = None
    english_text: Optional[str] = None
    confidence: float = 0.0
    transcription: Optional[str] = None  # For future ASR


@dataclass
class VideoManifest:
    """Complete manifest for a processed video."""
    video_id: str
    title: str
    youtube_url: str
    channel_name: Optional[str] = None
    duration_ms: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    scenes: List[SceneInfo] = field(default_factory=list)
    
    # Paths
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    frames_dir: Optional[str] = None
    audio_segments_dir: Optional[str] = None
    
    # Processing stats
    total_frames: int = 0
    nko_frames: int = 0
    total_audio_segments: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "youtube_url": self.youtube_url,
            "channel_name": self.channel_name,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
            "scenes": [asdict(s) for s in self.scenes],
            "paths": {
                "video": self.video_path,
                "audio": self.audio_path,
                "frames_dir": self.frames_dir,
                "audio_segments_dir": self.audio_segments_dir,
            },
            "stats": {
                "total_frames": self.total_frames,
                "nko_frames": self.nko_frames,
                "total_audio_segments": self.total_audio_segments,
            },
        }
    
    def save(self, path: str):
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'VideoManifest':
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        manifest = cls(
            video_id=data["video_id"],
            title=data["title"],
            youtube_url=data["youtube_url"],
            channel_name=data.get("channel_name"),
            duration_ms=data["duration_ms"],
            created_at=data["created_at"],
        )
        
        # Load paths
        paths = data.get("paths", {})
        manifest.video_path = paths.get("video")
        manifest.audio_path = paths.get("audio")
        manifest.frames_dir = paths.get("frames_dir")
        manifest.audio_segments_dir = paths.get("audio_segments_dir")
        
        # Load stats
        stats = data.get("stats", {})
        manifest.total_frames = stats.get("total_frames", 0)
        manifest.nko_frames = stats.get("nko_frames", 0)
        manifest.total_audio_segments = stats.get("total_audio_segments", 0)
        
        # Load scenes
        for scene_data in data.get("scenes", []):
            manifest.scenes.append(SceneInfo(**scene_data))
        
        return manifest


class AudioExtractor:
    """
    Audio extraction and segmentation for video analysis pipeline.
    
    Usage:
        extractor = AudioExtractor()
        manifest = await extractor.process_video(
            video_path="path/to/video.mp4",
            output_dir="path/to/output",
            scene_timestamps=[0.0, 15.5, 30.2, ...],  # From scene detection
        )
    """
    
    def __init__(
        self,
        audio_format: str = "m4a",
        audio_bitrate: str = "128k",
        audio_sample_rate: int = 16000,  # Optimal for speech recognition
    ):
        self.audio_format = audio_format
        self.audio_bitrate = audio_bitrate
        self.audio_sample_rate = audio_sample_rate
    
    def get_video_duration_ms(self, video_path: str) -> Optional[int]:
        """Get video duration in milliseconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                duration_sec = float(result.stdout.strip())
                return int(duration_sec * 1000)
        except Exception as e:
            print(f"  Warning: Could not get video duration: {e}")
        return None
    
    def extract_full_audio(
        self,
        video_path: str,
        output_path: str,
    ) -> Optional[str]:
        """
        Extract full audio track from video.
        
        Args:
            video_path: Path to video file
            output_path: Path to save audio file
            
        Returns:
            Path to extracted audio file, or None on failure
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",  # Overwrite output
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "aac" if self.audio_format == "m4a" else "libmp3lame",
            "-ab", self.audio_bitrate,
            "-ar", str(self.audio_sample_rate),
            output_path,
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=600
            )
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                print(f"  Audio extraction failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"  Audio extraction error: {e}")
        
        return None
    
    def segment_audio(
        self,
        audio_path: str,
        output_dir: str,
        scene_timestamps: List[float],  # In seconds
        duration_sec: float,
    ) -> List[Dict[str, Any]]:
        """
        Split audio into segments based on scene timestamps.
        
        Args:
            audio_path: Path to full audio file
            output_dir: Directory to save segments
            scene_timestamps: List of scene start times in seconds
            duration_sec: Total video duration in seconds
            
        Returns:
            List of segment info dicts with paths and timestamps
        """
        os.makedirs(output_dir, exist_ok=True)
        segments = []
        
        # Create segment boundaries
        boundaries = list(scene_timestamps)
        if not boundaries or boundaries[0] > 0.1:
            boundaries.insert(0, 0.0)
        boundaries.append(duration_sec)
        
        for i in range(len(boundaries) - 1):
            start_sec = boundaries[i]
            end_sec = boundaries[i + 1]
            duration = end_sec - start_sec
            
            # Skip very short segments (< 0.5 seconds)
            if duration < 0.5:
                continue
            
            segment_filename = f"segment_{i:04d}.{self.audio_format}"
            segment_path = os.path.join(output_dir, segment_filename)
            
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i", audio_path,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-acodec", "copy",  # No re-encoding
                segment_path,
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                
                if result.returncode == 0 and os.path.exists(segment_path):
                    segments.append({
                        "index": i,
                        "start_ms": int(start_sec * 1000),
                        "end_ms": int(end_sec * 1000),
                        "duration_ms": int(duration * 1000),
                        "path": segment_path,
                    })
                else:
                    print(f"  Segment {i} failed: {result.stderr[:100]}")
            except Exception as e:
                print(f"  Segment {i} error: {e}")
        
        return segments
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        video_id: str,
        title: str,
        youtube_url: str,
        scene_timestamps: List[float],
        channel_name: Optional[str] = None,
    ) -> VideoManifest:
        """
        Process a video: extract audio and segment by scenes.
        
        Args:
            video_path: Path to video file
            output_dir: Base output directory for this video
            video_id: YouTube video ID
            title: Video title
            youtube_url: Full YouTube URL
            scene_timestamps: Scene change timestamps in seconds
            channel_name: Channel name
            
        Returns:
            VideoManifest with all metadata
        """
        # Get duration
        duration_ms = self.get_video_duration_ms(video_path)
        if not duration_ms:
            duration_ms = int(scene_timestamps[-1] * 1000) if scene_timestamps else 0
        
        duration_sec = duration_ms / 1000.0
        
        # Create manifest
        manifest = VideoManifest(
            video_id=video_id,
            title=title,
            youtube_url=youtube_url,
            channel_name=channel_name,
            duration_ms=duration_ms,
        )
        
        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        audio_segments_dir = os.path.join(output_dir, "audio_segments")
        
        manifest.video_path = video_path
        manifest.frames_dir = frames_dir
        manifest.audio_segments_dir = audio_segments_dir
        
        # Extract full audio
        audio_path = os.path.join(output_dir, f"audio.{self.audio_format}")
        print(f"  Extracting full audio...")
        extracted_audio = self.extract_full_audio(video_path, audio_path)
        
        if extracted_audio:
            manifest.audio_path = extracted_audio
            
            # Segment audio by scenes
            print(f"  Segmenting audio into {len(scene_timestamps)} scenes...")
            segments = self.segment_audio(
                audio_path=extracted_audio,
                output_dir=audio_segments_dir,
                scene_timestamps=scene_timestamps,
                duration_sec=duration_sec,
            )
            
            manifest.total_audio_segments = len(segments)
            
            # Create scene infos from segments
            for seg in segments:
                scene = SceneInfo(
                    index=seg["index"],
                    start_ms=seg["start_ms"],
                    end_ms=seg["end_ms"],
                    duration_ms=seg["duration_ms"],
                    audio_path=seg["path"],
                )
                manifest.scenes.append(scene)
            
            print(f"  Created {len(segments)} audio segments")
        else:
            print(f"  Warning: Audio extraction failed")
        
        # Save manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        manifest.save(manifest_path)
        print(f"  Manifest saved: {manifest_path}")
        
        return manifest
    
    def link_frames_to_scenes(
        self,
        manifest: VideoManifest,
        frame_infos: List[Dict[str, Any]],
    ) -> VideoManifest:
        """
        Link frame analysis results to scenes based on timestamps.
        
        Args:
            manifest: VideoManifest to update
            frame_infos: List of frame analysis results with timestamps
            
        Returns:
            Updated manifest with frame links
        """
        for frame in frame_infos:
            timestamp_ms = int(frame.get("timestamp", 0) * 1000)
            
            # Find matching scene
            for scene in manifest.scenes:
                if scene.start_ms <= timestamp_ms < scene.end_ms:
                    # Update scene with frame info
                    scene.frame_path = frame.get("path")
                    scene.has_nko = frame.get("has_nko", False)
                    scene.nko_text = frame.get("nko_text")
                    scene.latin_text = frame.get("latin_text")
                    scene.english_text = frame.get("english_text")
                    scene.confidence = frame.get("confidence", 0.0)
                    break
        
        # Update stats
        manifest.total_frames = len(frame_infos)
        manifest.nko_frames = sum(1 for f in frame_infos if f.get("has_nko", False))
        
        return manifest


# Convenience function for direct use
def extract_audio_segments(
    video_path: str,
    output_dir: str,
    video_id: str,
    title: str,
    youtube_url: str,
    scene_timestamps: List[float],
    channel_name: Optional[str] = None,
    audio_format: str = "m4a",
) -> VideoManifest:
    """
    Convenience function to extract audio segments from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        video_id: YouTube video ID
        title: Video title
        youtube_url: Full YouTube URL
        scene_timestamps: Scene change timestamps in seconds
        channel_name: Channel name
        audio_format: Output audio format (m4a, mp3)
        
    Returns:
        VideoManifest with all metadata
    """
    extractor = AudioExtractor(audio_format=audio_format)
    return extractor.process_video(
        video_path=video_path,
        output_dir=output_dir,
        video_id=video_id,
        title=title,
        youtube_url=youtube_url,
        scene_timestamps=scene_timestamps,
        channel_name=channel_name,
    )


if __name__ == "__main__":
    # Test with a sample video
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    extractor = AudioExtractor()
    
    # Get duration
    duration = extractor.get_video_duration_ms(video_path)
    print(f"Video duration: {duration}ms")
    
    # Test with fake scene timestamps (every 60 seconds)
    if duration:
        scene_timestamps = [i * 60.0 for i in range(int(duration / 1000 / 60) + 1)]
        print(f"Scene timestamps: {scene_timestamps}")
        
        manifest = extractor.process_video(
            video_path=video_path,
            output_dir="./test_output",
            video_id="test",
            title="Test Video",
            youtube_url="https://youtube.com/watch?v=test",
            scene_timestamps=scene_timestamps,
        )
        
        print(f"\nManifest: {manifest.to_dict()}")

