#!/usr/bin/env python3
"""
N'Ko Video Analyzer with 5-World Generation (Production Grade)

Enhanced local frame extraction + Gemini analysis pipeline with:
- Smart frame sampling across ENTIRE video duration
- Scene change detection for slide-based content
- Duplicate frame detection to avoid redundant analysis
- N'Ko OCR using Gemini multimodal API
- 5-world variant generation using text-only API
- Supabase storage for trajectories
- Robust error handling and retry logic

Flow:
1. Download video from YouTube using yt-dlp
2. Get video duration and calculate optimal sampling
3. Extract frames evenly distributed across full video
4. Optional: Scene detection for slide changes
5. Send frames to Gemini API for N'Ko text analysis (~$0.002/frame)
6. Generate 5 world variants for each detection (~$0.0001/call)
7. Store results in Supabase (sources, frames, detections, trajectories)

Cost: ~$0.20-0.30 per video (100 frames + 5 worlds per detection)
"""

import asyncio
import aiohttp
import subprocess
import json
import os
import sys
import base64
import yaml
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re

try:
    import scrapetube
except ImportError:
    print("Run: pip install scrapetube aiohttp pyyaml httpx")
    sys.exit(1)

# Import our modules
from world_generator import WorldGenerator, WorldVariant, WorldGenerationResult, WORLDS
from supabase_client import (
    SupabaseClient,
    SourceData,
    FrameData,
    DetectionData,
    TrajectoryData,
    TrajectoryNodeData,
)
from frame_filter import SmartFrameExtractor, FrameInfo
from retry_utils import retry_with_backoff, RetryConfig, RetryError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
CHANNEL_URL = "https://www.youtube.com/@babamamadidiane"

# Gemini settings
GEMINI_MODEL = "gemini-3-flash-preview"  # Updated: 100% N'Ko detection vs 83% for 2.5-flash, no hallucination vs GPT-4.1
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Default worlds to generate
DEFAULT_WORLDS = ["world_everyday", "world_formal", "world_storytelling", "world_proverbs", "world_educational"]

# Default configuration
DEFAULT_CONFIG = {
    "extraction": {
        "target_frames": 100,
        "use_scene_detection": True,
        "use_deduplication": True,
        "skip_intro_seconds": 10,
        "skip_credits_seconds": 30,
    },
    "api": {
        "gemini": {
            "model": "gemini-2.0-flash",
            "timeout_seconds": 90,
            "max_retries": 3,
            "retry_base_delay": 2.0,
            "retry_max_delay": 60.0,
            "rate_limit_delay": 0.5,
        }
    },
    "storage": {
        "supabase": {"enabled": True},
        "local": {"temp_dir": "./data/temp", "keep_frames": False},
    },
    "worlds": {
        "enabled": True,
        "selected": DEFAULT_WORLDS,
    },
    "processing": {
        "stop_on_error": False,
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load production config from YAML.
    
    Args:
        config_path: Path to YAML config file. If None, looks for
                    training/config/production.yaml
    
    Returns:
        Configuration dictionary merged with defaults
    """
    if config_path is None:
        # Look for config in standard locations
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / "config" / "production.yaml",
            script_dir / "config" / "production.yaml",
            Path("./production.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        
        # Deep merge with defaults
        return _deep_merge(DEFAULT_CONFIG, file_config)
    
    logger.info("Using default configuration")
    return DEFAULT_CONFIG


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_index: int
    timestamp: float
    frame_path: Optional[str] = None
    has_nko: bool = False
    nko_text: Optional[str] = None
    latin_transliteration: Optional[str] = None
    english_translation: Optional[str] = None
    confidence: float = 0.0
    raw_response: Optional[str] = None
    worlds: List[WorldVariant] = field(default_factory=list)
    # Database IDs (populated after storage)
    frame_id: Optional[str] = None
    detection_id: Optional[str] = None


@dataclass
class VideoAnalysis:
    """Complete analysis result for a video."""
    video_id: str
    title: str
    youtube_url: str
    channel_name: Optional[str] = None
    frames_analyzed: int = 0
    frames_with_nko: int = 0
    total_world_variants: int = 0
    frames: List[FrameAnalysis] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None
    processing_time_ms: int = 0
    # Database ID (populated after storage)
    source_id: Optional[str] = None


class NkoAnalyzer:
    """
    N'Ko Video Analyzer with world generation and Supabase storage.
    
    Supports async context manager for session reuse:
    
        async with NkoAnalyzer(...) as analyzer:
            result = await analyzer.analyze_video(...)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        store_supabase: bool = False,
        generate_worlds: bool = True,
        worlds: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            api_key: Gemini API key
            store_supabase: Whether to store results in Supabase
            generate_worlds: Whether to generate world variants
            worlds: Which worlds to generate (defaults to all 5)
            config: Configuration dictionary (overrides config_path)
            config_path: Path to YAML config file
        """
        # Load configuration
        if config is not None:
            self.config = _deep_merge(DEFAULT_CONFIG, config)
        else:
            self.config = load_config(config_path)
        
        # API settings
        self.api_key = api_key or GEMINI_API_KEY
        api_config = self.config.get("api", {}).get("gemini", {})
        self.api_timeout = api_config.get("timeout_seconds", 90)
        self.rate_limit_delay = api_config.get("rate_limit_delay", 0.5)
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_retries=api_config.get("max_retries", 3),
            base_delay=api_config.get("retry_base_delay", 2.0),
            max_delay=api_config.get("retry_max_delay", 60.0),
        )
        
        # Feature flags
        self.store_supabase = store_supabase or self.config.get("storage", {}).get("supabase", {}).get("enabled", False)
        self.generate_worlds = generate_worlds if generate_worlds is not None else self.config.get("worlds", {}).get("enabled", True)
        self.worlds = worlds or self.config.get("worlds", {}).get("selected", DEFAULT_WORLDS)
        
        # Session for connection reuse (initialized in __aenter__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session: bool = False
        
        # Initialize sub-components
        self.world_generator: Optional[WorldGenerator] = None
        self.supabase: Optional[SupabaseClient] = None
        
        if self.generate_worlds:
            self.world_generator = WorldGenerator(api_key=self.api_key)
        
        if self.store_supabase:
            try:
                self.supabase = SupabaseClient()
            except ValueError as e:
                logger.warning(f"Supabase not configured: {e}")
                self.store_supabase = False
    
    async def __aenter__(self) -> 'NkoAnalyzer':
        """Enter async context - create shared session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.api_timeout)
            )
            self._owns_session = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context - close session if we own it."""
        if self._session and self._owns_session:
            await self._session.close()
            self._session = None
        return False
    
    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.api_timeout)
            )
            self._owns_session = True
        return self._session
    
    async def analyze_frame(
        self,
        frame_path: str,
        frame_index: int,
        timestamp: Optional[float] = None,
    ) -> FrameAnalysis:
        """
        Analyze a single frame with Gemini multimodal API.
        
        Uses retry logic with exponential backoff for robustness.
        Reuses HTTP session for connection pooling.
        
        Args:
            frame_path: Path to the frame image
            frame_index: Index of the frame
            timestamp: Frame timestamp in seconds (optional)
            
        Returns:
            FrameAnalysis with OCR results
        """
        if timestamp is None:
            timestamp = float(frame_index)  # Fallback to index as timestamp
        
        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=timestamp,
            frame_path=frame_path,
        )
        
        # Read frame data
        try:
            with open(frame_path, "rb") as f:
                frame_data = f.read()
        except IOError as e:
            analysis.raw_response = f"File read error: {e}"
            return analysis
        
        # Encode as base64
        image_b64 = base64.b64encode(frame_data).decode("utf-8")
        
        prompt = """Analyze this video frame for N'Ko script (ߒߞߏ) text.

If you find N'Ko text:
1. Extract all N'Ko text exactly as written
2. Provide Latin transliteration  
3. Provide English translation

Respond in this exact JSON format:
{
    "has_nko_text": true/false,
    "nko_text": "extracted N'Ko text or null",
    "latin_transliteration": "transliteration or null", 
    "english_translation": "translation or null",
    "confidence": 0.0-1.0,
    "notes": "any additional observations"
}

Focus on clear, visible text. Ignore blurry or partial text."""

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }
        
        # Get or create session
        session = self._get_session()
        
        async def _call_gemini_api():
            """Inner function for retry logic."""
            async with session.post(
                f"{GEMINI_API_URL}?key={self.api_key}",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited - raise to trigger retry
                    raise aiohttp.ClientError(f"Rate limited (429)")
                elif response.status >= 500:
                    # Server error - raise to trigger retry
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"Server error {response.status}: {error_text[:100]}")
                else:
                    # Client error - don't retry
                    error_text = await response.text()
                    return {"error": f"API error {response.status}: {error_text[:200]}"}
        
        try:
            # Call API with retry logic
            data = await retry_with_backoff(
                _call_gemini_api,
                max_retries=self.retry_config.max_retries,
                base_delay=self.retry_config.base_delay,
                max_delay=self.retry_config.max_delay,
                exceptions=(aiohttp.ClientError, asyncio.TimeoutError),
            )
            
            if "error" in data:
                analysis.raw_response = data["error"]
            else:
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    # Handle markdown code blocks
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                    result = json.loads(text.strip())
                    
                    analysis.has_nko = result.get("has_nko_text", False)
                    analysis.nko_text = result.get("nko_text")
                    analysis.latin_transliteration = result.get("latin_transliteration")
                    analysis.english_translation = result.get("english_translation")
                    analysis.confidence = result.get("confidence", 0.0)
                    analysis.raw_response = json.dumps(result)
                    
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    analysis.raw_response = f"Parse error: {e}"
                    
        except RetryError as e:
            logger.error(f"Frame {frame_index} failed after {e.attempts} attempts: {e.last_exception}")
            analysis.raw_response = f"Retry exhausted: {e.last_exception}"
        except Exception as e:
            logger.error(f"Frame {frame_index} unexpected error: {e}")
            analysis.raw_response = f"Unexpected error: {e}"
        
        return analysis
    
    async def generate_worlds_for_frame(
        self,
        frame: FrameAnalysis,
    ) -> FrameAnalysis:
        """
        Generate 5-world variants for a frame with N'Ko text.
        
        Args:
            frame: FrameAnalysis with N'Ko text
            
        Returns:
            Updated FrameAnalysis with world variants
        """
        if not frame.nko_text or not self.world_generator:
            return frame
        
        result = await self.world_generator.generate_worlds(
            nko_text=frame.nko_text,
            latin_text=frame.latin_transliteration,
            translation=frame.english_translation,
            worlds=self.worlds,
        )
        
        frame.worlds = result.worlds
        return frame
    
    async def analyze_video(
        self,
        video_id: str,
        title: str,
        youtube_url: str,
        temp_dir: str,
        target_frames: int = 100,
        use_smart_extraction: bool = True,
        use_scene_detection: bool = True,
        channel_name: Optional[str] = None,
    ) -> VideoAnalysis:
        """
        Full analysis pipeline for a single video.
        
        Args:
            video_id: YouTube video ID
            title: Video title
            youtube_url: Full YouTube URL
            temp_dir: Temporary directory for downloads
            target_frames: Target number of frames to extract
            use_smart_extraction: Use SmartFrameExtractor with dedup
            use_scene_detection: Use scene detection for slides
            channel_name: YouTube channel name
            
        Returns:
            VideoAnalysis with all results
        """
        start_time = datetime.now()
        
        analysis = VideoAnalysis(
            video_id=video_id,
            title=title,
            youtube_url=youtube_url,
            channel_name=channel_name,
        )
        
        print(f"\n[{video_id}] {title[:50]}...")
        
        # Step 1: Download video
        print("  Downloading video...")
        video_path = self._download_video(youtube_url, temp_dir)
        
        if not video_path:
            analysis.status = "download_failed"
            analysis.error = "Failed to download video"
            return analysis
        
        print(f"  Video: {video_path}")
        
        # Step 2: Extract frames with smart filtering
        print("  Extracting frames with intelligent filtering...")
        frames_dir = os.path.join(temp_dir, video_id, "frames")
        
        if use_smart_extraction:
            # Use SmartFrameExtractor for perceptual hash dedup + scene detection
            extractor = SmartFrameExtractor(
                target_frames=target_frames,
                use_scene_detection=use_scene_detection,
                use_deduplication=True,
                skip_intro=True,
                skip_credits=True,
            )
            
            frame_infos = extractor.extract_frames(video_path, frames_dir)
            
            # Print filter stats
            stats = extractor.get_stats()
            print(f"  Filter stats: {stats['total_frames']} total → {stats['unique_frames']} unique")
            print(f"    Duplicates removed: {stats['duplicates_removed']}")
            print(f"    Intro/credits skipped: {stats['intro_skipped'] + stats['credits_skipped']}")
            print(f"    Reduction: {stats['reduction']}")
            
            if not frame_infos:
                analysis.status = "extraction_failed"
                analysis.error = "Failed to extract frames"
                return analysis
            
            print(f"  Processing {len(frame_infos)} unique frames...")
            
            # Step 3: Analyze frames with Gemini (OCR)
            print("  Analyzing with Gemini OCR...")
            for frame_info in frame_infos:
                frame_analysis = await self.analyze_frame(
                    frame_path=frame_info.path,
                    frame_index=frame_info.index,
                    timestamp=frame_info.timestamp,
                )
                
                analysis.frames.append(frame_analysis)
                analysis.frames_analyzed += 1
                
                if frame_analysis.has_nko:
                    analysis.frames_with_nko += 1
                    nko = frame_analysis.nko_text[:30] if frame_analysis.nko_text else ""
                    print(f"    Frame {frame_info.index} @{frame_info.timestamp:.0f}s: ✓ N'Ko: {nko}...")
                else:
                    if frame_info.index % 10 == 0:  # Only log every 10th frame
                        print(f"    Frame {frame_info.index} @{frame_info.timestamp:.0f}s: No N'Ko text")
                
                # Rate limiting
                await asyncio.sleep(0.5)
        else:
            # Legacy extraction (for backwards compatibility)
            frame_paths = self._extract_frames(video_path, frames_dir, target_frames=target_frames)
            
            if not frame_paths:
                analysis.status = "extraction_failed"
                analysis.error = "Failed to extract frames"
                return analysis
            
            print(f"  Extracted {len(frame_paths)} frames")
            
            # Step 3: Analyze frames with Gemini (OCR)
            print("  Analyzing with Gemini OCR...")
            for i, frame_path in enumerate(frame_paths):
                frame_analysis = await self.analyze_frame(
                    frame_path=frame_path,
                    frame_index=i,
                )
                
                analysis.frames.append(frame_analysis)
                analysis.frames_analyzed += 1
                
                if frame_analysis.has_nko:
                    analysis.frames_with_nko += 1
                    nko = frame_analysis.nko_text[:30] if frame_analysis.nko_text else ""
                    print(f"    Frame {i}: ✓ N'Ko: {nko}...")
                else:
                    print(f"    Frame {i}: No N'Ko text")
                
                # Rate limiting
                await asyncio.sleep(0.5)
        
        # Step 4: Generate worlds for frames with N'Ko text
        if self.generate_worlds:
            nko_frames = [f for f in analysis.frames if f.nko_text]
            if nko_frames:
                print(f"  Generating 5 worlds for {len(nko_frames)} frames...")
                for frame in nko_frames:
                    await self.generate_worlds_for_frame(frame)
                    variant_count = sum(len(w.variants) for w in frame.worlds)
                    analysis.total_world_variants += variant_count
                    print(f"    Frame {frame.frame_index}: {variant_count} variants generated")
        
        # Step 5: Store in Supabase
        if self.store_supabase and self.supabase:
            print("  Storing in Supabase...")
            await self._store_in_supabase(analysis)
            print(f"    Source ID: {analysis.source_id}")
        
        analysis.status = "completed"
        analysis.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        print(f"  ✓ Completed in {analysis.processing_time_ms}ms")
        print(f"    Frames: {analysis.frames_analyzed}, N'Ko: {analysis.frames_with_nko}, Variants: {analysis.total_world_variants}")
        
        return analysis
    
    def _download_video(self, youtube_url: str, output_path: str) -> Optional[str]:
        """
        Download video using yt-dlp with multiple fallback strategies.
        
        Handles YouTube's SABR streaming restrictions by trying multiple
        format selectors and extractor arguments.
        
        Uses cookies file if available for authenticated downloads.
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Check for cookies file
        script_dir = Path(__file__).parent
        cookies_file = script_dir / "cookies.txt"
        has_cookies = cookies_file.exists()
        
        if has_cookies:
            logger.info(f"Using cookies from: {cookies_file}")
        else:
            logger.warning("No cookies.txt found - downloads may fail. Run: ./export-cookies-local.sh")
        
        # Multiple format strategies to handle SABR restrictions
        format_strategies = [
            # Strategy 1: Standard best quality with cookies
            {
                "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                "extra_args": [],
            },
            # Strategy 2: Single stream (avoids merge issues)
            {
                "format": "best[height<=720]",
                "extra_args": [],
            },
            # Strategy 3: Force iOS client (often bypasses restrictions)
            {
                "format": "best",
                "extra_args": ["--extractor-args", "youtube:player_client=ios"],
            },
            # Strategy 4: Force Android client
            {
                "format": "best",
                "extra_args": ["--extractor-args", "youtube:player_client=android"],
            },
            # Strategy 5: HLS streams only
            {
                "format": "best",
                "extra_args": ["--extractor-args", "youtube:player_client=web", "--prefer-free-formats"],
            },
        ]
        
        for i, strategy in enumerate(format_strategies, 1):
            cmd = [
                "yt-dlp",
                "-f", strategy["format"],
                "-o", os.path.join(output_path, "%(id)s.%(ext)s"),
                "--no-playlist",
                "--merge-output-format", "mp4",  # Ensure mp4 output
                "--retries", "3",
                "--fragment-retries", "3",
            ]
            
            # Add cookies if available
            if has_cookies:
                cmd.extend(["--cookies", str(cookies_file)])
            
            cmd.extend(strategy["extra_args"])
            cmd.append(youtube_url)
            
            try:
                logger.debug(f"Download attempt {i}/{len(format_strategies)}: {strategy['format']}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                
                if result.returncode == 0:
                    # Check for downloaded file
                    for ext in [".mp4", ".webm", ".mkv"]:
                        files = list(Path(output_path).glob(f"*{ext}"))
                        if files:
                            logger.info(f"Downloaded: {files[0]}")
                            return str(files[0])
                else:
                    logger.warning(f"Strategy {i} failed: {result.stderr[:200]}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Strategy {i} timed out")
            except Exception as e:
                logger.warning(f"Strategy {i} error: {e}")
        
        logger.error(f"All download strategies failed for {youtube_url}")
        return None
    
    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds using ffprobe."""
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
                return float(result.stdout.strip())
        except Exception as e:
            print(f"  Warning: Could not get video duration: {e}")
        return None
    
    def _extract_frames(
        self,
        video_path: str,
        output_dir: str,
        target_frames: int = 100,
        use_scene_detection: bool = False,
    ) -> List[str]:
        """
        Extract frames evenly distributed across the ENTIRE video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            target_frames: Target number of frames to extract
            use_scene_detection: Use scene change detection for slides
            
        Returns:
            List of frame file paths with timestamps
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        if not duration:
            print("  Warning: Using fallback extraction (duration unknown)")
            return self._extract_frames_fallback(video_path, output_dir, target_frames)
        
        print(f"  Video duration: {int(duration)}s ({int(duration/60)}m {int(duration%60)}s)")
        
        # Calculate frame interval for even distribution
        interval = duration / target_frames
        print(f"  Sampling interval: {interval:.1f}s ({target_frames} frames across full video)")
        
        if use_scene_detection:
            # Use scene detection for slide-based content
            return self._extract_frames_scene_detection(
                video_path, output_dir, target_frames, duration
            )
        
        # Extract frames at calculated intervals across full video
        # Using select filter to pick frames at specific timestamps
        timestamps = [i * interval for i in range(target_frames)]
        
        # Build select filter for specific timestamps
        # fps=1/{interval} gives us approximately the right sampling
        calculated_fps = 1.0 / interval if interval > 0 else 0.1
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", video_path,
            "-vf", f"fps={calculated_fps:.6f},scale='min(720,iw)':-1",
            "-frames:v", str(target_frames),
            "-q:v", "2",
            os.path.join(output_dir, "frame_%04d.jpg"),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  FFmpeg warning: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("  FFmpeg timed out")
            return []
        
        frames = sorted(Path(output_dir).glob("frame_*.jpg"))
        return [str(f) for f in frames]
    
    def _extract_frames_scene_detection(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int,
        duration: float,
    ) -> List[str]:
        """
        Extract frames using scene change detection.
        Better for slide-based educational content.
        """
        print("  Using scene detection for slide content...")
        
        # Scene detection threshold (0.3-0.5 works well for slides)
        threshold = 0.3
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',scale='min(720,iw)':-1",
            "-vsync", "vfr",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            os.path.join(output_dir, "frame_%04d.jpg"),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  Scene detection warning: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("  Scene detection timed out")
            return []
        
        frames = sorted(Path(output_dir).glob("frame_*.jpg"))
        
        # If scene detection found too few frames, fall back to even sampling
        if len(frames) < 10:
            print(f"  Scene detection found only {len(frames)} frames, using even sampling...")
            # Clear and retry with even sampling
            for f in frames:
                Path(f).unlink()
            return self._extract_frames(video_path, output_dir, max_frames, use_scene_detection=False)
        
        return [str(f) for f in frames]
    
    def _extract_frames_fallback(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int,
    ) -> List[str]:
        """Fallback extraction when duration is unknown."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", video_path,
            "-vf", "fps=0.1,scale='min(720,iw)':-1",  # 1 frame per 10 seconds
            "-frames:v", str(max_frames),
            "-q:v", "2",
            os.path.join(output_dir, "frame_%04d.jpg"),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  FFmpeg warning: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("  FFmpeg timed out")
            return []
        
        frames = sorted(Path(output_dir).glob("frame_*.jpg"))
        return [str(f) for f in frames]
    
    async def _store_in_supabase(self, analysis: VideoAnalysis) -> None:
        """Store all analysis results in Supabase."""
        if not self.supabase:
            return
        
        async with aiohttp.ClientSession() as session:
            # 1. Check if source already exists
            existing = await self.supabase.get_source_by_external_id(
                analysis.video_id, session=session
            )
            if existing:
                analysis.source_id = existing["id"]
                print(f"    Source already exists: {analysis.source_id}")
                return
            
            # 2. Insert source
            source_id = await self.supabase.insert_source(
                SourceData(
                    source_type="youtube",
                    url=analysis.youtube_url,
                    external_id=analysis.video_id,
                    title=analysis.title,
                    channel_name=analysis.channel_name,
                    status="completed",
                    metadata={
                        "frames_analyzed": analysis.frames_analyzed,
                        "frames_with_nko": analysis.frames_with_nko,
                        "total_world_variants": analysis.total_world_variants,
                    },
                ),
                session=session,
            )
            analysis.source_id = source_id
            
            # 3. Insert frames and detections
            for frame in analysis.frames:
                frame_id = await self.supabase.insert_frame(
                    FrameData(
                        source_id=source_id,
                        frame_index=frame.frame_index,
                        timestamp_ms=int(frame.timestamp * 1000),
                        has_nko=frame.has_nko,
                        confidence=frame.confidence,
                    ),
                    session=session,
                )
                frame.frame_id = frame_id
                
                # Insert detection if N'Ko text found
                if frame.nko_text:
                    detection_id = await self.supabase.insert_detection(
                        DetectionData(
                            frame_id=frame_id,
                            nko_text=frame.nko_text,
                            latin_text=frame.latin_transliteration,
                            english_text=frame.english_translation,
                            confidence=frame.confidence,
                            gemini_model=GEMINI_MODEL,
                            raw_response={"raw": frame.raw_response} if frame.raw_response else None,
                        ),
                        session=session,
                    )
                    frame.detection_id = detection_id
                    
                    # Insert trajectories for worlds
                    if frame.worlds:
                        await self.supabase.store_world_trajectories(
                            detection_id=detection_id,
                            worlds=[asdict(w) for w in frame.worlds],
                            session=session,
                        )
            
            # 4. Update source status
            await self.supabase.update_source_status(
                source_id,
                status="completed",
                frame_count=analysis.frames_analyzed,
                nko_frame_count=analysis.frames_with_nko,
                total_detections=analysis.frames_with_nko,
                session=session,
            )


def get_channel_videos(limit: Optional[int] = None) -> List[Dict[str, str]]:
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
            
            videos.append({
                "video_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": title,
            })
            
    except Exception as e:
        print(f"Error fetching videos: {e}")
    
    print(f"Found {len(videos)} videos")
    return videos


async def run_pipeline(
    limit: Optional[int] = None,
    frame_rate: float = 0.2,
    max_frames: int = 50,
    output_file: str = "analysis_results.json",
    temp_dir: str = "./temp",
    single_video: Optional[str] = None,
    store_supabase: bool = False,
    skip_worlds: bool = False,
    worlds: Optional[List[str]] = None,
):
    """
    Run the full analysis pipeline.
    
    Args:
        limit: Max number of videos to process
        frame_rate: Frame extraction rate (fps)
        max_frames: Max frames per video
        output_file: JSON output file
        temp_dir: Temp directory for downloads
        single_video: Single video URL to process
        store_supabase: Store results in Supabase
        skip_worlds: Skip world generation
        worlds: Which worlds to generate
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    # Initialize analyzer
    analyzer = NkoAnalyzer(
        store_supabase=store_supabase,
        generate_worlds=not skip_worlds,
        worlds=worlds,
    )
    
    print("=" * 60)
    print("N'Ko Video Analyzer with 5-World Generation")
    print("=" * 60)
    print(f"Frame rate: {frame_rate} fps (1 frame every {1/frame_rate:.1f}s)")
    print(f"Max frames: {max_frames}")
    print(f"World generation: {'Disabled' if skip_worlds else 'Enabled (' + ', '.join(worlds or DEFAULT_WORLDS) + ')'}")
    print(f"Supabase storage: {'Enabled' if store_supabase else 'Disabled'}")
    print("=" * 60)
    
    # Get video list
    if single_video:
        match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', single_video)
        video_id = match.group(1) if match else "unknown"
        videos = [{
            "video_id": video_id,
            "youtube_url": single_video,
            "title": "Test Video",
        }]
    else:
        videos = get_channel_videos(limit)
    
    if not videos:
        print("No videos to process!")
        return
    
    # Estimate cost
    estimated_cost = len(videos) * max_frames * 0.002  # OCR cost
    if not skip_worlds:
        # Assume 30% of frames have N'Ko text, 5 worlds each
        estimated_cost += len(videos) * max_frames * 0.3 * 5 * 0.0001
    print(f"\nEstimated cost: ~${estimated_cost:.2f}")
    print(f"Processing {len(videos)} videos...\n")
    
    # Process videos
    results = []
    for video in videos:
        analysis = await analyzer.analyze_video(
            video_id=video["video_id"],
            title=video["title"],
            youtube_url=video["youtube_url"],
            temp_dir=temp_dir,
            frame_rate=frame_rate,
            max_frames=max_frames,
        )
        results.append(analysis)
    
    # Summary
    completed = sum(1 for r in results if r.status == "completed")
    total_frames = sum(r.frames_analyzed for r in results)
    total_nko = sum(r.frames_with_nko for r in results)
    total_variants = sum(r.total_world_variants for r in results)
    
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Videos processed: {completed}/{len(results)}")
    print(f"  Total frames:     {total_frames}")
    print(f"  Frames with N'Ko: {total_nko}")
    print(f"  World variants:   {total_variants}")
    if store_supabase:
        stored = sum(1 for r in results if r.source_id)
        print(f"  Stored in Supabase: {stored}")
    
    # Save results to JSON
    output_data = {
        "run_time": datetime.now().isoformat(),
        "config": {
            "frame_rate": frame_rate,
            "max_frames": max_frames,
            "worlds": worlds or DEFAULT_WORLDS if not skip_worlds else [],
            "store_supabase": store_supabase,
        },
        "summary": {
            "videos_processed": completed,
            "total_frames": total_frames,
            "frames_with_nko": total_nko,
            "total_world_variants": total_variants,
        },
        "results": [
            {
                "video_id": r.video_id,
                "title": r.title,
                "youtube_url": r.youtube_url,
                "status": r.status,
                "frames_analyzed": r.frames_analyzed,
                "frames_with_nko": r.frames_with_nko,
                "total_world_variants": r.total_world_variants,
                "source_id": r.source_id,
                "error": r.error,
                "detections": [
                    {
                        "frame_index": f.frame_index,
                        "timestamp": f.timestamp,
                        "nko_text": f.nko_text,
                        "latin_transliteration": f.latin_transliteration,
                        "english_translation": f.english_translation,
                        "confidence": f.confidence,
                        "detection_id": f.detection_id,
                        "worlds": [
                            {
                                "world_name": w.world_name,
                                "variant_count": len(w.variants),
                                "cultural_notes": w.cultural_notes,
                                "error": w.error,
                            }
                            for w in f.worlds
                        ] if f.worlds else [],
                    }
                    for f in r.frames
                    if f.nko_text
                ],
            }
            for r in results
        ]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="N'Ko Video Analyzer with 5-World Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with world generation and Supabase storage
  python nko_analyzer.py --limit 5 --store-supabase

  # Single video test
  python nko_analyzer.py --video "https://www.youtube.com/watch?v=VIDEO_ID" --store-supabase

  # Cheaper: OCR only, no worlds
  python nko_analyzer.py --limit 10 --skip-worlds

  # Custom worlds
  python nko_analyzer.py --worlds world_everyday world_proverbs
        """
    )
    
    parser.add_argument("--limit", "-l", type=int, 
                        help="Limit number of videos to process")
    parser.add_argument("--frame-rate", type=float, default=0.2, 
                        help="Frames per second (default: 0.2 = 1 frame per 5 seconds)")
    parser.add_argument("--max-frames", type=int, default=50, 
                        help="Max frames per video (default: 50)")
    parser.add_argument("--output", "-o", type=str, default="analysis_results.json",
                        help="Output JSON file")
    parser.add_argument("--temp-dir", type=str, default="./temp",
                        help="Temporary directory for downloads")
    parser.add_argument("--video", "-v", type=str, 
                        help="Analyze single video URL")
    
    # New flags for world generation and Supabase
    parser.add_argument("--store-supabase", action="store_true",
                        help="Store results in Supabase database")
    parser.add_argument("--skip-worlds", action="store_true",
                        help="Skip world generation (faster, cheaper)")
    parser.add_argument("--worlds", nargs="+", 
                        choices=["world_everyday", "world_formal", "world_storytelling", 
                                "world_proverbs", "world_educational"],
                        help="Which worlds to generate (default: all 5)")
    
    args = parser.parse_args()
    
    asyncio.run(run_pipeline(
        limit=args.limit,
        frame_rate=args.frame_rate,
        max_frames=args.max_frames,
        output_file=args.output,
        temp_dir=args.temp_dir,
        single_video=args.video,
        store_supabase=args.store_supabase,
        skip_worlds=args.skip_worlds,
        worlds=args.worlds,
    ))


if __name__ == "__main__":
    main()
