#!/usr/bin/env python3
"""
Pass 4: ASR Transcription (Optional, Non-Blocking)

Transcribes audio segments using Whisper ASR and links to N'Ko slides
for curriculum building. This pass is OPTIONAL and non-blocking -
the main pipeline (Pass 1-3) works without it.

When ASR is ready, this pass:
1. Loads audio segments from Supabase that lack transcriptions
2. Sends audio to Whisper API (or local model)
3. Stores transcriptions with timestamps
4. Links transcribed audio to visual slides for curriculum

Usage:
    python run_transcription.py                    # Process all pending segments
    python run_transcription.py --limit 100        # Process first 100 segments
    python run_transcription.py --local            # Use local Whisper model
    python run_transcription.py --dry-run          # List segments without processing
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from supabase_client import SupabaseClient

# Check for OpenAI (Whisper API)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Check for local Whisper
try:
    import whisper
    HAS_LOCAL_WHISPER = True
except ImportError:
    HAS_LOCAL_WHISPER = False


# Progress tracking
PROGRESS_FILE = Path(__file__).parent.parent / "data" / "transcription_progress.json"


class WhisperTranscriber:
    """
    ASR transcription using Whisper (API or local).
    
    Supports:
    - OpenAI Whisper API (cloud, paid)
    - Local Whisper model (free, requires GPU)
    """
    
    def __init__(
        self,
        use_local: bool = False,
        model_name: str = "whisper-1",  # API model name
        local_model: str = "base",       # Local model: tiny, base, small, medium, large
    ):
        self.use_local = use_local
        self.model_name = model_name
        self.local_model_name = local_model
        
        if use_local:
            if not HAS_LOCAL_WHISPER:
                raise ImportError(
                    "Local Whisper not installed. "
                    "Run: pip install openai-whisper"
                )
            print(f"Loading local Whisper model: {local_model}...")
            self.local_model = whisper.load_model(local_model)
        else:
            if not HAS_OPENAI:
                raise ImportError(
                    "OpenAI not installed. "
                    "Run: pip install openai"
                )
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            self.client = OpenAI(api_key=api_key)
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "nko",  # N'Ko language code
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (nko for N'Ko, fr for French, etc.)
            
        Returns:
            Dict with transcription, language, confidence
        """
        if not os.path.exists(audio_path):
            return {
                "transcription": None,
                "error": f"Audio file not found: {audio_path}",
                "confidence": 0.0,
            }
        
        if self.use_local:
            return self._transcribe_local(audio_path, language)
        else:
            return self._transcribe_api(audio_path, language)
    
    def _transcribe_api(
        self,
        audio_path: str,
        language: str,
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        try:
            with open(audio_path, "rb") as audio_file:
                # Note: Whisper API doesn't officially support N'Ko
                # We'll use French as fallback since many N'Ko speakers are French
                api_language = language if language in ["en", "fr", "es", "de", "it"] else "fr"
                
                transcript = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                    response_format="verbose_json",
                    language=api_language,
                )
                
                return {
                    "transcription": transcript.text,
                    "language": transcript.language,
                    "confidence": getattr(transcript, "confidence", 0.8),
                    "duration": getattr(transcript, "duration", 0.0),
                    "model": self.model_name,
                }
        except Exception as e:
            return {
                "transcription": None,
                "error": str(e),
                "confidence": 0.0,
            }
    
    def _transcribe_local(
        self,
        audio_path: str,
        language: str,
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        try:
            # Local Whisper supports auto language detection
            result = self.local_model.transcribe(
                audio_path,
                task="transcribe",
                fp16=False,  # Use FP32 for CPU
            )
            
            return {
                "transcription": result["text"],
                "language": result.get("language", "unknown"),
                "confidence": 0.7,  # Local model doesn't provide confidence
                "segments": result.get("segments", []),
                "model": f"whisper-local-{self.local_model_name}",
            }
        except Exception as e:
            return {
                "transcription": None,
                "error": str(e),
                "confidence": 0.0,
            }


def save_progress(progress: Dict[str, Any]):
    """Save progress statistics."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


async def run_transcription(
    limit: int = 100,
    use_local: bool = False,
    local_model: str = "base",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run Pass 4: ASR Transcription on pending segments.
    
    Args:
        limit: Maximum segments to process
        use_local: Use local Whisper model
        local_model: Local model size (tiny, base, small, medium, large)
        dry_run: List segments without processing
        
    Returns:
        Progress statistics
    """
    # Initialize Supabase client
    try:
        supabase = SupabaseClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
        sys.exit(1)
    
    # Get pending segments
    print("Fetching segments needing transcription...")
    segments = await supabase.get_untranscribed_segments(limit=limit)
    
    if not segments:
        print("No segments need transcription!")
        return {"status": "no_pending", "count": 0}
    
    print(f"Found {len(segments)} segments to transcribe")
    
    if dry_run:
        print("\n=== DRY RUN: Segments to process ===")
        for i, seg in enumerate(segments[:20], 1):  # Show first 20
            print(f"  {i}. Segment {seg['segment_index']} ({seg['start_ms']}-{seg['end_ms']}ms)")
            print(f"     Audio: {seg.get('audio_path', 'N/A')}")
        if len(segments) > 20:
            print(f"  ... and {len(segments) - 20} more")
        return {"dry_run": True, "count": len(segments)}
    
    # Initialize transcriber
    try:
        transcriber = WhisperTranscriber(
            use_local=use_local,
            local_model=local_model,
        )
    except (ImportError, ValueError) as e:
        print(f"Error initializing transcriber: {e}")
        sys.exit(1)
    
    # Progress tracking
    progress = {
        "start_time": datetime.now().isoformat(),
        "total_segments": len(segments),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "use_local": use_local,
        "model": local_model if use_local else "whisper-1",
    }
    
    print(f"\n{'='*60}")
    print(f"PASS 4: ASR TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"Segments to process: {len(segments)}")
    print(f"Mode: {'Local Whisper' if use_local else 'OpenAI API'}")
    if not use_local:
        estimated_cost = len(segments) * 1 * 0.006  # ~1 min avg @ $0.006/min
        print(f"Estimated cost: ${estimated_cost:.2f} (assuming ~1 min per segment)")
    print(f"{'='*60}\n")
    
    # Process segments
    for i, segment in enumerate(segments, 1):
        segment_id = segment["id"]
        audio_path = segment.get("audio_path")
        
        print(f"[{i}/{len(segments)}] Segment {segment['segment_index']}: ", end="")
        
        if not audio_path or not os.path.exists(audio_path):
            print(f"⚠ Audio file missing: {audio_path}")
            progress["skipped"] += 1
            continue
        
        # Transcribe
        result = transcriber.transcribe(audio_path)
        
        if result.get("transcription"):
            # Store in Supabase
            try:
                await supabase.update_audio_segment_transcription(
                    segment_id=segment_id,
                    transcription=result["transcription"],
                    language=result.get("language", "unknown"),
                    confidence=result.get("confidence", 0.0),
                    model=result.get("model", "whisper"),
                )
                
                preview = result["transcription"][:50] + "..." if len(result["transcription"]) > 50 else result["transcription"]
                print(f"✓ \"{preview}\"")
                progress["completed"] += 1
            except Exception as e:
                print(f"✗ DB error: {e}")
                progress["failed"] += 1
        else:
            print(f"✗ {result.get('error', 'Unknown error')}")
            progress["failed"] += 1
        
        # Rate limiting for API
        if not use_local:
            await asyncio.sleep(0.5)
    
    progress["end_time"] = datetime.now().isoformat()
    save_progress(progress)
    
    print(f"\n{'='*60}")
    print(f"PASS 4 COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {progress['completed']}/{progress['total_segments']}")
    print(f"Failed: {progress['failed']}")
    print(f"Skipped (missing audio): {progress['skipped']}")
    print(f"{'='*60}\n")
    
    return progress


async def build_curriculum(
    source_id: Optional[str] = None,
    output_dir: str = "./data/curriculum",
) -> Dict[str, Any]:
    """
    Build curriculum from transcribed segments.
    
    Creates lesson files that combine:
    - N'Ko text from slides
    - Transliteration and translation
    - Teacher's spoken explanation (transcription)
    - Timestamps for review
    
    Args:
        source_id: Optional source ID to limit to single video
        output_dir: Directory to save curriculum files
        
    Returns:
        Curriculum statistics
    """
    # This is a placeholder for future curriculum building
    # Will be implemented when transcriptions are available
    print("Curriculum building not yet implemented.")
    print("Run transcription first to populate audio segments with transcriptions.")
    return {"status": "not_implemented"}


def main():
    parser = argparse.ArgumentParser(description="Pass 4: ASR Transcription (Optional)")
    parser.add_argument("--limit", type=int, default=100, help="Max segments to process")
    parser.add_argument("--local", action="store_true", help="Use local Whisper model")
    parser.add_argument("--model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Local Whisper model size")
    parser.add_argument("--dry-run", action="store_true", help="List segments without processing")
    parser.add_argument("--build-curriculum", action="store_true", help="Build curriculum from transcriptions")
    args = parser.parse_args()
    
    if args.build_curriculum:
        asyncio.run(build_curriculum())
    else:
        progress = asyncio.run(run_transcription(
            limit=args.limit,
            use_local=args.local,
            local_model=args.model,
            dry_run=args.dry_run,
        ))
        
        print(f"\nProgress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()

