"""
Supabase Pipeline Reporter

Reports pipeline progress and events to Supabase for real-time dashboard visibility.
Works alongside local JSON checkpoints for redundancy.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """Current pipeline progress state."""
    videos_total: int = 0
    videos_completed: int = 0
    videos_failed: int = 0
    frames_extracted: int = 0
    detections_found: int = 0
    audio_segments: int = 0
    cost_estimate: float = 0.0


class SupabaseReporter:
    """
    Reports pipeline progress to Supabase for dashboard observability.
    
    Usage:
        reporter = SupabaseReporter()
        run_id = reporter.start_run("extraction", channel_name="babamamadidiane", videos_total=532)
        
        # During processing
        reporter.event("video_start", video_id="abc123", message="Starting video...")
        reporter.update_progress(videos_completed=1, frames_extracted=10, detections_found=5)
        reporter.event("video_complete", video_id="abc123", message="Completed", data={"frames": 10})
        
        # On completion
        reporter.complete_run()
    """
    
    def __init__(self):
        """Initialize Supabase client if credentials are available."""
        self.client: Optional[Client] = None
        self.run_id: Optional[str] = None
        self.progress = PipelineProgress()
        self._enabled = False
        
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase package not installed. Pipeline reporting disabled.")
            return
        
        supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not found. Pipeline reporting disabled.")
            return
        
        try:
            self.client = create_client(supabase_url, supabase_key)
            self._enabled = True
            logger.info("Supabase reporter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
    
    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled."""
        return self._enabled and self.client is not None
    
    def start_run(
        self,
        run_type: str,
        channel_name: Optional[str] = None,
        videos_total: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Start a new pipeline run.
        
        Args:
            run_type: Type of run ('extraction', 'transcription', 'worlds', 'enrichment')
            channel_name: Optional channel being processed
            videos_total: Total videos to process
            metadata: Additional metadata
        
        Returns:
            Run ID if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        self.progress = PipelineProgress(videos_total=videos_total)
        
        try:
            result = self.client.table("pipeline_runs").insert({
                "run_type": run_type,
                "channel_name": channel_name,
                "status": "running",
                "videos_total": videos_total,
                "metadata": metadata or {}
            }).execute()
            
            if result.data and len(result.data) > 0:
                self.run_id = result.data[0]["id"]
                logger.info(f"Started pipeline run: {self.run_id}")
                
                # Log start event
                self.event("run_start", message=f"Started {run_type} pipeline", data={
                    "channel": channel_name,
                    "videos_total": videos_total
                })
                
                return self.run_id
        except Exception as e:
            logger.error(f"Failed to start pipeline run: {e}")
        
        return None
    
    def update_progress(
        self,
        videos_completed: Optional[int] = None,
        videos_failed: Optional[int] = None,
        frames_extracted: Optional[int] = None,
        detections_found: Optional[int] = None,
        audio_segments: Optional[int] = None,
        cost_estimate: Optional[float] = None
    ):
        """
        Update pipeline progress counters.
        
        Args:
            videos_completed: Total videos completed
            videos_failed: Total videos failed
            frames_extracted: Total frames extracted
            detections_found: Total N'Ko detections
            audio_segments: Total audio segments
            cost_estimate: Estimated API cost
        """
        # Update local state
        if videos_completed is not None:
            self.progress.videos_completed = videos_completed
        if videos_failed is not None:
            self.progress.videos_failed = videos_failed
        if frames_extracted is not None:
            self.progress.frames_extracted = frames_extracted
        if detections_found is not None:
            self.progress.detections_found = detections_found
        if audio_segments is not None:
            self.progress.audio_segments = audio_segments
        if cost_estimate is not None:
            self.progress.cost_estimate = cost_estimate
        
        if not self.enabled or not self.run_id:
            return
        
        try:
            self.client.table("pipeline_runs").update({
                "videos_completed": self.progress.videos_completed,
                "videos_failed": self.progress.videos_failed,
                "frames_extracted": self.progress.frames_extracted,
                "detections_found": self.progress.detections_found,
                "audio_segments": self.progress.audio_segments,
                "cost_estimate": self.progress.cost_estimate
            }).eq("id", self.run_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")
    
    def increment_progress(
        self,
        videos_completed: int = 0,
        videos_failed: int = 0,
        frames_extracted: int = 0,
        detections_found: int = 0,
        audio_segments: int = 0,
        cost_estimate: float = 0.0
    ):
        """
        Increment pipeline progress counters (adds to existing values).
        """
        self.update_progress(
            videos_completed=self.progress.videos_completed + videos_completed,
            videos_failed=self.progress.videos_failed + videos_failed,
            frames_extracted=self.progress.frames_extracted + frames_extracted,
            detections_found=self.progress.detections_found + detections_found,
            audio_segments=self.progress.audio_segments + audio_segments,
            cost_estimate=self.progress.cost_estimate + cost_estimate
        )
    
    def event(
        self,
        event_type: str,
        video_id: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Log a pipeline event.
        
        Args:
            event_type: Type of event (video_start, video_complete, detection_found, etc.)
            video_id: Optional video ID associated with event
            message: Human-readable message
            data: Additional event data
        """
        if not self.enabled or not self.run_id:
            return
        
        try:
            self.client.table("pipeline_events").insert({
                "run_id": self.run_id,
                "event_type": event_type,
                "video_id": video_id,
                "message": message,
                "data": data or {}
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")
    
    def video_start(self, video_id: str, title: str):
        """Log video processing start."""
        self.event("video_start", video_id=video_id, message=f"Processing: {title[:50]}")
    
    def video_complete(
        self,
        video_id: str,
        frames: int = 0,
        detections: int = 0,
        audio_segments: int = 0,
        duration_ms: int = 0
    ):
        """Log video processing completion."""
        self.event("video_complete", video_id=video_id, message="Completed", data={
            "frames": frames,
            "detections": detections,
            "audio_segments": audio_segments,
            "duration_ms": duration_ms
        })
        self.increment_progress(
            videos_completed=1,
            frames_extracted=frames,
            detections_found=detections,
            audio_segments=audio_segments
        )
    
    def video_failed(self, video_id: str, error: str):
        """Log video processing failure."""
        self.event("video_failed", video_id=video_id, message=f"Failed: {error[:100]}")
        self.increment_progress(videos_failed=1)
    
    def detection_found(
        self,
        video_id: str,
        nko_text: str,
        latin_text: Optional[str] = None,
        confidence: float = 0.0
    ):
        """Log a N'Ko detection event."""
        self.event("detection_found", video_id=video_id, message=nko_text[:50], data={
            "nko_text": nko_text,
            "latin_text": latin_text,
            "confidence": confidence
        })
    
    def complete_run(self, error_message: Optional[str] = None):
        """
        Mark the pipeline run as complete.
        
        Args:
            error_message: Optional error message if run failed
        """
        if not self.enabled or not self.run_id:
            return
        
        status = "failed" if error_message else "completed"
        
        try:
            self.client.table("pipeline_runs").update({
                "status": status,
                "ended_at": datetime.now().isoformat(),
                "error_message": error_message
            }).eq("id", self.run_id).execute()
            
            self.event(
                "run_complete" if not error_message else "run_failed",
                message=error_message or "Pipeline completed successfully",
                data={
                    "videos_completed": self.progress.videos_completed,
                    "videos_failed": self.progress.videos_failed,
                    "detections_found": self.progress.detections_found,
                    "cost_estimate": self.progress.cost_estimate
                }
            )
            
            logger.info(f"Pipeline run {self.run_id} {status}")
        except Exception as e:
            logger.error(f"Failed to complete run: {e}")
    
    def stop_run(self, reason: str = "User requested stop"):
        """Mark the pipeline run as stopped (graceful shutdown)."""
        if not self.enabled or not self.run_id:
            return
        
        try:
            self.client.table("pipeline_runs").update({
                "status": "stopped",
                "ended_at": datetime.now().isoformat(),
                "error_message": reason
            }).eq("id", self.run_id).execute()
            
            self.event("run_stopped", message=reason)
            logger.info(f"Pipeline run {self.run_id} stopped: {reason}")
        except Exception as e:
            logger.error(f"Failed to stop run: {e}")


# Singleton instance for easy access
_reporter: Optional[SupabaseReporter] = None


def get_reporter() -> SupabaseReporter:
    """Get or create the global reporter instance."""
    global _reporter
    if _reporter is None:
        _reporter = SupabaseReporter()
    return _reporter

