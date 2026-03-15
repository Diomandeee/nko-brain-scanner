#!/usr/bin/env python3
"""
Supabase Client for N'Ko Learning Pipeline

Handles all database operations for storing analysis results:
- nko_sources: Video metadata
- nko_frames: Individual frames
- nko_detections: OCR results
- nko_trajectories: Learning paths (cc-rag++ compatible)
- nko_trajectory_nodes: Individual trajectory nodes

Usage:
    from supabase_client import SupabaseClient
    
    client = SupabaseClient()
    source_id = await client.insert_source(video_data)
    frame_id = await client.insert_frame(source_id, frame_data)
"""

import asyncio
import aiohttp
import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Union


# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")


@dataclass
class SourceData:
    """Data for nko_sources table."""
    source_type: str = "youtube"
    url: str = ""
    external_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    duration_ms: Optional[int] = None
    language: str = "nko"
    quality: Optional[str] = None
    status: str = "pending"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FrameData:
    """Data for nko_frames table."""
    source_id: str
    frame_index: int
    timestamp_ms: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: str = "jpeg"
    has_nko: bool = False
    detection_count: int = 0
    confidence: float = 0.0
    is_keyframe: bool = False
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    image_hash: Optional[str] = None
    storage_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DetectionData:
    """Data for nko_detections table."""
    frame_id: str
    nko_text: str
    nko_text_normalized: Optional[str] = None
    latin_text: Optional[str] = None
    english_text: Optional[str] = None
    char_count: int = 0
    word_count: int = 0
    confidence: float = 0.0
    nko_coverage: float = 0.0
    is_valid_structure: bool = True
    bounding_box: Optional[Dict] = None
    text_region: Optional[str] = None
    context_type: Optional[str] = None
    gemini_model: Optional[str] = None
    raw_response: Optional[Dict] = None
    processing_time_ms: Optional[int] = None
    status: str = "raw"


@dataclass
class TrajectoryData:
    """Data for nko_trajectories table."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    trajectory_type: str = "content_flow"
    max_depth: int = 0
    total_nodes: int = 0
    avg_homogeneity: float = 1.0
    total_complexity: int = 0
    avg_temporal: float = 0.5
    dominant_phase: Optional[str] = None
    phase_distribution: Dict[str, float] = None
    avg_salience: float = 0.5
    high_salience_count: int = 0
    terminal_node_count: int = 0
    coherence_score: Optional[float] = None
    is_complete: bool = False
    is_successful: Optional[bool] = None
    
    def __post_init__(self):
        if self.phase_distribution is None:
            self.phase_distribution = {}


@dataclass
class TrajectoryNodeData:
    """Data for nko_trajectory_nodes table."""
    trajectory_id: str
    node_index: int
    learning_event_id: Optional[str] = None
    detection_id: Optional[str] = None
    vocabulary_id: Optional[str] = None
    trajectory_depth: int = 0
    trajectory_sibling_order: int = 0
    trajectory_homogeneity: float = 1.0
    trajectory_temporal: float = 0.0
    trajectory_complexity: int = 1
    trajectory_phase: Optional[str] = None
    trajectory_phase_confidence: float = 0.0
    salience_score: float = 0.5
    is_phase_transition: bool = False
    is_terminal: bool = False
    parent_node_id: Optional[str] = None
    child_count: int = 0
    content_preview: Optional[str] = None
    content_type: Optional[str] = None
    outcome: Optional[str] = None
    timestamp_ms: Optional[int] = None


@dataclass
class AudioSegmentData:
    """Data for nko_audio_segments table."""
    source_id: str
    segment_index: int
    start_ms: int
    end_ms: int
    frame_id: Optional[str] = None
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    audio_format: str = "m4a"
    transcription: Optional[str] = None
    transcription_language: Optional[str] = None
    transcription_confidence: Optional[float] = None
    transcription_model: Optional[str] = None
    has_nko: bool = False
    nko_text: Optional[str] = None
    latin_text: Optional[str] = None
    english_text: Optional[str] = None
    is_reviewed: bool = False
    lesson_notes: Optional[str] = None
    difficulty_level: Optional[int] = None


class SupabaseClient:
    """
    Async Supabase client for N'Ko pipeline.
    
    Uses PostgREST API directly for better async support.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """
        Initialize the Supabase client.
        
        Args:
            url: Supabase URL (defaults to SUPABASE_URL env var)
            key: Supabase service key (defaults to SUPABASE_SERVICE_KEY env var)
        """
        self.url = url or SUPABASE_URL
        self.key = key or SUPABASE_KEY
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials required. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
            )
        
        self.rest_url = f"{self.url}/rest/v1"
        self._headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    
    async def _request(
        self,
        method: str,
        table: str,
        data: Optional[Union[Dict, List[Dict]]] = None,
        params: Optional[Dict[str, str]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[Dict]:
        """
        Make a request to Supabase REST API.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            table: Table name
            data: Request body
            params: Query parameters
            session: Optional aiohttp session
            
        Returns:
            Response data as list of dicts
        """
        url = f"{self.rest_url}/{table}"
        
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            async with session.request(
                method=method,
                url=url,
                headers=self._headers,
                json=data,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"Supabase error {response.status}: {error_text}")
                
                if response.status == 204:
                    return []
                
                return await response.json()
        finally:
            if close_session:
                await session.close()
    
    def _clean_data(self, data: Dict) -> Dict:
        """Remove None values and convert special types."""
        result = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, dict):
                result[key] = json.dumps(value) if key not in ("metadata", "raw_response", "bounding_box", "phase_distribution") else value
            else:
                result[key] = value
        return result
    
    # ==================== Sources ====================
    
    async def insert_source(
        self,
        data: Union[SourceData, Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Insert a video source.
        
        Args:
            data: SourceData or dict with source fields
            session: Optional aiohttp session
            
        Returns:
            UUID of inserted source
        """
        if isinstance(data, SourceData):
            data = asdict(data)
        
        data = self._clean_data(data)
        result = await self._request("POST", "nko_sources", data, session=session)
        return result[0]["id"]
    
    async def update_source_status(
        self,
        source_id: str,
        status: str,
        error_message: Optional[str] = None,
        frame_count: Optional[int] = None,
        nko_frame_count: Optional[int] = None,
        total_detections: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Update source processing status."""
        data = {"status": status}
        if error_message:
            data["error_message"] = error_message
        if frame_count is not None:
            data["frame_count"] = frame_count
        if nko_frame_count is not None:
            data["nko_frame_count"] = nko_frame_count
        if total_detections is not None:
            data["total_detections"] = total_detections
        if status == "completed":
            data["processed_at"] = datetime.utcnow().isoformat()
        
        await self._request(
            "PATCH",
            "nko_sources",
            data,
            params={"id": f"eq.{source_id}"},
            session=session,
        )
    
    async def get_source_by_external_id(
        self,
        external_id: str,
        source_type: str = "youtube",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[Dict]:
        """Get source by external ID (e.g., YouTube video ID)."""
        result = await self._request(
            "GET",
            "nko_sources",
            params={
                "external_id": f"eq.{external_id}",
                "source_type": f"eq.{source_type}",
            },
            session=session,
        )
        return result[0] if result else None
    
    # ==================== Frames ====================
    
    async def insert_frame(
        self,
        data: Union[FrameData, Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Insert a frame.
        
        Args:
            data: FrameData or dict with frame fields
            session: Optional aiohttp session
            
        Returns:
            UUID of inserted frame
        """
        if isinstance(data, FrameData):
            data = asdict(data)
        
        data = self._clean_data(data)
        result = await self._request("POST", "nko_frames", data, session=session)
        return result[0]["id"]
    
    async def insert_frames_batch(
        self,
        frames: List[Union[FrameData, Dict]],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[str]:
        """
        Insert multiple frames in a batch.
        
        Args:
            frames: List of FrameData or dicts
            session: Optional aiohttp session
            
        Returns:
            List of inserted frame UUIDs
        """
        data_list = []
        for frame in frames:
            if isinstance(frame, FrameData):
                frame = asdict(frame)
            data_list.append(self._clean_data(frame))
        
        result = await self._request("POST", "nko_frames", data_list, session=session)
        return [r["id"] for r in result]
    
    # ==================== Detections ====================
    
    async def insert_detection(
        self,
        data: Union[DetectionData, Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Insert a detection.
        
        Args:
            data: DetectionData or dict with detection fields
            session: Optional aiohttp session
            
        Returns:
            UUID of inserted detection
        """
        if isinstance(data, DetectionData):
            data = asdict(data)
        
        # Calculate char_count if not set
        if data.get("nko_text") and not data.get("char_count"):
            data["char_count"] = len(data["nko_text"])
        
        data = self._clean_data(data)
        result = await self._request("POST", "nko_detections", data, session=session)
        return result[0]["id"]
    
    async def insert_detections_batch(
        self,
        detections: List[Union[DetectionData, Dict]],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[str]:
        """Insert multiple detections in a batch."""
        data_list = []
        for det in detections:
            if isinstance(det, DetectionData):
                det = asdict(det)
            if det.get("nko_text") and not det.get("char_count"):
                det["char_count"] = len(det["nko_text"])
            data_list.append(self._clean_data(det))
        
        result = await self._request("POST", "nko_detections", data_list, session=session)
        return [r["id"] for r in result]
    
    # ==================== Trajectories ====================
    
    async def insert_trajectory(
        self,
        data: Union[TrajectoryData, Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Insert a trajectory.
        
        Args:
            data: TrajectoryData or dict with trajectory fields
            session: Optional aiohttp session
            
        Returns:
            UUID of inserted trajectory
        """
        if isinstance(data, TrajectoryData):
            data = asdict(data)
        
        data = self._clean_data(data)
        result = await self._request("POST", "nko_trajectories", data, session=session)
        return result[0]["id"]
    
    async def update_trajectory(
        self,
        trajectory_id: str,
        updates: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Update trajectory fields."""
        updates = self._clean_data(updates)
        await self._request(
            "PATCH",
            "nko_trajectories",
            updates,
            params={"id": f"eq.{trajectory_id}"},
            session=session,
        )
    
    # ==================== Trajectory Nodes ====================
    
    async def insert_trajectory_node(
        self,
        data: Union[TrajectoryNodeData, Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Insert a trajectory node.
        
        Args:
            data: TrajectoryNodeData or dict with node fields
            session: Optional aiohttp session
            
        Returns:
            UUID of inserted node
        """
        if isinstance(data, TrajectoryNodeData):
            data = asdict(data)
        
        data = self._clean_data(data)
        result = await self._request("POST", "nko_trajectory_nodes", data, session=session)
        return result[0]["id"]
    
    async def insert_trajectory_nodes_batch(
        self,
        nodes: List[Union[TrajectoryNodeData, Dict]],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[str]:
        """Insert multiple trajectory nodes in a batch."""
        data_list = []
        for node in nodes:
            if isinstance(node, TrajectoryNodeData):
                node = asdict(node)
            data_list.append(self._clean_data(node))
        
        result = await self._request("POST", "nko_trajectory_nodes", data_list, session=session)
        return [r["id"] for r in result]
    
    # ==================== Audio Segments ====================
    
    async def insert_audio_segment(
        self,
        data: Union[AudioSegmentData, Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        Insert an audio segment.
        
        Args:
            data: AudioSegmentData or dict with segment fields
            session: Optional aiohttp session
            
        Returns:
            UUID of inserted segment
        """
        if isinstance(data, AudioSegmentData):
            data = asdict(data)
        
        data = self._clean_data(data)
        result = await self._request("POST", "nko_audio_segments", data, session=session)
        return result[0]["id"]
    
    async def insert_audio_segments_batch(
        self,
        segments: List[Union[AudioSegmentData, Dict]],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[str]:
        """Insert multiple audio segments in a batch."""
        data_list = []
        for seg in segments:
            if isinstance(seg, AudioSegmentData):
                seg = asdict(seg)
            data_list.append(self._clean_data(seg))
        
        result = await self._request("POST", "nko_audio_segments", data_list, session=session)
        return [r["id"] for r in result]
    
    async def update_audio_segment_transcription(
        self,
        segment_id: str,
        transcription: str,
        language: str = "nko",
        confidence: float = 0.0,
        model: str = "whisper-1",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Update audio segment with transcription result."""
        data = {
            "transcription": transcription,
            "transcription_language": language,
            "transcription_confidence": confidence,
            "transcription_model": model,
            "transcribed_at": datetime.utcnow().isoformat(),
        }
        await self._request(
            "PATCH",
            "nko_audio_segments",
            data,
            params={"id": f"eq.{segment_id}"},
            session=session,
        )
    
    async def get_untranscribed_segments(
        self,
        limit: int = 100,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[Dict]:
        """Get audio segments that need transcription."""
        result = await self._request(
            "GET",
            "nko_audio_segments",
            params={
                "transcription": "is.null",
                "audio_path": "not.is.null",
                "order": "created_at.asc",
                "limit": str(limit),
            },
            session=session,
        )
        return result
    
    # ==================== Convenience Methods ====================
    
    async def store_video_analysis(
        self,
        video_id: str,
        video_url: str,
        title: str,
        frames: List[Dict],
        channel_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store complete video analysis results.
        
        Args:
            video_id: YouTube video ID
            video_url: Full video URL
            title: Video title
            frames: List of frame analysis results
            channel_name: YouTube channel name
            
        Returns:
            Dict with source_id and counts
        """
        async with aiohttp.ClientSession() as session:
            # Check if already exists
            existing = await self.get_source_by_external_id(video_id, session=session)
            if existing:
                return {
                    "source_id": existing["id"],
                    "status": "already_exists",
                    "frame_count": existing.get("frame_count", 0),
                }
            
            # Insert source
            source_id = await self.insert_source(
                SourceData(
                    source_type="youtube",
                    url=video_url,
                    external_id=video_id,
                    title=title,
                    channel_name=channel_name,
                    status="processing",
                ),
                session=session,
            )
            
            frame_count = 0
            nko_frame_count = 0
            detection_count = 0
            
            # Insert frames and detections
            for frame in frames:
                has_nko = bool(frame.get("nko_text"))
                
                frame_id = await self.insert_frame(
                    FrameData(
                        source_id=source_id,
                        frame_index=frame.get("frame_index", 0),
                        timestamp_ms=int(frame.get("timestamp", 0) * 1000),
                        has_nko=has_nko,
                        confidence=frame.get("confidence", 0.0),
                    ),
                    session=session,
                )
                frame_count += 1
                
                if has_nko:
                    nko_frame_count += 1
                    await self.insert_detection(
                        DetectionData(
                            frame_id=frame_id,
                            nko_text=frame["nko_text"],
                            latin_text=frame.get("latin_transliteration"),
                            english_text=frame.get("english_translation"),
                            confidence=frame.get("confidence", 0.0),
                            gemini_model="gemini-2.0-flash",
                        ),
                        session=session,
                    )
                    detection_count += 1
            
            # Update source status
            await self.update_source_status(
                source_id,
                status="completed",
                frame_count=frame_count,
                nko_frame_count=nko_frame_count,
                total_detections=detection_count,
                session=session,
            )
            
            return {
                "source_id": source_id,
                "status": "created",
                "frame_count": frame_count,
                "nko_frame_count": nko_frame_count,
                "detection_count": detection_count,
            }
    
    async def store_world_trajectories(
        self,
        detection_id: str,
        worlds: List[Dict],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        """
        Store world generation results as trajectories.
        
        Args:
            detection_id: UUID of the detection
            worlds: List of world generation results
            session: Optional aiohttp session
            
        Returns:
            Dict with trajectory_ids and node counts
        """
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            trajectory_ids = []
            total_nodes = 0
            
            for world in worlds:
                world_name = world.get("world_name", "unknown")
                variants = world.get("variants", [])
                
                if not variants and world.get("error"):
                    continue
                
                # Create trajectory for this world
                trajectory_id = await self.insert_trajectory(
                    TrajectoryData(
                        name=f"world_{world_name}",
                        description=f"World exploration: {world_name}",
                        trajectory_type="content_flow",
                        total_nodes=len(variants),
                        dominant_phase="exploration",
                        is_complete=True,
                        is_successful=not bool(world.get("error")),
                    ),
                    session=session,
                )
                trajectory_ids.append(trajectory_id)
                
                # Create nodes for each variant
                nodes = []
                for i, variant in enumerate(variants):
                    nodes.append(TrajectoryNodeData(
                        trajectory_id=trajectory_id,
                        detection_id=detection_id,
                        node_index=i,
                        trajectory_depth=1,
                        trajectory_sibling_order=i,
                        trajectory_phase="exploration",
                        content_preview=variant.get("nko_text", "")[:100],
                        content_type=world_name,
                        outcome="success",
                        is_terminal=(i == len(variants) - 1),
                    ))
                
                if nodes:
                    await self.insert_trajectory_nodes_batch(nodes, session=session)
                    total_nodes += len(nodes)
            
            return {
                "trajectory_ids": trajectory_ids,
                "trajectory_count": len(trajectory_ids),
                "total_nodes": total_nodes,
            }
        finally:
            if close_session:
                await session.close()


async def test_supabase_client():
    """Test the Supabase client."""
    print("=" * 60)
    print("Supabase Client Test")
    print("=" * 60)
    
    try:
        client = SupabaseClient()
        print(f"Connected to: {client.url}")
        
        # Test insert source
        source_id = await client.insert_source(
            SourceData(
                source_type="youtube",
                url="https://www.youtube.com/watch?v=test123",
                external_id="test123",
                title="Test Video",
                status="pending",
            )
        )
        print(f"Inserted source: {source_id}")
        
        # Test insert frame
        frame_id = await client.insert_frame(
            FrameData(
                source_id=source_id,
                frame_index=0,
                timestamp_ms=0,
                has_nko=True,
                confidence=0.95,
            )
        )
        print(f"Inserted frame: {frame_id}")
        
        # Test insert detection
        detection_id = await client.insert_detection(
            DetectionData(
                frame_id=frame_id,
                nko_text="ߒߞߏ",
                latin_text="N'Ko",
                english_text="I declare",
                confidence=0.95,
            )
        )
        print(f"Inserted detection: {detection_id}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_supabase_client())

