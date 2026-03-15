"""
Training Orchestrator for learnnko

This module coordinates:
- cc-music-pipeline for video analysis (frame extraction, OCR)
- RAG++ for context retrieval and ingestion
- Prompt loading for world generation
- The overall training loop

Usage:
    from training.orchestrator import TrainingOrchestrator, OrchestratorConfig
    
    config = OrchestratorConfig.from_yaml("training/scheduler/config.yaml")
    orchestrator = TrainingOrchestrator(config)
    
    result = await orchestrator.process_video("https://youtube.com/watch?v=...")
"""

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx
import yaml
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rag_client import RagClient, RagClientError, RelatedTurn, TrainingTurn

# Try to import prompt loader
try:
    from prompts.loader import PromptLoader, PromptDefinition
except ImportError:
    PromptLoader = None
    PromptDefinition = None

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the Training Orchestrator."""
    
    # cc-music-pipeline backend
    analyzer_url: str = "https://cc-music-pipeline-513507963446.us-central1.run.app"
    analyzer_submit_endpoint: str = "/api/batch/submit"
    analyzer_status_endpoint: str = "/api/batch/status"
    analyzer_results_endpoint: str = "/api/batch/results"
    
    # RAG++ integration
    rag_enabled: bool = True
    rag_url: str = "https://rag-plusplus-274020562532.us-central1.run.app"
    rag_project_name: str = "NKo-Training"
    rag_ingest_training_data: bool = True
    rag_context_limit: int = 5
    
    # Prompts
    prompts_source: str = "yaml"  # "yaml" or "supabase"
    prompts_yaml_dir: str = "./prompts/nko"
    
    # Processing options
    max_frames_per_video: int = 50
    max_cost_per_video_usd: float = 0.50
    enabled_worlds: List[str] = field(default_factory=lambda: [
        "WorldEveryday",
        "WorldFormal", 
        "WorldStorytelling",
        "WorldProverbs",
        "WorldEducational",
    ])
    
    # Timeouts
    timeout_seconds: float = 300.0  # 5 minutes per video
    poll_interval_seconds: float = 5.0
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OrchestratorConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Analyzer settings
        if "analyzer" in data:
            a = data["analyzer"]
            config.analyzer_url = a.get("url", config.analyzer_url)
        elif "backend_api" in data:
            a = data["backend_api"]
            config.analyzer_url = a.get("url", config.analyzer_url)
            config.analyzer_submit_endpoint = a.get("submit_endpoint", config.analyzer_submit_endpoint)
            config.analyzer_status_endpoint = a.get("status_endpoint", config.analyzer_status_endpoint)
        
        # RAG settings
        if "rag" in data:
            r = data["rag"]
            config.rag_enabled = r.get("enabled", config.rag_enabled)
            config.rag_url = r.get("url", config.rag_url)
            config.rag_project_name = r.get("project_name", config.rag_project_name)
            config.rag_ingest_training_data = r.get("ingest_training_data", config.rag_ingest_training_data)
            config.rag_context_limit = r.get("context_limit", config.rag_context_limit)
        
        # Prompts settings
        if "prompts" in data:
            p = data["prompts"]
            config.prompts_source = p.get("source", config.prompts_source)
            config.prompts_yaml_dir = p.get("yaml_dir", config.prompts_yaml_dir)
        
        # Processing settings
        if "cost_per_video" in data:
            c = data["cost_per_video"]
            config.max_frames_per_video = c.get("frames_per_video_estimate", config.max_frames_per_video)
        
        return config
    
    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Load config from environment variables."""
        config = cls()
        
        config.analyzer_url = os.environ.get("ANALYZER_URL", config.analyzer_url)
        config.rag_enabled = os.environ.get("RAG_ENABLED", "true").lower() == "true"
        config.rag_url = os.environ.get("RAG_SERVICE_URL", config.rag_url)
        config.rag_project_name = os.environ.get("RAG_PROJECT_NAME", config.rag_project_name)
        
        return config


@dataclass
class NkoDetection:
    """A detected N'Ko text from video analysis."""
    text: str
    latin_transliteration: str
    english_translation: str
    confidence: float
    frame_timestamp_ms: int
    bounding_box: Optional[List[float]] = None
    source_url: Optional[str] = None
    frame_index: int = 0


@dataclass 
class WorldVariant:
    """A generated world variant for N'Ko text."""
    world_type: str
    content: Dict[str, Any]
    cost_usd: float = 0.0
    prompt_id: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single video."""
    video_url: str
    job_id: str
    status: str
    detections: List[NkoDetection]
    world_variants: List[WorldVariant]
    total_cost_usd: float
    processing_time_seconds: float
    error: Optional[str] = None
    rag_turn_ids: List[str] = field(default_factory=list)


class OrchestratorError(Exception):
    """Error from orchestrator."""
    pass


class TrainingOrchestrator:
    """
    Main orchestration module that coordinates:
    - cc-music-pipeline for video analysis
    - RAG++ for context retrieval and ingestion
    - Prompt loading for world generation
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.rag_client: Optional[RagClient] = None
        self.prompt_loader = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        
        # Initialize RAG client if enabled
        if config.rag_enabled:
            self.rag_client = RagClient(
                base_url=config.rag_url,
                project_name=config.rag_project_name,
            )
        
        # Initialize prompt loader if available
        if PromptLoader is not None:
            try:
                self.prompt_loader = PromptLoader(yaml_dir=config.prompts_yaml_dir)
                if config.prompts_source == "supabase":
                    supabase_url = os.environ.get("SUPABASE_URL")
                    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
                    if supabase_url and supabase_key:
                        self.prompt_loader.with_supabase(supabase_url, supabase_key)
            except Exception as e:
                logger.warning(f"Failed to initialize PromptLoader: {e}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._http_client
    
    async def close(self):
        """Close all clients."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        if self.rag_client:
            await self.rag_client.close()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new training session."""
        self._session_id = session_id or f"nko-training-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.rag_client:
            self.rag_client.start_session(self._session_id)
        return self._session_id
    
    async def process_video(self, video_url: str) -> ProcessingResult:
        """
        Process a single video through the full pipeline.
        
        Steps:
        1. Submit to cc-music-pipeline for frame analysis
        2. Wait for analysis to complete
        3. For each detection, query RAG++ for context
        4. Generate world variants with context
        5. Ingest results to RAG++
        
        Args:
            video_url: YouTube video URL to process.
        
        Returns:
            ProcessingResult with detections and variants.
        """
        start_time = datetime.now()
        
        # Ensure we have a session
        if not self._session_id:
            self.start_session()
        
        try:
            # Step 1: Submit to analyzer
            job_id = await self._submit_analysis(video_url)
            logger.info(f"Submitted video {video_url} as job {job_id}")
            
            # Step 2: Wait for completion
            analysis_result = await self._wait_for_analysis(job_id)
            
            if analysis_result.get("status") == "FAILED":
                return ProcessingResult(
                    video_url=video_url,
                    job_id=job_id,
                    status="FAILED",
                    detections=[],
                    world_variants=[],
                    total_cost_usd=analysis_result.get("total_cost_usd", 0.0),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    error=analysis_result.get("error", "Analysis failed"),
                )
            
            # Step 3: Extract detections
            detections = self._extract_detections(analysis_result, video_url)
            logger.info(f"Found {len(detections)} N'Ko detections")
            
            # Steps 4-5: Generate worlds and ingest
            world_variants = []
            rag_turn_ids = []
            
            for detection in detections:
                # Get RAG++ context
                context = await self._get_rag_context(detection.text)
                
                # Generate world variants (if we have results with worlds)
                variants = self._extract_world_variants(analysis_result, detection)
                world_variants.extend(variants)
                
                # Ingest to RAG++
                if self.config.rag_ingest_training_data and self.rag_client:
                    turn_ids = await self._ingest_to_rag(detection, variants, context)
                    rag_turn_ids.extend(turn_ids)
            
            total_cost = analysis_result.get("total_cost_usd", 0.0)
            
            return ProcessingResult(
                video_url=video_url,
                job_id=job_id,
                status="COMPLETED",
                detections=detections,
                world_variants=world_variants,
                total_cost_usd=total_cost,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                rag_turn_ids=rag_turn_ids,
            )
            
        except Exception as e:
            logger.error(f"Error processing video {video_url}: {e}")
            return ProcessingResult(
                video_url=video_url,
                job_id="",
                status="ERROR",
                detections=[],
                world_variants=[],
                total_cost_usd=0.0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
    
    async def _submit_analysis(self, video_url: str) -> str:
        """Submit video to cc-music-pipeline for analysis."""
        client = await self._get_client()
        
        payload = {
            "video_url": video_url,
            "max_frames": self.config.max_frames_per_video,
            "max_cost_usd": self.config.max_cost_per_video_usd,
            "schedule_mode": "CostOptimized",
            "enabled_worlds": self.config.enabled_worlds,
        }
        
        try:
            response = await client.post(
                f"{self.config.analyzer_url}{self.config.analyzer_submit_endpoint}",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            job_id = data.get("job_id")
            if not job_id:
                raise OrchestratorError(f"No job_id in response: {data}")
            
            return job_id
            
        except httpx.HTTPError as e:
            raise OrchestratorError(f"Failed to submit analysis: {e}")
    
    async def _wait_for_analysis(self, job_id: str) -> Dict[str, Any]:
        """Poll for analysis completion."""
        client = await self._get_client()
        
        start = datetime.now()
        timeout = self.config.timeout_seconds
        
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > timeout:
                raise OrchestratorError(f"Analysis timeout after {elapsed:.0f}s")
            
            try:
                response = await client.get(
                    f"{self.config.analyzer_url}{self.config.analyzer_status_endpoint}/{job_id}"
                )
                response.raise_for_status()
                data = response.json()
                
                status = data.get("status", "UNKNOWN")
                
                if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    # Get full results
                    results_response = await client.get(
                        f"{self.config.analyzer_url}{self.config.analyzer_results_endpoint}/{job_id}"
                    )
                    if results_response.status_code == 200:
                        return results_response.json()
                    return data
                
                logger.debug(f"Job {job_id} status: {status}")
                
            except httpx.HTTPError as e:
                logger.warning(f"Error checking status: {e}")
            
            await asyncio.sleep(self.config.poll_interval_seconds)
    
    def _extract_detections(
        self,
        analysis_result: Dict[str, Any],
        source_url: str,
    ) -> List[NkoDetection]:
        """Extract N'Ko detections from analysis result."""
        detections = []
        
        # Handle different result formats
        results = analysis_result.get("results", [])
        if isinstance(results, dict):
            results = results.get("frames", [])
        
        for idx, frame in enumerate(results):
            frame_detections = frame.get("detections", []) or frame.get("nko_detections", [])
            timestamp = frame.get("timestamp_ms", 0) or frame.get("timestamp", 0)
            
            for det in frame_detections:
                detections.append(NkoDetection(
                    text=det.get("text", "") or det.get("nko_text", ""),
                    latin_transliteration=det.get("latin_transliteration", "") or det.get("latin_text", ""),
                    english_translation=det.get("english_translation", "") or det.get("english", ""),
                    confidence=det.get("confidence", 0.0),
                    frame_timestamp_ms=timestamp,
                    bounding_box=det.get("bounding_box"),
                    source_url=source_url,
                    frame_index=idx,
                ))
        
        return detections
    
    def _extract_world_variants(
        self,
        analysis_result: Dict[str, Any],
        detection: NkoDetection,
    ) -> List[WorldVariant]:
        """Extract world variants for a detection."""
        variants = []
        
        # World variants might be in results
        results = analysis_result.get("results", {})
        worlds = results.get("world_variants", []) if isinstance(results, dict) else []
        
        for world in worlds:
            # Match by text if available
            if world.get("source_text") == detection.text:
                variants.append(WorldVariant(
                    world_type=world.get("world_type", "unknown"),
                    content=world.get("content", {}),
                    cost_usd=world.get("cost_usd", 0.0),
                    prompt_id=world.get("prompt_id"),
                ))
        
        return variants
    
    async def _get_rag_context(self, nko_text: str) -> List[RelatedTurn]:
        """Get related context from RAG++."""
        if not self.rag_client:
            return []
        
        try:
            return await self.rag_client.search(
                nko_text,
                limit=self.config.rag_context_limit,
            )
        except RagClientError as e:
            logger.warning(f"RAG++ search failed: {e}")
            return []
    
    async def _ingest_to_rag(
        self,
        detection: NkoDetection,
        variants: List[WorldVariant],
        context: List[RelatedTurn],
    ) -> List[str]:
        """Ingest detection and variants to RAG++."""
        if not self.rag_client:
            return []
        
        turn_ids = []
        
        # Create project ID from session
        project_id = f"nko-{self._session_id}" if self._session_id else f"nko-{uuid.uuid4().hex[:8]}"
        
        # Ingest the main detection
        turn = TrainingTurn(
            prompt_id=f"detection-{detection.frame_index}",
            prompt_text=f"Analyze N'Ko text: {detection.text}",
            response_text=f"Text: {detection.text}\nLatin: {detection.latin_transliteration}\nEnglish: {detection.english_translation}",
            session_id=self._session_id or "unknown",
            project_id=project_id,
            project_name=self.config.rag_project_name,
            turn_index=self.rag_client.next_turn_index() if self.rag_client else 0,
            source="nko_video_analysis",
            metadata={
                "confidence": detection.confidence,
                "frame_timestamp_ms": detection.frame_timestamp_ms,
                "source_url": detection.source_url,
                "related_context_count": len(context),
            },
        )
        
        try:
            turn_id = await self.rag_client.ingest(turn)
            turn_ids.append(turn_id)
        except RagClientError as e:
            logger.warning(f"Failed to ingest detection: {e}")
        
        # Ingest each world variant
        for variant in variants:
            variant_turn = TrainingTurn(
                prompt_id=f"world-{variant.world_type}-{detection.frame_index}",
                prompt_text=f"Generate {variant.world_type} variant for: {detection.text}",
                response_text=str(variant.content),
                session_id=self._session_id or "unknown",
                project_id=project_id,
                project_name=self.config.rag_project_name,
                turn_index=self.rag_client.next_turn_index() if self.rag_client else 0,
                source=f"nko_world_{variant.world_type.lower()}",
                metadata={
                    "world_type": variant.world_type,
                    "cost_usd": variant.cost_usd,
                    "source_nko_text": detection.text,
                },
            )
            
            try:
                turn_id = await self.rag_client.ingest(variant_turn)
                turn_ids.append(turn_id)
            except RagClientError as e:
                logger.warning(f"Failed to ingest world variant: {e}")
        
        return turn_ids
    
    async def end_session(self) -> Dict[str, Any]:
        """End the current session and trigger RAG++ training."""
        if not self.rag_client or not self._session_id:
            return {"status": "no_session"}
        
        try:
            result = await self.rag_client.end_session(
                session_id=self._session_id,
            )
            self._session_id = None
            return result
        except RagClientError as e:
            logger.error(f"Failed to end session: {e}")
            return {"status": "error", "error": str(e)}


# Convenience function
async def process_video(video_url: str, config: Optional[OrchestratorConfig] = None) -> ProcessingResult:
    """Process a single video with default orchestrator."""
    config = config or OrchestratorConfig.from_env()
    orchestrator = TrainingOrchestrator(config)
    try:
        return await orchestrator.process_video(video_url)
    finally:
        await orchestrator.close()

