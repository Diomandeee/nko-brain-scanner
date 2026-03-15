"""
RAG++ API Client for learnnko

This module provides a client for interacting with the RAG++ Cloud Run service.
It handles:
- Semantic search for related content
- Ingesting training data as memory turns
- Retrieving trajectory context
- Health checks

Usage:
    from training.rag_client import RagClient
    
    client = RagClient("https://rag-plusplus-274020562532.us-central1.run.app")
    
    # Search for related content
    results = await client.search("ߒߞߏ", limit=5)
    
    # Ingest training data
    turn_id = await client.ingest(TrainingTurn(...))
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx


@dataclass
class TrajectoryCoords:
    """5D trajectory coordinates from RAG++."""
    depth: float = 0.0
    sibling_order: float = 0.0
    homogeneity: float = 1.0
    temporal: float = 0.0
    complexity: float = 1.0
    phase: Optional[str] = None


@dataclass
class RelatedTurn:
    """A related turn from RAG++ semantic search."""
    turn_id: str
    content: str
    role: str
    similarity: float
    conversation_id: str
    trajectory: TrajectoryCoords
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingTurn:
    """Training data to ingest into RAG++."""
    prompt_id: str
    prompt_text: str
    response_text: str
    session_id: str
    project_id: str
    project_name: Optional[str] = None
    working_directory: Optional[str] = None
    turn_index: int = 0
    source: str = "nko_training"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextTurn:
    """A turn from trajectory context."""
    id: str
    content: str
    role: str
    created_at: str
    trajectory: TrajectoryCoords
    metadata: Dict[str, Any] = field(default_factory=dict)


class RagClientError(Exception):
    """Error from RAG++ API."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RagClient:
    """
    Client for RAG++ Cloud Run API.
    
    Provides methods for:
    - Semantic search
    - Training data ingestion
    - Trajectory context retrieval
    - Health checks
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        project_id: Optional[str] = None,
        project_name: str = "NKo-Training",
    ):
        """
        Initialize RAG++ client.
        
        Args:
            base_url: RAG++ service URL. Defaults to RAG_SERVICE_URL env var.
            timeout: HTTP request timeout in seconds.
            project_id: Default project ID for operations.
            project_name: Default project name for ingestion.
        """
        self.base_url = (
            base_url or 
            os.environ.get("RAG_SERVICE_URL") or
            os.environ.get("RAG_PLUSPLUS_URL") or
            "https://rag-plusplus-274020562532.us-central1.run.app"
        )
        self.timeout = timeout
        self.project_id = project_id
        self.project_name = project_name
        self._client: Optional[httpx.AsyncClient] = None
        
        # Session tracking
        self._session_id: Optional[str] = None
        self._turn_index: int = 0
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def health(self) -> Dict[str, Any]:
        """
        Check RAG++ service health.
        
        Returns:
            Health status including supabase, ingester, trainer status.
        """
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/api/rag/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise RagClientError(f"Health check failed: {e}", getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        project_id: Optional[str] = None,
    ) -> List[RelatedTurn]:
        """
        Semantic search for related content in RAG++.
        
        Args:
            query: Search query (e.g., N'Ko text).
            limit: Maximum number of results.
            project_id: Filter by project ID.
        
        Returns:
            List of related turns with similarity scores.
        """
        client = await self._get_client()
        
        params = {"query": query, "limit": limit}
        if project_id or self.project_id:
            params["project_id"] = project_id or self.project_id
        
        try:
            response = await client.get(
                f"{self.base_url}/api/rag/search",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                traj_data = item.get("trajectory", {})
                trajectory = TrajectoryCoords(
                    depth=traj_data.get("depth", 0.0),
                    sibling_order=traj_data.get("sibling_order", 0.0),
                    homogeneity=traj_data.get("homogeneity", 1.0),
                    temporal=traj_data.get("temporal", 0.0),
                    complexity=traj_data.get("complexity", 1.0),
                    phase=traj_data.get("phase"),
                )
                results.append(RelatedTurn(
                    turn_id=item.get("turn_id", ""),
                    content=item.get("content", ""),
                    role=item.get("role", "assistant"),
                    similarity=item.get("similarity", 0.0),
                    conversation_id=item.get("conversation_id", ""),
                    trajectory=trajectory,
                    metadata=item.get("metadata", {}),
                ))
            
            return results
            
        except httpx.HTTPError as e:
            raise RagClientError(f"Search failed: {e}", getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    async def ingest(self, turn: TrainingTurn) -> str:
        """
        Ingest training data into RAG++.
        
        Args:
            turn: Training turn to ingest.
        
        Returns:
            The memory_turn_id from RAG++.
        """
        client = await self._get_client()
        
        payload = {
            "prompt_id": turn.prompt_id,
            "prompt_text": turn.prompt_text,
            "response_text": turn.response_text,
            "session_id": turn.session_id,
            "project_id": turn.project_id,
            "project_name": turn.project_name or self.project_name,
            "working_directory": turn.working_directory or "/learnnko/training",
            "turn_index": turn.turn_index,
            "source": turn.source,
            "metadata": turn.metadata,
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/api/rag/ingest",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("memory_turn_id", "")
            
        except httpx.HTTPError as e:
            raise RagClientError(f"Ingest failed: {e}", getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    async def get_context(
        self,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[ContextTurn]:
        """
        Get trajectory context for a project.
        
        Args:
            project_id: Project ID to get context for.
            session_id: Optional session ID filter.
            limit: Maximum number of turns.
        
        Returns:
            List of context turns with trajectory coordinates.
        """
        client = await self._get_client()
        
        pid = project_id or self.project_id
        if not pid:
            raise RagClientError("project_id required for get_context")
        
        params = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        
        try:
            response = await client.get(
                f"{self.base_url}/api/rag/context/{pid}",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            
            turns = []
            for item in data.get("turns", []):
                traj_data = item.get("trajectory", {})
                trajectory = TrajectoryCoords(
                    depth=traj_data.get("depth", 0.0),
                    sibling_order=traj_data.get("sibling_order", 0.0),
                    homogeneity=traj_data.get("homogeneity", 1.0),
                    temporal=traj_data.get("temporal", 0.0),
                    complexity=traj_data.get("complexity", 1.0),
                    phase=traj_data.get("phase"),
                )
                turns.append(ContextTurn(
                    id=item.get("id", ""),
                    content=item.get("content", ""),
                    role=item.get("role", "assistant"),
                    created_at=item.get("created_at", ""),
                    trajectory=trajectory,
                    metadata=item.get("metadata", {}),
                ))
            
            return turns
            
        except httpx.HTTPError as e:
            raise RagClientError(f"Get context failed: {e}", getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    async def trigger_training(
        self,
        mode: str = "batch_turns",
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """
        Trigger CognitiveTwin training on recent turns.
        
        Args:
            mode: Training mode ("batch_turns" or "incremental").
            project_id: Filter by project.
            session_id: Filter by session.
            limit: Number of turns to train on.
        
        Returns:
            Training result with triggered status and sample count.
        """
        client = await self._get_client()
        
        payload = {
            "mode": mode,
            "project_id": project_id or self.project_id,
            "session_id": session_id,
            "limit": limit,
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/api/rag/train",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise RagClientError(f"Training trigger failed: {e}", getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    async def end_session(
        self,
        session_id: str,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Signal end of session, triggering batch training.
        
        Args:
            session_id: Session ID that ended.
            project_id: Project ID.
        
        Returns:
            Training result.
        """
        client = await self._get_client()
        
        payload = {
            "session_id": session_id,
            "project_id": project_id or self.project_id,
            "ended_at": datetime.now().isoformat(),
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/api/rag/session-end",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise RagClientError(f"Session end failed: {e}", getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None)
    
    # Session helpers
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new training session."""
        self._session_id = session_id or f"nko-training-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._turn_index = 0
        return self._session_id
    
    def next_turn_index(self) -> int:
        """Get and increment turn index."""
        idx = self._turn_index
        self._turn_index += 1
        return idx
    
    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._session_id


# Convenience functions

_default_client: Optional[RagClient] = None


def get_client() -> RagClient:
    """Get the default RAG++ client."""
    global _default_client
    if _default_client is None:
        _default_client = RagClient()
    return _default_client


async def search(query: str, limit: int = 5) -> List[RelatedTurn]:
    """Search RAG++ for related content."""
    return await get_client().search(query, limit)


async def ingest(turn: TrainingTurn) -> str:
    """Ingest training data to RAG++."""
    return await get_client().ingest(turn)

