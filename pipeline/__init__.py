"""
N'Ko Training Pipeline

This package provides:
- orchestrator: Coordinates cc-music-pipeline and RAG++ for video processing
- rag_client: Client for RAG++ Cloud Run API
- scheduler: Streaming scheduler for continuous training
- pipeline: Data transformation pipeline
"""

from .rag_client import RagClient, TrainingTurn, RelatedTurn, RagClientError
from .orchestrator import TrainingOrchestrator, OrchestratorConfig, ProcessingResult

__all__ = [
    "RagClient",
    "TrainingTurn", 
    "RelatedTurn",
    "RagClientError",
    "TrainingOrchestrator",
    "OrchestratorConfig",
    "ProcessingResult",
]

