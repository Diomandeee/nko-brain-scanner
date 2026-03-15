"""
Composite Scorer for N'Ko Benchmark.

Calculates weighted overall scores across all benchmark tasks.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .translation_metrics import TranslationMetrics, TranslationScores
from .accuracy_metrics import AccuracyMetrics, AccuracyScores


@dataclass
class TaskScore:
    """Score for a single task."""
    task_name: str
    weight: float
    raw_score: float  # 0-100
    weighted_score: float  # raw_score * weight
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelScore:
    """Complete score for a model."""
    model_id: str
    provider: str
    task_scores: List[TaskScore] = field(default_factory=list)
    overall_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "overall_score": self.overall_score,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_estimate": self.total_cost_estimate,
            "task_scores": {
                ts.task_name: {
                    "weight": ts.weight,
                    "raw_score": ts.raw_score,
                    "weighted_score": ts.weighted_score,
                    "metrics": ts.metrics,
                }
                for ts in self.task_scores
            },
        }


class CompositeScorer:
    """
    Calculate composite scores across all benchmark tasks.
    
    Weights:
    - Translation: 40%
    - Script Knowledge: 15%
    - Vocabulary: 20%
    - Cultural: 10%
    - Compositional: 15%
    """
    
    DEFAULT_WEIGHTS = {
        "translation": 0.40,
        "script_knowledge": 0.15,
        "vocabulary": 0.20,
        "cultural": 0.10,
        "compositional": 0.15,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.translation_metrics = TranslationMetrics()
        self.accuracy_metrics = AccuracyMetrics()
    
    def score_translation(
        self,
        predictions: List[str],
        references: List[str],
        latency_data: Optional[List[float]] = None,
    ) -> TaskScore:
        """
        Score translation task.
        
        Uses BLEU and chrF++ for scoring.
        """
        scores = self.translation_metrics.calculate_all(predictions, references)
        
        # Combine BLEU and chrF++ (chrF++ is better for morphologically rich languages)
        # Scale to 0-100
        raw_score = (scores.bleu * 0.4 + scores.chrf * 0.6)
        
        return TaskScore(
            task_name="translation",
            weight=self.weights["translation"],
            raw_score=raw_score,
            weighted_score=raw_score * self.weights["translation"],
            metrics={
                "bleu": scores.bleu,
                "chrf": scores.chrf,
                "bleu_1": scores.bleu_1,
                "bleu_4": scores.bleu_4,
                "brevity_penalty": scores.brevity_penalty,
            },
        )
    
    def score_script_knowledge(
        self,
        predictions: List[str],
        references: List[str],
        successes: List[bool],
    ) -> TaskScore:
        """Score script knowledge task."""
        scores = self.accuracy_metrics.calculate_all(predictions, references, successes)
        
        # Script knowledge is primarily about exact/partial matching
        raw_score = (
            scores.exact_match * 50 +
            scores.partial_match * 30 +
            scores.response_rate * 20
        )
        
        return TaskScore(
            task_name="script_knowledge",
            weight=self.weights["script_knowledge"],
            raw_score=raw_score,
            weighted_score=raw_score * self.weights["script_knowledge"],
            metrics={
                "exact_match": scores.exact_match,
                "partial_match": scores.partial_match,
                "response_rate": scores.response_rate,
            },
        )
    
    def score_vocabulary(
        self,
        predictions: List[str],
        references: List[str],
        successes: List[bool],
    ) -> TaskScore:
        """Score vocabulary task."""
        scores = self.accuracy_metrics.calculate_all(predictions, references, successes)
        
        # Vocabulary focuses on content overlap and partial matching
        raw_score = (
            scores.partial_match * 40 +
            scores.content_overlap * 40 +
            scores.response_rate * 20
        )
        
        return TaskScore(
            task_name="vocabulary",
            weight=self.weights["vocabulary"],
            raw_score=raw_score,
            weighted_score=raw_score * self.weights["vocabulary"],
            metrics={
                "partial_match": scores.partial_match,
                "content_overlap": scores.content_overlap,
                "response_rate": scores.response_rate,
            },
        )
    
    def score_cultural(
        self,
        predictions: List[str],
        references: List[str],
        successes: List[bool],
    ) -> TaskScore:
        """Score cultural task."""
        scores = self.accuracy_metrics.calculate_all(predictions, references, successes)
        
        # Cultural focuses on content understanding
        raw_score = (
            scores.partial_match * 35 +
            scores.content_overlap * 45 +
            scores.response_rate * 20
        )
        
        return TaskScore(
            task_name="cultural",
            weight=self.weights["cultural"],
            raw_score=raw_score,
            weighted_score=raw_score * self.weights["cultural"],
            metrics={
                "partial_match": scores.partial_match,
                "content_overlap": scores.content_overlap,
                "response_rate": scores.response_rate,
            },
        )
    
    def score_compositional(
        self,
        predictions: List[str],
        references: List[str],
        successes: List[bool],
    ) -> TaskScore:
        """Score compositional task."""
        scores = self.accuracy_metrics.calculate_all(predictions, references, successes)
        
        # Compositional requires more precise matching
        raw_score = (
            scores.exact_match * 30 +
            scores.partial_match * 40 +
            scores.content_overlap * 20 +
            scores.response_rate * 10
        )
        
        return TaskScore(
            task_name="compositional",
            weight=self.weights["compositional"],
            raw_score=raw_score,
            weighted_score=raw_score * self.weights["compositional"],
            metrics={
                "exact_match": scores.exact_match,
                "partial_match": scores.partial_match,
                "content_overlap": scores.content_overlap,
                "response_rate": scores.response_rate,
            },
        )
    
    def calculate_overall_score(
        self,
        task_scores: List[TaskScore],
    ) -> float:
        """Calculate weighted overall score."""
        return sum(ts.weighted_score for ts in task_scores)
    
    def create_model_score(
        self,
        model_id: str,
        provider: str,
        task_scores: List[TaskScore],
        avg_latency_ms: float = 0.0,
        total_cost: float = 0.0,
    ) -> ModelScore:
        """Create complete model score."""
        overall = self.calculate_overall_score(task_scores)
        
        return ModelScore(
            model_id=model_id,
            provider=provider,
            task_scores=task_scores,
            overall_score=overall,
            avg_latency_ms=avg_latency_ms,
            total_cost_estimate=total_cost,
        )
    
    def rank_models(
        self,
        model_scores: List[ModelScore],
    ) -> List[ModelScore]:
        """Rank models by overall score (highest first)."""
        return sorted(model_scores, key=lambda m: m.overall_score, reverse=True)
    
    def get_recommendation(
        self,
        model_scores: List[ModelScore],
    ) -> Dict[str, Any]:
        """Get recommendation for best model."""
        if not model_scores:
            return {"recommendation": None, "reason": "No models evaluated"}
        
        ranked = self.rank_models(model_scores)
        best = ranked[0]
        
        # Consider cost-effectiveness
        cost_effective = min(
            model_scores,
            key=lambda m: m.total_cost_estimate / max(m.overall_score, 1)
        )
        
        return {
            "best_overall": {
                "model_id": best.model_id,
                "provider": best.provider,
                "overall_score": best.overall_score,
                "reason": f"Highest overall score ({best.overall_score:.1f})",
            },
            "best_value": {
                "model_id": cost_effective.model_id,
                "provider": cost_effective.provider,
                "overall_score": cost_effective.overall_score,
                "cost": cost_effective.total_cost_estimate,
                "reason": "Best score-to-cost ratio",
            },
            "recommendation": best.model_id,
            "all_rankings": [
                {
                    "rank": i + 1,
                    "model_id": m.model_id,
                    "provider": m.provider,
                    "overall_score": m.overall_score,
                }
                for i, m in enumerate(ranked)
            ],
        }

