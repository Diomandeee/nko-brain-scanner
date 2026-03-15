"""
Semantic Kernel Python Wrapper for LearnN'Ko Pipeline.

This module provides a Python-native wrapper around the Rust cc-semantic-language
crate, with fallback implementations when the Rust extension is not available.

The wrapper integrates with the existing video extraction and Supabase pipeline,
enabling operator-based vocabulary compilation and lifecycle management.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

# Schema version must match Rust crate
SCHEMA_VERSION = "1.0.0"
PROBE_CONFIG_VERSION = "1.0.0"
REALIZATION_RULESET_VERSION = "1.0.0"


class SemanticOperator(Enum):
    """The 7-operator semantic alphabet."""
    STABILIZE = 0
    SHIFT = 1
    SCALE = 2
    INVERT = 3
    BIND = 4
    REPEAT = 5
    CLOSE = 6


class ContextDomain(Enum):
    """Domain from which an observation originated."""
    VIDEO = "video"
    DICTIONARY = "dictionary"
    FORUM = "forum"
    SYNTHETIC = "synthetic"
    MANUAL = "manual"


class GenerationType(Enum):
    """Type of generation context for 5-World expansion."""
    EVERYDAY = "everyday"
    FORMAL = "formal"
    STORYTELLING = "storytelling"
    PROVERBS = "proverbs"
    EDUCATIONAL = "educational"
    NOT_APPLICABLE = "not_applicable"


class InvarianceFailureMode(Enum):
    """Categorizes why invariance scoring failed."""
    INSUFFICIENT_COVERAGE = auto()
    HIGH_VARIANCE = auto()
    INCONSISTENT_DIRECTION = auto()
    CURVATURE_NOISE = auto()
    CONTEXT_COLLAPSE = auto()
    LOW_CONTEXT_ENTROPY = auto()
    PROBE_CONFIG_MISMATCH = auto()
    INSUFFICIENT_LENGTH = auto()


@dataclass
class TraceStats:
    """Reduced statistics from ΔZ probe observations."""
    n: int
    mean_norm: float
    mean_direction: List[float]
    directional_concentration: float
    curvature_consistency: float
    context_entropy: float
    probe_config_hash: int
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "meanNorm": self.mean_norm,
            "meanDirection": self.mean_direction,
            "directionalConcentration": self.directional_concentration,
            "curvatureConsistency": self.curvature_consistency,
            "contextEntropy": self.context_entropy,
            "probeConfigHash": self.probe_config_hash,
            "schemaVersion": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceStats":
        return cls(
            n=data["n"],
            mean_norm=data.get("meanNorm", data.get("mean_norm", 0.0)),
            mean_direction=data.get("meanDirection", data.get("mean_direction", [])),
            directional_concentration=data.get("directionalConcentration", data.get("directional_concentration", 0.0)),
            curvature_consistency=data.get("curvatureConsistency", data.get("curvature_consistency", 0.0)),
            context_entropy=data.get("contextEntropy", data.get("context_entropy", 0.0)),
            probe_config_hash=data.get("probeConfigHash", data.get("probe_config_hash", 0)),
            schema_version=data.get("schemaVersion", data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class ContextFeatures:
    """Named, typed context dimensions."""
    domain: ContextDomain
    generation_type: GenerationType
    speaker_confidence: float
    timestamp_seconds: Optional[float] = None
    source_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_video(cls, confidence: float, timestamp: float, source_id: str) -> "ContextFeatures":
        return cls(
            domain=ContextDomain.VIDEO,
            generation_type=GenerationType.NOT_APPLICABLE,
            speaker_confidence=confidence,
            timestamp_seconds=timestamp,
            source_id=source_id,
        )

    @classmethod
    def from_dictionary(cls, source_id: Optional[str] = None) -> "ContextFeatures":
        return cls(
            domain=ContextDomain.DICTIONARY,
            generation_type=GenerationType.NOT_APPLICABLE,
            speaker_confidence=1.0,
            source_id=source_id,
        )

    @classmethod
    def from_synthetic(cls, generation_type: GenerationType) -> "ContextFeatures":
        return cls(
            domain=ContextDomain.SYNTHETIC,
            generation_type=generation_type,
            speaker_confidence=0.9,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "generationType": self.generation_type.value,
            "speakerConfidence": self.speaker_confidence,
            "timestampSeconds": self.timestamp_seconds,
            "sourceId": self.source_id,
            "schemaVersion": self.schema_version,
        }


@dataclass
class CompiledForm:
    """The canonical output of morphological compilation."""
    surface_string: str
    operator_codes: List[int]
    signature: int
    canonicalization_hash: int
    schema_version: str = SCHEMA_VERSION

    @property
    def operator_sequence(self) -> List[SemanticOperator]:
        return [SemanticOperator(code) for code in self.operator_codes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surfaceString": self.surface_string,
            "operatorCodes": self.operator_codes,
            "signature": self.signature,
            "canonicalizationHash": self.canonicalization_hash,
            "schemaVersion": self.schema_version,
        }


@dataclass
class InvarianceResult:
    """Result of invariance scoring."""
    passed: bool
    score: float
    failure_mode: Optional[InvarianceFailureMode] = None
    evidence_summary: str = ""
    probe_config_hash: int = 0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "failureMode": self.failure_mode.name if self.failure_mode else None,
            "evidenceSummary": self.evidence_summary,
            "probeConfigHash": self.probe_config_hash,
            "schemaVersion": self.schema_version,
        }


class SemanticKernelPython:
    """
    Python-native implementation of the semantic kernel.
    
    Used when the Rust extension is not available. Provides the same
    interface but with simplified operator assignment logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self._event_log: List[Dict[str, Any]] = []
        self._ledger: Dict[int, Dict[str, Any]] = {}
        logger.info("Initialized Python-native SemanticKernel")

    def compile(self, text: str, confidence: float) -> CompiledForm:
        """Compiles N'Ko text into a canonical form."""
        if confidence < self.min_confidence:
            raise ValueError(f"Confidence {confidence} below threshold {self.min_confidence}")

        # Simple operator assignment
        operators = [SemanticOperator.STABILIZE.value]
        
        # Length-based scaling
        if len(text) > 3:
            operators.append(SemanticOperator.SCALE.value)
        
        # Check for N'Ko characters
        has_nko = any('\u07C0' <= c <= '\u07FF' for c in text)
        if has_nko and len(text) > 1:
            operators.append(SemanticOperator.BIND.value)
        
        # End with CLOSE
        operators.append(SemanticOperator.CLOSE.value)

        # Compute signature
        sig_input = f"ops:{','.join(map(str, operators))}|surf:{text.strip()}"
        signature = int(hashlib.sha256(sig_input.encode()).hexdigest()[:16], 16)

        # Canonicalization hash
        canon_input = f"{operators}:{REALIZATION_RULESET_VERSION}"
        canonicalization_hash = int(hashlib.sha256(canon_input.encode()).hexdigest()[:16], 16)

        return CompiledForm(
            surface_string=text,
            operator_codes=operators,
            signature=signature,
            canonicalization_hash=canonicalization_hash,
        )

    def score_invariance(
        self,
        trace_stats: TraceStats,
        config: Optional[Dict[str, Any]] = None,
    ) -> InvarianceResult:
        """Scores invariance for the given trace statistics."""
        cfg = config or {}
        min_observations = cfg.get("min_observations", 10)
        min_concentration = cfg.get("min_directional_concentration", 0.5)
        min_curvature = cfg.get("min_curvature_consistency", 0.4)
        min_entropy = cfg.get("min_context_entropy", 0.5)

        # Check coverage
        if trace_stats.n < min_observations:
            return InvarianceResult(
                passed=False,
                score=trace_stats.n / min_observations,
                failure_mode=InvarianceFailureMode.INSUFFICIENT_COVERAGE,
                evidence_summary=f"Insufficient coverage: {trace_stats.n}/{min_observations}",
                probe_config_hash=trace_stats.probe_config_hash,
            )

        # Check directional concentration
        if trace_stats.directional_concentration < min_concentration:
            return InvarianceResult(
                passed=False,
                score=0.5 * (trace_stats.directional_concentration / min_concentration),
                failure_mode=InvarianceFailureMode.INCONSISTENT_DIRECTION,
                evidence_summary=f"Low concentration: {trace_stats.directional_concentration:.2f}",
                probe_config_hash=trace_stats.probe_config_hash,
            )

        # Check curvature
        if trace_stats.curvature_consistency < min_curvature:
            return InvarianceResult(
                passed=False,
                score=0.5 * (trace_stats.curvature_consistency / min_curvature),
                failure_mode=InvarianceFailureMode.CURVATURE_NOISE,
                evidence_summary=f"Curvature noise: {trace_stats.curvature_consistency:.2f}",
                probe_config_hash=trace_stats.probe_config_hash,
            )

        # Check context entropy
        if trace_stats.context_entropy < min_entropy:
            return InvarianceResult(
                passed=False,
                score=0.5 * (trace_stats.context_entropy / min_entropy),
                failure_mode=InvarianceFailureMode.LOW_CONTEXT_ENTROPY,
                evidence_summary=f"Low entropy: {trace_stats.context_entropy:.2f}",
                probe_config_hash=trace_stats.probe_config_hash,
            )

        # Passed
        score = min(1.0, 0.5 + 0.1 * (trace_stats.n / min_observations - 1))
        return InvarianceResult(
            passed=True,
            score=score,
            evidence_summary=f"Passed: n={trace_stats.n}, conc={trace_stats.directional_concentration:.2f}",
            probe_config_hash=trace_stats.probe_config_hash,
        )

    def record_observation(
        self,
        signature: int,
        trace_stats: TraceStats,
        context_features: ContextFeatures,
    ) -> None:
        """Records a trace observation for a word."""
        event = {
            "type": "trace_observed",
            "signature": signature,
            "traceStats": trace_stats.to_dict(),
            "contextFeatures": context_features.to_dict(),
        }
        self._event_log.append(event)

        # Update ledger
        if signature in self._ledger:
            self._ledger[signature]["observation_count"] += trace_stats.n
        else:
            self._ledger[signature] = {
                "signature": signature,
                "stage": "proto",
                "observation_count": trace_stats.n,
            }

    @property
    def ledger_size(self) -> int:
        return len(self._ledger)

    @property
    def event_count(self) -> int:
        return len(self._event_log)


def get_semantic_kernel(config: Optional[Dict[str, Any]] = None) -> SemanticKernelPython:
    """
    Factory function to get a semantic kernel instance.
    
    Attempts to use the Rust extension if available, falls back to Python.
    """
    try:
        from cc_semantic_language import SemanticKernel as RustKernel
        config_json = json.dumps(config) if config else None
        return RustKernel(config_json)
    except ImportError:
        logger.warning("Rust cc-semantic-language not available, using Python fallback")
        return SemanticKernelPython(config)


# Convenience function for pipeline integration
def compile_nko_detection(
    text: str,
    confidence: float,
    video_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compiles an N'Ko detection from video extraction.
    
    Returns a dictionary suitable for Supabase insertion, or None if
    compilation fails (e.g., low confidence).
    
    Args:
        text: The detected N'Ko text
        confidence: OCR confidence score
        video_id: Optional video source ID
        timestamp: Optional timestamp in video (seconds)
    
    Returns:
        Dictionary with compiled form data, or None
    """
    try:
        kernel = get_semantic_kernel()
        compiled = kernel.compile(text, confidence)
        
        result = compiled.to_dict()
        result["videoId"] = video_id
        result["timestamp"] = timestamp
        result["operatorNames"] = [SemanticOperator(c).name for c in compiled.operator_codes]
        
        return result
        
    except ValueError as e:
        logger.debug(f"Compilation skipped: {e}")
        return None
    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return None

