"""
N'Ko and Manding Language AI Model Benchmark Pipeline

Evaluates state-of-the-art AI models (Claude, GPT, Gemini) on N'Ko and Manding 
language tasks to determine the optimal base model for the translation system.

Features:
- N'Ko script translation and recognition
- Bambara-French translation evaluation
- Cross-Manding language tests (N'Ko ↔ Bambara)
- Curriculum-based progressive difficulty (CEFR A1-C2)
- Complex tests: proverbs, compound words, cognates

Usage:
    # N'Ko-focused benchmark
    python -m training.benchmarks.nko_benchmark --all
    
    # Full Manding multilingual benchmark
    python -m training.benchmarks.manding_benchmark --all --multilingual
    
    # Curriculum-based evaluation
    python -m training.benchmarks.manding_benchmark --curriculum --levels A1 A2 B1
"""

from .config import (
    BenchmarkConfig, 
    ModelConfig,
    LANGUAGES,
    TRANSLATION_PAIRS,
    CURRICULUM_LEVELS,
)
from .nko_benchmark import NkoBenchmark
from .manding_benchmark import MandingBenchmark

__all__ = [
    # Config
    "BenchmarkConfig",
    "ModelConfig",
    "LANGUAGES",
    "TRANSLATION_PAIRS",
    "CURRICULUM_LEVELS",
    # Benchmarks
    "NkoBenchmark",
    "MandingBenchmark",
]

__version__ = "2.0.0"

