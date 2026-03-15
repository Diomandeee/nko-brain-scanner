"""
Data loading and sampling for N'Ko and Manding language benchmarks.
"""

from .sampler import TestDataSampler
from .supabase_loader import SupabaseLoader
from .complex_tests import (
    ComplexTestGenerator,
    NovelWordTest,
    SentenceConstructionTest,
    DisambiguationTest,
    ErrorCorrectionTest,
    ProverbTest,
    CompoundWordTest,
    DialectVariationTest,
    CognateMatchTest,
    generate_compositional_tests,
)
from .manding_loader import (
    MandingDataLoader,
    MandingTestSet,
    Language,
    VocabEntry,
    TranslationPair,
    CognatePair,
    create_manding_test_set,
    sample_translation_pairs,
)

__all__ = [
    "TestDataSampler",
    "SupabaseLoader",
    # Complex test generators
    "ComplexTestGenerator",
    "NovelWordTest",
    "SentenceConstructionTest",
    "DisambiguationTest",
    "ErrorCorrectionTest",
    "ProverbTest",
    "CompoundWordTest",
    "DialectVariationTest",
    "CognateMatchTest",
    "generate_compositional_tests",
    # Manding data loaders
    "MandingDataLoader",
    "MandingTestSet",
    "Language",
    "VocabEntry",
    "TranslationPair",
    "CognatePair",
    "create_manding_test_set",
    "sample_translation_pairs",
]

