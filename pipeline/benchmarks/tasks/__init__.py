"""
Benchmark task implementations for N'Ko and Manding languages.
"""

from .translation import TranslationTask
from .script_knowledge import ScriptKnowledgeTask
from .vocabulary import VocabularyTask
from .cultural import CulturalTask
from .compositional import CompositionalTask
from .cross_language import (
    CrossMandingTranslationTask,
    ScriptTransliterationTask,
    DialectIdentificationTask,
    CognateRecognitionTask,
    BackTranslationTask,
    CrossLanguageResult,
    CrossLanguageTaskType,
    calculate_cross_language_metrics,
)
from .curriculum import (
    CurriculumTask,
    CurriculumTestGenerator,
    CurriculumTestItem,
    CurriculumResult,
    CEFRLevel,
    calculate_curriculum_metrics,
)

__all__ = [
    # Core tasks
    "TranslationTask",
    "ScriptKnowledgeTask",
    "VocabularyTask",
    "CulturalTask",
    "CompositionalTask",
    # Cross-language tasks
    "CrossMandingTranslationTask",
    "ScriptTransliterationTask",
    "DialectIdentificationTask",
    "CognateRecognitionTask",
    "BackTranslationTask",
    "CrossLanguageResult",
    "CrossLanguageTaskType",
    "calculate_cross_language_metrics",
    # Curriculum tasks
    "CurriculumTask",
    "CurriculumTestGenerator",
    "CurriculumTestItem",
    "CurriculumResult",
    "CEFRLevel",
    "calculate_curriculum_metrics",
]

