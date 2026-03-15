"""
Configuration for N'Ko and Manding Language AI Model Benchmark Pipeline.

Contains model definitions, API configurations, language pairs, and benchmark settings
for comprehensive multilingual evaluation of AI models on Manding languages.

Supports:
- N'Ko (ߒ'ߞߏ) script
- Bambara (Bamanankan) - Latin script
- Malinke (Maninka) - Latin script
- Jula (Dioula) - Latin script
- English and French colonial languages
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum


@dataclass
class ModelConfig:
    """Configuration for a single AI model."""
    name: str
    provider: str  # 'anthropic', 'openai', 'google'
    model_id: str
    max_tokens: int = 2048
    temperature: float = 0.3
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    
    # Sample sizes per task
    translation_samples: int = 2000
    script_samples: int = 200
    vocabulary_samples: int = 500
    cultural_samples: int = 100
    compositional_samples: int = 200
    
    # Multilingual/Cross-language samples
    cross_language_samples: int = 200
    curriculum_samples_per_level: int = 50
    
    # Task weights (must sum to 1.0)
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "translation": 0.35,
        "script_knowledge": 0.10,
        "vocabulary": 0.15,
        "cultural": 0.10,
        "compositional": 0.10,
        "cross_language": 0.10,
        "curriculum": 0.10,
    })
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("training/benchmarks/reports"))
    save_detailed_results: bool = True
    
    # API settings
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout_seconds: int = 60
    rate_limit_delay: float = 0.5
    
    # Concurrent requests per provider
    max_concurrent_requests: int = 5
    
    # Multilingual benchmark settings
    enable_multilingual: bool = True
    curriculum_levels: List[str] = field(default_factory=lambda: ["A1", "A2", "B1", "B2", "C1", "C2"])


# =============================================================================
# Language Configuration for Manding Language Benchmark
# =============================================================================

class LanguageCode(Enum):
    """ISO 639-3 language codes for Manding and related languages."""
    NKO = "nqo"      # N'Ko script (unified Manding orthography)
    BAMBARA = "bam"  # Bambara (Bamanankan)
    MALINKE = "man"  # Malinke (Maninka)
    JULA = "dyu"     # Jula (Dioula)
    ENGLISH = "eng"  # English
    FRENCH = "fra"   # French
    ARABIC = "ara"   # Arabic (for loanword analysis)


# Language metadata
LANGUAGES: Dict[str, Dict] = {
    "nko": {
        "code": LanguageCode.NKO,
        "name": "N'Ko",
        "native_name": "ߒ'ߞߏ",
        "script": "N'Ko",
        "family": "Manding",
        "is_primary": True,
        "direction": "rtl",
    },
    "bambara": {
        "code": LanguageCode.BAMBARA,
        "name": "Bambara",
        "native_name": "Bamanankan",
        "script": "Latin",
        "family": "Manding",
        "is_primary": True,
        "direction": "ltr",
    },
    "malinke": {
        "code": LanguageCode.MALINKE,
        "name": "Malinke",
        "native_name": "Maninkakan",
        "script": "Latin",
        "family": "Manding",
        "is_primary": False,
        "direction": "ltr",
    },
    "jula": {
        "code": LanguageCode.JULA,
        "name": "Jula",
        "native_name": "Julakan",
        "script": "Latin",
        "family": "Manding",
        "is_primary": False,
        "direction": "ltr",
    },
    "french": {
        "code": LanguageCode.FRENCH,
        "name": "French",
        "native_name": "Français",
        "script": "Latin",
        "family": "Romance",
        "is_primary": True,
        "direction": "ltr",
    },
    "english": {
        "code": LanguageCode.ENGLISH,
        "name": "English",
        "native_name": "English",
        "script": "Latin",
        "family": "Germanic",
        "is_primary": True,
        "direction": "ltr",
    },
}

# Translation pairs to benchmark (source, target)
# 12 directions total covering all major language pairs
TRANSLATION_PAIRS: List[Tuple[str, str]] = [
    # N'Ko to colonial languages
    ("nko", "english"),
    ("english", "nko"),
    ("nko", "french"),
    ("french", "nko"),
    
    # N'Ko to other Manding variants
    ("nko", "bambara"),
    ("bambara", "nko"),
    
    # Bambara to colonial languages
    ("bambara", "french"),
    ("french", "bambara"),
    ("bambara", "english"),
    ("english", "bambara"),
    
    # Malinke/Jula pairs (secondary)
    ("malinke", "french"),
    ("french", "malinke"),
]

# Cross-Manding evaluation pairs
CROSS_MANDING_PAIRS: List[Tuple[str, str]] = [
    ("nko", "bambara"),
    ("bambara", "nko"),
    ("bambara", "malinke"),
    ("malinke", "bambara"),
    ("bambara", "jula"),
    ("jula", "bambara"),
]

# Script conversion pairs
SCRIPT_CONVERSION_PAIRS: List[Tuple[str, str]] = [
    ("nko", "latin"),     # N'Ko → Latin transliteration
    ("latin", "nko"),     # Latin → N'Ko script
    ("bambara", "nko"),   # Bambara (Latin) → N'Ko script
]


# =============================================================================
# Curriculum Levels (CEFR-aligned)
# =============================================================================

CURRICULUM_LEVELS = {
    "A1": {
        "name": "Beginner",
        "description": "Basic greetings, numbers, common words",
        "vocab_range": (0, 500),
        "sentence_complexity": "simple",
        "topics": ["greetings", "numbers", "colors", "family"],
    },
    "A2": {
        "name": "Elementary", 
        "description": "Simple sentences, questions, basic grammar",
        "vocab_range": (500, 1500),
        "sentence_complexity": "compound",
        "topics": ["daily_life", "food", "travel", "work"],
    },
    "B1": {
        "name": "Intermediate",
        "description": "Complex sentences, proverbs, intermediate vocabulary",
        "vocab_range": (1500, 3000),
        "sentence_complexity": "complex",
        "topics": ["culture", "traditions", "proverbs", "stories"],
    },
    "B2": {
        "name": "Upper Intermediate",
        "description": "Idiomatic expressions, cultural references, formal/informal registers",
        "vocab_range": (3000, 5000),
        "sentence_complexity": "idiomatic",
        "topics": ["idioms", "formal_speech", "literature", "religion"],
    },
    "C1": {
        "name": "Advanced",
        "description": "Error correction, novel compositions, nuanced meanings",
        "vocab_range": (5000, 10000),
        "sentence_complexity": "nuanced",
        "topics": ["advanced_grammar", "composition", "technical", "debate"],
    },
    "C2": {
        "name": "Proficient",
        "description": "Proverb completion, literary translation, dialectal variations",
        "vocab_range": (10000, None),
        "sentence_complexity": "literary",
        "topics": ["poetry", "dialects", "etymology", "linguistics"],
    },
}


# =============================================================================
# Model Definitions - API-verified model IDs (listed via keys on 2026-01-01)
# =============================================================================

MODELS: Dict[str, ModelConfig] = {
    # ==========================================================================
    # Anthropic Claude Models (Latest 2025)
    # Pricing: per 1K tokens (converted from per 1M tokens)
    # ==========================================================================
    "claude-4.5-sonnet": ModelConfig(
        name="Claude 4.5 Sonnet",
        provider="anthropic",
        # Verified via Anthropic SDK models.list()
        model_id="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.003,    # $3.00/1M = $0.003/1K
        cost_per_1k_output=0.015,    # $15.00/1M = $0.015/1K
    ),
    "claude-4.5-opus": ModelConfig(
        name="Claude 4.5 Opus",
        provider="anthropic",
        # Verified via Anthropic SDK models.list()
        model_id="claude-opus-4-5-20251101",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.005,     # $5.00/1M = $0.005/1K
        cost_per_1k_output=0.025,    # $25.00/1M = $0.025/1K
    ),
    
    # ==========================================================================
    # OpenAI GPT Models (Latest 2025)
    # ==========================================================================
    "gpt-5.2": ModelConfig(
        name="GPT-5.2",
        provider="openai",
        # Verified via OpenAI models.list()
        model_id="gpt-5.2",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00175,   # $1.75/1M = $0.00175/1K
        cost_per_1k_output=0.014,     # $14.00/1M = $0.014/1K
    ),
    "gpt-5-mini": ModelConfig(
        name="GPT-5-mini",
        provider="openai",
        # Verified via OpenAI models.list()
        model_id="gpt-5-mini",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.0,  # TODO: fill from OpenAI pricing docs
        cost_per_1k_output=0.0, # TODO: fill from OpenAI pricing docs
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",  # Fallback: Currently available GPT-4o
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.0025,     # $2.50/1M = $0.0025/1K
        cost_per_1k_output=0.01,      # $10.00/1M = $0.01/1K
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o mini",
        provider="openai",
        model_id="gpt-4o-mini",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.0,  # TODO: fill from OpenAI pricing docs
        cost_per_1k_output=0.0, # TODO: fill from OpenAI pricing docs
    ),
    
    # ==========================================================================
    # Google Gemini Models (Latest 2025)
    # ==========================================================================
    "gemini-3-pro": ModelConfig(
        name="Gemini 3 Pro",
        provider="google",
        # Verified via Google genai.list_models()
        model_id="models/gemini-3-pro-preview",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00125,   # $1.25/1M = $0.00125/1K (estimated)
        cost_per_1k_output=0.005,    # $5.00/1M = $0.005/1K (estimated)
    ),
    "gemini-3-flash": ModelConfig(
        name="Gemini 3 Flash",
        provider="google",
        # Verified via Google genai.list_models()
        model_id="models/gemini-3-flash-preview",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.0005,    # $0.50/1M = $0.0005/1K
        cost_per_1k_output=0.003,    # $3.00/1M = $0.003/1K
    ),
    
    # ==========================================================================
    # Fallback Models (Currently Available - Use if latest models unavailable)
    # ==========================================================================
    "claude-sonnet-4": ModelConfig(
        name="Claude Sonnet 4 (Fallback)",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "gemini-2.5-pro": ModelConfig(
        name="Gemini 2.5 Pro (Fallback)",
        provider="google",
        model_id="gemini-2.5-pro",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
    ),
    "gemini-2.5-flash": ModelConfig(
        name="Gemini 2.5 Flash (Fallback)",
        provider="google",
        model_id="gemini-2.5-flash",
        max_tokens=4096,
        temperature=0.3,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
}

# Provider groupings (API-verified defaults)
PROVIDERS = {
    "anthropic": ["claude-4.5-sonnet", "claude-4.5-opus", "claude-sonnet-4"],
    "openai": ["gpt-5.2", "gpt-5-mini", "gpt-4o", "gpt-4o-mini"],
    "google": ["gemini-3-pro", "gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
}


# =============================================================================
# Data Paths
# =============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = TRAINING_DIR / "data"
NKO_DATA_DIR = PROJECT_ROOT / "nko" / "data"

# nicolingua-0005 N'Ko data (130K+ parallel pairs)
NICOLINGUA_DIR = DATA_DIR / "exports" / "nicolingua"
NICOLINGUA_TRANSLATIONS = NICOLINGUA_DIR / "nicolingua_translations.jsonl"
NICOLINGUA_VOCABULARY = NICOLINGUA_DIR / "nicolingua_vocabulary.jsonl"
NICOLINGUA_MONOLINGUAL = NICOLINGUA_DIR / "nicolingua_monolingual.txt"
NICOLINGUA_RAW_DIR = DATA_DIR / "nicolingua" / "data"

# Bayelemabaga Bambara-French corpus (47K+ pairs)
BAYELEMABAGA_DIR = NKO_DATA_DIR / "bayelemabaga" / "bayelemabaga"
BAYELEMABAGA_TRAIN = BAYELEMABAGA_DIR / "train" / "train.tsv"
BAYELEMABAGA_VALID = BAYELEMABAGA_DIR / "dev" / "dev.tsv"
BAYELEMABAGA_TEST = BAYELEMABAGA_DIR / "test" / "test.tsv"

# Unified corpus Bambara-French (56K+ pairs)
UNIFIED_CORPUS_DIR = NKO_DATA_DIR / "unified_corpus"
UNIFIED_CORPUS_ALL = UNIFIED_CORPUS_DIR / "all.tsv"
UNIFIED_CORPUS_TRAIN = UNIFIED_CORPUS_DIR / "train.tsv"
UNIFIED_CORPUS_VALID = UNIFIED_CORPUS_DIR / "valid.tsv"
UNIFIED_CORPUS_TEST = UNIFIED_CORPUS_DIR / "test.tsv"

# Ankataa dictionary (2K+ entries)
DICTIONARY_DIR = DATA_DIR / "dictionary"

# Bidirectional training data
BIDIRECTIONAL_DATA = NKO_DATA_DIR / "training_data_bidirectional.json"

# Output
REPORTS_DIR = TRAINING_DIR / "benchmarks" / "reports"


# =============================================================================
# Data Source Statistics (as of 2025)
# =============================================================================

DATA_SOURCE_STATS = {
    "nicolingua": {
        "name": "nicolingua-0005",
        "description": "N'Ko-English-French trilingual parallel corpus",
        "pairs": 130850,
        "languages": ["nko", "english", "french"],
        "verified": True,
    },
    "bayelemabaga": {
        "name": "Bayelemabaga",
        "description": "Bambara-French parallel corpus",
        "pairs": 46976,
        "languages": ["bambara", "french"],
        "verified": True,
    },
    "unified_corpus": {
        "name": "Unified Corpus",
        "description": "Additional Bambara-French aligned pairs",
        "pairs": 55793,
        "languages": ["bambara", "french"],
        "verified": True,
    },
    "ankataa": {
        "name": "Ankataa Dictionary",
        "description": "N'Ko dictionary entries with definitions",
        "entries": 2101,
        "languages": ["nko", "english", "french"],
        "verified": True,
    },
    "supabase_nko_vocabulary": {
        "name": "Supabase N'Ko Vocabulary",
        "description": "Verified N'Ko vocabulary entries",
        "entries": 198,
        "languages": ["nko"],
        "verified": True,
        "gold_standard": True,
    },
    "supabase_nko_translations": {
        "name": "Supabase N'Ko Translations",
        "description": "Validated N'Ko translation pairs",
        "entries": 34,
        "languages": ["nko", "english", "french"],
        "verified": True,
        "gold_standard": True,
    },
}


# =============================================================================
# Environment Variables
# =============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from environment variables."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var:
        return os.getenv(env_var)
    return None


def get_supabase_config() -> Dict[str, Optional[str]]:
    """Get Supabase configuration from environment."""
    return {
        "url": os.getenv("SUPABASE_URL"),
        "key": os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY"),
    }


# =============================================================================
# N'Ko Script Reference Data
# =============================================================================

# N'Ko alphabet for script knowledge tests
NKO_ALPHABET = {
    # Basic letters
    "ߊ": {"latin": "a", "name": "a"},
    "ߋ": {"latin": "o", "name": "o"},
    "ߌ": {"latin": "i", "name": "i"},
    "ߍ": {"latin": "e", "name": "e"},
    "ߎ": {"latin": "u", "name": "u"},
    "ߏ": {"latin": "ɔ", "name": "open o"},
    "ߐ": {"latin": "ɛ", "name": "open e"},
    "ߓ": {"latin": "b", "name": "ba"},
    "ߔ": {"latin": "p", "name": "pa"},
    "ߕ": {"latin": "t", "name": "ta"},
    "ߖ": {"latin": "j", "name": "ja"},
    "ߗ": {"latin": "c", "name": "cha"},
    "ߘ": {"latin": "d", "name": "da"},
    "ߙ": {"latin": "r", "name": "ra"},
    "ߚ": {"latin": "rr", "name": "rra"},
    "ߛ": {"latin": "s", "name": "sa"},
    "ߜ": {"latin": "gb", "name": "gba"},
    "ߝ": {"latin": "f", "name": "fa"},
    "ߞ": {"latin": "k", "name": "ka"},
    "ߟ": {"latin": "l", "name": "la"},
    "ߠ": {"latin": "na velaire", "name": "na velaire"},
    "ߡ": {"latin": "m", "name": "ma"},
    "ߢ": {"latin": "ny", "name": "nya"},
    "ߣ": {"latin": "n", "name": "na"},
    "ߤ": {"latin": "h", "name": "ha"},
    "ߥ": {"latin": "w", "name": "wa"},
    "ߦ": {"latin": "y", "name": "ya"},
    "ߧ": {"latin": "ny palatal", "name": "nya palatal"},
}

# Common N'Ko morphemes for compositional tests
NKO_MORPHEMES = {
    "prefixes": {
        "ߡߊ߬": {"meaning": "causative/place", "example": "ߡߊ߬ߟߐ߲"},
        "ߓߟߏ߬": {"meaning": "un-/dis-", "example": "ߓߟߏ߬ߟߊ"},
    },
    "suffixes": {
        "ߟߊ": {"meaning": "place of action", "example": "ߞߍߟߊ"},
        "ߟߌ": {"meaning": "agent/doer", "example": "ߞߊ߬ߙߊ߲߬ߟߌ"},
        "ߦߊ": {"meaning": "abstract quality", "example": "ߢߌ߬ߡߊ߬ߦߊ"},
        "ߓߊ": {"meaning": "augmentative/big", "example": "ߡߏ߬ߛߏ߬ߓߊ"},
    },
    "roots": {
        "ߞߍ": {"meaning": "to do/make", "class": "verb"},
        "ߕߊ߯": {"meaning": "to go", "class": "verb"},
        "ߣߊ߬": {"meaning": "to come", "class": "verb"},
        "ߝߐ߫": {"meaning": "to say/speak", "class": "verb"},
        "ߘߎ߲": {"meaning": "to eat", "class": "verb"},
        "ߡߐ߰": {"meaning": "person", "class": "noun"},
        "ߘߋ߲": {"meaning": "child", "class": "noun"},
        "ߛߏ": {"meaning": "house/city", "class": "noun"},
    }
}


# =============================================================================
# Quick Config Presets
# =============================================================================

def get_quick_config() -> BenchmarkConfig:
    """Get configuration for quick benchmark run (~200 samples)."""
    return BenchmarkConfig(
        translation_samples=50,
        script_samples=20,
        vocabulary_samples=30,
        cultural_samples=10,
        compositional_samples=20,
        cross_language_samples=30,
        curriculum_samples_per_level=10,
        enable_multilingual=True,
        curriculum_levels=["A1", "A2", "B1"],
    )


def get_medium_config() -> BenchmarkConfig:
    """Get configuration for medium benchmark run (~1000 samples)."""
    return BenchmarkConfig(
        translation_samples=500,
        script_samples=50,
        vocabulary_samples=100,
        cultural_samples=30,
        compositional_samples=50,
        cross_language_samples=100,
        curriculum_samples_per_level=30,
        enable_multilingual=True,
        curriculum_levels=["A1", "A2", "B1", "B2"],
    )


def get_full_config() -> BenchmarkConfig:
    """Get configuration for full benchmark run (~2700 samples as per plan)."""
    return BenchmarkConfig(
        translation_samples=2000,
        script_samples=200,
        vocabulary_samples=500,
        cultural_samples=100,
        compositional_samples=200,
        cross_language_samples=200,
        curriculum_samples_per_level=50,
        enable_multilingual=True,
        curriculum_levels=["A1", "A2", "B1", "B2", "C1", "C2"],
    )


def get_multilingual_config() -> BenchmarkConfig:
    """Get configuration focused on multilingual/cross-Manding evaluation."""
    return BenchmarkConfig(
        translation_samples=1500,  # Distributed across language pairs
        script_samples=200,
        vocabulary_samples=300,
        cultural_samples=100,
        compositional_samples=100,
        cross_language_samples=300,  # Extra cross-language samples
        curriculum_samples_per_level=50,
        enable_multilingual=True,
        curriculum_levels=["A1", "A2", "B1", "B2", "C1", "C2"],
        task_weights={
            "translation": 0.30,
            "script_knowledge": 0.10,
            "vocabulary": 0.10,
            "cultural": 0.10,
            "compositional": 0.10,
            "cross_language": 0.20,  # Higher weight for cross-language
            "curriculum": 0.10,
        },
    )


def get_curriculum_config() -> BenchmarkConfig:
    """Get configuration focused on curriculum-based evaluation."""
    return BenchmarkConfig(
        translation_samples=500,
        script_samples=100,
        vocabulary_samples=200,
        cultural_samples=50,
        compositional_samples=100,
        cross_language_samples=100,
        curriculum_samples_per_level=100,  # More curriculum samples
        enable_multilingual=True,
        curriculum_levels=["A1", "A2", "B1", "B2", "C1", "C2"],
        task_weights={
            "translation": 0.25,
            "script_knowledge": 0.10,
            "vocabulary": 0.10,
            "cultural": 0.10,
            "compositional": 0.10,
            "cross_language": 0.10,
            "curriculum": 0.25,  # Higher weight for curriculum
        },
    )


def get_comprehensive_config() -> BenchmarkConfig:
    """
    Get configuration for COMPREHENSIVE Manding language evaluation.
    
    Uses the full nicolingua dataset (130K+ pairs), Supabase data,
    Bayelemabaga corpus, and all other available sources.
    
    This is the most thorough evaluation mode.
    """
    return BenchmarkConfig(
        # Use MAXIMUM available data
        translation_samples=5000,  # 5K samples per language pair
        script_samples=1000,
        vocabulary_samples=2000,
        cultural_samples=500,
        compositional_samples=500,
        cross_language_samples=1000,
        curriculum_samples_per_level=200,
        enable_multilingual=True,
        curriculum_levels=["A1", "A2", "B1", "B2", "C1", "C2"],
        
        # Balanced weights for comprehensive evaluation
        task_weights={
            "translation": 0.30,
            "script_knowledge": 0.10,
            "vocabulary": 0.15,
            "cultural": 0.10,
            "compositional": 0.15,  # Novel word building, compound words
            "cross_language": 0.10,
            "curriculum": 0.10,
        },
        
        # Performance settings for large-scale evaluation
        max_retries=5,
        timeout_seconds=120,
        rate_limit_delay=0.3,  # Faster for large runs
        max_concurrent_requests=10,
    )


def get_quick_config() -> BenchmarkConfig:
    """Quick config for testing (small samples)."""
    return BenchmarkConfig(
        translation_samples=100,
        script_samples=50,
        vocabulary_samples=50,
        cultural_samples=20,
        compositional_samples=20,
        cross_language_samples=50,
        curriculum_samples_per_level=10,
        enable_multilingual=True,
        curriculum_levels=["A1", "B1", "C1"],  # Only 3 levels for quick test
    )


def estimate_benchmark_cost(
    config: BenchmarkConfig,
    model_keys: List[str],
    avg_input_tokens: int = 150,
    avg_output_tokens: int = 100,
) -> Dict[str, Any]:
    """
    Estimate the cost of running a benchmark.
    
    Args:
        config: Benchmark configuration
        model_keys: List of model keys to test
        avg_input_tokens: Average input tokens per sample (prompt)
        avg_output_tokens: Average output tokens per sample (response)
        
    Returns:
        Dictionary with cost breakdown
    """
    # Calculate total samples
    # Translation: samples per language pair × 12 pairs
    translation_total = config.translation_samples * 8  # 8 main pairs have data
    script_total = config.script_samples
    vocab_total = config.vocabulary_samples
    cultural_total = config.cultural_samples
    compositional_total = config.compositional_samples
    cross_lang_total = config.cross_language_samples
    curriculum_total = config.curriculum_samples_per_level * len(config.curriculum_levels)
    
    total_samples = (
        translation_total +
        script_total +
        vocab_total +
        cultural_total +
        compositional_total +
        cross_lang_total +
        curriculum_total
    )
    
    # Estimate tokens per model
    total_input_tokens = total_samples * avg_input_tokens
    total_output_tokens = total_samples * avg_output_tokens
    
    # Calculate cost per model
    model_costs = {}
    total_cost = 0.0
    
    for model_key in model_keys:
        if model_key not in MODELS:
            continue
        model = MODELS[model_key]
        
        input_cost = (total_input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (total_output_tokens / 1000) * model.cost_per_1k_output
        model_total = input_cost + output_cost
        
        model_costs[model_key] = {
            "name": model.name,
            "provider": model.provider,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(model_total, 4),
        }
        total_cost += model_total
    
    return {
        "config_summary": {
            "translation_samples": config.translation_samples,
            "total_samples_per_model": total_samples,
            "total_samples_all_models": total_samples * len(model_keys),
        },
        "token_estimates": {
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "total_input_per_model": total_input_tokens,
            "total_output_per_model": total_output_tokens,
        },
        "model_costs": model_costs,
        "total_cost": round(total_cost, 2),
        "models_count": len(model_keys),
    }


def print_cost_estimate(
    config: BenchmarkConfig,
    model_keys: List[str],
) -> None:
    """Print a formatted cost estimate."""
    estimate = estimate_benchmark_cost(config, model_keys)
    
    print("\n" + "=" * 60)
    print("   BENCHMARK COST ESTIMATE")
    print("=" * 60)
    
    print(f"\n📊 Sample Counts:")
    print(f"   Translation samples/pair: {estimate['config_summary']['translation_samples']}")
    print(f"   Samples per model: {estimate['config_summary']['total_samples_per_model']:,}")
    print(f"   Total samples (all models): {estimate['config_summary']['total_samples_all_models']:,}")
    
    print(f"\n📝 Token Estimates:")
    print(f"   Avg input tokens/sample: {estimate['token_estimates']['avg_input_tokens']}")
    print(f"   Avg output tokens/sample: {estimate['token_estimates']['avg_output_tokens']}")
    
    print(f"\n💰 Cost Breakdown by Model:")
    for model_key, costs in estimate['model_costs'].items():
        print(f"   {costs['name']} ({costs['provider']}):")
        print(f"      Input:  ${costs['input_cost']:.4f}")
        print(f"      Output: ${costs['output_cost']:.4f}")
        print(f"      Total:  ${costs['total_cost']:.4f}")
    
    print(f"\n🔢 TOTAL ESTIMATED COST: ${estimate['total_cost']:.2f}")
    print("=" * 60)

