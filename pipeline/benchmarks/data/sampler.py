"""
Stratified Test Data Sampler for N'Ko Benchmark.

Samples test data from nicolingua corpus with balanced distribution
across translation directions and complexity levels.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ..config import (
    NICOLINGUA_TRANSLATIONS,
    NICOLINGUA_VOCABULARY,
    DICTIONARY_DIR,
    BenchmarkConfig,
)


@dataclass
class TranslationSample:
    """A single translation test sample."""
    nko_text: str
    english: str
    french: str
    source: str
    source_file: str
    complexity: str = "simple"  # simple, medium, complex
    has_english: bool = True
    has_french: bool = True


@dataclass
class VocabularySample:
    """A single vocabulary test sample."""
    word: str
    word_class: Optional[str] = None
    definitions_en: List[str] = field(default_factory=list)
    definitions_fr: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source: str = "ankataa"


@dataclass
class TestDataset:
    """Complete test dataset for benchmarking."""
    # Translation samples by direction
    nko_to_en: List[TranslationSample] = field(default_factory=list)
    nko_to_fr: List[TranslationSample] = field(default_factory=list)
    en_to_nko: List[TranslationSample] = field(default_factory=list)
    fr_to_nko: List[TranslationSample] = field(default_factory=list)
    
    # Vocabulary samples
    vocabulary: List[VocabularySample] = field(default_factory=list)
    
    # Script knowledge samples (N'Ko text for recognition)
    script_samples: List[Dict[str, str]] = field(default_factory=list)
    
    # Cultural samples (proverbs, idioms)
    cultural_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Compositional samples
    compositional_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def total_samples(self) -> int:
        """Get total number of samples."""
        return (
            len(self.nko_to_en) + len(self.nko_to_fr) +
            len(self.en_to_nko) + len(self.fr_to_nko) +
            len(self.vocabulary) + len(self.script_samples) +
            len(self.cultural_samples) + len(self.compositional_samples)
        )


class TestDataSampler:
    """
    Stratified sampler for N'Ko benchmark test data.
    
    Loads data from:
    - nicolingua translations (130K+ parallel pairs)
    - nicolingua vocabulary (53K+ words)
    - ankataa dictionary (37K+ entries)
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._translations: List[Dict] = []
        self._vocabulary: List[Dict] = []
        self._dictionary: List[Dict] = []
        self._loaded = False
    
    def load_data(self) -> None:
        """Load all data sources."""
        if self._loaded:
            return
        
        print("Loading benchmark data sources...")
        
        # Load nicolingua translations
        if NICOLINGUA_TRANSLATIONS.exists():
            print(f"  Loading translations from {NICOLINGUA_TRANSLATIONS.name}...")
            with open(NICOLINGUA_TRANSLATIONS, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._translations.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            print(f"    Loaded {len(self._translations):,} translation pairs")
        
        # Load nicolingua vocabulary
        if NICOLINGUA_VOCABULARY.exists():
            print(f"  Loading vocabulary from {NICOLINGUA_VOCABULARY.name}...")
            with open(NICOLINGUA_VOCABULARY, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._vocabulary.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            print(f"    Loaded {len(self._vocabulary):,} vocabulary entries")
        
        # Load ankataa dictionary
        dict_files = list(DICTIONARY_DIR.glob("ankataa_dictionary_*.json"))
        if dict_files:
            latest_dict = max(dict_files, key=lambda p: p.stat().st_mtime)
            print(f"  Loading dictionary from {latest_dict.name}...")
            with open(latest_dict, 'r', encoding='utf-8') as f:
                self._dictionary = json.load(f)
            print(f"    Loaded {len(self._dictionary):,} dictionary entries")
        
        self._loaded = True
        print(f"  Total data loaded: {len(self._translations) + len(self._vocabulary) + len(self._dictionary):,} entries")
    
    def _classify_complexity(self, text: str) -> str:
        """Classify text complexity based on length and structure."""
        words = text.split()
        if len(words) <= 3:
            return "simple"
        elif len(words) <= 10:
            return "medium"
        else:
            return "complex"
    
    def _has_quality_translations(self, entry: Dict) -> bool:
        """Check if entry has quality translations (non-empty)."""
        nko = entry.get("nko_text", "").strip()
        en = entry.get("english", "").strip()
        fr = entry.get("french", "").strip()
        
        # Must have N'Ko and at least one translation
        if not nko:
            return False
        if not en and not fr:
            return False
        # Skip very short entries (likely just letters)
        if len(nko) < 2:
            return False
        return True
    
    def sample_translations(self) -> Dict[str, List[TranslationSample]]:
        """
        Sample translation pairs with stratified distribution.
        
        Returns:
            Dict with keys: nko_to_en, nko_to_fr, en_to_nko, fr_to_nko
        """
        self.load_data()
        
        # Filter quality entries
        quality_entries = [e for e in self._translations if self._has_quality_translations(e)]
        
        # Separate by available translations
        en_available = [e for e in quality_entries if e.get("english", "").strip()]
        fr_available = [e for e in quality_entries if e.get("french", "").strip()]
        
        # Stratify by complexity
        complexity_buckets = defaultdict(list)
        for entry in quality_entries:
            complexity = self._classify_complexity(entry.get("nko_text", ""))
            complexity_buckets[complexity].append(entry)
        
        # Sample with complexity balance
        def balanced_sample(entries: List[Dict], n: int) -> List[Dict]:
            """Sample with balanced complexity distribution."""
            if len(entries) <= n:
                return entries
            
            # Group by complexity
            by_complexity = defaultdict(list)
            for e in entries:
                c = self._classify_complexity(e.get("nko_text", ""))
                by_complexity[c].append(e)
            
            # Target: 30% simple, 50% medium, 20% complex
            targets = {
                "simple": int(n * 0.3),
                "medium": int(n * 0.5),
                "complex": int(n * 0.2),
            }
            
            result = []
            for complexity, target in targets.items():
                available = by_complexity[complexity]
                sample_size = min(target, len(available))
                result.extend(random.sample(available, sample_size))
            
            # Fill remainder if needed
            remaining = n - len(result)
            if remaining > 0:
                unused = [e for e in entries if e not in result]
                if unused:
                    result.extend(random.sample(unused, min(remaining, len(unused))))
            
            return result
        
        # Sample for each direction
        nko_to_en_raw = balanced_sample(en_available, self.config.translation_samples // 2)
        nko_to_fr_raw = balanced_sample(fr_available, self.config.translation_samples // 4)
        en_to_nko_raw = balanced_sample(en_available, self.config.translation_samples // 6)
        fr_to_nko_raw = balanced_sample(fr_available, self.config.translation_samples // 10)
        
        # Convert to TranslationSample objects
        def to_sample(entry: Dict) -> TranslationSample:
            return TranslationSample(
                nko_text=entry.get("nko_text", ""),
                english=entry.get("english", ""),
                french=entry.get("french", ""),
                source=entry.get("source", "nicolingua-0005"),
                source_file=entry.get("source_file", "unknown"),
                complexity=self._classify_complexity(entry.get("nko_text", "")),
                has_english=bool(entry.get("english", "").strip()),
                has_french=bool(entry.get("french", "").strip()),
            )
        
        return {
            "nko_to_en": [to_sample(e) for e in nko_to_en_raw],
            "nko_to_fr": [to_sample(e) for e in nko_to_fr_raw],
            "en_to_nko": [to_sample(e) for e in en_to_nko_raw],
            "fr_to_nko": [to_sample(e) for e in fr_to_nko_raw],
        }
    
    def sample_vocabulary(self) -> List[VocabularySample]:
        """Sample vocabulary entries for definition tests."""
        self.load_data()
        
        # Prefer dictionary entries (more complete)
        samples = []
        
        if self._dictionary:
            # Sample from ankataa dictionary
            sample_size = min(self.config.vocabulary_samples, len(self._dictionary))
            sampled = random.sample(self._dictionary, sample_size)
            
            for entry in sampled:
                samples.append(VocabularySample(
                    word=entry.get("word", ""),
                    word_class=entry.get("word_class"),
                    definitions_en=entry.get("definitions_en", []),
                    definitions_fr=entry.get("definitions_fr", []),
                    examples=entry.get("examples", []),
                    source="ankataa",
                ))
        
        # Supplement with nicolingua vocabulary if needed
        remaining = self.config.vocabulary_samples - len(samples)
        if remaining > 0 and self._vocabulary:
            additional = random.sample(
                self._vocabulary, 
                min(remaining, len(self._vocabulary))
            )
            for entry in additional:
                samples.append(VocabularySample(
                    word=entry.get("word", ""),
                    source="nicolingua",
                ))
        
        return samples
    
    def sample_script_knowledge(self) -> List[Dict[str, str]]:
        """Sample N'Ko text for script recognition tests."""
        self.load_data()
        
        samples = []
        
        # Get unique N'Ko words
        nko_words = set()
        for entry in self._vocabulary[:10000]:  # Limit for efficiency
            word = entry.get("word", "").strip()
            if word and len(word) >= 2:
                nko_words.add(word)
        
        # Sample diverse lengths
        word_list = list(nko_words)
        if word_list:
            sampled = random.sample(
                word_list, 
                min(self.config.script_samples, len(word_list))
            )
            for word in sampled:
                samples.append({
                    "nko_text": word,
                    "task": "recognize",
                })
        
        return samples
    
    def sample_cultural(self) -> List[Dict[str, Any]]:
        """Sample cultural content (proverbs, idioms) for testing."""
        self.load_data()
        
        samples = []
        
        # Look for longer, meaningful text that might be proverbs/idioms
        cultural_candidates = []
        for entry in self._translations:
            nko = entry.get("nko_text", "")
            en = entry.get("english", "")
            
            # Proverbs tend to be medium length with meaningful content
            words = nko.split()
            if 5 <= len(words) <= 20:
                cultural_candidates.append({
                    "nko_text": nko,
                    "english": en,
                    "french": entry.get("french", ""),
                    "type": "proverb_candidate",
                })
        
        if cultural_candidates:
            sampled = random.sample(
                cultural_candidates,
                min(self.config.cultural_samples, len(cultural_candidates))
            )
            samples.extend(sampled)
        
        return samples
    
    def create_dataset(self) -> TestDataset:
        """Create complete test dataset with all sample types."""
        print("\nCreating benchmark test dataset...")
        
        # Sample translations
        trans = self.sample_translations()
        print(f"  Translation samples: {sum(len(v) for v in trans.values())}")
        
        # Sample vocabulary
        vocab = self.sample_vocabulary()
        print(f"  Vocabulary samples: {len(vocab)}")
        
        # Sample script knowledge
        script = self.sample_script_knowledge()
        print(f"  Script knowledge samples: {len(script)}")
        
        # Sample cultural content
        cultural = self.sample_cultural()
        print(f"  Cultural samples: {len(cultural)}")
        
        dataset = TestDataset(
            nko_to_en=trans["nko_to_en"],
            nko_to_fr=trans["nko_to_fr"],
            en_to_nko=trans["en_to_nko"],
            fr_to_nko=trans["fr_to_nko"],
            vocabulary=vocab,
            script_samples=script,
            cultural_samples=cultural,
            compositional_samples=[],  # Will be filled by ComplexTestGenerator
        )
        
        print(f"  Total samples: {dataset.total_samples()}")
        return dataset


def create_test_dataset(config: Optional[BenchmarkConfig] = None) -> TestDataset:
    """Convenience function to create a test dataset."""
    sampler = TestDataSampler(config)
    return sampler.create_dataset()

