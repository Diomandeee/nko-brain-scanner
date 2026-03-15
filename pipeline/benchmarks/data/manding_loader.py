"""
Manding Language Unified Data Loader.

Combines data from multiple sources for comprehensive Manding language benchmarking:
- Supabase: Verified N'Ko vocabulary and translations
- nicolingua-0005: N'Ko-English-French trilingual corpus (130K+ pairs)
- Bayelemabaga: Bambara-French parallel corpus (47K+ pairs)
- unified_corpus: Additional Bambara-French pairs (56K+ pairs)
- Ankataa Dictionary: N'Ko dictionary entries

Supports cross-Manding evaluation across N'Ko, Bambara, Malinke, and Jula.
"""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class Language(Enum):
    """Manding language family and colonial languages."""
    NKO = "nko"           # N'Ko script
    BAMBARA = "bambara"   # Latin script (bam)
    MALINKE = "malinke"   # Latin script (man)
    JULA = "jula"         # Latin script (dyu)
    ENGLISH = "english"   # Latin script (eng)
    FRENCH = "french"     # Latin script (fra)
    ARABIC = "arabic"     # Arabic script (ara)


@dataclass
class VocabEntry:
    """A vocabulary entry with multilingual meanings."""
    id: str
    word: str
    word_normalized: str
    language: Language
    script: str  # "nko", "latin", "arabic"
    
    # Meanings
    meaning_primary: Optional[str] = None
    meanings: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    
    # Linguistic properties
    pos: Optional[str] = None  # Part of speech
    latin_transcription: Optional[str] = None
    
    # Cross-language equivalents
    nko_equivalent: Optional[str] = None
    bambara_equivalent: Optional[str] = None
    english_equivalent: Optional[str] = None
    french_equivalent: Optional[str] = None
    
    # Metadata
    frequency: int = 0
    cefr_level: Optional[str] = None
    difficulty: float = 0.5
    topics: List[str] = field(default_factory=list)
    is_verified: bool = False
    source: str = "unknown"


@dataclass
class TranslationPair:
    """A parallel translation pair between two languages."""
    id: str
    source_lang: Language
    target_lang: Language
    source_text: str
    target_text: str
    
    # Additional translations if available
    nko_text: Optional[str] = None
    bambara_text: Optional[str] = None
    english_text: Optional[str] = None
    french_text: Optional[str] = None
    
    # Metadata
    text_type: str = "sentence"  # word, phrase, sentence, paragraph
    domain: Optional[str] = None  # religious, technical, conversational, etc.
    formality: str = "neutral"
    confidence: float = 1.0
    is_verified: bool = False
    source: str = "unknown"


@dataclass
class CognatePair:
    """Cognate words across Manding variants."""
    id: str
    nko_form: Optional[str] = None
    bambara_form: Optional[str] = None
    malinke_form: Optional[str] = None
    jula_form: Optional[str] = None
    meaning: str = ""
    pos: Optional[str] = None
    is_verified: bool = False


@dataclass
class MandingTestSet:
    """Complete test dataset for Manding language benchmarking."""
    # N'Ko data
    nko_vocabulary: List[VocabEntry] = field(default_factory=list)
    nko_translations: List[TranslationPair] = field(default_factory=list)
    
    # Bambara data
    bambara_vocabulary: List[VocabEntry] = field(default_factory=list)
    bambara_french_pairs: List[TranslationPair] = field(default_factory=list)
    bambara_english_pairs: List[TranslationPair] = field(default_factory=list)
    
    # Cross-language data
    nko_bambara_cognates: List[CognatePair] = field(default_factory=list)
    cross_manding_pairs: List[TranslationPair] = field(default_factory=list)
    
    # Metadata
    total_samples: int = 0
    sources: List[str] = field(default_factory=list)


class MandingDataLoader:
    """
    Unified loader for all Manding language data sources.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        self.project_root = project_root
        self.training_dir = project_root / "training"
        self.nko_data_dir = project_root / "nko" / "data"
        
        # Data paths
        self.nicolingua_dir = self.training_dir / "data" / "nicolingua" / "data"
        self.nicolingua_exports = self.training_dir / "data" / "exports" / "nicolingua"
        self.bayelemabaga_dir = self.nko_data_dir / "bayelemabaga" / "bayelemabaga"
        self.unified_corpus_dir = self.nko_data_dir / "unified_corpus"
        self.dictionary_path = self.training_dir / "data" / "dictionary"
        
    def load_nicolingua_translations(self, limit: Optional[int] = None) -> List[TranslationPair]:
        """
        Load N'Ko-English-French translations from nicolingua corpus.
        
        Args:
            limit: Maximum number of pairs to load
            
        Returns:
            List of TranslationPair objects
        """
        pairs = []
        
        # Try exported JSONL first (preprocessed)
        jsonl_path = self.nicolingua_exports / "nicolingua_translations.jsonl"
        if jsonl_path.exists():
            print(f"  Loading from {jsonl_path}...")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        data = json.loads(line)
                        nko_text = data.get("nko_text", "")
                        english_text = data.get("english", "")
                        french_text = data.get("french", "")
                        
                        # Use English if available, otherwise French
                        target_text = english_text if english_text else french_text
                        target_lang = Language.ENGLISH if english_text else Language.FRENCH
                        
                        pair = TranslationPair(
                            id=f"nicolingua_{i}",
                            source_lang=Language.NKO,
                            target_lang=target_lang,
                            source_text=nko_text,
                            target_text=target_text,
                            nko_text=nko_text,
                            english_text=english_text if english_text else None,
                            french_text=french_text if french_text else None,
                            text_type=data.get("text_type", "sentence"),
                            source="nicolingua-0005",
                            is_verified=True,
                        )
                        if pair.source_text and pair.target_text:
                            pairs.append(pair)
                    except json.JSONDecodeError:
                        continue
            
            print(f"    Loaded {len(pairs)} N'Ko translation pairs")
            return pairs
        
        # Fall back to raw parallel files
        print(f"  Loading from raw nicolingua files...")
        parallel_files = self._find_nicolingua_parallel_files()
        
        for nko_file, eng_file, fra_file in parallel_files:
            if limit and len(pairs) >= limit:
                break
                
            try:
                with open(nko_file, 'r', encoding='utf-8') as f:
                    nko_lines = f.readlines()
                with open(eng_file, 'r', encoding='utf-8') as f:
                    eng_lines = f.readlines()
                fra_lines = []
                if fra_file and fra_file.exists():
                    with open(fra_file, 'r', encoding='utf-8') as f:
                        fra_lines = f.readlines()
                
                for i, (nko, eng) in enumerate(zip(nko_lines, eng_lines)):
                    if limit and len(pairs) >= limit:
                        break
                    fra = fra_lines[i].strip() if i < len(fra_lines) else None
                    
                    pair = TranslationPair(
                        id=f"nicolingua_{nko_file.stem}_{i}",
                        source_lang=Language.NKO,
                        target_lang=Language.ENGLISH,
                        source_text=nko.strip(),
                        target_text=eng.strip(),
                        nko_text=nko.strip(),
                        english_text=eng.strip(),
                        french_text=fra,
                        source="nicolingua-0005",
                        is_verified=True,
                    )
                    if pair.source_text and pair.target_text:
                        pairs.append(pair)
                        
            except Exception as e:
                print(f"    Warning: Error reading {nko_file}: {e}")
                
        print(f"    Loaded {len(pairs)} N'Ko translation pairs from raw files")
        return pairs
    
    def _find_nicolingua_parallel_files(self) -> List[Tuple[Path, Path, Optional[Path]]]:
        """Find parallel file triplets in nicolingua data."""
        triplets = []
        
        if not self.nicolingua_dir.exists():
            return triplets
            
        nko_files = list(self.nicolingua_dir.glob("*.nqo_Nkoo"))
        
        for nko_file in nko_files:
            base_name = nko_file.stem.replace(".nqo_Nkoo", "")
            eng_file = self.nicolingua_dir / f"{base_name}.eng_Latn"
            fra_file = self.nicolingua_dir / f"{base_name}.fra_Latn"
            
            if eng_file.exists():
                triplets.append((nko_file, eng_file, fra_file if fra_file.exists() else None))
                
        return triplets
    
    def load_nicolingua_vocabulary(self, limit: Optional[int] = None) -> List[VocabEntry]:
        """Load N'Ko vocabulary from nicolingua exports."""
        vocab = []
        
        jsonl_path = self.nicolingua_exports / "nicolingua_vocabulary.jsonl"
        if jsonl_path.exists():
            print(f"  Loading vocabulary from {jsonl_path}...")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        data = json.loads(line)
                        entry = VocabEntry(
                            id=f"nicolingua_vocab_{i}",
                            word=data.get("word", ""),
                            word_normalized=data.get("word_normalized", data.get("word", "")),
                            language=Language.NKO,
                            script="nko",
                            meaning_primary=data.get("meaning"),
                            frequency=data.get("frequency", 1),
                            source="nicolingua-0005",
                            is_verified=True,
                        )
                        if entry.word:
                            vocab.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            print(f"    Loaded {len(vocab)} N'Ko vocabulary entries")
        
        return vocab
    
    def load_bayelemabaga(self, split: str = "all", limit: Optional[int] = None) -> List[TranslationPair]:
        """
        Load Bambara-French pairs from Bayelemabaga corpus.
        
        Args:
            split: "train", "valid", "test", or "all"
            limit: Maximum number of pairs to load
            
        Returns:
            List of TranslationPair objects
        """
        pairs = []
        
        if split == "all":
            splits = ["train", "valid", "test"]
        else:
            splits = [split]
        
        for s in splits:
            tsv_path = self.bayelemabaga_dir / s / f"{s if s != 'valid' else 'dev'}.tsv"
            if not tsv_path.exists():
                print(f"    Warning: {tsv_path} not found")
                continue
                
            print(f"  Loading Bayelemabaga {s} split from {tsv_path}...")
            
            with open(tsv_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and len(pairs) >= limit:
                        break
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        bam_text = parts[0].strip()
                        fra_text = parts[1].strip()
                        
                        pair = TranslationPair(
                            id=f"bayelemabaga_{s}_{i}",
                            source_lang=Language.BAMBARA,
                            target_lang=Language.FRENCH,
                            source_text=bam_text,
                            target_text=fra_text,
                            bambara_text=bam_text,
                            french_text=fra_text,
                            text_type="sentence",
                            source="bayelemabaga",
                            is_verified=True,
                        )
                        if pair.source_text and pair.target_text:
                            pairs.append(pair)
        
        print(f"    Loaded {len(pairs)} Bambara-French pairs from Bayelemabaga")
        return pairs
    
    def load_unified_corpus(self, split: str = "all", limit: Optional[int] = None) -> List[TranslationPair]:
        """
        Load Bambara-French pairs from unified corpus.
        
        Args:
            split: "train", "valid", "test", or "all"
            limit: Maximum number of pairs to load
            
        Returns:
            List of TranslationPair objects
        """
        pairs = []
        
        if split == "all":
            tsv_path = self.unified_corpus_dir / "all.tsv"
        else:
            tsv_path = self.unified_corpus_dir / f"{split}.tsv"
            
        if not tsv_path.exists():
            print(f"    Warning: {tsv_path} not found")
            return pairs
            
        print(f"  Loading unified corpus from {tsv_path}...")
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and len(pairs) >= limit:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    bam_text = parts[0].strip()
                    fra_text = parts[1].strip()
                    
                    pair = TranslationPair(
                        id=f"unified_{i}",
                        source_lang=Language.BAMBARA,
                        target_lang=Language.FRENCH,
                        source_text=bam_text,
                        target_text=fra_text,
                        bambara_text=bam_text,
                        french_text=fra_text,
                        text_type="sentence",
                        source="unified_corpus",
                        is_verified=True,
                    )
                    if pair.source_text and pair.target_text:
                        pairs.append(pair)
        
        print(f"    Loaded {len(pairs)} pairs from unified corpus")
        return pairs
    
    def load_ankataa_dictionary(self, limit: Optional[int] = None) -> List[VocabEntry]:
        """Load N'Ko vocabulary from Ankataa dictionary."""
        vocab = []
        
        # Find the latest dictionary file
        dict_files = list(self.dictionary_path.glob("ankataa_dictionary_*.json"))
        if not dict_files:
            print("    Warning: No Ankataa dictionary found")
            return vocab
            
        dict_path = sorted(dict_files)[-1]  # Latest file
        print(f"  Loading Ankataa dictionary from {dict_path}...")
        
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            entries = data if isinstance(data, list) else data.get("entries", [])
            
            for i, entry in enumerate(entries):
                if limit and i >= limit:
                    break
                    
                vocab_entry = VocabEntry(
                    id=f"ankataa_{i}",
                    word=entry.get("nkoText", entry.get("word", "")),
                    word_normalized=entry.get("nkoText", entry.get("word", "")),
                    language=Language.NKO,
                    script="nko",
                    meaning_primary=entry.get("definition", entry.get("meaning_primary")),
                    latin_transcription=entry.get("latin", entry.get("latinText")),
                    pos=entry.get("pos", entry.get("partOfSpeech")),
                    english_equivalent=entry.get("english"),
                    french_equivalent=entry.get("french"),
                    source="ankataa_dictionary",
                    is_verified=True,
                )
                if vocab_entry.word:
                    vocab.append(vocab_entry)
                    
            print(f"    Loaded {len(vocab)} entries from Ankataa dictionary")
            
        except Exception as e:
            print(f"    Error loading dictionary: {e}")
            
        return vocab
    
    def load_bidirectional_training_data(self, limit: Optional[int] = None) -> List[TranslationPair]:
        """Load Bambara-English pairs from bidirectional training data."""
        pairs = []
        
        json_path = self.nko_data_dir / "training_data_bidirectional.json"
        if not json_path.exists():
            return pairs
            
        print(f"  Loading bidirectional training data from {json_path}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            entries = data.get("pairs", [])
            
            for i, entry in enumerate(entries):
                if limit and i >= limit:
                    break
                    
                pair = TranslationPair(
                    id=f"bidirectional_{i}",
                    source_lang=Language.BAMBARA,
                    target_lang=Language.ENGLISH,
                    source_text=entry.get("bambara", ""),
                    target_text=entry.get("english", ""),
                    bambara_text=entry.get("bambara"),
                    english_text=entry.get("english"),
                    text_type="phrase",
                    source="bidirectional_training",
                    is_verified=True,
                )
                if pair.source_text and pair.target_text:
                    pairs.append(pair)
                    
            print(f"    Loaded {len(pairs)} Bambara-English pairs")
            
        except Exception as e:
            print(f"    Error loading bidirectional data: {e}")
            
        return pairs
    
    def generate_cognate_pairs(
        self,
        nko_vocab: List[VocabEntry],
        bambara_pairs: List[TranslationPair],
        limit: int = 200,
    ) -> List[CognatePair]:
        """
        Generate N'Ko-Bambara cognate pairs by matching meanings.
        
        This is an approximation - real cognates would need linguistic analysis.
        We match based on similar French translations.
        """
        cognates = []
        
        # Build Bambara vocab from translation pairs
        bambara_french_map: Dict[str, str] = {}
        for pair in bambara_pairs:
            if pair.bambara_text and pair.french_text:
                # Simple: use first word as key
                bam_word = pair.bambara_text.split()[0] if pair.bambara_text else ""
                if bam_word and len(bam_word) > 2:
                    bambara_french_map[pair.french_text.lower()[:30]] = bam_word
        
        # Match N'Ko vocab to Bambara via French
        for vocab in nko_vocab[:limit * 2]:  # Sample more to get enough matches
            if len(cognates) >= limit:
                break
                
            french = vocab.french_equivalent or vocab.meaning_primary
            if french:
                french_key = french.lower()[:30]
                if french_key in bambara_french_map:
                    cognate = CognatePair(
                        id=f"cognate_{len(cognates)}",
                        nko_form=vocab.word,
                        bambara_form=bambara_french_map[french_key],
                        meaning=vocab.meaning_primary or french,
                        pos=vocab.pos,
                        is_verified=False,  # These are inferred, not verified
                    )
                    cognates.append(cognate)
        
        print(f"    Generated {len(cognates)} cognate pairs")
        return cognates
    
    async def load_supabase_data(
        self,
        vocab_limit: int = 1000,
        trans_limit: int = 1000,
    ) -> Tuple[List[VocabEntry], List[TranslationPair]]:
        """Load ALL data from Supabase (not just verified)."""
        try:
            from .supabase_loader import SupabaseLoader
            
            async with SupabaseLoader() as loader:
                if not loader.is_configured:
                    print("    Supabase not configured")
                    return [], []
                    
                print("  Loading from Supabase (all entries, no filters)...")
                
                # Get ALL vocabulary, not just verified
                supabase_vocab = await loader.fetch_verified_vocabulary(
                    limit=vocab_limit,
                    verified_only=False,
                )
                # Get ALL translations, not just gold
                supabase_trans = await loader.fetch_verified_translations(
                    limit=trans_limit,
                    validated_only=False,
                    quality_tier=None,
                )
                
                # Convert to our dataclasses
                vocab = []
                for v in supabase_vocab:
                    entry = VocabEntry(
                        id=v.id,
                        word=v.word,
                        word_normalized=v.word_normalized,
                        language=Language.NKO,
                        script="nko",
                        meaning_primary=v.meaning_primary,
                        meanings=v.meanings,
                        definition=v.definition,
                        pos=v.pos,
                        latin_transcription=v.latin,
                        frequency=v.frequency,
                        cefr_level=v.cefr_level,
                        is_verified=v.is_dictionary_verified,
                        source="supabase",
                    )
                    vocab.append(entry)
                
                trans = []
                for t in supabase_trans:
                    pair = TranslationPair(
                        id=t.id,
                        source_lang=Language.NKO,
                        target_lang=Language.ENGLISH,
                        source_text=t.nko_text,
                        target_text=t.english_text or "",
                        nko_text=t.nko_text,
                        english_text=t.english_text,
                        french_text=t.french_text,
                        text_type=t.text_type or "sentence",
                        confidence=t.confidence,
                        is_verified=t.validated,
                        source="supabase",
                    )
                    if pair.source_text:
                        trans.append(pair)
                
                print(f"    Loaded {len(vocab)} vocab, {len(trans)} translations from Supabase")
                return vocab, trans
                
        except ImportError:
            print("    Warning: Supabase loader not available")
            return [], []
        except Exception as e:
            print(f"    Warning: Supabase error: {e}")
            return [], []
    
    def load_supabase_data_sync(self) -> Tuple[List[VocabEntry], List[TranslationPair]]:
        """Synchronous wrapper for loading Supabase data."""
        import asyncio
        try:
            return asyncio.run(self.load_supabase_data())
        except Exception:
            return [], []
    
    def load_all(
        self,
        nko_translation_limit: int = 10000,  # Increased from 2000
        bambara_limit: int = 5000,  # Increased from 1000
        vocab_limit: int = 2000,  # Increased from 500
        include_supabase: bool = True,
    ) -> MandingTestSet:
        """
        Load all available Manding language data.
        
        Args:
            nko_translation_limit: Max N'Ko translations to load
            bambara_limit: Max Bambara pairs to load
            vocab_limit: Max vocabulary entries to load
            include_supabase: Whether to include Supabase data
            
        Returns:
            Complete MandingTestSet with all data sources
        """
        print("Loading Manding language data from all sources...")
        
        testset = MandingTestSet()
        
        # N'Ko data from nicolingua
        testset.nko_translations = self.load_nicolingua_translations(limit=nko_translation_limit)
        testset.nko_vocabulary = self.load_nicolingua_vocabulary(limit=vocab_limit)
        
        # Add Ankataa dictionary
        ankataa_vocab = self.load_ankataa_dictionary(limit=vocab_limit)
        testset.nko_vocabulary.extend(ankataa_vocab)
        
        # Bambara data
        testset.bambara_french_pairs = self.load_bayelemabaga(split="test", limit=bambara_limit)
        if len(testset.bambara_french_pairs) < bambara_limit:
            additional = self.load_unified_corpus(split="test", limit=bambara_limit - len(testset.bambara_french_pairs))
            testset.bambara_french_pairs.extend(additional)
        
        # Bambara-English
        testset.bambara_english_pairs = self.load_bidirectional_training_data(limit=200)
        
        # Build vocabulary from Bambara pairs
        bambara_words = set()
        for pair in testset.bambara_french_pairs[:500]:
            for word in pair.bambara_text.split() if pair.bambara_text else []:
                if len(word) > 2:
                    bambara_words.add(word)
        
        testset.bambara_vocabulary = [
            VocabEntry(
                id=f"bam_vocab_{i}",
                word=word,
                word_normalized=word.lower(),
                language=Language.BAMBARA,
                script="latin",
                source="extracted",
            )
            for i, word in enumerate(list(bambara_words)[:vocab_limit])
        ]
        
        # Supabase data
        if include_supabase:
            supabase_vocab, supabase_trans = self.load_supabase_data_sync()
            testset.nko_vocabulary.extend(supabase_vocab)
            testset.nko_translations.extend(supabase_trans)
        
        # Generate cross-Manding cognates
        testset.nko_bambara_cognates = self.generate_cognate_pairs(
            testset.nko_vocabulary,
            testset.bambara_french_pairs,
            limit=200,
        )
        
        # Calculate total samples
        testset.total_samples = (
            len(testset.nko_translations) +
            len(testset.nko_vocabulary) +
            len(testset.bambara_french_pairs) +
            len(testset.bambara_english_pairs) +
            len(testset.bambara_vocabulary) +
            len(testset.nko_bambara_cognates)
        )
        
        testset.sources = [
            "nicolingua-0005",
            "bayelemabaga",
            "unified_corpus",
            "ankataa_dictionary",
            "supabase" if include_supabase else None,
            "bidirectional_training",
        ]
        testset.sources = [s for s in testset.sources if s]
        
        print(f"\nManding Test Set Summary:")
        print(f"  N'Ko translations: {len(testset.nko_translations)}")
        print(f"  N'Ko vocabulary: {len(testset.nko_vocabulary)}")
        print(f"  Bambara-French pairs: {len(testset.bambara_french_pairs)}")
        print(f"  Bambara-English pairs: {len(testset.bambara_english_pairs)}")
        print(f"  Bambara vocabulary: {len(testset.bambara_vocabulary)}")
        print(f"  Cognate pairs: {len(testset.nko_bambara_cognates)}")
        print(f"  Total samples: {testset.total_samples}")
        print(f"  Sources: {', '.join(testset.sources)}")
        
        return testset


def create_manding_test_set(
    nko_limit: int = 10000,  # Increased for comprehensive evaluation
    bambara_limit: int = 5000,
    vocab_limit: int = 2000,
) -> MandingTestSet:
    """
    Convenience function to create a test set.
    
    Default limits are set for comprehensive evaluation using the full
    nicolingua dataset (130K pairs), Supabase, and Bayelemabaga.
    """
    loader = MandingDataLoader()
    return loader.load_all(
        nko_translation_limit=nko_limit,
        bambara_limit=bambara_limit,
        vocab_limit=vocab_limit,
        include_supabase=True,
    )


def sample_translation_pairs(
    pairs: List[TranslationPair],
    n: int,
    stratify_by: str = "text_type",
) -> List[TranslationPair]:
    """Sample translation pairs with optional stratification."""
    if n >= len(pairs):
        return pairs
    
    if stratify_by == "text_type":
        # Group by text type
        by_type: Dict[str, List[TranslationPair]] = {}
        for pair in pairs:
            key = pair.text_type or "unknown"
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(pair)
        
        # Sample proportionally
        result = []
        per_type = max(1, n // len(by_type))
        for type_pairs in by_type.values():
            sample_size = min(per_type, len(type_pairs))
            result.extend(random.sample(type_pairs, sample_size))
        
        # Fill remaining
        if len(result) < n:
            remaining = [p for p in pairs if p not in result]
            result.extend(random.sample(remaining, min(n - len(result), len(remaining))))
        
        return result[:n]
    else:
        return random.sample(pairs, n)

