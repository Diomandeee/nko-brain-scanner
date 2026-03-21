#!/usr/bin/env python3
"""
Dictionary-Augmented Bambara Translator
========================================
Replaces blind LLM zero-shot translation with dictionary-grounded approach.

Pipeline:
  1. Tokenize Latin Bambara into words
  2. Morphological decomposition (handle n-, a-, ka, etc.)
  3. Dictionary lookup for each word/morpheme
  4. Inject glosses into LLM prompt for grounded translation
  5. Return translation + confidence (dictionary coverage %)

The dictionary transforms LLM hallucination into guided assembly.
Without dictionary: "n fa ani a kɔrɔkɛ" → "don't be afraid" (wrong)
With dictionary:    "n fa ani a kɔrɔkɛ" + glosses → "my father and my older brother" (correct)
"""

import json
import re
from pathlib import Path
from typing import Optional


# ── Bambara Grammar Particles ────────────────────────────────────────

# Common prefixes/suffixes that modify word meaning
POSSESSIVE_PRONOUNS = {
    "n": "my", "i": "your", "a": "his/her/its",
    "an": "our", "aw": "your (pl)", "u": "their",
}

POSTPOSITIONS = {
    "la": "in/at", "na": "in/to", "ma": "to/for",
    "kan": "on/about", "kɔnɔ": "inside", "kɔfɛ": "behind",
    "ɲɛ": "in front of", "kɔrɔ": "under/meaning",
}

CONJUNCTIONS = {
    "ani": "and", "walima": "or", "nka": "but",
    "ka": "to (infinitive)", "bɛ": "is/are (present)",
    "tɛ": "is not/are not", "ye": "is/did (past)",
    "ne": "I", "ale": "he/she (emphatic)", "olu": "they",
}

GREETINGS = {
    "i ni ce": "hello",
    "i ni sogoma": "good morning",
    "i ni tile": "good afternoon",
    "i ni wula": "good evening",
    "i ka kɛnɛ": "how are you",
    "nba": "thank you",
    "nse": "goodbye",
    "aw ni ce": "hello everyone",
    "aw ni sogoma": "good morning everyone",
    "aw ni tile": "good afternoon everyone",
    "aw ni wula": "good evening everyone",
    "aw ni baara": "well done",
}

COMMON_WORDS = {
    # Merge particles + very common words for baseline coverage
    **POSSESSIVE_PRONOUNS,
    **POSTPOSITIONS,
    **CONJUNCTIONS,
    # Common nouns
    "fa": "father", "ba": "mother", "den": "child",
    "cɛ": "man", "muso": "woman", "mɔgɔ": "person",
    "so": "house", "dugu": "village/city", "ji": "water",
    "dumuni": "food", "wari": "money", "baara": "work",
    "sira": "road/path", "waati": "time", "tile": "day/sun",
    "su": "night", "san": "year/rain", "kalo": "month/moon",
    # Common verbs
    "taa": "go", "naa": "come", "bɔ": "leave/exit",
    "don": "enter", "fɔ": "say/tell", "ye": "see",
    "dɔn": "know", "kɛ": "do/make", "di": "give",
    "ta": "take", "sɔrɔ": "get/obtain", "se": "arrive/can",
    # Family
    "kɔrɔkɛ": "older brother", "kɔrɔmuso": "older sister",
    "dɔgɔkɛ": "younger brother", "dɔgɔmuso": "younger sister",
    "facɛ": "father", "bamuso": "mother",
    # Numbers
    "kelen": "one", "fila": "two", "saba": "three",
    "naani": "four", "duuru": "five",
    # Adjectives
    "ɲuman": "good", "jugu": "bad/enemy", "kɔrɔ": "old",
    "kura": "new", "jan": "tall/long", "surun": "short",
    "bon": "big", "dɔgɔ": "small",
}


class BambaraDictionary:
    """Bambara → English dictionary with morphological awareness."""

    # Default paths for data files
    DEFAULT_DICT = str(Path(__file__).parent.parent / "data" / "bambara_english_dict.json")
    DEFAULT_PARALLEL = str(Path(__file__).parent.parent / "data" / "parallel_corpus.jsonl")
    DEFAULT_ANKATAA = str(Path(__file__).parent.parent / "pipeline" / "data" / "dictionary" / "ankataa_dictionary_20260319.json")

    def __init__(self, dict_path: Optional[str] = None):
        self.entries = {}
        self.sentence_cache = {}  # IPA sentence → English

        # Load extended dictionary first (lowest priority)
        dict_path = dict_path or self.DEFAULT_DICT
        if dict_path and Path(dict_path).exists():
            self._load_dictionary(dict_path)

        # Load Ankataa dictionary (medium priority — 1,977 entries)
        # Check default path first, then same directory as this file
        ankataa_path = self.DEFAULT_ANKATAA
        if not Path(ankataa_path).exists():
            ankataa_path = str(Path(__file__).parent / "ankataa_dictionary_20260319.json")
        if Path(ankataa_path).exists():
            print(f"Loading Ankataa dictionary...")
            self._load_dictionary(ankataa_path)

        # Load parallel corpus for sentence-level lookup
        if Path(self.DEFAULT_PARALLEL).exists():
            self._load_parallel_corpus(self.DEFAULT_PARALLEL)

        # COMMON_WORDS last — highest priority (curated grammar particles)
        self.entries.update(COMMON_WORDS)

        # Pre-compute normalized lookup caches (O(1) instead of O(n))
        self.normalized_entries = {}
        self.normalized_sentences = {}
        self._build_normalized_cache()

    def _load_dictionary(self, path):
        """Load dictionary JSON (IPA→English, Bambara→English, or Ankataa format)."""
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                # Ankataa format: {"word": "...", "definitions_en": [...]}
                if "definitions_en" in entry and entry["definitions_en"]:
                    word = entry.get("word", "").strip().lower()
                    # Join definitions, take first one for primary meaning
                    eng = entry["definitions_en"][0]
                    if word and eng:
                        self.entries[word] = eng
                    continue
                # Legacy format: {"bambara": "...", "english": "..."}
                word = entry.get("bambara", entry.get("ipa", "")).strip().lower()
                eng = (entry.get("english") or "").strip()
                if word and eng:
                    self.entries[word] = eng
        elif isinstance(data, dict):
            for word, definition in data.items():
                if isinstance(definition, str):
                    self.entries[word.lower()] = definition
                elif isinstance(definition, dict):
                    self.entries[word.lower()] = definition.get("english", str(definition))

        print(f"Dictionary loaded: {len(self.entries)} entries")

    def _load_parallel_corpus(self, path):
        """Load parallel corpus for sentence-level fuzzy matching."""
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ipa = entry.get("ipa", "").strip().lower()
                eng = entry.get("english", "").strip()
                if ipa and eng:
                    self.sentence_cache[ipa] = eng
        print(f"Parallel corpus loaded: {len(self.sentence_cache)} sentence pairs")

    def _build_normalized_cache(self):
        """Pre-compute normalized keys for O(1) fuzzy lookup."""
        for key, val in self.entries.items():
            norm = key.replace("ɛ", "e").replace("ɔ", "o").replace("ɲ", "ny").replace("ŋ", "ng")
            self.normalized_entries[norm] = val
        for key, val in self.sentence_cache.items():
            norm = key.replace("ɛ", "e").replace("ɔ", "o").replace("ɲ", "ny").replace("ŋ", "ng")
            self.normalized_sentences[norm] = val

    def lookup(self, word):
        """Look up a word, trying exact match then morphological decomposition."""
        w = word.lower().strip()

        # Exact match
        if w in self.entries:
            return self.entries[w]

        # Try without tone marks (ɛ→e, ɔ→o for fuzzy match) — O(1) via pre-computed cache
        normalized = w.replace("ɛ", "e").replace("ɔ", "o").replace("ɲ", "ny").replace("ŋ", "ng")
        if normalized in self.normalized_entries:
            return self.normalized_entries[normalized]

        # Try morphological decomposition: prefix + stem
        # Bambara possessive/pronoun prefixes: n (my), i (your), a (his/her)
        for prefix in ["n", "i", "a"]:
            if w.startswith(prefix) and len(w) > len(prefix):
                stem = w[len(prefix):]
                if stem in self.entries:
                    prefix_meaning = POSSESSIVE_PRONOUNS.get(prefix, prefix)
                    return f"{prefix_meaning} {self.entries[stem]}"
                # Also try with normalized stem — O(1) via pre-computed cache
                stem_norm = stem.replace("ɛ", "e").replace("ɔ", "o").replace("ɲ", "ny").replace("ŋ", "ng")
                if stem_norm in self.normalized_entries:
                    prefix_meaning = POSSESSIVE_PRONOUNS.get(prefix, prefix)
                    return f"{prefix_meaning} {self.normalized_entries[stem_norm]}"

        # Try stripping common suffixes
        for suffix in ["w", "lu", "ya", "li", "ni", "ba", "nin", "den", "la", "na"]:
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                stem = w[:-len(suffix)]
                if stem in self.entries:
                    return f"{self.entries[stem]} ({suffix}-suffix)"

        return None

    def gloss_sentence(self, text):
        """Gloss each word in a sentence. Returns list of (word, gloss_or_None)."""
        words = text.strip().split()
        glossed = []
        for word in words:
            gloss = self.lookup(word)
            glossed.append((word, gloss))
        return glossed

    def coverage(self, text):
        """What % of words have dictionary entries."""
        glossed = self.gloss_sentence(text)
        if not glossed:
            return 0.0
        found = sum(1 for _, g in glossed if g is not None)
        return found / len(glossed)


class NLLBTranslator:
    """NLLB-200 translation backend for Bambara → English/French."""

    # NLLB language codes
    LANG_CODES = {
        "English": "eng_Latn",
        "French": "fra_Latn",
        "Bambara": "bam_Latn",
    }

    def __init__(self, model_name=None):
        self._model = None
        self._tokenizer = None
        # Prefer fine-tuned model if available, otherwise fall back to stock
        if model_name:
            self._model_name = model_name
        else:
            finetuned = str(Path(__file__).parent / "nllb_bambara_merged")
            if Path(finetuned).exists() and (Path(finetuned) / "config.json").exists():
                self._model_name = finetuned
            else:
                self._model_name = "facebook/nllb-200-distilled-600M"

    def _load(self):
        """Lazy-load NLLB model on first use."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            print(f"Loading NLLB model ({self._model_name})...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(device)
            self._device = device
            print(f"NLLB loaded on {device}")
        except Exception as e:
            print(f"NLLB load failed: {e}")
            self._model = None

    def translate(self, text, src="Bambara", tgt="English"):
        """Translate text using NLLB-200."""
        self._load()
        if self._model is None:
            return None

        src_code = self.LANG_CODES.get(src, "bam_Latn")
        tgt_code = self.LANG_CODES.get(tgt, "eng_Latn")

        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        import torch
        tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=256,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)


class BambaraTranslator:
    """Dictionary-augmented Bambara → English translator with NLLB backend."""

    def __init__(self, dict_path: Optional[str] = None, ollama_url="http://localhost:11434", model="qwen2.5:14b", use_nllb=True):
        self.dictionary = BambaraDictionary(dict_path)
        self.ollama_url = ollama_url
        self.model = model
        self.nllb = NLLBTranslator() if use_nllb else None
        self._cache = {}  # Translation result cache

    def translate(self, bambara_text, target="English"):
        """Translate Bambara text using tiered approach: corpus > dictionary > NLLB > Ollama."""
        # Check translation cache first
        cache_key = (bambara_text.lower().strip(), target)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check greetings/phrases first (exact multi-word matches)
        normalized = bambara_text.lower().strip()
        if normalized in GREETINGS:
            result = {
                "translation": GREETINGS[normalized],
                "method": "greeting",
                "confidence": 1.0,
                "glosses": [(bambara_text, GREETINGS[normalized])],
            }
            self._cache[cache_key] = result
            return result

        # Try sentence-level match (parallel corpus) — O(1) via pre-computed cache
        match_key = normalized.replace("ɔ", "o").replace("ɛ", "e").replace("ɲ", "ny").replace("ŋ", "ng")
        # Check exact match first
        if normalized in self.dictionary.sentence_cache:
            result = {
                "translation": self.dictionary.sentence_cache[normalized],
                "method": "parallel_corpus",
                "confidence": 1.0,
                "glosses": [(bambara_text, self.dictionary.sentence_cache[normalized])],
            }
            self._cache[cache_key] = result
            return result
        # Check normalized match
        if match_key in self.dictionary.normalized_sentences:
            cached_eng = self.dictionary.normalized_sentences[match_key]
            result = {
                "translation": cached_eng,
                "method": "parallel_corpus",
                "confidence": 1.0,
                "glosses": [(bambara_text, cached_eng)],
            }
            self._cache[cache_key] = result
            return result

        glossed = self.dictionary.gloss_sentence(bambara_text)
        coverage = self.dictionary.coverage(bambara_text)

        # Detect if sentence has verbs/complex structure
        verb_markers = {"bɛ", "tɛ", "ye", "ka", "kɛ", "na", "sera", "bɛna"}
        words = set(bambara_text.lower().split())
        has_verb = bool(words & verb_markers) or len(words) > 4

        # Short noun phrases with high coverage: dictionary assembly
        if coverage >= 0.8 and not has_verb:
            simple = self._assemble_from_glosses(glossed)
            if simple:
                result = {
                    "translation": simple,
                    "method": "dictionary",
                    "confidence": coverage,
                    "glosses": glossed,
                }
                self._cache[cache_key] = result
                return result

        # NLLB-200 for full sentences (purpose-built for low-resource languages)
        if self.nllb:
            nllb_result = self.nllb.translate(bambara_text, tgt=target)
            if nllb_result:
                result = {
                    "translation": nllb_result,
                    "method": "nllb",
                    "confidence": coverage,
                    "glosses": glossed,
                }
                self._cache[cache_key] = result
                return result

        # High-coverage fallback: dictionary assembly even for verb sentences
        if coverage >= 0.8:
            simple = self._assemble_from_glosses(glossed)
            if simple:
                result = {
                    "translation": simple,
                    "method": "dictionary",
                    "confidence": coverage,
                    "glosses": glossed,
                }
                self._cache[cache_key] = result
                return result

        # Final fallback: Ollama LLM with dictionary context
        gloss_lines = [f"  {w} = {g}" for w, g in glossed if g]
        if gloss_lines:
            prompt = (
                f"Translate this Bambara sentence to {target}. "
                f"Use the dictionary glosses below to ensure accuracy.\n\n"
                f"Bambara: {bambara_text}\n\n"
                f"Dictionary glosses:\n" + "\n".join(gloss_lines) + "\n\n"
                f"Translation:"
            )
        else:
            prompt = (
                f"Translate this Bambara sentence to {target}. "
                f"Only output the translation.\n\n{bambara_text}"
            )

        translation = self._query_ollama(prompt)

        result = {
            "translation": translation,
            "method": "dictionary+llm" if gloss_lines else "llm_only",
            "confidence": coverage,
            "glosses": glossed,
        }
        self._cache[cache_key] = result
        return result

    def _assemble_from_glosses(self, glossed):
        """Try to assemble a natural English sentence from glosses alone."""
        parts = []
        for i, (word, gloss) in enumerate(glossed):
            if gloss is None:
                return None  # Can't do pure dictionary if any word is missing
            meaning = gloss.split(";")[0].split("(")[0].strip()
            if "/" in meaning:
                options = [m.strip() for m in meaning.split("/")]
                meaning = options[0]
            parts.append(meaning)

        if not parts:
            return None

        return " ".join(parts)

    def _query_ollama(self, prompt):
        """Query local Ollama for translation (last resort fallback)."""
        try:
            import urllib.request
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
                return result.get("response", "").strip()
        except Exception as e:
            return f"[Translation unavailable: {e}]"


# ── Improved N'Ko → Latin Bridge ─────────────────────────────────────
# Uses the correct mappings from cross-script-bridge/core/nko.py

NKO_TO_LATIN_CORRECT = {
    # Vowels — CORRECT Bambara mapping
    '\u07CA': 'a',   # ߊ
    '\u07CB': 'e',   # ߋ (was 'ee' in demo — wrong)
    '\u07CC': 'i',   # ߌ
    '\u07CD': 'ɛ',   # ߍ (open-e)
    '\u07CE': 'u',   # ߎ
    '\u07CF': 'ɔ',   # ߏ (open-o — was 'o' in demo — wrong)
    '\u07D0': 'ɛ',   # ߐ (open-e in Bambara — bridge says 'ə' but Bambara has no schwa)

    # Consonants
    '\u07D1': 'a',   # ߑ Dagbamma (rare)
    '\u07D2': 'ŋ',   # ߒ N (the N in N'Ko!)
    '\u07D3': 'b',   # ߓ
    '\u07D4': 'p',   # ߔ
    '\u07D5': 't',   # ߕ
    '\u07D6': 'j',   # ߖ
    '\u07D7': 'c',   # ߗ
    '\u07D8': 'd',   # ߘ
    '\u07D9': 'r',   # ߙ
    '\u07DA': 'rr',  # ߚ (trilled)
    '\u07DB': 's',   # ߛ
    '\u07DC': 'gb',  # ߜ
    '\u07DD': 'f',   # ߝ
    '\u07DE': 'k',   # ߞ
    '\u07DF': 'l',   # ߟ
    '\u07E0': 'n',   # ߠ Na Woloso
    '\u07E1': 'm',   # ߡ
    '\u07E2': 'ɲ',   # ߢ
    '\u07E3': 'n',   # ߣ
    '\u07E4': 'h',   # ߤ
    '\u07E5': 'w',   # ߥ
    '\u07E6': 'y',   # ߦ
    '\u07E7': 'ŋ',   # ߧ
}

NKO_TONE_MARKS = {'\u07EB', '\u07EC', '\u07ED', '\u07EE', '\u07EF'}
NKO_NASALS = {'\u07F2', '\u07F3'}


def nko_to_latin_correct(nko_text):
    """Convert N'Ko text to Latin Bambara using correct phonemic mapping."""
    result = []
    for c in nko_text:
        if c in NKO_TO_LATIN_CORRECT:
            result.append(NKO_TO_LATIN_CORRECT[c])
        elif c == ' ':
            result.append(' ')
        elif c in NKO_TONE_MARKS or c in NKO_NASALS:
            pass  # Drop diacritics for Latin
    return ''.join(result)


def demo():
    """Test the translator with the user's example."""
    print("=== N'Ko → Latin Bridge Test ===")
    nko = "ߣߝߊ ߊߣߌ ߊ ߞߏߙߏߞߐ"
    print(f"N'Ko input: {nko}")

    # Old (wrong) mapping
    old_map = {
        '\u07CA': 'a', '\u07CB': 'ee', '\u07CC': 'i', '\u07CD': 'e', '\u07CE': 'u',
        '\u07CF': 'o', '\u07D0': 'ɛ',
        '\u07D3': 'b', '\u07D5': 't', '\u07D6': 'j', '\u07D9': 'r', '\u07DB': 's',
        '\u07DC': 'g', '\u07DD': 'f', '\u07DE': 'k', '\u07DF': 'l',
        '\u07E1': 'm', '\u07E3': 'n', '\u07E4': 'h', '\u07E5': 'w', '\u07E6': 'y',
    }
    old_result = ''.join(old_map.get(c, c if c == ' ' else '') for c in nko)
    print(f"Old bridge:  {old_result}")

    new_result = nko_to_latin_correct(nko)
    print(f"New bridge:  {new_result}")

    print("\n=== Dictionary-Augmented Translation ===")
    translator = BambaraTranslator()
    result = translator.translate(new_result)
    print(f"Input:       {new_result}")
    print(f"Translation: {result['translation']}")
    print(f"Method:      {result['method']}")
    print(f"Coverage:    {result['confidence']*100:.0f}%")
    print(f"Glosses:")
    for word, gloss in result['glosses']:
        print(f"  {word:15s} → {gloss or '???'}")


if __name__ == "__main__":
    demo()
