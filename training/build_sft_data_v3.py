#!/usr/bin/env python3
"""
Build V3 SFT training data for N'Ko fine-tuning.

Extends V2 data with new sources:
  - V2 sources (parallel corpus, Wikipedia, proverbs, greetings, blessings, cultural, vocab, LearnNKo ML)
  - OLDI-Seed: 6,193 N'Ko Wikipedia text entries (text completion + knowledge tasks)
  - Bayelemabaga pilot: 500 Bambara-French-N'Ko triples + 1,001 SFT examples
  - N'Ko Wikipedia corpus JSONL: 1,693 articles with actual N'Ko text
  - Enhanced Wikipedia: longer text segments, sentence-level splitting
  - training_data_large.json: 17,648 training examples + 8,824 parallel pairs

Output: data/sft_v3_combined.jsonl (merged V2 + V3 new examples)

Target: 40K+ minimum, 50K+ ideal

Usage:
    python3 training/build_sft_data_v3.py
"""

import json
import random
import hashlib
import sys
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
NKO_DATA = Path.home() / "Desktop" / "NKo" / "nko" / "data"
LEARNNKO_DATA = Path.home() / "projects" / "LearnNKo"
LEARNNKO_ML = LEARNNKO_DATA / "ml" / "data"
SCANNER_DATA = PROJECT_ROOT / "data"
NICOLINGUA_RAW = SCANNER_DATA / "nicolingua" / "raw"
NICOLINGUA_OUT = SCANNER_DATA / "nicolingua"
OUT_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = PROJECT_ROOT / "training"

random.seed(42)

# Try to import cross-script bridge for Latin -> N'Ko conversion
BRIDGE_AVAILABLE = False
try:
    sys.path.insert(0, str(Path.home() / "Desktop" / "NKo"))
    from nko.transliterate import transliterate
    BRIDGE_AVAILABLE = True
except ImportError:
    try:
        sys.path.insert(0, str(Path.home() / "Desktop" / "cross-script-bridge"))
        from core.bridge import Bridge
        _bridge = Bridge()
        def transliterate(text, source="latin", target="nko"):
            return _bridge.convert(text, source, target)
        BRIDGE_AVAILABLE = True
    except ImportError:
        pass

NKO_START, NKO_END = 0x07C0, 0x07FF


def has_nko(text):
    """Check if text contains N'Ko characters."""
    return any(NKO_START <= ord(ch) <= NKO_END for ch in text)


def nko_purity(text):
    """Calculate fraction of text that is N'Ko script (excluding spaces/punctuation)."""
    if not text:
        return 0.0
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    nko_chars = sum(1 for ch in alpha_chars if NKO_START <= ord(ch) <= NKO_END)
    return nko_chars / len(alpha_chars)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def msg(user_text, assistant_text):
    """Create a messages-format SFT example."""
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def content_hash(example):
    """Hash an example for deduplication."""
    return hashlib.md5(
        json.dumps(example["messages"], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


# ============================================================
# V2 Source extractors (from build_sft_data_v2.py)
# ============================================================

def extract_parallel_corpus():
    """460 parallel NKo/English pairs."""
    examples = []
    path = SCANNER_DATA / "parallel_corpus.jsonl"
    if not path.exists():
        print("  SKIP parallel_corpus.jsonl (not found)")
        return examples

    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            en = (entry.get("english") or "").strip()
            nko = (entry.get("nko") or "").strip()
            if not en or not nko:
                continue
            examples.append(msg(f"Translate this to N'Ko script: {en}", nko))
            examples.append(msg(f"Translate this N'Ko text to English: {nko}", en))
            ipa = (entry.get("ipa") or "").strip()
            if ipa:
                examples.append(msg(f"What is the IPA transcription of this N'Ko text: {nko}", ipa))

    print(f"  parallel_corpus: {len(examples)} examples")
    return examples


def extract_wikipedia_txt():
    """N'Ko Wikipedia text -> completion tasks."""
    examples = []
    wiki_path = NKO_DATA / "corpus" / "nko-wikipedia.txt"
    if not wiki_path.exists():
        wiki_path = SCANNER_DATA / "nko_wikipedia_corpus.txt"
    if not wiki_path.exists():
        print("  SKIP wikipedia txt (not found)")
        return examples

    lines = load_lines(wiki_path)
    for line in lines:
        if not has_nko(line) or len(line) < 20:
            continue
        midpoint = len(line) // 2
        prompt = line[:midpoint]
        completion = line[midpoint:]
        examples.append(msg(
            f"Continue this N'Ko text: {prompt}",
            completion
        ))

    print(f"  wikipedia txt: {len(examples)} examples")
    return examples


def extract_cultural_sources():
    """Proverbs, greetings, blessings, cultural concepts from NKo/LearnNKo."""
    examples = []

    # --- Proverbs (unified JSON) ---
    for prov_path in [NKO_DATA / "cultural" / "proverbs-unified.json",
                       LEARNNKO_DATA / "nko" / "data" / "cultural" / "proverbs-unified.json"]:
        if prov_path.exists():
            data = load_json(prov_path)
            proverbs = data.get("proverbs", data) if isinstance(data, dict) else data
            if isinstance(proverbs, list):
                for p in proverbs:
                    nko = (p.get("text_nko") or p.get("nko") or p.get("text") or "").strip()
                    meaning = (p.get("english_translation") or p.get("meaning") or p.get("english") or "").strip()
                    latin = (p.get("text_latin") or p.get("latin") or "").strip()
                    if nko and meaning:
                        examples.append(msg(f"What does this N'Ko proverb mean: {nko}", meaning))
                        examples.append(msg(f"Share a N'Ko proverb about: {meaning}", nko))
                    elif latin and meaning and BRIDGE_AVAILABLE:
                        try:
                            nko_conv = transliterate(latin, source="latin", target="nko")
                            if nko_conv and has_nko(nko_conv):
                                examples.append(msg(f"What does this proverb mean: {nko_conv}", meaning))
                        except Exception:
                            pass
            break

    # --- Keyboard proverbs ---
    kb_vocab_path = NKO_DATA / "keyboard" / "vocabulary.json"
    if kb_vocab_path.exists():
        kb = load_json(kb_vocab_path)
        kb_proverbs = kb.get("proverbs", [])
        for p in kb_proverbs:
            nko = (p.get("nko") or "").strip()
            meaning = (p.get("meaning") or "").strip()
            if nko and meaning:
                examples.append(msg(f"Explain this N'Ko proverb: {nko}", meaning))

    # --- LearnNKo proverbs.json ---
    lnko_prov = LEARNNKO_DATA / "data" / "proverbs" / "proverbs.json"
    if lnko_prov.exists():
        data = load_json(lnko_prov)
        for p in data.get("proverbs", []):
            original = (p.get("original") or "").strip()
            english = (p.get("english") or "").strip()
            meaning = (p.get("meaning") or "").strip()
            if original and english:
                examples.append(msg(f"Translate this Manding proverb: {original}", english))
                if meaning:
                    examples.append(msg(f"What is the deeper meaning of: {english}", meaning))

    # --- Greetings ---
    for greet_path in [NKO_DATA / "cultural" / "greetings-unified.json",
                        LEARNNKO_DATA / "nko" / "data" / "cultural" / "greetings-unified.json"]:
        if greet_path.exists():
            data = load_json(greet_path)
            greetings = data.get("greetings", data) if isinstance(data, dict) else data
            if isinstance(greetings, list):
                for g in greetings:
                    nko = (g.get("nko") or g.get("greeting") or "").strip()
                    en = (g.get("english") or g.get("meaning") or "").strip()
                    context = (g.get("context") or g.get("usage") or "").strip()
                    if nko and en:
                        examples.append(msg(f"How do you say this greeting in N'Ko: {en}", nko))
                        if context:
                            examples.append(msg(f"When do you use this N'Ko greeting: {nko}", f"{en}. {context}"))
            break

    # --- Blessings ---
    for bless_path in [NKO_DATA / "cultural" / "blessings-unified.json",
                        LEARNNKO_DATA / "nko" / "data" / "cultural" / "blessings-unified.json"]:
        if bless_path.exists():
            data = load_json(bless_path)
            blessings = data.get("blessings", data) if isinstance(data, dict) else data
            if isinstance(blessings, list):
                for b in blessings:
                    nko = (b.get("nko") or b.get("text") or "").strip()
                    en = (b.get("english") or b.get("meaning") or "").strip()
                    if nko and en:
                        examples.append(msg(f"Write this blessing in N'Ko: {en}", nko))
            break

    # --- Cultural concepts ---
    for cc_path in [NKO_DATA / "cultural" / "cultural-concepts.json",
                     LEARNNKO_DATA / "nko" / "data" / "cultural" / "cultural-concepts.json"]:
        if cc_path.exists():
            data = load_json(cc_path)
            concepts = data.get("concepts", data) if isinstance(data, dict) else data
            if isinstance(concepts, list):
                for c in concepts:
                    name = (c.get("name") or c.get("concept") or "").strip()
                    desc = (c.get("description") or c.get("meaning") or "").strip()
                    nko = (c.get("nko") or "").strip()
                    if name and desc:
                        if nko:
                            examples.append(msg(f"Explain the Manding concept '{name}' ({nko})", desc))
                        else:
                            examples.append(msg(f"Explain the Manding concept '{name}'", desc))
            break

    # --- Cognates ---
    cognates_path = NKO_DATA / "nko-cognates.json"
    if not cognates_path.exists():
        cognates_path = LEARNNKO_DATA / "nko" / "data" / "nko-cognates.json"
    if cognates_path.exists():
        cognates = load_json(cognates_path)
        if isinstance(cognates, list):
            for cog in cognates:
                nko = (cog.get("nko") or "").strip()
                en = (cog.get("english") or "").strip()
                if nko and en:
                    examples.append(msg(f"What is '{en}' in N'Ko script?", nko))

    print(f"  cultural sources: {len(examples)} examples")
    return examples


def extract_vocabulary():
    """Vocabulary and keyboard data."""
    examples = []

    # nko-vocabulary.txt
    vocab_path = NKO_DATA / "corpus" / "nko-vocabulary.txt"
    if vocab_path.exists():
        lines = load_lines(vocab_path)
        for line in lines:
            if has_nko(line) and len(line) > 5:
                examples.append(msg(
                    f"Continue this N'Ko vocabulary: {line[:len(line)//2]}",
                    line[len(line)//2:]
                ))

    # Keyboard vocabulary sections
    kb_path = NKO_DATA / "keyboard" / "vocabulary.json"
    if kb_path.exists():
        vocab = load_json(kb_path)

        # Greetings: dict of nko_key -> {meaning, transliteration, ...}
        greetings_dict = vocab.get("greetings", {})
        if isinstance(greetings_dict, dict):
            for nko_key, info in greetings_dict.items():
                if isinstance(info, dict):
                    en = (info.get("meaning") or "").strip()
                    response = (info.get("response") or "").strip()
                    if nko_key and en:
                        examples.append(msg(f"What does '{nko_key}' mean in English?", en))
                        examples.append(msg(f"How do you say '{en}' in N'Ko?", nko_key))
                        if response:
                            examples.append(msg(f"What is the response to the N'Ko greeting '{nko_key}'?", response))

        # Pronouns: dict of nko -> {meaning, trans}
        pronouns_dict = vocab.get("pronouns", {})
        if isinstance(pronouns_dict, dict):
            for nko_key, info in pronouns_dict.items():
                if isinstance(info, dict):
                    en = (info.get("meaning") or "").strip()
                    if nko_key and en:
                        examples.append(msg(f"What does the N'Ko pronoun '{nko_key}' mean?", en))

        # Verbs: dict of nko -> {meaning, trans}
        verbs_dict = vocab.get("verbs", {})
        if isinstance(verbs_dict, dict):
            for nko_key, info in verbs_dict.items():
                if isinstance(info, dict):
                    en = (info.get("meaning") or "").strip()
                    if nko_key and en:
                        examples.append(msg(f"What does the N'Ko verb '{nko_key}' mean?", en))
                        examples.append(msg(f"How do you write '{en}' in N'Ko?", nko_key))

        # Nouns: dict of category -> dict of nko -> {meaning, trans}
        nouns_dict = vocab.get("nouns", {})
        if isinstance(nouns_dict, dict):
            for category, entries in nouns_dict.items():
                if isinstance(entries, dict):
                    for nko_key, info in entries.items():
                        if isinstance(info, dict):
                            en = (info.get("meaning") or "").strip()
                            if nko_key and en:
                                examples.append(msg(f"What does '{nko_key}' mean in English?", en))
                                examples.append(msg(f"How do you write '{en}' in N'Ko?", nko_key))

        # Numbers: dict of nko_digit -> {value, trans}
        numbers_dict = vocab.get("numbers", {})
        if isinstance(numbers_dict, dict):
            for nko_key, info in numbers_dict.items():
                if isinstance(info, dict):
                    value = info.get("value", "")
                    trans = (info.get("trans") or "").strip()
                    if nko_key and trans:
                        examples.append(msg(f"What is the N'Ko number '{nko_key}'?", f"{value} ({trans})"))

        # Phrases: dict of category -> list of {nko, meaning, trans}
        phrases_dict = vocab.get("phrases", {})
        if isinstance(phrases_dict, dict):
            for category, phrase_list in phrases_dict.items():
                if isinstance(phrase_list, list):
                    for p in phrase_list:
                        if isinstance(p, dict):
                            nko = (p.get("nko") or "").strip()
                            en = (p.get("meaning") or "").strip()
                            if nko and en:
                                examples.append(msg(f"What does '{nko}' mean?", en))
                                examples.append(msg(f"How do you say '{en}' in N'Ko?", nko))

    print(f"  vocabulary: {len(examples)} examples")
    return examples


def extract_learnnko_ml_data():
    """LearnNKo ML training data: 8,817 Bambara-French pairs."""
    examples = []

    pairs_path = LEARNNKO_ML / "robotsmali_text_pairs.json"
    if not pairs_path.exists():
        print("  SKIP LearnNKo ML data (not found)")
        return examples

    data = load_json(pairs_path)
    pairs = data.get("pairs", [])

    nko_converted = 0
    for pair in pairs:
        bambara = (pair.get("bambara") or "").strip()
        french = (pair.get("french") or "").strip()
        if not bambara or not french:
            continue

        if BRIDGE_AVAILABLE:
            try:
                nko = transliterate(bambara, source="latin", target="nko")
                if nko and has_nko(nko):
                    examples.append(msg(f"Translate to N'Ko: {french}", nko))
                    examples.append(msg(f"Translate this N'Ko text to French: {nko}", french))
                    nko_converted += 1
                    continue
            except Exception:
                pass

        # Fallback: Bambara Latin <-> French
        examples.append(msg(f"Translate to Bambara: {french}", bambara))
        examples.append(msg(f"Translate this Bambara text to French: {bambara}", french))

    if BRIDGE_AVAILABLE:
        print(f"  LearnNKo ML data: {len(examples)} examples ({nko_converted} N'Ko-converted)")
    else:
        print(f"  LearnNKo ML data: {len(examples)} examples (Latin script, bridge unavailable)")
    return examples


def extract_learnnko_training_data():
    """LearnNKo bidirectional training data."""
    examples = []
    path = LEARNNKO_ML / "training_data_bidirectional.json"
    if not path.exists():
        return examples

    data = load_json(path)
    pairs = data.get("pairs", [])
    for pair in pairs:
        bambara = (pair.get("bambara") or "").strip()
        english = (pair.get("english") or "").strip()
        if not bambara or not english:
            continue

        if BRIDGE_AVAILABLE:
            try:
                nko = transliterate(bambara, source="latin", target="nko")
                if nko and has_nko(nko):
                    examples.append(msg(f"Translate to N'Ko: {english}", nko))
                    examples.append(msg(f"What does this mean in English: {nko}", english))
                    continue
            except Exception:
                pass

        examples.append(msg(f"Translate to Bambara: {english}", bambara))

    print(f"  LearnNKo bidirectional: {len(examples)} examples")
    return examples


# ============================================================
# V3 NEW Source extractors
# ============================================================

def extract_oldi_seed_nko():
    """OLDI-Seed N'Ko Wikipedia entries: 6,193 texts for completion/knowledge tasks."""
    examples = []
    path = NICOLINGUA_RAW / "oldi_seed_nko.jsonl"
    if not path.exists():
        print("  SKIP oldi_seed_nko (not found)")
        return examples

    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    for entry in entries:
        text = (entry.get("text") or "").strip()
        url = (entry.get("url") or "").strip()
        if not text or len(text) < 30:
            continue
        if not has_nko(text):
            continue

        # Extract article title from URL if available
        title = ""
        if url and "wikipedia.org/wiki/" in url:
            title = url.split("/wiki/")[-1].replace("_", " ")

        # Task 1: Text completion (split at various points)
        if len(text) > 60:
            # Split at ~40% for a good prompt-completion ratio
            split_point = int(len(text) * 0.4)
            # Try to split at a word boundary
            space_idx = text.rfind(" ", max(0, split_point - 20), split_point + 20)
            if space_idx > 0:
                split_point = space_idx
            prompt_text = text[:split_point].strip()
            completion_text = text[split_point:].strip()
            if prompt_text and completion_text:
                examples.append(msg(
                    f"Continue this N'Ko text: {prompt_text}",
                    completion_text
                ))

        # Task 2: If we have an English title, create a knowledge prompt
        if title and len(text) > 50:
            examples.append(msg(
                f"Write about {title} in N'Ko script",
                text
            ))

        # Task 3: N'Ko reading comprehension (for longer texts)
        if len(text) > 150:
            # Split into sentences (N'Ko uses , . and ߸ as separators)
            sentences = re.split(r'[.߸]\s*', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
            if len(sentences) >= 2:
                # Use first sentence as context, ask to continue
                examples.append(msg(
                    f"Given this N'Ko context, what comes next: {sentences[0]}",
                    ". ".join(sentences[1:3])
                ))

    print(f"  oldi_seed_nko: {len(examples)} examples")
    return examples


def extract_nko_wiki_jsonl():
    """N'Ko Wikipedia JSONL corpus: articles with actual N'Ko content."""
    examples = []
    path = SCANNER_DATA / "nko_wikipedia_corpus.jsonl"
    if not path.exists():
        print("  SKIP nko_wikipedia_corpus.jsonl (not found)")
        return examples

    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            title = (entry.get("title") or "").strip()
            text = (entry.get("text") or "").strip()
            nko_chars = entry.get("nko_chars", 0)

            if nko_chars < 50 or not text or len(text) < 50:
                continue

            # Only use text that has substantial N'Ko content
            if nko_purity(text) < 0.3:
                continue

            # Task 1: Article generation
            if title and has_nko(title):
                # Use title as prompt
                if len(text) > 200:
                    # Use first ~300 chars as response
                    examples.append(msg(
                        f"Write about {title} in N'Ko",
                        text[:300].strip()
                    ))

            # Task 2: Text completion from N'Ko-rich segments
            # Extract segments that are mostly N'Ko
            segments = text.split("\n")
            for seg in segments:
                seg = seg.strip()
                if len(seg) > 40 and has_nko(seg) and nko_purity(seg) > 0.5:
                    mid = len(seg) // 2
                    space_idx = seg.rfind(" ", max(0, mid - 15), mid + 15)
                    if space_idx > 0:
                        mid = space_idx
                    examples.append(msg(
                        f"Continue this N'Ko text: {seg[:mid].strip()}",
                        seg[mid:].strip()
                    ))

    print(f"  nko_wiki_jsonl: {len(examples)} examples")
    return examples


def extract_bayelemabaga_pairs():
    """Bayelemabaga pilot pairs: 500 Bambara-French-N'Ko triples."""
    examples = []
    path = SCANNER_DATA / "bayelemabaga" / "pilot_pairs.jsonl"
    if not path.exists():
        print("  SKIP bayelemabaga pilot_pairs (not found)")
        return examples

    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            bambara = (entry.get("bambara") or "").strip()
            french = (entry.get("french") or "").strip()
            nko = (entry.get("nko") or "").strip()

            if not nko or len(nko) < 3:
                continue

            # Bambara -> N'Ko (script conversion)
            if bambara:
                examples.append(msg(
                    f"Write this Bambara sentence in N'Ko script: {bambara}",
                    nko
                ))

            # French -> N'Ko translation
            if french:
                examples.append(msg(
                    f"Translate to N'Ko: {french}",
                    nko
                ))
                examples.append(msg(
                    f"Translate this N'Ko text to French: {nko}",
                    french
                ))

            # If we have both bambara and french, create a trilingual example
            if bambara and french:
                examples.append(msg(
                    f"Translate this Bambara to French: {bambara}",
                    french
                ))

    print(f"  bayelemabaga_pairs: {len(examples)} examples")
    return examples


def extract_bayelemabaga_sft():
    """Bayelemabaga pilot SFT: 1,001 pre-formatted examples."""
    examples = []
    path = SCANNER_DATA / "bayelemabaga" / "pilot_sft.jsonl"
    if not path.exists():
        print("  SKIP bayelemabaga pilot_sft (not found)")
        return examples

    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if "messages" in entry and len(entry["messages"]) >= 2:
                examples.append(entry)

    print(f"  bayelemabaga_sft: {len(examples)} examples")
    return examples


def extract_training_data_large():
    """LearnNKo training_data_large.json: 17,648 training examples + 8,824 parallel pairs."""
    examples = []
    path = LEARNNKO_ML / "training_data_large.json"
    if not path.exists():
        print("  SKIP training_data_large (not found)")
        return examples

    data = load_json(path)

    # Training examples (already in input/target format)
    for ex in data.get("training_examples", []):
        inp = (ex.get("input") or "").strip()
        target = (ex.get("target") or "").strip()
        direction = (ex.get("direction") or "").strip()
        if not inp or not target:
            continue

        # Convert to messages format with cleaner prompts
        if direction == "fr_to_bam":
            examples.append(msg(f"Translate from French to Bambara: {inp.replace('translate french to bambara: ', '')}", target))
        elif direction == "bam_to_fr":
            examples.append(msg(f"Translate from Bambara to French: {inp.replace('translate bambara to french: ', '')}", target))
        else:
            # Clean up the input prefix
            clean_inp = inp
            for prefix in ["translate french to bambara: ", "translate bambara to french: "]:
                if clean_inp.startswith(prefix):
                    clean_inp = clean_inp[len(prefix):]
                    break
            examples.append(msg(f"Translate: {clean_inp}", target))

    # Parallel pairs (create bidirectional examples)
    for pair in data.get("parallel_pairs", []):
        bambara = (pair.get("bambara") or "").strip()
        french = (pair.get("french") or "").strip()
        if not bambara or not french:
            continue

        # Try N'Ko conversion if bridge available
        if BRIDGE_AVAILABLE:
            try:
                nko = transliterate(bambara, source="latin", target="nko")
                if nko and has_nko(nko):
                    examples.append(msg(f"Write this in N'Ko script: {french}", nko))
                    continue
            except Exception:
                pass

    print(f"  training_data_large: {len(examples)} examples")
    return examples


def extract_nko_cultural_txt():
    """N'Ko cultural text corpus."""
    examples = []
    path = NKO_DATA / "corpus" / "nko-cultural.txt"
    if not path.exists():
        path = LEARNNKO_DATA / "nko" / "data" / "corpus" / "nko-cultural.txt"
    if not path.exists():
        print("  SKIP nko-cultural.txt (not found)")
        return examples

    lines = load_lines(path)
    for line in lines:
        if has_nko(line) and len(line) > 20:
            mid = len(line) // 2
            examples.append(msg(
                f"Complete this N'Ko cultural text: {line[:mid]}",
                line[mid:]
            ))

    print(f"  nko_cultural_txt: {len(examples)} examples")
    return examples


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Building V3 SFT Training Data")
    print("=" * 70)
    print(f"Cross-script bridge: {'AVAILABLE' if BRIDGE_AVAILABLE else 'NOT AVAILABLE'}")
    print()

    all_v3_examples = []

    # V2 Sources (rebuild fresh)
    print("V2 Sources:")
    all_v3_examples.extend(extract_parallel_corpus())
    all_v3_examples.extend(extract_wikipedia_txt())
    all_v3_examples.extend(extract_cultural_sources())
    all_v3_examples.extend(extract_vocabulary())
    all_v3_examples.extend(extract_learnnko_ml_data())
    all_v3_examples.extend(extract_learnnko_training_data())

    # V3 NEW Sources
    print("\nV3 New Sources:")
    all_v3_examples.extend(extract_oldi_seed_nko())
    all_v3_examples.extend(extract_nko_wiki_jsonl())
    all_v3_examples.extend(extract_bayelemabaga_pairs())
    all_v3_examples.extend(extract_bayelemabaga_sft())
    all_v3_examples.extend(extract_training_data_large())
    all_v3_examples.extend(extract_nko_cultural_txt())

    # Deduplicate V3 examples
    seen = set()
    unique_v3 = []
    for ex in all_v3_examples:
        h = content_hash(ex)
        if h not in seen:
            seen.add(h)
            unique_v3.append(ex)

    print(f"\nV3 raw: {len(all_v3_examples)}, V3 unique (deduped): {len(unique_v3)}")

    # Save nicolingua train/test splits
    nicolingua_examples = []
    for ex in unique_v3:
        # Track examples from V3-new sources by checking for specific patterns
        user_content = ex["messages"][0]["content"]
        if any(kw in user_content for kw in [
            "Continue this N'Ko text:",
            "Write about",
            "Given this N'Ko context",
            "Write this Bambara sentence in N'Ko",
        ]):
            nicolingua_examples.append(ex)

    if nicolingua_examples:
        random.shuffle(nicolingua_examples)
        test_size = max(1, int(len(nicolingua_examples) * 0.1))
        test_data = nicolingua_examples[:test_size]
        train_data = nicolingua_examples[test_size:]

        NICOLINGUA_OUT.mkdir(parents=True, exist_ok=True)
        for path, data, label in [
            (NICOLINGUA_OUT / "train.jsonl", train_data, "train"),
            (NICOLINGUA_OUT / "test.jsonl", test_data, "test"),
        ]:
            with open(path, "w") as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  nicolingua {label}: {len(data)} examples -> {path}")

    # Load existing V2 combined data
    v2_combined_path = TRAINING_DIR / "combined_train_v2.jsonl"
    v2_examples = []
    v2_hashes = set()
    if v2_combined_path.exists():
        with open(v2_combined_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                v2_examples.append(ex)
                v2_hashes.add(content_hash(ex))
        print(f"\nExisting V2 data: {len(v2_examples)} examples")

    # Merge: V2 + new V3 examples (avoid duplicates)
    new_only = [ex for ex in unique_v3 if content_hash(ex) not in v2_hashes]
    print(f"New examples not in V2: {len(new_only)}")

    combined = v2_examples + new_only
    random.shuffle(combined)

    # Save combined V3
    out_path = OUT_DIR / "sft_v3_combined.jsonl"
    with open(out_path, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    nko_count = sum(
        1 for ex in combined
        if any(has_nko(m["content"]) for m in ex["messages"])
    )

    print(f"\n{'=' * 70}")
    print(f"V3 SFT DATA READY")
    print(f"{'=' * 70}")
    print(f"  Total combined: {len(combined)} examples")
    print(f"  From V2: {len(v2_examples)}")
    print(f"  New in V3: {len(new_only)}")
    print(f"  N'Ko content: {nko_count}/{len(combined)} ({nko_count/len(combined)*100:.1f}%)")
    print(f"  Output: {out_path}")

    # Validate format
    print("\nValidating format...")
    errors = 0
    for i, ex in enumerate(combined):
        if "messages" not in ex:
            print(f"  ERROR at index {i}: missing 'messages' key, has: {list(ex.keys())}")
            errors += 1
        elif not isinstance(ex["messages"], list) or len(ex["messages"]) < 2:
            print(f"  ERROR at index {i}: messages not a list or too short")
            errors += 1
        else:
            for m in ex["messages"]:
                if "role" not in m or "content" not in m:
                    print(f"  ERROR at index {i}: message missing role/content")
                    errors += 1
                    break

    if errors == 0:
        print(f"  Format validation PASSED (all {len(combined)} examples have correct schema)")
    else:
        print(f"  Format validation FAILED: {errors} errors")

    target_met = len(combined) >= 40000
    print(f"\n  Target check: {len(combined)}/40000 minimum {'PASS' if target_met else 'FAIL'}")
    if len(combined) >= 50000:
        print(f"  Stretch target: {len(combined)}/50000 PASS")


if __name__ == "__main__":
    main()
