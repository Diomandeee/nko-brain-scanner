#!/usr/bin/env python3
"""
Build V2 SFT training data for N'Ko fine-tuning.

Extends V1 data with LearnNKo sources:
  - V1 sources (parallel corpus, Wikipedia, proverbs, greetings, blessings, cultural, vocab)
  - LearnNKo ML data: 8,817 Bambara-French pairs (via cross-script bridge -> N'Ko)
  - LearnNKo cultural data: proverbs, greetings, blessings, cognates, cultural concepts
  - Five-worlds generated content (if available from pipeline/data/)

Output: training/train_v2.jsonl, training/valid_v2.jsonl (90/10 split)

Usage:
    python3 training/build_sft_data_v2.py
    python3 training/build_sft_data_v2.py --include-worlds  # include five-worlds if available
"""

import json
import random
import sys
from pathlib import Path

NKO_DATA = Path.home() / "Desktop" / "NKo" / "nko" / "data"
LEARNNKO_DATA = Path.home() / "projects" / "LearnNKo"
LEARNNKO_ML = LEARNNKO_DATA / "ml" / "data"
LEARNNKO_PIPELINE = LEARNNKO_DATA / "pipeline" / "data"
SCANNER_DATA = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent

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
    return any(NKO_START <= ord(ch) <= NKO_END for ch in text)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def msg(user_text, assistant_text):
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def extract_parallel_corpus():
    """460 parallel NKo/English pairs."""
    examples = []
    path = SCANNER_DATA / "parallel_corpus.jsonl"
    if not path.exists():
        print(f"  SKIP parallel_corpus.jsonl (not found)")
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


def extract_wikipedia():
    """N'Ko Wikipedia text -> completion tasks."""
    examples = []
    wiki_path = NKO_DATA / "corpus" / "nko-wikipedia.txt"
    if not wiki_path.exists():
        wiki_path = SCANNER_DATA / "nko_wikipedia_corpus.txt"
    if not wiki_path.exists():
        print(f"  SKIP wikipedia (not found)")
        return examples

    lines = load_lines(wiki_path)
    for line in lines:
        if not has_nko(line) or len(line) < 20:
            continue
        midpoint = len(line) // 2
        prompt = line[:midpoint]
        completion = line[midpoint:]
        examples.append(msg(
            f"ߞߊ߬ ߣߌ߲߬ ߟߊߓߊ߲߬: {prompt}",
            completion
        ))

    print(f"  wikipedia: {len(examples)} examples")
    return examples


def extract_cultural_sources():
    """Proverbs, greetings, blessings, cultural concepts."""
    examples = []

    # Proverbs
    for prov_path in [NKO_DATA / "cultural" / "proverbs-unified.json",
                       LEARNNKO_DATA / "nko" / "data" / "cultural" / "proverbs-unified.json"]:
        if prov_path.exists():
            proverbs = load_json(prov_path)
            if isinstance(proverbs, list):
                for p in proverbs:
                    nko = (p.get("nko") or p.get("text") or "").strip()
                    meaning = (p.get("meaning") or p.get("english") or "").strip()
                    if nko and meaning:
                        examples.append(msg(f"What does this N'Ko proverb mean: {nko}", meaning))
                        examples.append(msg(f"Share a N'Ko proverb about: {meaning}", nko))
            break

    # Greetings
    for greet_path in [NKO_DATA / "cultural" / "greetings-unified.json",
                        LEARNNKO_DATA / "nko" / "data" / "cultural" / "greetings-unified.json"]:
        if greet_path.exists():
            greetings = load_json(greet_path)
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

    # Blessings
    for bless_path in [NKO_DATA / "cultural" / "blessings-unified.json",
                        LEARNNKO_DATA / "nko" / "data" / "cultural" / "blessings-unified.json"]:
        if bless_path.exists():
            blessings = load_json(bless_path)
            if isinstance(blessings, list):
                for b in blessings:
                    nko = (b.get("nko") or b.get("text") or "").strip()
                    en = (b.get("english") or b.get("meaning") or "").strip()
                    if nko and en:
                        examples.append(msg(f"Write this blessing in N'Ko: {en}", nko))
            break

    # Cultural concepts
    for cc_path in [NKO_DATA / "cultural" / "cultural-concepts.json",
                     LEARNNKO_DATA / "nko" / "data" / "cultural" / "cultural-concepts.json"]:
        if cc_path.exists():
            concepts = load_json(cc_path)
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

    # Cognates
    cognates_path = NKO_DATA / "nko-cognates.json"
    if not cognates_path.exists():
        cognates_path = LEARNNKO_DATA / "nko" / "data" / "nko-cognates.json"
    if cognates_path.exists():
        cognates = load_json(cognates_path)
        if isinstance(cognates, list):
            for cog in cognates:
                nko = (cog.get("nko") or "").strip()
                en = (cog.get("english") or "").strip()
                latin = (cog.get("latin") or cog.get("bambara") or "").strip()
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
                    f"ߞߊ ߘߐ߬ߝߐ ߟߊ߬ߓߊ: {line[:len(line)//2]}",
                    line[len(line)//2:]
                ))

    # Keyboard vocabulary
    kb_path = NKO_DATA / "keyboard" / "vocabulary.json"
    if not kb_path.exists():
        kb_path = LEARNNKO_DATA / "nko" / "data" / "keyboard" / "vocabulary.json"
    if kb_path.exists():
        vocab = load_json(kb_path)
        if isinstance(vocab, list):
            for v in vocab:
                nko = (v.get("nko") or v.get("word") or "").strip()
                en = (v.get("english") or v.get("meaning") or "").strip()
                if nko and en:
                    examples.append(msg(f"What does '{nko}' mean in English?", en))

    print(f"  vocabulary: {len(examples)} examples")
    return examples


def extract_learnnko_ml_data():
    """LearnNKo ML training data: 8,817 Bambara-French pairs.

    If cross-script bridge is available, converts Bambara Latin -> N'Ko.
    Otherwise creates Bambara<->French translation examples.
    """
    examples = []

    pairs_path = LEARNNKO_ML / "robotsmali_text_pairs.json"
    if not pairs_path.exists():
        print(f"  SKIP LearnNKo ML data (not found)")
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

        # Fallback: Bambara Latin <-> French (still useful for multilingual understanding)
        examples.append(msg(f"Translate to Bambara: {french}", bambara))
        examples.append(msg(f"Translate this Bambara text to French: {bambara}", french))

    if BRIDGE_AVAILABLE:
        print(f"  LearnNKo ML data: {len(examples)} examples ({nko_converted} N'Ko-converted)")
    else:
        print(f"  LearnNKo ML data: {len(examples)} examples (Latin script, bridge unavailable)")
    return examples


def extract_learnnko_training_data():
    """LearnNKo bidirectional training data (39 user-audio pairs)."""
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


def extract_five_worlds():
    """Five-worlds generated content from LearnNKo pipeline (if available)."""
    examples = []

    # Check for worlds output from the pipeline
    worlds_dir = LEARNNKO_PIPELINE
    worlds_files = list(worlds_dir.glob("worlds_*.json")) if worlds_dir.exists() else []

    if not worlds_files:
        # Check for vocabulary with world expansions
        vocab_path = worlds_dir / "vocabulary.json" if worlds_dir.exists() else None
        if vocab_path and vocab_path.exists():
            vocab = load_json(vocab_path)
            # Extract any world-generated content
            if isinstance(vocab, dict):
                for word_key, word_data in vocab.items():
                    if isinstance(word_data, dict) and "worlds" in word_data:
                        for world in word_data["worlds"]:
                            nko = (world.get("nko_text") or "").strip()
                            context = (world.get("context") or "").strip()
                            world_name = (world.get("world") or "").strip()
                            if nko and context:
                                examples.append(msg(
                                    f"Write a {world_name} text in N'Ko about: {context}",
                                    nko
                                ))

    if not examples:
        print(f"  five-worlds: 0 examples (pipeline not yet run)")
    else:
        print(f"  five-worlds: {len(examples)} examples")
    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build V2 SFT training data")
    parser.add_argument("--include-worlds", action="store_true",
                        help="Include five-worlds data if available")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    args = parser.parse_args()

    print("=" * 60)
    print("Building V2 SFT Training Data")
    print("=" * 60)
    print(f"Cross-script bridge: {'AVAILABLE' if BRIDGE_AVAILABLE else 'NOT AVAILABLE'}")
    print()

    all_examples = []

    # V1 sources
    print("V1 Sources:")
    all_examples.extend(extract_parallel_corpus())
    all_examples.extend(extract_wikipedia())
    all_examples.extend(extract_cultural_sources())
    all_examples.extend(extract_vocabulary())

    # V2 sources (LearnNKo)
    print("\nV2 Sources (LearnNKo):")
    all_examples.extend(extract_learnnko_ml_data())
    all_examples.extend(extract_learnnko_training_data())
    if args.include_worlds:
        all_examples.extend(extract_five_worlds())

    # Deduplicate by content hash
    seen = set()
    unique = []
    for ex in all_examples:
        key = json.dumps(ex["messages"], ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"\nTotal: {len(all_examples)} raw, {len(unique)} unique (deduped)")

    # Shuffle and split
    random.shuffle(unique)
    val_size = max(1, int(len(unique) * args.val_ratio))
    val_data = unique[:val_size]
    train_data = unique[val_size:]

    # Save
    train_path = OUT_DIR / "train_v2.jsonl"
    val_path = OUT_DIR / "valid_v2.jsonl"

    for path, data, label in [(train_path, train_data, "train"), (val_path, val_data, "valid")]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {label}: {len(data)} examples -> {path}")

    # Also create combined with CPT data
    cpt_train = OUT_DIR / "cpt_train.jsonl"
    combined_path = OUT_DIR / "combined_train_v2.jsonl"
    combined_val = OUT_DIR / "combined_valid_v2.jsonl"

    if cpt_train.exists():
        cpt_examples = []
        with open(cpt_train) as f:
            for line in f:
                cpt_examples.append(json.loads(line))
        combined = cpt_examples + train_data
        random.shuffle(combined)
        with open(combined_path, "w") as f:
            for ex in combined:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  combined (CPT+SFT): {len(combined)} examples -> {combined_path}")

        # Combined valid
        cpt_val = OUT_DIR / "cpt_valid.jsonl"
        if cpt_val.exists():
            with open(cpt_val) as f:
                cpt_val_examples = [json.loads(line) for line in f]
            combined_v = cpt_val_examples + val_data
            with open(combined_val, "w") as f:
                for ex in combined_v:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  combined valid: {len(combined_v)} examples -> {combined_val}")

    # Summary
    print(f"\n{'='*60}")
    print(f"V2 SFT DATA READY")
    print(f"{'='*60}")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Valid: {len(val_data)} examples")
    nko_count = sum(1 for ex in unique if any(has_nko(m["content"]) for m in ex["messages"]))
    print(f"  N'Ko content: {nko_count}/{len(unique)} ({nko_count/len(unique)*100:.1f}%)")


if __name__ == "__main__":
    main()
