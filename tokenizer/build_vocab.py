#!/usr/bin/env python3
"""
Build vocab.json from nko-unified.json and morphology inventories.

Reads the 232-character canonical dataset and the morphological analyzer's
built-in affix/pronoun/particle inventories, assigns IDs, and writes
tokenizer/vocab.json.

Usage:
    python -m tokenizer.build_vocab
"""

import json
import sys
from pathlib import Path

_NKO_ROOT = Path.home() / "Desktop" / "NKo"
if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))

from nko.morphology import MorphologicalAnalyzer
from nko.phonetics import NKoPhonetics, LETTER_CHARS, DIGIT_CHARS


def build_vocab() -> dict:
    """Build the full vocabulary mapping."""
    token_to_id = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "[SEP]": 4,
    }
    idx = 5

    # 1. All NKO letters (vowels + consonants) from phonetics tables
    for ch in sorted(LETTER_CHARS):
        if ch not in token_to_id:
            token_to_id[ch] = idx
            idx += 1

    # 2. NKO digits
    for ch in sorted(DIGIT_CHARS):
        if ch not in token_to_id:
            token_to_id[ch] = idx
            idx += 1

    # 3. Load nko-unified.json for any characters we missed
    unified_path = _NKO_ROOT / "data" / "nko-unified.json"
    if unified_path.exists():
        with open(unified_path) as f:
            unified = json.load(f)
        for category in ["vowels", "consonants", "digits", "tone_marks", "punctuation"]:
            chars = unified.get("characters", {}).get(category, [])
            for entry in chars:
                ch = entry.get("char", "")
                if ch and ch not in token_to_id:
                    token_to_id[ch] = idx
                    idx += 1

    # 3b. Ensure ALL NKO Unicode combining marks are covered (U+07EB-U+07F5)
    for codepoint in range(0x07EB, 0x07F6):
        ch = chr(codepoint)
        if ch not in token_to_id:
            token_to_id[ch] = idx
            idx += 1

    # 4. Common morphemes from analyzer
    analyzer = MorphologicalAnalyzer()
    seen = set(token_to_id.keys())

    for source_name, source_list in [
        ("pronouns", analyzer.subject_pronouns),
        ("tense_markers", analyzer.tense_markers),
        ("postpositions", analyzer.postpositions),
        ("derivation_suffixes", analyzer.derivation_suffixes),
    ]:
        for morph in source_list:
            for text in [morph.nko, morph.latin]:
                text = (text or "").strip()
                if text and text not in seen:
                    # Strip tones for clean lookup
                    clean = NKoPhonetics.strip_tones(text)
                    if clean and clean not in seen:
                        token_to_id[clean] = idx
                        seen.add(clean)
                        idx += 1

    # 5. Common vocabulary from keyboard AI lexicon
    vocab_path = _NKO_ROOT / "data" / "keyboard" / "vocabulary.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        # Extract NKO words from all categories
        for category_name, entries in vocab_data.items():
            if isinstance(entries, list):
                for entry in entries:
                    nko = entry.get("nko", "") if isinstance(entry, dict) else ""
                    nko = NKoPhonetics.strip_tones(nko).strip()
                    if nko and nko not in seen and len(nko) > 1:
                        token_to_id[nko] = idx
                        seen.add(nko)
                        idx += 1

    return {
        "token_to_id": token_to_id,
        "vocab_size": len(token_to_id),
        "special_tokens": {
            "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4,
        },
        "character_count": sum(1 for k in token_to_id if len(k) == 1),
        "morpheme_count": sum(1 for k in token_to_id if len(k) > 1 and not k.startswith("[")),
    }


def main():
    vocab = build_vocab()
    out_path = Path(__file__).parent / "vocab.json"
    with open(out_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary built: {vocab['vocab_size']} tokens")
    print(f"  Characters: {vocab['character_count']}")
    print(f"  Morphemes:  {vocab['morpheme_count']}")
    print(f"  Special:    5")
    print(f"  Written to: {out_path}")


if __name__ == "__main__":
    main()
