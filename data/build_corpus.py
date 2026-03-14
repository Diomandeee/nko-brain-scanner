#!/usr/bin/env python3
"""
Build parallel NKO <-> English evaluation corpus from existing data.

Sources:
  - nko-unified.json: vocabulary, proverbs, wisdom mappings
  - nko-cognates.json: 18 cross-language cognate entries
  - proverbs-unified.json: 62 proverbs (text_nko + english_translation)
  - greetings-unified.json: 23 greeting protocols (text_nko + english_translation)
  - blessings-unified.json: 29 blessings/prayers (text_nko + english_translation)
  - cultural-concepts.json: 12 concepts (text_nko + english_translation)
  - clans-unified.json: 9 clans (name_nko + jamu + jamu_meaning)
  - cultural-calendar.json: 7 events (name_nko + description + greetings)
  - keyboard/vocabulary.json: ~80 entries (NKO key + meaning)
  - cross-script-bridge/data/cognates.json: 10 cognates + 5 phrases
  - Conjugation paradigms: verbs x pronouns x tenses

Output: data/parallel_corpus.jsonl

Usage:
    python -m data.build_corpus
"""

import json
import sys
from pathlib import Path

_NKO_ROOT = Path.home() / "Desktop" / "NKo"
if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))

from nko.phonetics import NKoPhonetics


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"  [skip] {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def extract_from_unified(unified: dict) -> list:
    """Extract parallel pairs from nko-unified.json."""
    pairs = []

    # Vocabulary entries
    vocab = unified.get("vocabulary", {})
    for category, entries in vocab.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            nko = entry.get("nko", "")
            english = entry.get("en", "") or entry.get("english", "")
            bambara = entry.get("bambara", "")
            if nko and english:
                pairs.append({
                    "english": english,
                    "nko": nko,
                    "ipa": _safe_ipa(nko),
                    "bambara": bambara,
                    "category": f"vocabulary_{category}",
                    "source": "nko-unified.json",
                })

    # Proverbs (nested by language in unified)
    proverbs = unified.get("proverbs", {})
    for lang, section in proverbs.items():
        if not isinstance(section, dict):
            continue
        entries = section.get("entries", [])
        for entry in entries:
            nko = entry.get("nko", "")
            english = entry.get("english", "") or entry.get("meaning", "")
            if nko and english:
                pairs.append({
                    "english": english,
                    "nko": nko,
                    "ipa": entry.get("transliteration", ""),
                    "category": f"proverb_{lang}",
                    "source": "nko-unified.json",
                })

    # Wisdom mappings (programming proverbs)
    wisdom = unified.get("wisdom", {})
    concepts = wisdom.get("concepts", {})
    for concept_name, concept_data in concepts.items():
        if not isinstance(concept_data, dict):
            continue
        for prov in concept_data.get("proverbs", []):
            nko = prov.get("nko", "")
            english = prov.get("english", "")
            if nko and english:
                pairs.append({
                    "english": english,
                    "nko": nko,
                    "ipa": prov.get("latin", ""),
                    "category": f"wisdom_{concept_name}",
                    "source": "nko-unified.json",
                })

    return pairs


def extract_from_cognates(cognates: dict) -> list:
    """Extract parallel pairs from nko-cognates.json."""
    pairs = []

    for section_name in ["manding_internal", "arabic_loans", "french_loans", "native_manding"]:
        entries = cognates.get(section_name, [])
        for entry in entries:
            nko = entry.get("nko", "")
            concept = entry.get("concept", "")
            if nko and concept:
                pairs.append({
                    "english": concept,
                    "nko": nko,
                    "ipa": _safe_ipa(nko),
                    "latin": entry.get("latin", ""),
                    "category": f"cognate_{section_name}",
                    "source": "nko-cognates.json",
                })

    return pairs


def extract_from_proverbs_unified(data_dir: Path) -> list:
    """Extract from proverbs-unified.json (flat list under 'proverbs' key)."""
    pairs = []
    path = data_dir / "cultural" / "proverbs-unified.json"
    data = load_json(path)
    if not data:
        return pairs

    for entry in data.get("proverbs", []):
        nko = entry.get("text_nko", "")
        english = entry.get("english_translation", "")
        meaning = entry.get("meaning", "")
        latin = entry.get("text_latin", "")
        lang = entry.get("language", "unknown")

        if nko and english:
            pairs.append({
                "english": english,
                "nko": nko,
                "ipa": latin,
                "meaning": meaning,
                "category": f"proverb_{lang.lower()}",
                "source": "proverbs-unified.json",
            })
        elif not nko and english and latin:
            # Latin-only proverbs, still useful for English side
            pairs.append({
                "english": english,
                "nko": "",
                "ipa": latin,
                "meaning": meaning,
                "latin_original": latin,
                "category": f"proverb_{lang.lower()}_latin_only",
                "source": "proverbs-unified.json",
            })

    return pairs


def extract_from_greetings(data_dir: Path) -> list:
    """Extract from greetings-unified.json (text_nko + english_translation)."""
    pairs = []
    path = data_dir / "cultural" / "greetings-unified.json"
    data = load_json(path)
    if not data:
        return pairs

    for entry in data.get("greetings", []):
        nko = entry.get("text_nko", "")
        english = entry.get("english_translation", "")
        latin = entry.get("text_latin", "")
        response = entry.get("expected_response", "")

        if nko and english:
            pairs.append({
                "english": english,
                "nko": nko,
                "ipa": latin,
                "category": f"greeting_{entry.get('phase', 'general')}",
                "source": "greetings-unified.json",
            })
            # Also add the expected response if available
            if response:
                pairs.append({
                    "english": f"Response: {english}",
                    "nko": response,
                    "ipa": "",
                    "category": "greeting_response",
                    "source": "greetings-unified.json",
                })

    return pairs


def extract_from_blessings(data_dir: Path) -> list:
    """Extract from blessings-unified.json (text_nko + english_translation)."""
    pairs = []
    path = data_dir / "cultural" / "blessings-unified.json"
    data = load_json(path)
    if not data:
        return pairs

    for entry in data.get("blessings", []):
        nko = entry.get("text_nko", "")
        english = entry.get("english_translation", "")
        latin = entry.get("text_latin", "")

        if nko and english:
            pairs.append({
                "english": english,
                "nko": nko,
                "ipa": latin,
                "category": f"blessing_{entry.get('life_event', 'general')}",
                "source": "blessings-unified.json",
            })

    return pairs


def extract_from_concepts(data_dir: Path) -> list:
    """Extract from cultural-concepts.json (text_nko + english_translation)."""
    pairs = []
    path = data_dir / "cultural" / "cultural-concepts.json"
    data = load_json(path)
    if not data:
        return pairs

    for entry in data.get("concepts", []):
        nko = entry.get("text_nko", "")
        english = entry.get("english_translation", "")
        latin = entry.get("text_latin", "")
        usage = entry.get("usage", "")

        if nko and english:
            pairs.append({
                "english": english,
                "nko": nko,
                "ipa": latin,
                "category": f"concept_{entry.get('type', 'concept')}",
                "source": "cultural-concepts.json",
            })
            # Add usage as a separate pair if distinct
            if usage and usage != english:
                pairs.append({
                    "english": usage,
                    "nko": nko,
                    "ipa": latin,
                    "category": f"concept_usage_{entry.get('type', 'concept')}",
                    "source": "cultural-concepts.json",
                })

    return pairs


def extract_from_clans(data_dir: Path) -> list:
    """Extract from clans-unified.json (name_nko + jamu + jamu_meaning)."""
    pairs = []
    path = data_dir / "cultural" / "clans-unified.json"
    data = load_json(path)
    if not data:
        return pairs

    for clan in data.get("clans", []):
        name_nko = clan.get("name_nko", "")
        name_latin = clan.get("name_latin", "")
        jamu = clan.get("jamu", "")
        jamu_meaning = clan.get("jamu_meaning", "")
        notes = clan.get("notes", "")

        # Clan name pair
        if name_nko and name_latin:
            pairs.append({
                "english": name_latin,
                "nko": name_nko,
                "ipa": "",
                "category": "clan_name",
                "source": "clans-unified.json",
            })

        # Jamu (praise name) pair
        if jamu and jamu_meaning:
            pairs.append({
                "english": jamu_meaning,
                "nko": jamu,
                "ipa": "",
                "category": "clan_jamu",
                "source": "clans-unified.json",
            })

        # Appropriate greetings
        for greeting in clan.get("appropriate_greetings", []):
            if greeting:
                pairs.append({
                    "english": f"Greeting for {name_latin}: {jamu_meaning}",
                    "nko": greeting,
                    "ipa": "",
                    "category": "clan_greeting",
                    "source": "clans-unified.json",
                })

    return pairs


def extract_from_calendar(data_dir: Path) -> list:
    """Extract from cultural-calendar.json (name_nko + description + greetings)."""
    pairs = []
    path = data_dir / "cultural" / "cultural-calendar.json"
    data = load_json(path)
    if not data:
        return pairs

    for event in data.get("events", []):
        name_nko = event.get("name_nko", "")
        description = event.get("description", "")

        # Event name pair
        if name_nko and description:
            pairs.append({
                "english": description,
                "nko": name_nko,
                "ipa": event.get("name_latin", ""),
                "category": "calendar_event",
                "source": "cultural-calendar.json",
            })

        # Event-specific greetings
        for greeting in event.get("greetings", []):
            if greeting:
                pairs.append({
                    "english": f"Greeting for {description}",
                    "nko": greeting,
                    "ipa": "",
                    "category": "calendar_greeting",
                    "source": "cultural-calendar.json",
                })

        # Associated proverbs
        for proverb in event.get("associated_proverbs", []):
            if proverb:
                pairs.append({
                    "english": f"Proverb associated with {description}",
                    "nko": proverb,
                    "ipa": "",
                    "category": "calendar_proverb",
                    "source": "cultural-calendar.json",
                })

    return pairs


def extract_from_keyboard_vocab(data_dir: Path) -> list:
    """Extract from keyboard/vocabulary.json (NKO key -> meaning dict)."""
    pairs = []
    path = data_dir / "keyboard" / "vocabulary.json"
    data = load_json(path)
    if not data:
        return pairs

    # Greetings: {"NKO text": {"transliteration": ..., "meaning": ...}}
    for nko_text, info in data.get("greetings", {}).items():
        if isinstance(info, dict):
            english = info.get("meaning", "")
            if nko_text and english:
                pairs.append({
                    "english": english,
                    "nko": nko_text,
                    "ipa": info.get("transliteration", ""),
                    "category": "keyboard_greeting",
                    "source": "keyboard/vocabulary.json",
                })
            # Also add response form
            response = info.get("response", "")
            if response:
                pairs.append({
                    "english": f"Response: {english}",
                    "nko": response,
                    "ipa": "",
                    "category": "keyboard_greeting_response",
                    "source": "keyboard/vocabulary.json",
                })

    # Pronouns: {"NKO": {"trans": ..., "meaning": ...}}
    for nko_text, info in data.get("pronouns", {}).items():
        if isinstance(info, dict):
            english = info.get("meaning", "")
            if nko_text and english:
                pairs.append({
                    "english": english,
                    "nko": nko_text,
                    "ipa": info.get("trans", ""),
                    "category": "keyboard_pronoun",
                    "source": "keyboard/vocabulary.json",
                })

    # Verbs: {"NKO": {"trans": ..., "meaning": ...}}
    for nko_text, info in data.get("verbs", {}).items():
        if isinstance(info, dict):
            english = info.get("meaning", "")
            if nko_text and english:
                pairs.append({
                    "english": english,
                    "nko": nko_text,
                    "ipa": info.get("trans", ""),
                    "category": "keyboard_verb",
                    "source": "keyboard/vocabulary.json",
                })

    # Nouns: nested {"family": {"NKO": {"trans": ..., "meaning": ...}}, ...}
    for subcategory, entries in data.get("nouns", {}).items():
        if not isinstance(entries, dict):
            continue
        for nko_text, info in entries.items():
            if isinstance(info, dict):
                english = info.get("meaning", "")
                if nko_text and english:
                    pairs.append({
                        "english": english,
                        "nko": nko_text,
                        "ipa": info.get("trans", ""),
                        "category": f"keyboard_noun_{subcategory}",
                        "source": "keyboard/vocabulary.json",
                    })

    # Numbers: {"NKO digit": {"value": N, "trans": ...}}
    for nko_digit, info in data.get("numbers", {}).items():
        if isinstance(info, dict):
            value = info.get("value", "")
            trans = info.get("trans", "")
            if nko_digit and trans:
                pairs.append({
                    "english": f"{value} ({trans})",
                    "nko": nko_digit,
                    "ipa": trans,
                    "category": "keyboard_number",
                    "source": "keyboard/vocabulary.json",
                })

    # Proverbs: [{"nko": ..., "meaning": ...}]
    for entry in data.get("proverbs", []):
        if isinstance(entry, dict):
            nko = entry.get("nko", "")
            meaning = entry.get("meaning", "")
            if nko and meaning:
                pairs.append({
                    "english": meaning,
                    "nko": nko,
                    "ipa": entry.get("trans", ""),
                    "literal": entry.get("literal", ""),
                    "category": "keyboard_proverb",
                    "source": "keyboard/vocabulary.json",
                })

    # Phrases: {"common": [...], "questions": [...]}
    for phrase_cat, phrase_list in data.get("phrases", {}).items():
        if not isinstance(phrase_list, list):
            continue
        for entry in phrase_list:
            if isinstance(entry, dict):
                nko = entry.get("nko", "")
                english = entry.get("meaning", "")
                if nko and english:
                    pairs.append({
                        "english": english,
                        "nko": nko,
                        "ipa": entry.get("trans", ""),
                        "category": f"keyboard_phrase_{phrase_cat}",
                        "source": "keyboard/vocabulary.json",
                    })

    return pairs


def extract_from_bridge_cognates() -> list:
    """Extract from cross-script-bridge/data/cognates.json."""
    pairs = []
    path = Path.home() / "Desktop" / "cross-script-bridge" / "data" / "cognates.json"
    data = load_json(path)
    if not data:
        return pairs

    # Cognates: {"concept_key": {"nko": ..., "arabic": ..., "latin": ...}}
    for concept, info in data.get("cognates", {}).items():
        if isinstance(info, dict):
            nko = info.get("nko", "")
            if nko and concept:
                pairs.append({
                    "english": concept.replace("_", " "),
                    "nko": nko,
                    "ipa": info.get("latin", ""),
                    "arabic": info.get("arabic", ""),
                    "category": "bridge_cognate",
                    "source": "cross-script-bridge/cognates.json",
                })

    # Phrases: {"key": {"nko": ..., "meaning": ...}}
    for key, info in data.get("phrases", {}).items():
        if isinstance(info, dict):
            nko = info.get("nko", "")
            english = info.get("meaning", "")
            if nko and english:
                pairs.append({
                    "english": english,
                    "nko": nko,
                    "ipa": info.get("latin", ""),
                    "arabic": info.get("arabic", ""),
                    "category": "bridge_phrase",
                    "source": "cross-script-bridge/cognates.json",
                })

    return pairs


def extract_from_keyboard_ai() -> list:
    """Extract parallel pairs from keyboard_ai.py MANDING_LEXICON."""
    pairs = []
    try:
        from nko.keyboard_ai import MANDING_LEXICON, PROVERBS
    except ImportError:
        print("  [skip] keyboard_ai.py import failed")
        return pairs

    if isinstance(MANDING_LEXICON, dict):
        for nko_text, info in MANDING_LEXICON.items():
            if isinstance(info, dict):
                english = info.get("en", "")
                bambara = info.get("bambara", "")
                if nko_text and english:
                    pairs.append({
                        "english": english,
                        "nko": nko_text,
                        "ipa": _safe_ipa(nko_text),
                        "bambara": bambara,
                        "category": info.get("context", "lexicon"),
                        "source": "keyboard_ai.py",
                    })

    if isinstance(PROVERBS, list):
        for item in PROVERBS:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                nko, translit, english = item[0], item[1], item[2]
                pairs.append({
                    "english": english,
                    "nko": nko,
                    "ipa": translit,
                    "category": "keyboard_ai_proverb",
                    "source": "keyboard_ai.py",
                })

    return pairs


def generate_conjugation_paradigms(keyboard_data: dict) -> list:
    """Generate conjugation paradigms from verbs x pronouns to expand corpus."""
    pairs = []

    verbs = keyboard_data.get("verbs", {})
    pronouns = keyboard_data.get("pronouns", {})

    if not verbs or not pronouns:
        return pairs

    # Tense markers
    tenses = [
        ("ߓߍ߬", "be", "present"),        # present/progressive
        ("ߕߍ", "te", "negative"),          # negative
        ("ߞߊ߬", "ka", "past/infinitive"),  # past/infinitive
    ]

    for v_nko, v_info in verbs.items():
        if not isinstance(v_info, dict):
            continue
        v_meaning = v_info.get("meaning", "")
        v_trans = v_info.get("trans", "")

        for p_nko, p_info in pronouns.items():
            if not isinstance(p_info, dict):
                continue
            p_meaning = p_info.get("meaning", "").split(",")[0].strip()
            p_trans = p_info.get("trans", "")

            for t_nko, t_trans, tense_name in tenses:
                # Build NKO sentence: pronoun + tense_marker + verb
                nko_sentence = f"{p_nko} {t_nko} {v_nko}"

                # Build English
                if tense_name == "present":
                    eng = f"{p_meaning} {v_meaning.replace('to ', '')}"
                elif tense_name == "negative":
                    eng = f"{p_meaning} do not {v_meaning.replace('to ', '')}"
                else:
                    eng = f"{p_meaning} {v_meaning}"

                pairs.append({
                    "english": eng,
                    "nko": nko_sentence,
                    "ipa": f"{p_trans} {t_trans} {v_trans}",
                    "category": f"conjugation_{tense_name}",
                    "source": "generated_paradigm",
                })

    return pairs


def _safe_ipa(nko_text: str) -> str:
    """Safely get IPA, returning empty string on failure."""
    try:
        return NKoPhonetics.to_ipa(nko_text) if nko_text else ""
    except Exception:
        return ""


def deduplicate(pairs: list) -> list:
    """Remove duplicate pairs based on NKO text."""
    seen = set()
    result = []
    for p in pairs:
        # Dedupe by NKO text (primary), fallback to English
        key = p.get("nko", "").strip()
        if not key:
            key = p.get("english", "").strip()
        if key and key not in seen:
            seen.add(key)
            result.append(p)
    return result


def main():
    data_dir = _NKO_ROOT / "data"

    # Load source files
    unified = load_json(data_dir / "nko-unified.json")
    cognates = load_json(data_dir / "nko-cognates.json")
    keyboard_data = load_json(data_dir / "keyboard" / "vocabulary.json")

    # Extract from all sources
    all_pairs = []

    print("Extracting from sources:")

    pairs = extract_from_unified(unified)
    print(f"  nko-unified.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_cognates(cognates)
    print(f"  nko-cognates.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_proverbs_unified(data_dir)
    print(f"  proverbs-unified.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_greetings(data_dir)
    print(f"  greetings-unified.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_blessings(data_dir)
    print(f"  blessings-unified.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_concepts(data_dir)
    print(f"  cultural-concepts.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_clans(data_dir)
    print(f"  clans-unified.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_calendar(data_dir)
    print(f"  cultural-calendar.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_keyboard_vocab(data_dir)
    print(f"  keyboard/vocabulary.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_bridge_cognates()
    print(f"  cross-script-bridge/cognates.json: {len(pairs)} entries")
    all_pairs.extend(pairs)

    pairs = extract_from_keyboard_ai()
    print(f"  keyboard_ai.py: {len(pairs)} entries")
    all_pairs.extend(pairs)

    # Generate conjugation paradigms for corpus expansion
    pairs = generate_conjugation_paradigms(keyboard_data)
    print(f"  conjugation paradigms: {len(pairs)} entries")
    all_pairs.extend(pairs)

    # Deduplicate
    pairs = deduplicate(all_pairs)

    # Filter: separate by NKO availability
    nko_pairs = [p for p in pairs if p.get("nko")]
    all_with_english = [p for p in pairs if p.get("english")]

    # Write JSONL
    out_path = Path(__file__).parent / "parallel_corpus.jsonl"
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Stats
    categories = {}
    for p in pairs:
        cat = p.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nParallel corpus built: {len(pairs)} total entries")
    print(f"  With NKO text: {len(nko_pairs)}")
    print(f"  With English:  {len(all_with_english)}")
    print(f"  Categories ({len(categories)}):")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print(f"  Written to: {out_path}")


if __name__ == "__main__":
    main()
