#!/usr/bin/env python3
"""
NLLB Fine-Tune Data Preparation
================================
Extracts parallel Bambara↔English/French pairs from all available sources
and prepares them for NLLB-200 LoRA fine-tuning.

Sources:
  1. Ankataa dictionary (2,101 entries → ~1,977 with English)
  2. Parallel corpus (460 IPA/N'Ko ↔ English pairs)
  3. Bayelemabaga pilot_pairs (500 Bambara ↔ French ↔ N'Ko)
  4. Bayelemabaga pilot_sft (1,001 French→N'Ko instruction pairs)
  5. Cultural data: proverbs, greetings, blessings
  6. COMMON_WORDS from bambara_translator.py

Outputs:
  - nllb_train.jsonl: Training set (90%)
  - nllb_valid.jsonl: Validation set (10%)
  Format: {"src": "bambara text", "tgt": "english text", "src_lang": "bam_Latn", "tgt_lang": "eng_Latn", "source": "..."}
"""

import json
import random
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
PIPELINE = BASE / "pipeline" / "data"
NKO = BASE / "nko" / "data"
OUTPUT = BASE / "asr" / "nllb_data"


def extract_ankataa():
    """Extract Bambara→English pairs from Ankataa dictionary."""
    pairs = []

    # Try both paths
    for p in [
        PIPELINE / "dictionary" / "ankataa_dictionary_20260319.json",
        BASE / "asr" / "ankataa_dictionary_20260319.json",
    ]:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for entry in data:
                word = entry.get("word", "").strip()
                defs_en = entry.get("definitions_en", [])
                defs_fr = entry.get("definitions_fr", [])

                # English pairs
                if word and defs_en:
                    # Use first definition as primary
                    pairs.append({
                        "src": word,
                        "tgt": defs_en[0],
                        "src_lang": "bam_Latn",
                        "tgt_lang": "eng_Latn",
                        "source": "ankataa_dict",
                    })

                # French pairs
                if word and defs_fr:
                    pairs.append({
                        "src": word,
                        "tgt": defs_fr[0],
                        "src_lang": "bam_Latn",
                        "tgt_lang": "fra_Latn",
                        "source": "ankataa_dict_fr",
                    })

                # If entry has examples with translations, add those too
                for ex in entry.get("examples", []):
                    bam = ex.get("bambara", "").strip()
                    en = (ex.get("english") or "").strip()
                    fr = (ex.get("french") or "").strip()
                    if bam and en:
                        pairs.append({
                            "src": bam, "tgt": en,
                            "src_lang": "bam_Latn", "tgt_lang": "eng_Latn",
                            "source": "ankataa_example",
                        })
                    if bam and fr:
                        pairs.append({
                            "src": bam, "tgt": fr,
                            "src_lang": "bam_Latn", "tgt_lang": "fra_Latn",
                            "source": "ankataa_example_fr",
                        })
            break
    return pairs


def extract_parallel_corpus():
    """Extract from parallel_corpus.jsonl (IPA/N'Ko ↔ English)."""
    pairs = []
    p = DATA / "parallel_corpus.jsonl"
    if not p.exists():
        return pairs

    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            ipa = entry.get("ipa", "").strip()
            eng = entry.get("english", "").strip()
            if ipa and eng:
                pairs.append({
                    "src": ipa,
                    "tgt": eng,
                    "src_lang": "bam_Latn",
                    "tgt_lang": "eng_Latn",
                    "source": "parallel_corpus",
                })
    return pairs


def extract_bayelemabaga():
    """Extract from Bayelemabaga pilot data (Bambara ↔ French)."""
    pairs = []

    # pilot_pairs.jsonl: trilinear Bambara/French/N'Ko
    p = DATA / "bayelemabaga" / "pilot_pairs.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                bam = entry.get("bambara", "").strip()
                fr = entry.get("french", "").strip()
                if bam and fr:
                    pairs.append({
                        "src": bam,
                        "tgt": fr,
                        "src_lang": "bam_Latn",
                        "tgt_lang": "fra_Latn",
                        "source": "bayelemabaga_pairs",
                    })

    # pilot_sft.jsonl: instruction pairs (French→N'Ko)
    p = DATA / "bayelemabaga" / "pilot_sft.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                messages = entry.get("messages", [])
                if len(messages) >= 2:
                    user_msg = messages[0].get("content", "")
                    # Extract French source from instruction
                    if "Translate to N'Ko:" in user_msg:
                        fr = user_msg.replace("Translate to N'Ko:", "").strip()
                        nko = messages[1].get("content", "").strip()
                        if fr and nko:
                            # We want Bambara→French direction for NLLB
                            # The French text is the source, keep as-is
                            pairs.append({
                                "src": fr,
                                "tgt": fr,  # Self-pair for French data
                                "src_lang": "fra_Latn",
                                "tgt_lang": "fra_Latn",
                                "source": "bayelemabaga_sft_skip",
                            })
    return [p for p in pairs if p["source"] != "bayelemabaga_sft_skip"]


def extract_cultural():
    """Extract from cultural JSON files (proverbs, greetings, blessings)."""
    pairs = []
    cultural = NKO / "cultural"
    if not cultural.exists():
        return pairs

    for fname, key in [
        ("proverbs-unified.json", "proverbs"),
        ("greetings-unified.json", "greetings"),
        ("blessings-unified.json", "blessings"),
    ]:
        p = cultural / fname
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        items = data.get(key, [])
        for item in items:
            latin = (item.get("text_latin") or "").strip()
            eng = (item.get("english_translation") or "").strip()
            if latin and eng:
                pairs.append({
                    "src": latin,
                    "tgt": eng,
                    "src_lang": "bam_Latn",
                    "tgt_lang": "eng_Latn",
                    "source": f"cultural_{key}",
                })
    return pairs


def extract_common_words():
    """Generate pairs from the curated COMMON_WORDS dictionary."""
    from bambara_translator import COMMON_WORDS
    pairs = []
    for bam, eng in COMMON_WORDS.items():
        pairs.append({
            "src": bam,
            "tgt": eng,
            "src_lang": "bam_Latn",
            "tgt_lang": "eng_Latn",
            "source": "common_words",
        })
    return pairs


def add_reverse_pairs(pairs):
    """Add reverse direction (English→Bambara) for bidirectional training."""
    reverse = []
    for p in pairs:
        if p["src_lang"] == "bam_Latn" and p["tgt_lang"] == "eng_Latn":
            reverse.append({
                "src": p["tgt"],
                "tgt": p["src"],
                "src_lang": "eng_Latn",
                "tgt_lang": "bam_Latn",
                "source": p["source"] + "_rev",
            })
        elif p["src_lang"] == "bam_Latn" and p["tgt_lang"] == "fra_Latn":
            reverse.append({
                "src": p["tgt"],
                "tgt": p["src"],
                "src_lang": "fra_Latn",
                "tgt_lang": "bam_Latn",
                "source": p["source"] + "_rev",
            })
    return reverse


def deduplicate(pairs):
    """Remove duplicate src+tgt pairs."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["src"].lower().strip(), p["tgt"].lower().strip(), p["src_lang"], p["tgt_lang"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    print("=" * 60)
    print("NLLB Fine-Tune Data Preparation")
    print("=" * 60)

    # Extract from all sources
    sources = {
        "Ankataa dictionary": extract_ankataa(),
        "Parallel corpus": extract_parallel_corpus(),
        "Bayelemabaga": extract_bayelemabaga(),
        "Cultural data": extract_cultural(),
        "Common words": extract_common_words(),
    }

    all_pairs = []
    for name, pairs in sources.items():
        print(f"  {name:25s}: {len(pairs):5d} pairs")
        all_pairs.extend(pairs)

    print(f"\n  {'Total (forward)':25s}: {len(all_pairs):5d} pairs")

    # Add reverse pairs for bidirectional training
    reverse = add_reverse_pairs(all_pairs)
    print(f"  {'Reverse pairs':25s}: {len(reverse):5d} pairs")
    all_pairs.extend(reverse)

    # Deduplicate
    all_pairs = deduplicate(all_pairs)
    print(f"  {'After dedup':25s}: {len(all_pairs):5d} pairs")

    # Stats by language pair
    lang_pairs = {}
    for p in all_pairs:
        key = f"{p['src_lang']}→{p['tgt_lang']}"
        lang_pairs[key] = lang_pairs.get(key, 0) + 1
    print(f"\n  Language pair distribution:")
    for lp, count in sorted(lang_pairs.items()):
        print(f"    {lp:25s}: {count:5d}")

    # Filter out very short pairs (< 3 chars) and very long (> 500 chars)
    filtered = [
        p for p in all_pairs
        if 3 <= len(p["src"]) <= 500 and 3 <= len(p["tgt"]) <= 500
    ]
    print(f"\n  After length filter:     {len(filtered):5d} pairs")

    # Split 90/10 train/valid
    random.seed(42)
    random.shuffle(filtered)
    split_idx = int(len(filtered) * 0.9)
    train = filtered[:split_idx]
    valid = filtered[split_idx:]

    print(f"  Train set:               {len(train):5d} pairs")
    print(f"  Valid set:               {len(valid):5d} pairs")

    # Save
    OUTPUT.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT / "nllb_train.jsonl"
    with open(train_path, "w") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    valid_path = OUTPUT / "nllb_valid.jsonl"
    with open(valid_path, "w") as f:
        for p in valid:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Saved: {train_path}")
    print(f"  Saved: {valid_path}")

    # Show samples
    print(f"\n  Sample pairs:")
    for p in train[:5]:
        print(f"    [{p['src_lang']}→{p['tgt_lang']}] {p['src'][:40]:40s} → {p['tgt'][:40]}")


if __name__ == "__main__":
    main()
