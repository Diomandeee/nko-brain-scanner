#!/usr/bin/env python3
"""
Translate cognitive twin SFT data from English/Latin to N'Ko.

Takes a JSONL file with SFT message pairs and transliterates content to N'Ko
using the canonical transliteration engine at ~/Desktop/NKo/nko/transliterate.py.

Strategy:
  - Bambara/Manding Latin text: direct transliteration via IPA bridge (high fidelity)
  - English text: phonetic transliteration (experimental, lower fidelity)
  - Mixed text: detect script per segment, transliterate each accordingly

The output preserves the SFT message structure:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python3 translate_sft_to_nko.py \
        --input ~/sft_training_data.jsonl \
        --output data/nko_sft.jsonl \
        --stats data/translation_stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the NKo package to path
NKO_PATH = os.path.expanduser("~/Desktop/NKo")
if NKO_PATH not in sys.path:
    sys.path.insert(0, NKO_PATH)

try:
    from nko.transliterate import (
        NkoTransliterator,
        detect_script,
        Script,
    )
    TRANSLITERATOR = NkoTransliterator()
    print("N'Ko transliterator loaded successfully")
except ImportError as e:
    print(f"ERROR: Could not import nko.transliterate: {e}")
    print(f"Ensure {NKO_PATH} exists and contains nko/transliterate.py")
    sys.exit(1)


def has_nko(text):
    """Check if text already contains N'Ko characters."""
    return any(0x07C0 <= ord(ch) <= 0x07FF for ch in text)


def transliterate_to_nko(text):
    """Transliterate text to N'Ko, handling mixed scripts.

    Returns:
        nko_text: transliterated string
        confidence: float 0-1 indicating quality
        source_script: detected source script
    """
    if not text or not text.strip():
        return text, 1.0, "empty"

    # Already N'Ko? Pass through.
    if has_nko(text):
        return text, 1.0, "nko"

    detected = detect_script(text)

    try:
        result = TRANSLITERATOR.convert(text, source=detected, target=Script.NKO)
        return result.target_text, result.confidence, str(detected)
    except Exception as e:
        # Fallback: character-by-character attempt
        return text, 0.0, f"error:{e}"


def translate_message(message):
    """Translate a single SFT message to N'Ko."""
    content = message.get("content", "")
    nko_content, confidence, source = transliterate_to_nko(content)
    return {
        "role": message["role"],
        "content": nko_content,
    }, confidence, source


def translate_example(example):
    """Translate a full SFT example (list of messages) to N'Ko."""
    messages = example.get("messages", [])
    translated_messages = []
    confidences = []
    sources = []

    for msg in messages:
        translated, conf, src = translate_message(msg)
        translated_messages.append(translated)
        confidences.append(conf)
        sources.append(src)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "messages": translated_messages,
    }, avg_confidence, sources


def main():
    parser = argparse.ArgumentParser(description="Translate SFT data to N'Ko")
    parser.add_argument("--input", required=True, help="Input SFT JSONL")
    parser.add_argument("--output", required=True, help="Output N'Ko SFT JSONL")
    parser.add_argument("--stats", default=None, help="Output translation statistics JSON")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum confidence to include example (0-1)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0=all)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Process
    total = 0
    kept = 0
    skipped = 0
    confidence_sum = 0
    source_counts = {}

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            total += 1
            if args.limit and total > args.limit:
                break

            try:
                example = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            translated, confidence, sources = translate_example(example)

            if confidence >= args.min_confidence:
                fout.write(json.dumps(translated, ensure_ascii=False) + "\n")
                kept += 1
                confidence_sum += confidence
            else:
                skipped += 1

            for src in sources:
                source_counts[src] = source_counts.get(src, 0) + 1

            if total % 500 == 0:
                print(f"  Processed {total} examples, kept {kept}, skipped {skipped}")

    avg_conf = confidence_sum / kept if kept > 0 else 0
    print(f"\nTranslation complete:")
    print(f"  Total processed: {total}")
    print(f"  Kept: {kept} (avg confidence: {avg_conf:.3f})")
    print(f"  Skipped: {skipped}")
    print(f"  Source scripts: {source_counts}")
    print(f"  Output: {args.output}")

    if args.stats:
        stats = {
            "total": total,
            "kept": kept,
            "skipped": skipped,
            "avg_confidence": avg_conf,
            "source_scripts": source_counts,
            "min_confidence_threshold": args.min_confidence,
        }
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats: {args.stats}")


if __name__ == "__main__":
    main()
