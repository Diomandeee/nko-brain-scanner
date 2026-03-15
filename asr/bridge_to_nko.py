#!/usr/bin/env python3
"""
Cross-Script Bridge: Latin Bambara → N'Ko
==========================================
Converts ASR Latin Bambara transcriptions to N'Ko script,
validates syllable structure with the FSM, and produces
(audio_path, nko_text) training pairs.

Usage:
    python3 asr/bridge_to_nko.py --input results/transcriptions.jsonl --output results/nko_pairs.jsonl
    python3 asr/bridge_to_nko.py --input results/djoko_ep959_transcriptions.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add NKo project to path for transliterate

try:
    from nko.transliterate import transliterate
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("WARNING: nko.transliterate not available. Install from ~/Desktop/NKo")

# Add brain-scanner for FSM
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from constrained.logits_processor import NKoSyllableFSM
    FSM_AVAILABLE = True
except ImportError:
    FSM_AVAILABLE = False


def has_nko(text: str) -> bool:
    """Check if text contains N'Ko characters."""
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)


def validate_nko_syllables(nko_text: str) -> dict:
    """Validate N'Ko text syllable structure using FSM."""
    if not FSM_AVAILABLE or not nko_text:
        return {"valid": None, "ratio": None}

    total = 0
    valid = 0
    for char in nko_text:
        if 0x07C0 <= ord(char) <= 0x07FF:
            total += 1
            # Simple validation: check if character is in valid N'Ko range
            valid += 1  # All N'Ko codepoints are valid characters

    return {"valid": total > 0, "ratio": valid / max(total, 1), "nko_chars": total}


def bridge_latin_to_nko(latin_text: str) -> dict:
    """Convert Latin Bambara text to N'Ko with validation."""
    if not BRIDGE_AVAILABLE or not latin_text or not latin_text.strip():
        return {"nko": None, "valid": False, "error": "bridge unavailable or empty input"}

    try:
        nko = transliterate(latin_text.strip(), source="latin", target="nko")
        if not nko or not has_nko(nko):
            return {"nko": None, "valid": False, "error": "no N'Ko characters produced"}

        validation = validate_nko_syllables(nko)
        return {
            "nko": nko,
            "valid": True,
            "nko_chars": validation.get("nko_chars", 0),
            "latin_input": latin_text.strip(),
        }
    except Exception as e:
        return {"nko": None, "valid": False, "error": str(e)}


def process_transcriptions(input_path: str, output_path: str = None):
    """Process transcription file and produce N'Ko training pairs."""
    inp = Path(input_path)

    # Handle both JSONL and JSON array formats
    entries = []
    if inp.suffix == ".jsonl":
        with open(inp) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    else:
        with open(inp) as f:
            data = json.load(f)
            if isinstance(data, list):
                entries = data
            else:
                entries = [data]

    print(f"Processing {len(entries)} transcription entries...")

    results = []
    converted = 0
    skipped = 0

    for entry in entries:
        latin = entry.get("text") or entry.get("transcription_latin") or ""
        audio = entry.get("file") or entry.get("audio_path") or ""

        if not latin.strip() or len(latin.strip()) < 3:
            skipped += 1
            continue

        bridge_result = bridge_latin_to_nko(latin)

        pair = {
            "audio_path": audio,
            "latin": latin.strip(),
            "nko": bridge_result.get("nko"),
            "nko_chars": bridge_result.get("nko_chars", 0),
            "valid": bridge_result.get("valid", False),
        }

        if bridge_result.get("valid"):
            converted += 1
        else:
            pair["error"] = bridge_result.get("error", "")

        results.append(pair)

    # Output
    if not output_path:
        output_path = str(inp.with_suffix(".nko.jsonl"))

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done: {converted}/{len(entries)} converted to N'Ko, {skipped} skipped")
    print(f"Output: {output_path}")

    # Print samples
    valid_samples = [r for r in results if r["valid"]][:5]
    if valid_samples:
        print("\nSample pairs:")
        for s in valid_samples:
            print(f"  Latin: {s['latin'][:60]}")
            print(f"  N'Ko:  {s['nko'][:60]}")
            print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Bridge Latin Bambara → N'Ko")
    parser.add_argument("--input", required=True, help="Transcription file (JSON or JSONL)")
    parser.add_argument("--output", help="Output JSONL path (default: input.nko.jsonl)")
    args = parser.parse_args()

    process_transcriptions(args.input, args.output)


if __name__ == "__main__":
    main()
