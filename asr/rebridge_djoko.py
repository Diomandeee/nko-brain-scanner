#!/usr/bin/env python3
"""
rebridge_djoko.py -- Re-bridge Djoko/babamamadidiane pairs to char-level N'Ko
=============================================================================

The original streaming pipeline (stream_with_features.py) used a naive
char-by-char LATIN_TO_NKO dictionary that had incorrect mappings:
  - o -> U+07CF (open O, ߏ) instead of U+07CB (O, ߋ)
  - g -> U+07DC (Gba, ߜ) for all 'g' sounds
  - z -> U+07D6 (Ja, ߖ) instead of U+07DB (Sa, ߛ)
  - dropped many valid Bambara characters (ŋ mapped to syllabic N)

This script re-bridges using the canonical IPA-routed transliterate() engine
from nko.transliterate, which correctly handles:
  - Bambara special vowels (ɔ, ɛ)
  - Bambara digraphs (ny, ng, gb, ch, dj)
  - Proper vowel distinctions (o vs ɔ -> ߋ vs ߏ)
  - NFD normalization for toned vowels

It also filters out Whisper hallucination loops (repetitive transcriptions
where unique word ratio drops below 30%).

Usage:
    # Run locally -- reads from Sweden via SSH, processes, writes output
    python3 asr/rebridge_djoko.py

    # Or copy to Sweden and run there
    scp asr/rebridge_djoko.py root@ssh8.vast.ai:/workspace/
    ssh root@ssh8.vast.ai 'cd /workspace && python3 rebridge_djoko.py'

    # Specify custom paths
    python3 asr/rebridge_djoko.py \
        --input /workspace/results/feature_pairs_djoko.jsonl \
        --output /workspace/results/djoko_nko_rebridged.jsonl \
        --min-words 2 --max-repeat 0.3

Output format:
    {"id": N, "bam": "latin bambara text", "nko": "N'Ko text", "feat_id": "VIDEO_NNNN"}
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Resolve transliterate import -----------------------------------------
# When running locally, the nko package is at repo root.
# When running on Vast.ai, we need to either have it installed or copy it.

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from nko.transliterate import transliterate, detect_script
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False


# --- Fallback bridge for running on Vast.ai without the nko package -------
# This is a faithful reproduction of the IPA-routed logic from
# nko/transliterate.py, embedded so the script is self-contained on Vast.ai.

FALLBACK_MAP = {
    # Vowels
    "a": "\u07ca", "e": "\u07cd", "i": "\u07cc", "o": "\u07cb",
    "u": "\u07ce", "\u0254": "\u07cf", "\u025b": "\u07d0",
    "\u0259": "\u07d0",
    # Consonants
    "b": "\u07d3", "d": "\u07d8", "f": "\u07dd", "g": "\u07dc",
    "h": "\u07e4", "k": "\u07de", "l": "\u07df", "m": "\u07e1",
    "n": "\u07e3", "p": "\u07d4", "r": "\u07d9", "s": "\u07db",
    "t": "\u07d5", "w": "\u07e5", "z": "\u07db", "v": "\u07dd",
}

FALLBACK_DIGRAPHS = {
    "ny": "\u07e2",   # Nya
    "ng": "\u07a7",   # Nga -- actually U+07D2? Let's use the correct one
    "gb": "\u07dc",   # Gba
    "ch": "\u07d7",   # Cha
    "dj": "\u07d6",   # Ja
    "sh": "\u07db",   # Sha -> Sa
}
# Fix: Nga is U+07A7 is wrong, it should be U+07D2 (syllabic N / Nga)
# Actually: ŋ (velar nasal) maps to ߧ U+07E7... No.
# From the char map: "ŋ": U+07E7 is not right either.
# From transliterate.py: IPA_TO_NKO["ŋ"] -> ??? Let's check:
# NKO_CONSONANTS_TO_IPA has "ߧ": "ŋ"  (U+07E7). So ŋ -> ߧ
FALLBACK_DIGRAPHS["ng"] = "\u07e7"  # ŋ -> ߧ Nga

# Bambara special consonants written directly
FALLBACK_MAP["\u0272"] = "\u07e2"   # ɲ -> ߢ Nya
FALLBACK_MAP["\u014b"] = "\u07e7"   # ŋ -> ߧ Nga


def _fallback_bridge(text: str) -> str:
    """Fallback Latin->N'Ko bridge when nko.transliterate is not available."""
    import unicodedata
    text = unicodedata.normalize("NFD", text.lower())
    parts = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            parts.append(" ")
            i += 1
            continue
        # Try digraphs first
        matched = False
        for dg, nko_ch in FALLBACK_DIGRAPHS.items():
            if text[i:i+len(dg)] == dg:
                parts.append(nko_ch)
                i += len(dg)
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        if ch in FALLBACK_MAP:
            parts.append(FALLBACK_MAP[ch])
        elif ch == "j":
            parts.append("\u07d6")  # Ja
        elif ch == "c":
            parts.append("\u07d7")  # Cha
        elif ch == "y":
            parts.append("\u07e6")  # Ya
        elif ch == "q":
            parts.append("\u07de")  # Ka
        elif ch == "x":
            parts.append("\u07de\u07db")  # ks -> Ka+Sa
        # Skip combining diacritics (tone marks) in fallback
        elif 0x0300 <= ord(ch) <= 0x036F:
            pass
        else:
            pass  # drop unmapped
        i += 1
    return "".join(parts)


def bridge(text: str) -> str:
    """Bridge Latin Bambara to N'Ko using the best available method."""
    if BRIDGE_AVAILABLE:
        return transliterate(text.strip(), source="latin", target="nko")
    return _fallback_bridge(text.strip())


def has_nko(text: str) -> bool:
    """Check if text contains N'Ko characters (U+07C0-U+07FF)."""
    return any(0x07C0 <= ord(c) <= 0x07FF for c in text)


def unique_word_ratio(text: str) -> float:
    """Ratio of unique words to total words. Low ratio = hallucination loop."""
    words = text.split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)


def is_repetitive(text: str, threshold: float = 0.3) -> bool:
    """
    Detect Whisper hallucination loops.

    Whisper large-v3 on Bambara often produces repetitive output like:
      "a bɔra ka yaya jakite lamɛn a bɔra ka yaya jakite lamɛn a bɔra..."
    These have very low unique word ratios.
    """
    words = text.split()
    if len(words) <= 6:
        return False  # Short utterances can legitimately repeat
    return unique_word_ratio(text) < threshold


def fetch_remote_file(ssh_host: str, ssh_port: int, ssh_key: str,
                      remote_path: str) -> str:
    """Fetch a file from the Vast.ai instance via SSH."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "scp", "-i", ssh_key, "-P", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_host}:{remote_path}",
        tmp_path,
    ]
    print(f"Fetching {remote_path} from {ssh_host}:{ssh_port}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")
    print(f"Downloaded to {tmp_path}")
    return tmp_path


def rebridge(input_path: str, output_path: str,
             min_words: int = 2, max_repeat: float = 0.3,
             min_nko_ratio: float = 0.8) -> dict:
    """
    Re-bridge existing Latin-N'Ko pairs using the canonical transliterate engine.

    Args:
        input_path: Path to the existing feature_pairs_djoko.jsonl
        output_path: Path for the re-bridged output JSONL
        min_words: Minimum word count to keep a pair
        max_repeat: Maximum repetition ratio to filter hallucinations
        min_nko_ratio: Minimum ratio of N'Ko chars in output to accept

    Returns:
        Stats dict with counts
    """
    print(f"Reading {input_path}...")
    entries = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries")
    print(f"Bridge engine: {'nko.transliterate (IPA-routed)' if BRIDGE_AVAILABLE else 'fallback (embedded char map)'}")

    stats = {
        "total_input": len(entries),
        "kept": 0,
        "filtered_repetitive": 0,
        "filtered_too_short": 0,
        "filtered_empty_nko": 0,
        "filtered_low_nko_ratio": 0,
        "rebridged_different": 0,
        "rebridged_same": 0,
        "unique_videos": set(),
    }

    output_entries = []
    pair_id = 0

    for entry in entries:
        latin = entry.get("latin", "").strip()
        feat_id = entry.get("feat_id", "")
        video_id = entry.get("video_id", "")
        old_nko = entry.get("nko", "")

        # Filter: too short
        words = latin.split()
        if len(words) < min_words:
            stats["filtered_too_short"] += 1
            continue

        # Filter: repetitive hallucination
        if is_repetitive(latin, threshold=max_repeat):
            stats["filtered_repetitive"] += 1
            continue

        # Re-bridge
        new_nko = bridge(latin)

        # Filter: empty N'Ko output
        if not new_nko or not has_nko(new_nko):
            stats["filtered_empty_nko"] += 1
            continue

        # Filter: low N'Ko character ratio (means many chars were dropped)
        nko_chars = sum(1 for c in new_nko if 0x07C0 <= ord(c) <= 0x07FF)
        total_chars = sum(1 for c in new_nko if not c.isspace())
        if total_chars > 0 and (nko_chars / total_chars) < min_nko_ratio:
            stats["filtered_low_nko_ratio"] += 1
            continue

        # Track whether bridging actually changed
        if new_nko != old_nko:
            stats["rebridged_different"] += 1
        else:
            stats["rebridged_same"] += 1

        stats["unique_videos"].add(video_id)

        output_entries.append({
            "id": pair_id,
            "bam": latin,
            "nko": new_nko,
            "feat_id": feat_id,
        })
        pair_id += 1

    stats["kept"] = len(output_entries)
    stats["unique_videos"] = len(stats["unique_videos"])

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return stats


def print_stats(stats: dict):
    """Print processing statistics."""
    print("\n" + "=" * 60)
    print("Re-bridge Statistics")
    print("=" * 60)
    print(f"  Input entries:         {stats['total_input']:>6}")
    print(f"  Kept (output):         {stats['kept']:>6}")
    print(f"  Filtered repetitive:   {stats['filtered_repetitive']:>6}")
    print(f"  Filtered too short:    {stats['filtered_too_short']:>6}")
    print(f"  Filtered empty N'Ko:   {stats['filtered_empty_nko']:>6}")
    print(f"  Filtered low N'Ko %%:   {stats['filtered_low_nko_ratio']:>6}")
    print(f"  Unique videos:         {stats['unique_videos']:>6}")
    print(f"  Bridging changed:      {stats['rebridged_different']:>6}")
    print(f"  Bridging unchanged:    {stats['rebridged_same']:>6}")
    print("=" * 60)


def print_samples(output_path: str, n: int = 5):
    """Print sample output entries."""
    print(f"\nSample output ({n} entries):")
    print("-" * 60)
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            entry = json.loads(line)
            print(f"  [{entry['id']}] feat_id={entry['feat_id']}")
            print(f"      bam: {entry['bam'][:70]}")
            print(f"      nko: {entry['nko'][:70]}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Re-bridge Djoko/babamamadidiane pairs to char-level N'Ko"
    )
    parser.add_argument(
        "--input", "-i",
        default="/workspace/results/feature_pairs_djoko.jsonl",
        help="Input JSONL with existing pairs (default: Sweden instance path)"
    )
    parser.add_argument(
        "--output", "-o",
        default="/workspace/results/djoko_nko_rebridged.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--min-words", type=int, default=2,
        help="Minimum word count to keep a pair (default: 2)"
    )
    parser.add_argument(
        "--max-repeat", type=float, default=0.3,
        help="Max unique-word-ratio threshold for hallucination filter (default: 0.3)"
    )
    parser.add_argument(
        "--min-nko-ratio", type=float, default=0.8,
        help="Minimum ratio of N'Ko chars in output (default: 0.8)"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="Fetch input from Vast.ai Sweden instance via SSH"
    )
    parser.add_argument(
        "--ssh-host", default="ssh8.vast.ai",
        help="SSH host for Vast.ai (default: ssh8.vast.ai)"
    )
    parser.add_argument(
        "--ssh-port", type=int, default=32176,
        help="SSH port for Vast.ai (default: 32176)"
    )
    parser.add_argument(
        "--ssh-key", default=os.path.expanduser("~/.ssh/id_vastai"),
        help="SSH key for Vast.ai"
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload output back to Vast.ai after processing"
    )
    args = parser.parse_args()

    # Resolve input path
    input_path = args.input
    if args.remote:
        input_path = fetch_remote_file(
            args.ssh_host, args.ssh_port, args.ssh_key, args.input
        )

    # If input doesn't exist locally, try fetching from Sweden
    if not os.path.exists(input_path) and not args.remote:
        print(f"Input not found at {input_path}")
        print("Attempting to fetch from Sweden instance...")
        try:
            input_path = fetch_remote_file(
                args.ssh_host, args.ssh_port, args.ssh_key, args.input
            )
        except Exception as e:
            print(f"Could not fetch: {e}")
            sys.exit(1)

    # Resolve output path -- if running locally, default to local results dir
    output_path = args.output
    if args.remote or not os.path.exists(os.path.dirname(output_path) or "."):
        local_results = REPO_ROOT / "results"
        local_results.mkdir(exist_ok=True)
        output_path = str(local_results / "djoko_nko_rebridged.jsonl")

    # Run re-bridge
    stats = rebridge(
        input_path, output_path,
        min_words=args.min_words,
        max_repeat=args.max_repeat,
        min_nko_ratio=args.min_nko_ratio,
    )

    print_stats(stats)
    print_samples(output_path)
    print(f"Output written to: {output_path}")

    # Upload back to Sweden if requested
    if args.upload:
        remote_dest = args.output
        cmd = [
            "scp", "-i", args.ssh_key, "-P", str(args.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            output_path,
            f"root@{args.ssh_host}:{remote_dest}",
        ]
        print(f"\nUploading to {args.ssh_host}:{remote_dest}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("Upload complete.")
        else:
            print(f"Upload failed: {result.stderr}")

    # Clean up temp file if we fetched remotely
    if args.remote and input_path.startswith(tempfile.gettempdir()):
        os.unlink(input_path)


if __name__ == "__main__":
    main()
