#!/usr/bin/env python3
"""
Sigil Encoder: Map conversation turns to N'Ko sigil sequences.

Takes raw text, transliterates to N'Ko, extracts semantic features,
and assigns a sigil sequence from the 10 canonical sigils.

The sigil assignment is based on information-theoretic properties
of the text, not keyword matching. Each sigil represents a dynamic
pattern in the conversational signal.

Usage:
    python3 sigil_encoder.py \
        --input conversations.jsonl \
        --output results/sigil_encoded.jsonl \
        --limit 1000
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Add NKo to path
NKO_PATH = os.path.expanduser("~/Desktop/NKo")
if NKO_PATH not in sys.path:
    sys.path.insert(0, NKO_PATH)

try:
    from nko.transliterate import transliterate, detect_script
except ImportError:
    print("WARNING: nko.transliterate not available. N'Ko transliteration will be skipped.")
    transliterate = None

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
except ImportError:
    print("WARNING: tiktoken not available. BPE counting will use word approximation.")
    ENC = None


# ---- The 10 Sigils ----

SIGILS = [
    {"char": "ߛ", "name": "stabilization", "description": "Dispersion decreased"},
    {"char": "ߜ", "name": "dispersion", "description": "Spread increased"},
    {"char": "ߕ", "name": "transition", "description": "Change point"},
    {"char": "ߙ", "name": "return", "description": "Re-entry to basin"},
    {"char": "ߡ", "name": "dwell", "description": "Sustained stay"},
    {"char": "ߚ", "name": "oscillation", "description": "Rapid alternation"},
    {"char": "ߞ", "name": "recovery", "description": "Return latency"},
    {"char": "ߣ", "name": "novelty", "description": "New basin"},
    {"char": "ߠ", "name": "place_shift", "description": "Location change"},
    {"char": "ߥ", "name": "echo", "description": "Pattern match"},
]

SIGIL_BY_NAME = {s["name"]: s["char"] for s in SIGILS}


# ---- Feature Extraction ----

def extract_features(text, prev_text=None):
    """Extract information-theoretic features from a text turn.

    Returns a dict of features used for sigil assignment.
    """
    words = text.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    char_count = len(text)

    # Lexical diversity (type-token ratio)
    ttr = unique_words / word_count if word_count > 0 else 0

    # Sentence count (rough)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Average sentence length
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else word_count

    # Character entropy (Shannon)
    char_freq = Counter(text.lower())
    total_chars = sum(char_freq.values())
    if total_chars > 0:
        char_entropy = -sum(
            (c / total_chars) * math.log2(c / total_chars)
            for c in char_freq.values() if c > 0
        )
    else:
        char_entropy = 0

    # Question density
    question_marks = text.count("?")
    question_density = question_marks / sentence_count if sentence_count > 0 else 0

    # Repetition (bigram overlap)
    bigrams = list(zip(words[:-1], words[1:])) if len(words) > 1 else []
    unique_bigrams = len(set(bigrams))
    bigram_novelty = unique_bigrams / len(bigrams) if bigrams else 1.0

    # Overlap with previous turn (if available)
    overlap_ratio = 0.0
    if prev_text:
        prev_words = set(prev_text.lower().split())
        curr_words = set(words)
        if prev_words and curr_words:
            overlap_ratio = len(prev_words & curr_words) / len(prev_words | curr_words)

    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "ttr": ttr,
        "char_count": char_count,
        "char_entropy": char_entropy,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "question_density": question_density,
        "bigram_novelty": bigram_novelty,
        "overlap_ratio": overlap_ratio,
    }


# ---- Sigil Assignment ----

def assign_sigils(features, max_sigils=5):
    """Assign a sequence of sigils based on extracted features.

    The assignment uses thresholds on information-theoretic features
    to determine which dynamic patterns characterize this turn.
    """
    sigil_seq = []

    ttr = features["ttr"]
    entropy = features["char_entropy"]
    overlap = features["overlap_ratio"]
    bigram_nov = features["bigram_novelty"]
    q_density = features["question_density"]
    avg_sent = features["avg_sentence_len"]

    # High overlap with previous = echo (pattern match)
    if overlap > 0.4:
        sigil_seq.append(SIGIL_BY_NAME["echo"])

    # Low TTR = repetitive = dwell (sustained stay)
    if ttr < 0.5 and features["word_count"] > 10:
        sigil_seq.append(SIGIL_BY_NAME["dwell"])

    # Very high TTR = diverse vocabulary = novelty
    if ttr > 0.85:
        sigil_seq.append(SIGIL_BY_NAME["novelty"])

    # High question density = transition/seeking
    if q_density > 0.3:
        sigil_seq.append(SIGIL_BY_NAME["transition"])

    # Long sentences = sustained focus = stabilization
    if avg_sent > 20:
        sigil_seq.append(SIGIL_BY_NAME["stabilization"])

    # Short, choppy sentences = oscillation
    if avg_sent < 8 and features["sentence_count"] > 3:
        sigil_seq.append(SIGIL_BY_NAME["oscillation"])

    # Low entropy = concentrated language = recovery
    if entropy < 3.5:
        sigil_seq.append(SIGIL_BY_NAME["recovery"])

    # High entropy = dispersed language = dispersion
    if entropy > 4.5:
        sigil_seq.append(SIGIL_BY_NAME["dispersion"])

    # Low overlap after high overlap = return
    if overlap < 0.1 and features.get("prev_overlap", 0) > 0.3:
        sigil_seq.append(SIGIL_BY_NAME["return"])

    # Topic shift indicators = place_shift
    if bigram_nov > 0.95 and features["word_count"] > 15:
        sigil_seq.append(SIGIL_BY_NAME["place_shift"])

    # Default: if no sigils matched, assign stabilization (neutral)
    if not sigil_seq:
        sigil_seq.append(SIGIL_BY_NAME["stabilization"])

    return sigil_seq[:max_sigils]


# ---- Main Pipeline ----

def encode_turn(text, prev_text=None):
    """Full pipeline: text -> features -> sigils -> measurements."""
    features = extract_features(text, prev_text)
    sigils = assign_sigils(features)

    # BPE token count
    if ENC:
        bpe_tokens = len(ENC.encode(text))
    else:
        bpe_tokens = len(text.split()) * 1.3  # rough approximation

    # N'Ko transliteration
    if transliterate:
        try:
            nko_text = transliterate(text, target="nko")
            nko_chars = len(nko_text)
        except Exception:
            nko_text = ""
            nko_chars = 0
    else:
        nko_text = ""
        nko_chars = 0

    return {
        "original": text,
        "nko_transliteration": nko_text,
        "sigil_sequence": "".join(sigils),
        "sigil_names": [s["name"] for s in SIGILS if s["char"] in sigils],
        "bpe_tokens": bpe_tokens,
        "nko_chars": nko_chars,
        "sigil_count": len(sigils),
        "features": features,
    }


def main():
    parser = argparse.ArgumentParser(description="N'Ko Sigil Encoder")
    parser.add_argument("--input", required=True, help="Input JSONL (one text per line or messages format)")
    parser.add_argument("--output", required=True, help="Output JSONL with sigil encodings")
    parser.add_argument("--limit", type=int, default=1000, help="Max turns to process")
    parser.add_argument("--text-field", default="content",
                        help="JSON field containing text (or 'messages' for SFT format)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    turns = []
    with open(args.input) as f:
        for line in f:
            entry = json.loads(line.strip())
            # Handle SFT message format
            if "messages" in entry:
                for msg in entry["messages"]:
                    text = msg.get("content", "")
                    if text.strip():
                        turns.append(text)
            elif args.text_field in entry:
                text = entry[args.text_field]
                if text and text.strip():
                    turns.append(text)
            elif isinstance(entry, str):
                turns.append(entry)
            if len(turns) >= args.limit:
                break

    print(f"Processing {len(turns)} conversation turns...")

    results = []
    prev_text = None
    for i, text in enumerate(turns):
        result = encode_turn(text, prev_text)
        results.append(result)
        prev_text = text

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(turns)}]")

    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    total_bpe = sum(r["bpe_tokens"] for r in results)
    total_nko = sum(r["nko_chars"] for r in results)
    total_sigils = sum(r["sigil_count"] for r in results)

    print(f"\nEncoding Summary:")
    print(f"  Turns processed: {len(results)}")
    print(f"  Total BPE tokens: {total_bpe}")
    print(f"  Total N'Ko characters: {total_nko}")
    print(f"  Total sigils: {total_sigils}")
    print(f"  BPE -> Sigil compression: {total_bpe / total_sigils:.1f}x")
    if total_nko > 0:
        print(f"  N'Ko -> Sigil compression: {total_nko / total_sigils:.1f}x")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
