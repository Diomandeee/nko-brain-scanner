#!/usr/bin/env python3
"""
Head-to-Head Tokenizer Evaluation

Compares standard BPE vs morpheme-aware BPE on N'Ko text across 5 metrics:
  1. Compression ratio: tokens per N'Ko text (lower = better)
  2. Morpheme boundary preservation: % of linguistic boundaries aligning with token boundaries
  3. Reconstruction accuracy: round-trip encode→decode fidelity
  4. Syllable integrity: % of tokens that are complete syllables
  5. Training efficiency: PPL at 100/250/500 iters (requires trained models)

Runs on the 100 N'Ko eval examples from eval/build_eval_set.py.

Usage:
    python3 tokenizer/eval_tokenizers.py --eval-data eval/nko_eval_100.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

_SCANNER_ROOT = Path(__file__).parent.parent
_NKO_ROOT = Path.home() / "Desktop" / "NKo"

if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))
if str(_SCANNER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCANNER_ROOT))

from nko.morphology import MorphologicalAnalyzer
from nko.phonetics import (
    VOWEL_CHARS, CONSONANT_CHARS, TONE_MARK_CHARS,
    COMBINING_CHARS, ALL_NKO_CHARS,
)
from tokenizer.tokenizer import NkoTokenizer
from tokenizer.morpheme_tokenizer import MorphemeAwareTokenizer
from tokenizer.train_bpe import is_nko_char, extract_nko_words, word_to_chars

NKO_START = 0x07C0
NKO_END = 0x07FF


def has_nko(text: str) -> bool:
    return any(NKO_START <= ord(ch) <= NKO_END for ch in text)


def extract_nko_text(text: str) -> str:
    """Extract only N'Ko characters and spaces."""
    result = []
    for ch in text:
        if NKO_START <= ord(ch) <= NKO_END or ch == " ":
            result.append(ch)
    return "".join(result).strip()


def measure_compression(tokenizer_fn, texts: List[str]) -> Dict:
    """Measure tokens-per-text compression ratio."""
    total_chars = 0
    total_tokens = 0
    per_text = []

    for text in texts:
        nko_text = extract_nko_text(text)
        if not nko_text:
            continue

        tokens = tokenizer_fn(nko_text)
        n_chars = len([ch for ch in nko_text if NKO_START <= ord(ch) <= NKO_END])
        n_tokens = len(tokens)

        total_chars += n_chars
        total_tokens += n_tokens
        per_text.append(n_chars / max(n_tokens, 1))

    return {
        "total_nko_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_token": total_chars / max(total_tokens, 1),
        "avg_compression": sum(per_text) / max(len(per_text), 1),
        "n_texts": len(per_text),
    }


def measure_morpheme_preservation(
    tokenizer_fn,
    texts: List[str],
    analyzer: MorphologicalAnalyzer,
) -> Dict:
    """Measure how well token boundaries align with morpheme boundaries."""
    total_morph_boundaries = 0
    aligned_boundaries = 0
    total_words = 0

    for text in texts:
        nko_text = extract_nko_text(text)
        words = nko_text.split()

        for word in words:
            if not any(NKO_START <= ord(ch) <= NKO_END for ch in word):
                continue

            analysis = analyzer.analyze_word(word)
            if analysis.confidence < 0.4 or len(analysis.morphemes) <= 1:
                continue

            total_words += 1

            # Get morpheme boundary positions (character offsets)
            morph_boundaries = set()
            pos = 0
            for morph in analysis.morphemes[:-1]:  # Boundaries between morphemes
                morph_text = morph.nko if morph.nko else morph.text
                pos += len(morph_text)
                morph_boundaries.add(pos)

            total_morph_boundaries += len(morph_boundaries)

            # Get token boundary positions
            tokens = tokenizer_fn(word)
            token_boundaries = set()
            tpos = 0
            for tok in tokens[:-1]:
                tpos += len(tok)
                token_boundaries.add(tpos)

            # Count aligned boundaries
            for mb in morph_boundaries:
                if mb in token_boundaries:
                    aligned_boundaries += 1

    return {
        "total_morpheme_boundaries": total_morph_boundaries,
        "aligned_boundaries": aligned_boundaries,
        "preservation_ratio": aligned_boundaries / max(total_morph_boundaries, 1),
        "words_analyzed": total_words,
    }


def measure_reconstruction(tokenizer, texts: List[str]) -> Dict:
    """Measure round-trip encode→decode fidelity."""
    total = 0
    exact_matches = 0
    char_correct = 0
    char_total = 0

    for text in texts:
        nko_text = extract_nko_text(text)
        if not nko_text:
            continue
        total += 1

        ids = tokenizer.encode(nko_text, add_special=False)
        reconstructed = tokenizer.decode(ids)

        if reconstructed == nko_text:
            exact_matches += 1

        # Character-level accuracy
        for i, ch in enumerate(nko_text):
            char_total += 1
            if i < len(reconstructed) and reconstructed[i] == ch:
                char_correct += 1

    return {
        "total_texts": total,
        "exact_matches": exact_matches,
        "exact_match_ratio": exact_matches / max(total, 1),
        "char_accuracy": char_correct / max(char_total, 1),
    }


def measure_syllable_integrity(tokenizer_fn, texts: List[str]) -> Dict:
    """Measure what fraction of tokens are complete syllables."""
    total_tokens = 0
    complete_syllables = 0
    partial_tokens = 0

    for text in texts:
        nko_text = extract_nko_text(text)
        if not nko_text:
            continue

        tokens = tokenizer_fn(nko_text)
        for tok in tokens:
            if not any(NKO_START <= ord(ch) <= NKO_END for ch in tok):
                continue

            total_tokens += 1
            has_vowel = any(ch in VOWEL_CHARS for ch in tok)
            has_consonant = any(ch in CONSONANT_CHARS for ch in tok)

            # A complete syllable has at least one vowel (V, CV, CVN patterns)
            if has_vowel:
                complete_syllables += 1
            else:
                partial_tokens += 1

    return {
        "total_nko_tokens": total_tokens,
        "complete_syllable_tokens": complete_syllables,
        "partial_tokens": partial_tokens,
        "syllable_integrity": complete_syllables / max(total_tokens, 1),
    }


def load_eval_texts(eval_path: str) -> List[str]:
    """Load N'Ko texts from eval JSONL."""
    texts = []
    path = Path(eval_path)
    if not path.exists():
        print(f"  Warning: eval file not found at {path}")
        return texts

    with open(path) as f:
        for line in f:
            try:
                ex = json.loads(line)
                for msg in ex.get("messages", []):
                    content = msg.get("content", "")
                    if has_nko(content):
                        texts.append(content)
            except (json.JSONDecodeError, KeyError):
                continue

    return texts


def main():
    parser = argparse.ArgumentParser(description="Compare N'Ko tokenizers head-to-head")
    parser.add_argument("--eval-data", type=str,
                        default=os.path.expanduser("~/nko-brain-scanner/eval/nko_eval.jsonl"),
                        help="Path to N'Ko eval JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    args = parser.parse_args()

    print("Loading eval texts...")
    eval_texts = load_eval_texts(args.eval_data)
    print(f"  Loaded {len(eval_texts)} N'Ko-containing texts")

    if not eval_texts:
        print("No eval texts found. Exiting.")
        return

    # Initialize tokenizers
    print("\nInitializing tokenizers...")
    standard_tokenizer = NkoTokenizer(use_bpe=True)
    morpheme_tokenizer = MorphemeAwareTokenizer()
    analyzer = MorphologicalAnalyzer()

    print(f"  Standard BPE vocab size: {standard_tokenizer.vocab_size}")
    print(f"  Morpheme-aware vocab size: {morpheme_tokenizer.vocab_size}")

    # Define tokenize functions for each
    def std_tokenize(text):
        return standard_tokenizer.tokenize(text)

    def morph_tokenize(text):
        return morpheme_tokenizer.tokenize(text)

    # Run all metrics
    results = {}

    # 1. Compression ratio
    print("\n[1/4] Measuring compression ratio...")
    std_compression = measure_compression(std_tokenize, eval_texts)
    morph_compression = measure_compression(morph_tokenize, eval_texts)
    results["compression"] = {
        "standard_bpe": std_compression,
        "morpheme_bpe": morph_compression,
    }
    print(f"  Standard BPE: {std_compression['chars_per_token']:.2f} chars/token")
    print(f"  Morpheme BPE: {morph_compression['chars_per_token']:.2f} chars/token")

    # 2. Morpheme boundary preservation
    print("\n[2/4] Measuring morpheme boundary preservation...")
    std_morph = measure_morpheme_preservation(std_tokenize, eval_texts, analyzer)
    morph_morph = measure_morpheme_preservation(morph_tokenize, eval_texts, analyzer)
    results["morpheme_preservation"] = {
        "standard_bpe": std_morph,
        "morpheme_bpe": morph_morph,
    }
    print(f"  Standard BPE: {std_morph['preservation_ratio']:.3f} ({std_morph['aligned_boundaries']}/{std_morph['total_morpheme_boundaries']})")
    print(f"  Morpheme BPE: {morph_morph['preservation_ratio']:.3f} ({morph_morph['aligned_boundaries']}/{morph_morph['total_morpheme_boundaries']})")

    # 3. Reconstruction accuracy
    print("\n[3/4] Measuring reconstruction accuracy...")
    std_recon = measure_reconstruction(standard_tokenizer, eval_texts)
    morph_recon = measure_reconstruction(morpheme_tokenizer, eval_texts)
    results["reconstruction"] = {
        "standard_bpe": std_recon,
        "morpheme_bpe": morph_recon,
    }
    print(f"  Standard BPE: {std_recon['exact_match_ratio']:.3f} exact, {std_recon['char_accuracy']:.3f} char-level")
    print(f"  Morpheme BPE: {morph_recon['exact_match_ratio']:.3f} exact, {morph_recon['char_accuracy']:.3f} char-level")

    # 4. Syllable integrity
    print("\n[4/4] Measuring syllable integrity...")
    std_syll = measure_syllable_integrity(std_tokenize, eval_texts)
    morph_syll = measure_syllable_integrity(morph_tokenize, eval_texts)
    results["syllable_integrity"] = {
        "standard_bpe": std_syll,
        "morpheme_bpe": morph_syll,
    }
    print(f"  Standard BPE: {std_syll['syllable_integrity']:.3f} ({std_syll['complete_syllable_tokens']}/{std_syll['total_nko_tokens']})")
    print(f"  Morpheme BPE: {morph_syll['syllable_integrity']:.3f} ({morph_syll['complete_syllable_tokens']}/{morph_syll['total_nko_tokens']})")

    # Summary table
    print(f"\n{'='*80}")
    print("TOKENIZER COMPARISON SUMMARY (N'Ko Eval Set)")
    print(f"{'='*80}")
    print(f"{'Metric':<40} {'Standard BPE':>18} {'Morpheme BPE':>18}")
    print(f"{'-'*80}")

    rows = [
        ("Chars per token", std_compression['chars_per_token'], morph_compression['chars_per_token']),
        ("Total tokens (lower=better)", std_compression['total_tokens'], morph_compression['total_tokens']),
        ("Morpheme boundary preservation", std_morph['preservation_ratio'], morph_morph['preservation_ratio']),
        ("Reconstruction (exact match)", std_recon['exact_match_ratio'], morph_recon['exact_match_ratio']),
        ("Reconstruction (char accuracy)", std_recon['char_accuracy'], morph_recon['char_accuracy']),
        ("Syllable integrity", std_syll['syllable_integrity'], morph_syll['syllable_integrity']),
    ]

    for label, std_val, morph_val in rows:
        better = " *" if morph_val > std_val else "  " if morph_val == std_val else ""
        # For total tokens, lower is better
        if "lower" in label:
            better = " *" if morph_val < std_val else "  " if morph_val == std_val else ""
        print(f"  {label:<38} {std_val:>16.4f} {morph_val:>16.4f}{better}")

    print(f"\n  * = morpheme-aware tokenizer wins")

    # Winner count
    morph_wins = sum(1 for _, s, m in rows if (m > s if "lower" not in rows[rows.index((_, s, m))][0] else m < s))
    print(f"\n  Morpheme BPE wins on {morph_wins}/{len(rows)} metrics")

    # Save results
    output_path = args.output or str(_SCANNER_ROOT / "results" / "tokenizer_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
