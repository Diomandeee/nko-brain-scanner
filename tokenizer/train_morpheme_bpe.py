#!/usr/bin/env python3
"""
Train Morpheme-Constrained BPE on N'Ko Corpus

Learns BPE merges that respect morphological boundaries:
1. Run morphological analyzer on all training text
2. Collect morpheme-level frequency statistics
3. Learn BPE merges within morpheme boundaries only
4. Output merges.json + vocab.json in same format as train_bpe.py

Comparison mode: measures token count on eval set vs pure BPE.

Usage:
    python3 tokenizer/train_morpheme_bpe.py \
        --corpus data/nko-corpus.jsonl \
        --merges 256 \
        --compare-bpe
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# N'Ko Unicode range
NKO_START = 0x07C0
NKO_END = 0x07FF

# Add NKo project to path
_NKO_ROOT = Path.home() / "Desktop" / "NKo"
if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))

from nko.morphology import MorphologicalAnalyzer, MorphemeType
from nko.phonetics import NKoPhonetics, TONE_MARK_CHARS, COMBINING_CHARS

_SCANNER_ROOT = Path(__file__).parent.parent

# Reuse utilities from train_bpe
sys.path.insert(0, str(_SCANNER_ROOT))
from tokenizer.train_bpe import (
    is_nko_char, is_nko_tone, extract_nko_words,
    word_to_chars, get_pair_counts, merge_pair,
    load_corpus, load_word_frequencies,
)


def categorize_morpheme(morph_type: str) -> str:
    """Map morpheme type to BPE category."""
    if morph_type in ("root",):
        return "root"
    elif morph_type in ("suffix", "prefix", "postposition", "derivation"):
        return "affix"
    elif morph_type in ("subject", "object", "tense", "aspect", "negation",
                        "determiner", "connector", "plural"):
        return "particle"
    return "unknown"


def analyze_corpus_morphemes(
    words: List[str],
    word_counts: Dict[str, int],
    analyzer: MorphologicalAnalyzer,
    confidence_threshold: float = 0.4,
) -> Dict[str, Dict[Tuple[str, ...], int]]:
    """
    Analyze corpus words and group morphemes by category.

    Returns dict mapping category → {morpheme_char_tuple → frequency}.
    """
    categories: Dict[str, Dict[Tuple[str, ...], int]] = {
        "root": {},
        "affix": {},
        "particle": {},
        "unknown": {},
    }

    analyzed = 0
    high_conf = 0

    for word in words:
        freq = word_counts.get(word, 1)
        analysis = analyzer.analyze_word(word)

        if analysis.confidence >= confidence_threshold and len(analysis.morphemes) > 0:
            high_conf += 1
            for morph in analysis.morphemes:
                morph_text = morph.nko if morph.nko else morph.text
                if not morph_text or not any(is_nko_char(ch) for ch in morph_text):
                    continue

                chars = word_to_chars(morph_text)
                if len(chars) >= 2:
                    category = categorize_morpheme(morph.morpheme_type.value)
                    key = tuple(chars)
                    categories[category][key] = categories[category].get(key, 0) + freq
        else:
            # Low confidence — treat as unknown
            chars = word_to_chars(word)
            if len(chars) >= 2:
                key = tuple(chars)
                categories["unknown"][key] = categories["unknown"].get(key, 0) + freq

        analyzed += 1

    print(f"  Analyzed {analyzed} words, {high_conf} high-confidence ({high_conf/max(analyzed,1)*100:.1f}%)")
    for cat, data in categories.items():
        total = sum(data.values())
        print(f"    {cat}: {len(data)} unique morphemes, {total} total occurrences")

    return categories


def train_category_bpe(
    word_freqs: Dict[Tuple[str, ...], int],
    num_merges: int,
    min_frequency: int = 3,
    label: str = "",
) -> List[Tuple[str, str, str]]:
    """Train BPE merges on a single morpheme category."""
    merges = []

    for i in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break

        best_pair = pair_counts.most_common(1)[0]
        pair, count = best_pair

        if count < min_frequency:
            break

        merged = pair[0] + pair[1]
        merges.append((pair[0], pair[1], merged))
        word_freqs = merge_pair(word_freqs, pair)

        if i < 5 or i % 20 == 0:
            print(f"    [{label}] Merge {i:3d}: '{pair[0]}'+'{pair[1]}' -> '{merged}' (count: {count})")

    print(f"    [{label}] Learned {len(merges)} merges")
    return merges


def compare_tokenizers(
    eval_texts: List[str],
    morpheme_merges: Dict[str, List[Tuple[str, str, str]]],
    standard_merges: List[Tuple[str, str, str]],
) -> Dict:
    """Compare morpheme-aware vs standard BPE on eval text."""
    from tokenizer.train_bpe import word_to_chars

    morph_total_tokens = 0
    standard_total_tokens = 0
    morph_total_chars = 0

    for text in eval_texts:
        words = extract_nko_words(text)
        for word in words:
            chars = word_to_chars(word)
            morph_total_chars += len(chars)

            # Standard BPE
            std_units = list(chars)
            for left, right, _ in standard_merges:
                new_units = []
                i = 0
                while i < len(std_units):
                    if i < len(std_units) - 1 and std_units[i] == left and std_units[i+1] == right:
                        new_units.append(left + right)
                        i += 2
                    else:
                        new_units.append(std_units[i])
                        i += 1
                std_units = new_units
            standard_total_tokens += len(std_units)

            # Morpheme BPE: apply all category merges
            morph_units = list(chars)
            all_merges = []
            for cat in ["root", "affix", "particle", "unknown"]:
                all_merges.extend(morpheme_merges.get(cat, []))
            for left, right, _ in all_merges:
                new_units = []
                i = 0
                while i < len(morph_units):
                    if i < len(morph_units) - 1 and morph_units[i] == left and morph_units[i+1] == right:
                        new_units.append(left + right)
                        i += 2
                    else:
                        new_units.append(morph_units[i])
                        i += 1
                morph_units = new_units
            morph_total_tokens += len(morph_units)

    return {
        "total_nko_chars": morph_total_chars,
        "standard_bpe_tokens": standard_total_tokens,
        "morpheme_bpe_tokens": morph_total_tokens,
        "standard_compression": morph_total_chars / max(standard_total_tokens, 1),
        "morpheme_compression": morph_total_chars / max(morph_total_tokens, 1),
        "token_reduction_pct": (
            (standard_total_tokens - morph_total_tokens) / max(standard_total_tokens, 1) * 100
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train morpheme-constrained BPE on N'Ko")
    parser.add_argument("--num-merges", type=int, default=256,
                        help="Total BPE merges to learn (split across categories)")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="Minimum pair frequency for merge")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Corpus JSONL path (default: training data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output vocab JSON path")
    parser.add_argument("--compare-bpe", action="store_true",
                        help="Compare with standard BPE compression")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Morpheme analysis confidence threshold")
    args = parser.parse_args()

    nko_root = Path.home() / "Desktop" / "NKo" / "data"

    # Load corpus
    corpus_paths = [
        str(nko_root / "corpus" / "nko-wikipedia.txt"),
        str(nko_root / "corpus" / "nko-cultural.txt"),
        str(nko_root / "corpus" / "nko-vocabulary.txt"),
    ]

    cpt_path = _SCANNER_ROOT / "training" / "cpt_train.jsonl"
    cpt_text = ""
    if cpt_path.exists():
        print("Extracting N'Ko text from CPT training data...")
        lines = []
        with open(cpt_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    if any(is_nko_char(ch) for ch in text):
                        lines.append(text)
                except json.JSONDecodeError:
                    continue
        cpt_text = "\n".join(lines)
        print(f"  CPT N'Ko text: {len(cpt_text):,} chars from {len(lines):,} entries")

    print("\nLoading corpus files...")
    corpus_text = load_corpus(corpus_paths)
    if cpt_text:
        corpus_text = corpus_text + "\n" + cpt_text

    print(f"\nTotal corpus: {len(corpus_text):,} characters")

    # Extract words and count frequencies
    all_words = extract_nko_words(corpus_text)
    word_counts = Counter(all_words)
    unique_words = list(word_counts.keys())
    print(f"Unique N'Ko words: {len(unique_words):,}")

    # Load pre-computed word frequencies
    freq_path = str(nko_root / "corpus" / "word-frequencies.json")
    precomputed = load_word_frequencies(freq_path)
    for word, count in precomputed.items():
        nko_only = "".join(ch for ch in word if is_nko_char(ch))
        if len(nko_only) >= 2:
            word_counts[nko_only] += count
            if nko_only not in unique_words:
                unique_words.append(nko_only)

    # Step 1: Morphological analysis of corpus
    print("\nAnalyzing corpus morphology...")
    analyzer = MorphologicalAnalyzer()
    categories = analyze_corpus_morphemes(
        unique_words, word_counts, analyzer, args.confidence,
    )

    # Step 2: Train BPE per category
    # Budget: 40% root, 25% affix, 15% particle, 20% unknown
    budget = {
        "root": int(args.num_merges * 0.40),
        "affix": int(args.num_merges * 0.25),
        "particle": int(args.num_merges * 0.15),
        "unknown": args.num_merges - int(args.num_merges * 0.80),
    }

    print(f"\nTraining morpheme-constrained BPE ({args.num_merges} total merges)...")
    print(f"  Budget: root={budget['root']}, affix={budget['affix']}, "
          f"particle={budget['particle']}, unknown={budget['unknown']}")

    category_merges: Dict[str, List[Tuple[str, str, str]]] = {}
    for cat in ["root", "affix", "particle", "unknown"]:
        if categories[cat]:
            print(f"\n  Training {cat} BPE ({budget[cat]} merges)...")
            category_merges[cat] = train_category_bpe(
                dict(categories[cat]),
                budget[cat],
                args.min_freq,
                label=cat,
            )
        else:
            category_merges[cat] = []

    # Step 3: Build vocabulary
    print("\nBuilding morpheme-aware vocabulary...")
    token_to_id = {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
        "[SEP]": 4, "[MORPH_SEP]": 5,
    }
    idx = 6

    # N'Ko characters
    for cp in range(NKO_START, NKO_END + 1):
        ch = chr(cp)
        if ch not in token_to_id:
            token_to_id[ch] = idx
            idx += 1

    token_to_id[" "] = idx
    idx += 1

    # Add merge tokens per category
    total_merge_tokens = 0
    for cat in ["root", "affix", "particle", "unknown"]:
        for _, _, merged in category_merges.get(cat, []):
            if merged not in token_to_id:
                token_to_id[merged] = idx
                idx += 1
                total_merge_tokens += 1

    print(f"  Vocab size: {len(token_to_id)}")
    print(f"  Merge tokens added: {total_merge_tokens}")

    # Save
    output_path = args.output or str(_SCANNER_ROOT / "tokenizer" / "morpheme_bpe_vocab.json")
    vocab_data = {
        "token_to_id": token_to_id,
        "vocab_size": len(token_to_id),
        "special_tokens": {
            "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
            "[SEP]": 4, "[MORPH_SEP]": 5,
        },
        "root_merges": [[l, r] for l, r, _ in category_merges.get("root", [])],
        "affix_merges": [[l, r] for l, r, _ in category_merges.get("affix", [])],
        "particle_merges": [[l, r] for l, r, _ in category_merges.get("particle", [])],
        "general_merges": [[l, r] for l, r, _ in category_merges.get("unknown", [])],
        "merge_counts": {cat: len(m) for cat, m in category_merges.items()},
        "total_merges": sum(len(m) for m in category_merges.values()),
        "confidence_threshold": args.confidence,
    }

    with open(output_path, "w") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved morpheme BPE vocab to {output_path}")

    # Print top merges per category
    for cat in ["root", "affix", "particle", "unknown"]:
        merges = category_merges.get(cat, [])
        if merges:
            print(f"\n  Top 10 {cat} merges:")
            for i, (l, r, m) in enumerate(merges[:10]):
                print(f"    {i+1:3d}. '{l}' + '{r}' -> '{m}'")

    # Step 4: Compare with standard BPE
    if args.compare_bpe:
        print("\n" + "=" * 60)
        print("COMPRESSION COMPARISON: Morpheme BPE vs Standard BPE")
        print("=" * 60)

        # Load eval texts
        eval_path = _SCANNER_ROOT / "eval" / "nko_eval.jsonl"
        eval_texts = []
        if eval_path.exists():
            with open(eval_path) as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        for msg in ex.get("messages", []):
                            content = msg.get("content", "")
                            if any(is_nko_char(ch) for ch in content):
                                eval_texts.append(content)
                    except json.JSONDecodeError:
                        continue
        else:
            eval_texts = [corpus_text[:5000]]

        print(f"  Eval texts: {len(eval_texts)}")

        # Load standard BPE merges
        std_bpe_path = _SCANNER_ROOT / "tokenizer" / "bpe_vocab.json"
        standard_merges = []
        if std_bpe_path.exists():
            with open(std_bpe_path) as f:
                std_data = json.load(f)
            for pair in std_data.get("merges", []):
                if len(pair) == 2:
                    standard_merges.append((pair[0], pair[1], pair[0] + pair[1]))

        comparison = compare_tokenizers(eval_texts, category_merges, standard_merges)

        print(f"\n  {'Metric':<30} {'Standard BPE':>15} {'Morpheme BPE':>15}")
        print(f"  {'-'*60}")
        nko_label = "Total NKo chars"
        print(f"  {nko_label:<30} {comparison['total_nko_chars']:>15,}")
        print(f"  {'Token count':<30} {comparison['standard_bpe_tokens']:>15,} {comparison['morpheme_bpe_tokens']:>15,}")
        print(f"  {'Compression ratio':<30} {comparison['standard_compression']:>15.2f} {comparison['morpheme_compression']:>15.2f}")
        print(f"  {'Token reduction':<30} {'':>15} {comparison['token_reduction_pct']:>14.1f}%")


if __name__ == "__main__":
    main()
