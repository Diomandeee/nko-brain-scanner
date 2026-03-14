#!/usr/bin/env python3
"""
Train BPE (Byte Pair Encoding) merges on N'Ko corpus.

Learns common multi-character N'Ko sequences that should be single tokens.
This addresses the core tokenizer bottleneck: Qwen3-8B has 32 single-char
N'Ko tokens but zero subword merges, creating 4x token inflation vs English.

The output is a merge table + expanded vocabulary that can be:
1. Used directly by NkoTokenizer for morpheme-aware tokenization
2. Fed into vocabulary extension training to add these tokens to Qwen3
3. Published as a standalone N'Ko BPE model

Usage:
    python3 train_bpe.py --vocab-size 512 --output bpe_vocab.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

# N'Ko Unicode range
NKO_START = 0x07C0
NKO_END = 0x07FF


def is_nko_char(ch: str) -> bool:
    """Check if character is in N'Ko Unicode block."""
    return NKO_START <= ord(ch) <= NKO_END


def is_nko_tone(ch: str) -> bool:
    """Check if character is an N'Ko combining/tone mark."""
    cp = ord(ch)
    return 0x07EB <= cp <= 0x07F5


def is_nko_base(ch: str) -> bool:
    """Check if character is an N'Ko base letter (not tone/combining)."""
    return is_nko_char(ch) and not is_nko_tone(ch)


def extract_nko_words(text: str) -> List[str]:
    """Extract sequences of N'Ko characters (words) from text."""
    words = []
    current = []
    for ch in text:
        if is_nko_char(ch):
            current.append(ch)
        elif current:
            word = ''.join(current)
            if len(word) >= 2:  # Only words with 2+ chars
                words.append(word)
            current = []
    if current:
        word = ''.join(current)
        if len(word) >= 2:
            words.append(word)
    return words


def word_to_chars(word: str) -> List[str]:
    """
    Split word into base characters with tone marks attached.
    Tone marks attach to the preceding base character.
    e.g. "ߊ߬" -> ["ߊ߬"] (not ["ߊ", "߬"])
    """
    units = []
    current = ""
    for ch in word:
        if is_nko_tone(ch):
            current += ch
        else:
            if current:
                units.append(current)
            current = ch
    if current:
        units.append(current)
    return units


def get_pair_counts(word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
    """Count adjacent pair frequencies across all words."""
    pairs = Counter()
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pairs[pair] += freq
    return pairs


def merge_pair(
    word_freqs: Dict[Tuple[str, ...], int],
    pair: Tuple[str, str]
) -> Dict[Tuple[str, ...], int]:
    """Merge all occurrences of a pair in the vocabulary."""
    new_word_freqs = {}
    merged = pair[0] + pair[1]

    for word_tuple, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if (i < len(word_tuple) - 1 and
                    word_tuple[i] == pair[0] and
                    word_tuple[i + 1] == pair[1]):
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def load_corpus(corpus_paths: List[str]) -> str:
    """Load and concatenate corpus text from multiple files."""
    texts = []
    for path in corpus_paths:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                texts.append(f.read())
            print(f"  Loaded {p.name}: {p.stat().st_size:,} bytes")
    return '\n'.join(texts)


def load_word_frequencies(freq_path: str) -> Dict[str, int]:
    """Load pre-computed word frequencies."""
    p = Path(freq_path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return {
        entry['word']: entry['count']
        for entry in data.get('frequencies', [])
    }


def train_bpe(
    corpus_text: str,
    word_freq_data: Dict[str, int],
    num_merges: int = 256,
    min_frequency: int = 5,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """
    Train BPE merges on N'Ko text.

    Returns:
        merges: List of (left, right, merged) tuples in merge order
        vocab: Final vocabulary with token frequencies
    """
    # Extract N'Ko words from corpus
    corpus_words = extract_nko_words(corpus_text)
    print(f"  Extracted {len(corpus_words):,} N'Ko words from corpus")

    # Build word frequency table (combine corpus + pre-computed)
    word_counts = Counter()
    for w in corpus_words:
        word_counts[w] += 1

    # Add pre-computed frequencies (weighted higher since they're from larger corpus)
    for word, count in word_freq_data.items():
        nko_only = ''.join(ch for ch in word if is_nko_char(ch))
        if len(nko_only) >= 2:
            word_counts[nko_only] += count

    print(f"  Unique N'Ko words: {len(word_counts):,}")
    print(f"  Total word occurrences: {sum(word_counts.values()):,}")

    # Convert words to character tuples (tone-attached)
    word_freqs: Dict[Tuple[str, ...], int] = {}
    for word, freq in word_counts.items():
        chars = word_to_chars(word)
        if len(chars) >= 2:
            word_freqs[tuple(chars)] = freq

    # Collect initial character vocabulary
    char_vocab = Counter()
    for word_tuple, freq in word_freqs.items():
        for ch in word_tuple:
            char_vocab[ch] += freq

    print(f"  Initial character vocab: {len(char_vocab)} unique chars")
    print(f"  Top 10 chars: {char_vocab.most_common(10)}")

    # BPE training loop
    merges = []
    for i in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            print(f"  No more pairs at merge {i}")
            break

        best_pair = pair_counts.most_common(1)[0]
        pair, count = best_pair

        if count < min_frequency:
            print(f"  Stopping at merge {i}: best pair count {count} < min {min_frequency}")
            break

        merged = pair[0] + pair[1]
        merges.append((pair[0], pair[1], merged))

        if i < 20 or i % 50 == 0:
            print(f"  Merge {i:3d}: '{pair[0]}' + '{pair[1]}' -> '{merged}' (count: {count})")

        word_freqs = merge_pair(word_freqs, pair)

    print(f"\n  Learned {len(merges)} BPE merges")

    # Build final vocabulary
    final_vocab = Counter()
    for word_tuple, freq in word_freqs.items():
        for token in word_tuple:
            final_vocab[token] += freq

    # Add single characters that might not appear in merged forms
    for ch, freq in char_vocab.items():
        if ch not in final_vocab:
            final_vocab[ch] = freq

    return merges, dict(final_vocab)


def build_tokenizer_vocab(
    merges: List[Tuple[str, str, str]],
    bpe_vocab: Dict[str, int],
    morpheme_vocab_path: str = None,
) -> Dict:
    """
    Build a complete tokenizer vocabulary combining BPE merges with
    morphological knowledge.
    """
    # Special tokens
    token_to_id = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "[SEP]": 4,
    }
    idx = 5

    # Add all single N'Ko characters (U+07C0 to U+07FF)
    for cp in range(NKO_START, NKO_END + 1):
        ch = chr(cp)
        if ch not in token_to_id:
            token_to_id[ch] = idx
            idx += 1

    # Add space
    token_to_id[" "] = idx
    idx += 1

    # Add BPE merge tokens (in merge order, most frequent first)
    bpe_tokens_added = 0
    for left, right, merged in merges:
        if merged not in token_to_id:
            token_to_id[merged] = idx
            idx += 1
            bpe_tokens_added += 1

    # Load existing morpheme vocab and add any not covered by BPE
    morpheme_tokens_added = 0
    if morpheme_vocab_path:
        mp = Path(morpheme_vocab_path)
        if mp.exists():
            with open(mp) as f:
                morph_data = json.load(f)
            for token in morph_data.get("token_to_id", {}):
                if token not in token_to_id and not token.startswith("["):
                    token_to_id[token] = idx
                    idx += 1
                    morpheme_tokens_added += 1

    return {
        "token_to_id": token_to_id,
        "vocab_size": len(token_to_id),
        "special_tokens": {
            "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4,
        },
        "merges": [[left, right] for left, right, _ in merges],
        "merge_count": len(merges),
        "bpe_tokens_added": bpe_tokens_added,
        "morpheme_tokens_added": morpheme_tokens_added,
        "nko_char_count": NKO_END - NKO_START + 1,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train BPE on N'Ko corpus")
    parser.add_argument("--num-merges", type=int, default=256,
                        help="Number of BPE merges to learn (default: 256)")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="Minimum pair frequency for merge (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output vocab JSON path")
    args = parser.parse_args()

    nko_root = Path.home() / "Desktop" / "NKo" / "data"
    scanner_root = Path.home() / "Desktop" / "nko-brain-scanner"

    # Corpus sources
    corpus_paths = [
        str(nko_root / "corpus" / "nko-wikipedia.txt"),
        str(nko_root / "corpus" / "nko-cultural.txt"),
        str(nko_root / "corpus" / "nko-vocabulary.txt"),
    ]

    # Also extract N'Ko text from training data
    cpt_path = scanner_root / "training" / "cpt_train.jsonl"
    if cpt_path.exists():
        print("Extracting N'Ko text from CPT training data...")
        cpt_text = []
        with open(cpt_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    if any(is_nko_char(ch) for ch in text):
                        cpt_text.append(text)
                except json.JSONDecodeError:
                    continue
        cpt_combined = '\n'.join(cpt_text)
        print(f"  CPT N'Ko text: {len(cpt_combined):,} chars from {len(cpt_text):,} entries")
    else:
        cpt_combined = ""

    print("\nLoading corpus files...")
    corpus_text = load_corpus(corpus_paths)
    if cpt_combined:
        corpus_text = corpus_text + '\n' + cpt_combined

    print(f"\nTotal corpus: {len(corpus_text):,} characters")

    # Load pre-computed word frequencies
    freq_path = str(nko_root / "corpus" / "word-frequencies.json")
    print("\nLoading word frequencies...")
    word_freqs = load_word_frequencies(freq_path)
    print(f"  Pre-computed: {len(word_freqs):,} unique words")

    # Train BPE
    print(f"\nTraining BPE ({args.num_merges} merges, min_freq={args.min_freq})...")
    merges, bpe_vocab = train_bpe(
        corpus_text, word_freqs,
        num_merges=args.num_merges,
        min_frequency=args.min_freq,
    )

    # Build complete vocabulary
    print("\nBuilding tokenizer vocabulary...")
    morpheme_vocab = str(scanner_root / "tokenizer" / "vocab.json")
    vocab_data = build_tokenizer_vocab(merges, bpe_vocab, morpheme_vocab)

    # Save
    output_path = args.output or str(scanner_root / "tokenizer" / "bpe_vocab.json")
    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Total vocab size: {vocab_data['vocab_size']}")
    print(f"  N'Ko single chars: {vocab_data['nko_char_count']}")
    print(f"  BPE merge tokens: {vocab_data['bpe_tokens_added']}")
    print(f"  Morpheme tokens: {vocab_data['morpheme_tokens_added']}")
    print(f"  Total merges learned: {vocab_data['merge_count']}")
    print(f"  Saved to: {output_path}")

    # Print top 20 BPE tokens by frequency
    print(f"\nTop 30 learned subword tokens:")
    for i, (left, right, merged) in enumerate(merges[:30]):
        freq = bpe_vocab.get(merged, 0)
        print(f"  {i+1:3d}. '{merged}' (from '{left}' + '{right}', freq: {freq})")


if __name__ == "__main__":
    main()
