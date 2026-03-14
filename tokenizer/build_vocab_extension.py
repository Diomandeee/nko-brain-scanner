#!/usr/bin/env python3
"""
Build vocabulary extension training data for Qwen3-8B.

Takes the BPE merge table and generates training examples that teach
the model to treat multi-character N'Ko sequences as single semantic units.

Two strategies:
1. Embedding initialization: Average the embeddings of constituent characters
2. SFT examples: Sentences where the BPE tokens appear in context

This script produces the training data. Actual vocabulary extension
requires modifying the model's embedding layer (done on Mac5 with MLX).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

NKO_ROOT = Path.home() / "Desktop" / "NKo"
SCANNER_ROOT = Path.home() / "Desktop" / "nko-brain-scanner"


def load_bpe_vocab(path: str = None) -> Dict:
    """Load BPE vocabulary."""
    if path is None:
        path = str(SCANNER_ROOT / "tokenizer" / "bpe_vocab.json")
    with open(path) as f:
        return json.load(f)


def load_corpus_sentences(max_sentences: int = 5000) -> List[str]:
    """Load N'Ko sentences from corpus."""
    sentences = []

    # Wikipedia
    wiki_path = NKO_ROOT / "data" / "corpus" / "nko-wikipedia.txt"
    if wiki_path.exists():
        with open(wiki_path) as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 10:
                    sentences.append(line)

    # Cultural text
    cultural_path = NKO_ROOT / "data" / "corpus" / "nko-cultural.txt"
    if cultural_path.exists():
        with open(cultural_path) as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 10:
                    sentences.append(line)

    # Parallel corpus
    parallel_path = SCANNER_ROOT / "data" / "parallel_corpus.jsonl"
    if parallel_path.exists():
        with open(parallel_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    nko = entry.get("nko", "")
                    if nko and len(nko) > 10:
                        sentences.append(nko)
                except json.JSONDecodeError:
                    continue

    return sentences[:max_sentences]


def find_sentences_containing(token: str, sentences: List[str], max_examples: int = 5) -> List[str]:
    """Find sentences containing a specific token."""
    matches = []
    for sent in sentences:
        if token in sent:
            matches.append(sent)
            if len(matches) >= max_examples:
                break
    return matches


def build_extension_data(bpe_vocab: Dict, sentences: List[str]) -> Dict:
    """
    Build vocabulary extension data.

    Returns a dict with:
    - new_tokens: List of new BPE tokens to add to vocabulary
    - constituent_chars: For each new token, the characters it's composed of
    - context_examples: Training sentences where each token appears
    - embedding_init: Strategy for initializing new token embeddings
    """
    merges = bpe_vocab.get("merges", [])
    token_to_id = bpe_vocab.get("token_to_id", {})

    # Only include BPE tokens that are multi-character N'Ko sequences
    NKO_START, NKO_END = 0x07C0, 0x07FF

    def is_nko_token(token: str) -> bool:
        return len(token) >= 2 and any(NKO_START <= ord(ch) <= NKO_END for ch in token)

    new_tokens = []
    for left, right in merges:
        merged = left + right
        if is_nko_token(merged) and merged in token_to_id:
            # Find context sentences
            examples = find_sentences_containing(merged, sentences, max_examples=3)
            if examples:  # Only include if we have usage examples
                new_tokens.append({
                    "token": merged,
                    "constituents": [left, right],
                    "token_id": token_to_id[merged],
                    "examples": examples,
                    "char_count": len(merged),
                })

    # Sort by frequency (lower token_id = more frequent merge)
    new_tokens.sort(key=lambda x: x["token_id"])

    # Build SFT training examples
    sft_examples = []
    for tok_info in new_tokens[:200]:  # Top 200 most frequent
        token = tok_info["token"]
        for example_sent in tok_info["examples"]:
            # Create a text-completion training example
            sft_examples.append({
                "text": example_sent,
                "target_token": token,
                "category": "vocab_extension",
            })

    return {
        "new_tokens": new_tokens,
        "total_new_tokens": len(new_tokens),
        "sft_examples": sft_examples,
        "total_sft_examples": len(sft_examples),
        "embedding_init_strategy": "average_constituents",
        "note": "Initialize new token embeddings as average of constituent char embeddings",
    }


def main():
    print("Loading BPE vocabulary...")
    bpe_vocab = load_bpe_vocab()
    print(f"  {bpe_vocab['merge_count']} merges, {bpe_vocab['vocab_size']} total tokens")

    print("\nLoading corpus sentences...")
    sentences = load_corpus_sentences()
    print(f"  {len(sentences)} sentences loaded")

    print("\nBuilding vocabulary extension data...")
    ext_data = build_extension_data(bpe_vocab, sentences)

    print(f"\nResults:")
    print(f"  New tokens to add: {ext_data['total_new_tokens']}")
    print(f"  SFT training examples: {ext_data['total_sft_examples']}")
    print(f"  Embedding init: {ext_data['embedding_init_strategy']}")

    # Save
    output_path = SCANNER_ROOT / "training" / "vocab_extension.json"
    with open(output_path, 'w') as f:
        json.dump(ext_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {output_path}")

    # Also save just the new token list for embedding initialization
    token_list_path = SCANNER_ROOT / "training" / "new_nko_tokens.json"
    token_list = [{
        "token": t["token"],
        "constituents": t["constituents"],
        "char_count": t["char_count"],
    } for t in ext_data["new_tokens"]]
    with open(token_list_path, 'w') as f:
        json.dump(token_list, f, ensure_ascii=False, indent=2)
    print(f"  Token list: {token_list_path}")

    # Print top 20
    print(f"\nTop 20 new tokens:")
    for i, tok in enumerate(ext_data["new_tokens"][:20]):
        print(f"  {i+1:3d}. '{tok['token']}' ({tok['char_count']} chars) "
              f"= '{tok['constituents'][0]}' + '{tok['constituents'][1]}' "
              f"({len(tok['examples'])} examples)")


if __name__ == "__main__":
    main()
