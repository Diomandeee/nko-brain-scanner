#!/usr/bin/env python3
"""
Generate synthetic evaluation pairs for the syllable retriever.

Creates (codebook_indices, nko_text) pairs using FSM-valid syllable sequences.
These test retrieval accuracy in the trivial case (perfect embeddings).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from constrained.nko_fsm import NKoSyllableFSM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_pair(codebook_nko, n_syllables=5):
    """
    Generate a synthetic (indices, nko_text) pair.

    Strategy: randomly pick syllables and check FSM validity.
    """
    fsm = NKoSyllableFSM()
    chosen_indices = []
    text = ""

    for _ in range(n_syllables * 3):  # extra tries for FSM rejections
        if len(chosen_indices) >= n_syllables:
            break

        idx = random.randint(0, len(codebook_nko) - 1)
        syllable = codebook_nko[idx]

        # Check FSM validity
        fsm_copy = fsm.clone()
        if fsm_copy.would_be_admissible(syllable):
            for ch in syllable:
                fsm_copy.advance(ch)
            fsm = fsm_copy
            chosen_indices.append(idx)
            text += syllable

    return {
        "indices": chosen_indices,
        "nko_text": text,
        "n_syllables": len(chosen_indices),
    }


if __name__ == "__main__":
    codebook_path = os.path.join(PROJECT_ROOT, "data", "syllable_codebook.json")
    with open(codebook_path) as f:
        raw = json.load(f)
    entries = raw.get("syllables", raw) if isinstance(raw, dict) else raw
    codebook_nko = [e["nko"] for e in entries]

    random.seed(42)
    pairs = [generate_synthetic_pair(codebook_nko) for _ in range(200)]

    # Filter out empty pairs
    valid_pairs = [p for p in pairs if p["n_syllables"] > 0]

    output_path = os.path.join(PROJECT_ROOT, "data", "synthetic_eval_pairs.json")
    with open(output_path, "w") as f:
        json.dump(valid_pairs, f, ensure_ascii=False, indent=2)

    avg_len = sum(p["n_syllables"] for p in valid_pairs) / len(valid_pairs) if valid_pairs else 0
    print(f"Generated {len(valid_pairs)} valid synthetic eval pairs (avg {avg_len:.1f} syllables)")
    print(f"Saved to {output_path}")
