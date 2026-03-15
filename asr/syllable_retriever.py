#!/usr/bin/env python3
"""
Syllable Retriever + FSM-Constrained Assembly for N'Ko ASR.

Loads the 3,024-entry syllable codebook, builds an embedding matrix,
performs k-NN retrieval via cosine similarity, and assembles valid N'Ko
text using FSM-constrained beam search.
"""

import sys
import os

# Cross-directory imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import mlx.core as mx
import mlx.nn as nn
from constrained.nko_fsm import NKoSyllableFSM

CODEBOOK_SIZE = 3024
EMBED_DIM = 512
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SyllableRetriever(nn.Module):
    """Retrieval-based syllable decoder with FSM-constrained beam search."""

    def __init__(self, codebook_path=None, embed_dim=EMBED_DIM):
        super().__init__()
        if codebook_path is None:
            codebook_path = os.path.join(PROJECT_ROOT, "data", "syllable_codebook.json")

        with open(codebook_path) as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            entries = raw.get("syllables", [])
        else:
            entries = raw

        self.codebook_entries = entries
        self.codebook_nko = [e["nko"] for e in entries]
        self.codebook_ipa = [e.get("ipa", "") for e in entries]
        self.codebook_size = len(entries)

        # Embedding matrix: (codebook_size, embed_dim)
        # Initialize with scaled random normal
        scale = 1.0 / (embed_dim ** 0.5)
        self.embeddings = mx.random.normal(shape=(self.codebook_size, embed_dim)) * scale

        # Pre-compute normalized embeddings for retrieval
        self._update_normed()

    def _update_normed(self):
        """Recompute normalized embedding cache."""
        norms = mx.sqrt(mx.sum(self.embeddings ** 2, axis=-1, keepdims=True))
        self.normed_embeddings = self.embeddings / (norms + 1e-8)

    def retrieve_top_k(self, query, k=10):
        """
        Retrieve top-k codebook entries by cosine similarity.

        Args:
            query: (embed_dim,) normalized audio embedding
            k: number of candidates to return

        Returns:
            indices: (k,) codebook indices
            scores: (k,) similarity scores
        """
        # Normalize query
        q_norm = mx.sqrt(mx.sum(query ** 2))
        q_normed = query / (q_norm + 1e-8)

        # Cosine similarity against all codebook entries
        sims = self.normed_embeddings @ q_normed  # (codebook_size,)

        # Top-k
        top_k_idx = mx.argsort(-sims)[:k]
        return top_k_idx, sims[top_k_idx]

    def fsm_beam_search(self, audio_embeddings, beam_width=5):
        """
        FSM-constrained beam search over retrieved syllables.

        Args:
            audio_embeddings: list of (embed_dim,) embeddings, one per audio frame
            beam_width: number of beams to maintain

        Returns:
            best text (str), best score (float)
        """
        # Each beam: (text, score, fsm_instance)
        beams = [("", 0.0, NKoSyllableFSM())]

        for frame_emb in audio_embeddings:
            top_k_idx, scores = self.retrieve_top_k(frame_emb, k=beam_width * 3)
            idx_list = top_k_idx.tolist()
            score_list = scores.tolist()

            new_beams = []
            for text, cumulative_score, fsm in beams:
                for idx, sim in zip(idx_list, score_list):
                    syllable = self.codebook_nko[idx]

                    # Check if this syllable is FSM-admissible
                    fsm_copy = fsm.clone()
                    if fsm_copy.would_be_admissible(syllable):
                        # Advance the FSM through all chars of the syllable
                        for ch in syllable:
                            fsm_copy.advance(ch)
                        new_beams.append((
                            text + syllable,
                            cumulative_score + sim,
                            fsm_copy,
                        ))

            if new_beams:
                # Keep top beam_width candidates by score
                beams = sorted(new_beams, key=lambda x: -x[1])[:beam_width]
            # If no valid beams found, keep previous beams unchanged

        if not beams:
            return "", 0.0

        best = beams[0]
        return best[0], best[1]

    def retrieve_syllable(self, index):
        """Get syllable info by codebook index."""
        if 0 <= index < self.codebook_size:
            return self.codebook_entries[index]
        return None

    def get_nko_text(self, indices):
        """Convert codebook indices to N'Ko text."""
        return "".join(self.codebook_nko[i] for i in indices if 0 <= i < self.codebook_size)


if __name__ == "__main__":
    print("=== Syllable Retriever Smoke Test ===")
    retriever = SyllableRetriever()
    print(f"Codebook size: {retriever.codebook_size}")
    print(f"Embedding shape: {retriever.embeddings.shape}")

    # Test retrieval with random query
    query = mx.random.normal(shape=(EMBED_DIM,))
    indices, scores = retriever.retrieve_top_k(query, k=5)
    print(f"\nTop-5 for random query:")
    for idx, score in zip(indices.tolist(), scores.tolist()):
        entry = retriever.codebook_entries[idx]
        print(f"  [{idx}] {entry['nko']} ({entry['ipa']}) score={score:.4f}")

    # Test beam search with synthetic embeddings
    # Use actual codebook embeddings as "perfect" audio embeddings
    test_indices = [0, 10, 20, 30, 40]  # arbitrary codebook entries
    test_embeddings = [retriever.embeddings[i] for i in test_indices]

    text, score = retriever.fsm_beam_search(test_embeddings, beam_width=5)
    print(f"\nBeam search result:")
    print(f"  Text: {text}")
    print(f"  Score: {score:.4f}")
    print(f"  Expected syllables: {[retriever.codebook_nko[i] for i in test_indices]}")

    print("\nSmoke test passed.")
