#!/usr/bin/env python3
"""
Retrieval-Centric N'Ko ASR Model
==================================
Audio → N'Ko syllable retrieval, bypassing Latin transcription entirely.

Architecture:
  1. Frozen Whisper encoder → audio embeddings (d=1280)
  2. Audio projector MLP → shared space (d=512)
  3. N'Ko syllable codebook (3,024 entries, pre-indexed in d=512)
  4. Contrastive retrieval: pull matching audio-syllable pairs together
  5. FSM-constrained assembly of retrieved syllables into valid N'Ko words

Training:
  - L1: Audio ↔ Text contrastive (InfoNCE)
  - L2: Syllable retrieval cross-entropy
  - Data: (audio_segment, nko_text) pairs from ASR + bridge pipeline

Why this beats seq2seq:
  - Finite retrieval space (3,024 syllables) vs open-ended generation
  - N'Ko 1:1 phoneme-grapheme = no ambiguity
  - FSM guarantees valid syllable structure
  - Tone marks preserved (Latin drops them)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioProjector(nn.Module):
    """Projects Whisper encoder output to shared embedding space."""

    def __init__(self, input_dim: int = 1280, hidden_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 1280) -> (batch, seq_len, 512)"""
        return self.proj(x)


class SyllableCodebook(nn.Module):
    """Pre-indexed N'Ko syllable embeddings for retrieval."""

    def __init__(self, codebook_path: str, embed_dim: int = 512):
        super().__init__()

        with open(codebook_path) as f:
            cb = json.load(f)

        self.syllables = cb["syllables"]
        self.num_syllables = len(self.syllables)
        self.embed_dim = embed_dim

        # Learnable syllable embeddings
        self.embeddings = nn.Embedding(self.num_syllables, embed_dim)

        # Build lookup tables
        self.nko_to_idx = {}
        self.idx_to_nko = {}
        for i, syl in enumerate(self.syllables):
            nko = syl["nko"]
            self.nko_to_idx[nko] = i
            self.idx_to_nko[i] = nko

    def forward(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return embeddings for given indices, or all embeddings if None."""
        if indices is not None:
            return self.embeddings(indices)
        return self.embeddings.weight  # (num_syllables, embed_dim)

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve nearest syllables for query embeddings.

        query: (batch, seq_len, embed_dim)
        Returns: (scores, indices) each (batch, seq_len, top_k)
        """
        all_embeds = self.embeddings.weight  # (num_syllables, embed_dim)
        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        embeds_norm = F.normalize(all_embeds, dim=-1)
        # Similarity: (batch, seq_len, num_syllables)
        sim = torch.matmul(query_norm, embeds_norm.T)
        return torch.topk(sim, top_k, dim=-1)

    def encode_text(self, nko_text: str) -> List[int]:
        """Convert N'Ko text to syllable indices."""
        indices = []
        for char in nko_text:
            if char in self.nko_to_idx:
                indices.append(self.nko_to_idx[char])
        return indices

    def decode_indices(self, indices: List[int]) -> str:
        """Convert syllable indices back to N'Ko text."""
        return "".join(self.idx_to_nko.get(i, "") for i in indices)


class RetrievalASR(nn.Module):
    """Full retrieval-centric ASR model.

    Takes Whisper encoder output and retrieves N'Ko syllables.
    """

    def __init__(
        self,
        codebook_path: str,
        whisper_dim: int = 1280,
        embed_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # Audio encoder projection
        self.audio_proj = AudioProjector(whisper_dim, 768, embed_dim)

        # Syllable codebook
        self.codebook = SyllableCodebook(codebook_path, embed_dim)

        # CTC-style output head for variable-length sequences
        self.ctc_head = nn.Linear(embed_dim, self.codebook.num_syllables + 1)  # +1 for blank

    def forward(self, audio_features: torch.Tensor) -> dict:
        """
        audio_features: (batch, seq_len, whisper_dim) from Whisper encoder

        Returns dict with:
          - projected: (batch, seq_len, embed_dim)
          - ctc_logits: (batch, seq_len, num_syllables+1)
          - retrieval_scores: (batch, seq_len, num_syllables)
        """
        # Project audio to shared space
        projected = self.audio_proj(audio_features)

        # CTC logits for sequence prediction
        ctc_logits = self.ctc_head(projected)

        # Retrieval scores (cosine similarity with codebook)
        all_embeds = self.codebook.embeddings.weight
        proj_norm = F.normalize(projected, dim=-1)
        embeds_norm = F.normalize(all_embeds, dim=-1)
        retrieval_scores = torch.matmul(proj_norm, embeds_norm.T) / self.temperature

        return {
            "projected": projected,
            "ctc_logits": ctc_logits,
            "retrieval_scores": retrieval_scores,
        }

    def contrastive_loss(self, audio_proj: torch.Tensor, text_indices: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss between audio projections and syllable embeddings.

        audio_proj: (batch, embed_dim) - mean-pooled audio projection
        text_indices: (batch,) - target syllable indices
        """
        text_embeds = self.codebook(text_indices)  # (batch, embed_dim)

        # Normalize
        audio_norm = F.normalize(audio_proj, dim=-1)
        text_norm = F.normalize(text_embeds, dim=-1)

        # Similarity matrix
        logits = torch.matmul(audio_norm, text_norm.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        # Symmetric InfoNCE
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)

        return (loss_a2t + loss_t2a) / 2

    def ctc_loss(self, ctc_logits: torch.Tensor, targets: torch.Tensor,
                 input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """CTC loss for sequence-level training."""
        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)  # (T, B, C)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.codebook.num_syllables)


def build_model(codebook_path: str = None) -> RetrievalASR:
    """Build the retrieval ASR model."""
    if codebook_path is None:
        codebook_path = str(Path(__file__).parent.parent / "data" / "syllable_codebook.json")

    model = RetrievalASR(codebook_path)
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RetrievalASR: {param_count:,} params ({trainable:,} trainable)")
    print(f"  Audio projector: {sum(p.numel() for p in model.audio_proj.parameters()):,}")
    print(f"  Codebook: {model.codebook.num_syllables} syllables x {model.codebook.embed_dim}d")
    print(f"  CTC head: {sum(p.numel() for p in model.ctc_head.parameters()):,}")

    return model


if __name__ == "__main__":
    model = build_model()
    # Test with dummy input
    batch = torch.randn(2, 100, 1280)  # 2 samples, 100 time steps, Whisper dim
    output = model(batch)
    print(f"\nTest forward pass:")
    print(f"  Input: {batch.shape}")
    print(f"  Projected: {output['projected'].shape}")
    print(f"  CTC logits: {output['ctc_logits'].shape}")
    print(f"  Retrieval scores: {output['retrieval_scores'].shape}")
