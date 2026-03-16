#!/usr/bin/env python3
"""
Retrieval-Centric N'Ko ASR Model — V2 (Fixed)
================================================
Fixes from V1 diagnosis:
  1. SYLLABLE-LEVEL tokenizer (greedy longest-match) instead of char-by-char
  2. Temporal modeling via Transformer encoder layers before CTC head
  3. Stride-4 downsampling to reduce CTC blank ratio from 93% to ~73%
  4. Increased model capacity (embed_dim=768, 4 transformer layers -> ~25M params)
  5. Focal CTC loss for class imbalance

Architecture:
  1. Frozen Whisper encoder -> audio embeddings (d=1280)
  2. Stride-4 Conv1d downsampler (1500 -> 375 frames)
  3. Audio projector MLP -> shared space (d=768)
  4. 4-layer Transformer encoder for temporal context
  5. CTC head -> 3,025 classes (3,024 syllables + blank)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NKoSyllableTokenizer:
    """Greedy longest-match syllable tokenizer for N'Ko text.

    The codebook contains multi-character syllable entries (e.g., "ߞߊ" = "ka").
    V1 iterated char-by-char, matching only the 7 bare vowels (0.2% of codebook).
    This tokenizer does greedy longest-match to use the full 3,024-entry codebook.

    The codebook has only V, VN, CV, CVN patterns (no VC, CVC).
    Standalone consonants (coda consonants like ߣ in ߊߣ "an") are given
    extra indices beyond the codebook so they aren't silently dropped.
    """

    # N'Ko consonant codepoints (U+07D2 to U+07E6, excluding vowels U+07CA-U+07D0)
    NKO_CONSONANTS = [chr(c) for c in range(0x07D2, 0x07E7)]

    def __init__(self, codebook: dict):
        self.syllables = codebook["syllables"]
        self.codebook_size = len(self.syllables)

        # Build lookup: nko_string -> index
        self.nko_to_idx = {}
        self.idx_to_nko = {}
        self.max_syl_len = 0

        for i, syl in enumerate(self.syllables):
            nko = syl["nko"]
            self.nko_to_idx[nko] = i
            self.idx_to_nko[i] = nko
            self.max_syl_len = max(self.max_syl_len, len(nko))

        # Add standalone consonant entries for coda consonants
        # These get indices starting at codebook_size
        self.extra_consonants = {}
        extra_idx = self.codebook_size
        for c in self.NKO_CONSONANTS:
            if c not in self.nko_to_idx:
                self.nko_to_idx[c] = extra_idx
                self.idx_to_nko[extra_idx] = c
                self.extra_consonants[c] = extra_idx
                extra_idx += 1

        self.num_syllables = extra_idx  # codebook_size + num_extra_consonants

        # Build trie for efficient longest-match
        self.trie = {}
        for nko_str, idx in self.nko_to_idx.items():
            node = self.trie
            for ch in nko_str:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node["_idx"] = idx

    def encode(self, nko_text: str) -> List[int]:
        """Encode N'Ko text to syllable indices using greedy longest-match.

        Skips only spaces and punctuation. Consonants that don't start a
        codebook syllable are encoded as standalone consonant tokens.
        """
        indices = []
        i = 0
        while i < len(nko_text):
            # Try longest match first via trie
            node = self.trie
            best_idx = None
            best_len = 0

            for j in range(i, min(i + self.max_syl_len, len(nko_text))):
                ch = nko_text[j]
                if ch not in node:
                    break
                node = node[ch]
                if "_idx" in node:
                    best_idx = node["_idx"]
                    best_len = j - i + 1

            if best_idx is not None:
                indices.append(best_idx)
                i += best_len
            else:
                # Skip spaces/punctuation, but NOT N'Ko characters
                i += 1

        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert syllable indices back to N'Ko text."""
        return "".join(self.idx_to_nko.get(i, "") for i in indices)


class TemporalDownsampler(nn.Module):
    """Stride-4 Conv1d to reduce sequence length before CTC.

    Whisper outputs 1500 frames for 30s audio (50 fps).
    With stride 4: 1500 -> 375 frames.
    This reduces the blank ratio from ~93% to ~73% for typical targets.
    """

    def __init__(self, input_dim: int = 1280, output_dim: int = 768, stride: int = 4):
        super().__init__()
        self.stride = stride
        # Conv1d expects channel-first (B, C, T); LayerNorm applied after permuting back
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=stride * 2 - 1,
                      stride=stride, padding=stride - 1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len//stride, output_dim)"""
        # Conv1d expects (batch, channels, seq_len)
        out = self.conv_layers(x.permute(0, 2, 1))  # (B, output_dim, T//stride)
        out = out.permute(0, 2, 1)  # (B, T//stride, output_dim)
        out = self.norm(out)
        return out

    def output_length(self, input_length: int) -> int:
        """Calculate output sequence length for a given input length."""
        kernel_size = self.stride * 2 - 1
        padding = self.stride - 1
        return (input_length + 2 * padding - kernel_size) // self.stride + 1


class AudioProjector(nn.Module):
    """Projects downsampled features to embedding space."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, output_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TemporalEncoder(nn.Module):
    """Transformer encoder for temporal context across frames.

    V1 had NO temporal modeling - each frame was classified independently.
    This makes CTC alignment nearly impossible because CTC requires
    the model to learn when to spike vs. emit blank, which requires
    knowing neighboring frame context.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8,
                 num_layers: int = 4, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        # enable_nested_tensor=False suppresses the UserWarning about norm_first
        # (nested tensor optimisation is disabled anyway when norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)"""
        x = self.pos_encoding(x)
        return self.encoder(x, src_key_padding_mask=padding_mask)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SyllableCodebook(nn.Module):
    """Pre-indexed N'Ko syllable embeddings for retrieval."""

    def __init__(self, codebook_path: str, embed_dim: int = 768):
        super().__init__()

        with open(codebook_path) as f:
            cb = json.load(f)

        self.syllables = cb["syllables"]
        self.embed_dim = embed_dim

        # Tokenizer for proper syllable-level encoding (includes extra consonant entries)
        self.tokenizer = NKoSyllableTokenizer(cb)
        self.num_syllables = self.tokenizer.num_syllables  # codebook + extra consonants

        # Learnable syllable embeddings (covers codebook + extra consonant entries)
        self.embeddings = nn.Embedding(self.num_syllables, embed_dim)

    def forward(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indices is not None:
            return self.embeddings(indices)
        return self.embeddings.weight

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeds = self.embeddings.weight
        query_norm = F.normalize(query, dim=-1)
        embeds_norm = F.normalize(all_embeds, dim=-1)
        sim = torch.matmul(query_norm, embeds_norm.T)
        return torch.topk(sim, top_k, dim=-1)

    def encode_text(self, nko_text: str) -> List[int]:
        """Convert N'Ko text to syllable indices using proper syllable tokenizer."""
        return self.tokenizer.encode(nko_text)

    def decode_indices(self, indices: List[int]) -> str:
        return self.tokenizer.decode(indices)


class RetrievalASRv2(nn.Module):
    """Fixed retrieval-centric ASR model.

    Changes from V1:
      - Stride-4 downsampler: 1500 -> 375 frames
      - 4-layer Transformer encoder for temporal context
      - embed_dim=768 (up from 512) for capacity
      - ~25M params (up from 2.9M)
    """

    def __init__(
        self,
        codebook_path: str,
        whisper_dim: int = 1280,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        downsample_stride: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.downsample_stride = downsample_stride

        # 1. Temporal downsampler (1500 -> 375 frames)
        self.downsampler = TemporalDownsampler(whisper_dim, embed_dim, stride=downsample_stride)

        # 2. Audio projector
        self.audio_proj = AudioProjector(embed_dim, embed_dim * 2, embed_dim)

        # 3. Temporal encoder (the key missing piece from V1)
        self.temporal_encoder = TemporalEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # 4. Syllable codebook
        self.codebook = SyllableCodebook(codebook_path, embed_dim)

        # 5. CTC head
        self.ctc_head = nn.Linear(embed_dim, self.codebook.num_syllables + 1)  # +1 for blank
        self.num_syllables = self.codebook.num_syllables

    def forward(self, audio_features: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        audio_features: (batch, seq_len, whisper_dim) from Whisper encoder

        Returns dict with:
          - ctc_logits: (batch, downsampled_len, num_syllables+1)
          - projected: (batch, downsampled_len, embed_dim)
        """
        # Downsample: (B, 1500, 1280) -> (B, 375, 768)
        downsampled = self.downsampler(audio_features)

        # Project
        projected = self.audio_proj(downsampled)

        # Temporal encoding (THE critical fix)
        encoded = self.temporal_encoder(projected, padding_mask)

        # CTC logits
        ctc_logits = self.ctc_head(encoded)

        return {
            "ctc_logits": ctc_logits,
            "projected": encoded,
        }

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after downsampling."""
        return torch.tensor([
            self.downsampler.output_length(l.item()) for l in input_lengths
        ], dtype=torch.long)

    def ctc_decode(self, ctc_logits: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decoding (collapse repeats, remove blanks)."""
        predictions = ctc_logits.argmax(dim=-1)  # (batch, time)
        results = []
        for pred in predictions:
            decoded = []
            prev = -1
            for idx in pred.tolist():
                if idx != self.num_syllables and idx != prev:  # not blank and not repeat
                    decoded.append(idx)
                prev = idx
            results.append(decoded)
        return results


class FocalCTCLoss(nn.Module):
    """CTC loss with focal weighting to counter class imbalance.

    Standard CTC loss is dominated by blank tokens when T >> U.
    Focal weighting reduces the contribution of easy (blank) predictions.
    """

    def __init__(self, blank: int, gamma: float = 2.0, zero_infinity: bool = True):
        super().__init__()
        self.blank = blank
        self.gamma = gamma
        self.zero_infinity = zero_infinity

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        # Standard CTC loss
        ctc_loss = F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=self.blank, reduction="none", zero_infinity=self.zero_infinity,
        )

        # Focal weighting: down-weight easy samples
        with torch.no_grad():
            # Approximate "easiness" as exp(-loss) -- well-predicted samples have low loss
            p = torch.exp(-ctc_loss)
            focal_weight = (1 - p) ** self.gamma

        return (focal_weight * ctc_loss).mean()


def build_model(codebook_path: str = None, **kwargs) -> RetrievalASRv2:
    """Build the V2 retrieval ASR model."""
    if codebook_path is None:
        # Try Vast.ai workspace path first, then fall back to repo-relative path
        vastai_path = Path("/workspace/syllable_codebook.json")
        repo_path = Path(__file__).parent.parent / "data" / "syllable_codebook.json"
        codebook_path = str(vastai_path if vastai_path.exists() else repo_path)

    model = RetrievalASRv2(codebook_path, **kwargs)
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"RetrievalASR V2: {param_count:,} params ({trainable:,} trainable)")
    print(f"  Downsampler:      {sum(p.numel() for p in model.downsampler.parameters()):,}")
    print(f"  Audio projector:  {sum(p.numel() for p in model.audio_proj.parameters()):,}")
    print(f"  Temporal encoder: {sum(p.numel() for p in model.temporal_encoder.parameters()):,}")
    print(f"  Codebook:         {model.codebook.num_syllables} syllables x {model.codebook.embed_dim}d = {sum(p.numel() for p in model.codebook.parameters()):,}")
    print(f"  CTC head:         {sum(p.numel() for p in model.ctc_head.parameters()):,}")

    return model


if __name__ == "__main__":
    model = build_model()

    # Test with dummy Whisper output (30s audio = 1500 frames)
    batch = torch.randn(2, 1500, 1280)
    output = model(batch)
    print(f"\nTest forward pass:")
    print(f"  Input:       {batch.shape}")
    print(f"  CTC logits:  {output['ctc_logits'].shape}")
    print(f"  Projected:   {output['projected'].shape}")

    # Test downsampled lengths
    input_lens = torch.tensor([1500, 800])
    output_lens = model.compute_output_lengths(input_lens)
    print(f"  Input lens:  {input_lens.tolist()}")
    print(f"  Output lens: {output_lens.tolist()}")

    # Test syllable tokenizer
    tokenizer = model.codebook.tokenizer
    test_text = "\u07de\u07ca \u07e3\u07cf\u07e3\u07ca"  # "ka nona" roughly
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n  Tokenizer test:")
    print(f"    Input:   '{test_text}' ({len(test_text)} chars)")
    print(f"    Encoded: {encoded} ({len(encoded)} tokens)")
    print(f"    Decoded: '{decoded}'")

    # Test CTC decode
    fake_logits = torch.randn(2, output['ctc_logits'].shape[1], model.num_syllables + 1)
    decoded_seqs = model.ctc_decode(fake_logits)
    print(f"\n  CTC decode test:")
    for i, seq in enumerate(decoded_seqs):
        text = tokenizer.decode(seq)
        print(f"    Sample {i}: {len(seq)} syllables -> '{text[:50]}...'")
