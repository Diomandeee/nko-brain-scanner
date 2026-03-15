#!/usr/bin/env python3
"""
Joint Embedding Space for multimodal N'Ko ASR.

Projects audio features (from Whisper), visual features (from SigLIP),
and text features (from N'Ko BPE tokenizer) into a shared d=512
embedding space. The syllable codebook lives in this space, enabling
cross-modal retrieval.

Architecture:
    Audio Encoder (Whisper)  ──┐
                               ├──→ Joint Space (d=512) ──→ Syllable Codebook Retrieval
    Scene Encoder (SigLIP)   ──┤                              + FSM Assembly
                               │
    Text Encoder (N'Ko BPE)  ──┘

Training uses contrastive loss (audio↔text alignment) plus
codebook retrieval loss (retrieved syllable should match ground truth).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBED_DIM = 512


class LinearProjector:
    """Simple linear projection layer using numpy (no framework dependency)."""

    def __init__(self, input_dim, output_dim, name=""):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weight = np.random.randn(output_dim, input_dim).astype(np.float32) * scale
        self.bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        """Project input to output space. x: (..., input_dim) -> (..., output_dim)"""
        return x @ self.weight.T + self.bias

    def l2_normalize(self, x):
        """L2 normalize along last dimension."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norms + 1e-8)

    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}

    def load_state_dict(self, state):
        self.weight = state["weight"]
        self.bias = state["bias"]


class JointEmbeddingSpace:
    """
    Shared embedding space for audio, visual, and text modalities.

    Each modality has a projector that maps its native features
    into the joint space. The syllable codebook embeddings also
    live in this space, enabling cross-modal retrieval.
    """

    def __init__(self, audio_dim=512, visual_dim=512, text_dim=512, joint_dim=EMBED_DIM):
        """
        Args:
            audio_dim: Whisper encoder output dimension
            visual_dim: Scene encoder output dimension
            text_dim: Text encoder output dimension
            joint_dim: Shared embedding space dimension
        """
        self.joint_dim = joint_dim

        # Projectors for each modality
        self.audio_projector = LinearProjector(audio_dim, joint_dim, "audio")
        self.visual_projector = LinearProjector(visual_dim, joint_dim, "visual")
        self.text_projector = LinearProjector(text_dim, joint_dim, "text")

        # Fusion weights for multimodal combination
        self.audio_weight = 0.6
        self.visual_weight = 0.2
        self.text_weight = 0.2

    def project_audio(self, audio_features, normalize=True):
        """
        Project audio features into joint space.

        Args:
            audio_features: (num_frames, audio_dim) from Whisper encoder
            normalize: L2 normalize output

        Returns:
            (num_frames, joint_dim) embeddings
        """
        projected = self.audio_projector.forward(audio_features)
        if normalize:
            projected = self.audio_projector.l2_normalize(projected)
        return projected

    def project_visual(self, visual_features, normalize=True):
        """
        Project visual features into joint space.

        Args:
            visual_features: (num_keyframes, visual_dim) from scene encoder
            normalize: L2 normalize output

        Returns:
            (num_keyframes, joint_dim) embeddings
        """
        projected = self.visual_projector.forward(visual_features)
        if normalize:
            projected = self.visual_projector.l2_normalize(projected)
        return projected

    def project_text(self, text_features, normalize=True):
        """
        Project text features into joint space.

        Args:
            text_features: (seq_len, text_dim) from text encoder
            normalize: L2 normalize output

        Returns:
            (seq_len, joint_dim) embeddings
        """
        projected = self.text_projector.forward(text_features)
        if normalize:
            projected = self.text_projector.l2_normalize(projected)
        return projected

    def fuse_multimodal(self, audio_emb, visual_emb=None, text_emb=None):
        """
        Fuse embeddings from multiple modalities.

        For each audio frame, combines with nearest visual keyframe
        and text context using weighted average.

        Args:
            audio_emb: (num_frames, joint_dim) projected audio
            visual_emb: (num_keyframes, joint_dim) projected visual, or None
            text_emb: (seq_len, joint_dim) projected text context, or None

        Returns:
            (num_frames, joint_dim) fused embeddings
        """
        fused = audio_emb * self.audio_weight

        if visual_emb is not None and len(visual_emb) > 0:
            # For each audio frame, find nearest visual keyframe
            # Simple approach: replicate visual to match audio frame count
            num_audio = len(audio_emb)
            num_visual = len(visual_emb)

            if num_visual == 1:
                visual_matched = np.tile(visual_emb, (num_audio, 1))
            else:
                # Interpolate visual embeddings to match audio frames
                indices = np.linspace(0, num_visual - 1, num_audio).astype(int)
                visual_matched = visual_emb[indices]

            fused = fused + visual_matched * self.visual_weight
        else:
            # Redistribute visual weight to audio
            fused = fused + audio_emb * self.visual_weight

        if text_emb is not None and len(text_emb) > 0:
            # Use mean text embedding as global context
            text_context = text_emb.mean(axis=0, keepdims=True)
            text_broadcast = np.tile(text_context, (len(audio_emb), 1))
            fused = fused + text_broadcast * self.text_weight
        else:
            fused = fused + audio_emb * self.text_weight

        # Re-normalize
        norms = np.linalg.norm(fused, axis=-1, keepdims=True)
        return fused / (norms + 1e-8)

    def contrastive_loss(self, audio_emb, text_emb, temperature=0.07):
        """
        Compute InfoNCE contrastive loss between audio and text pairs.

        Args:
            audio_emb: (batch, joint_dim) audio embeddings
            text_emb: (batch, joint_dim) text embeddings (paired)
            temperature: softmax temperature

        Returns:
            loss: scalar contrastive loss
        """
        # Cosine similarity matrix
        sim_matrix = audio_emb @ text_emb.T / temperature  # (batch, batch)

        # Labels: diagonal elements are positive pairs
        batch_size = len(audio_emb)
        labels = np.arange(batch_size)

        # Cross-entropy loss (audio->text and text->audio)
        # Softmax along rows (audio->text)
        log_probs_a2t = sim_matrix - np.log(np.exp(sim_matrix).sum(axis=1, keepdims=True))
        loss_a2t = -log_probs_a2t[np.arange(batch_size), labels].mean()

        # Softmax along columns (text->audio)
        log_probs_t2a = sim_matrix.T - np.log(np.exp(sim_matrix.T).sum(axis=1, keepdims=True))
        loss_t2a = -log_probs_t2a[np.arange(batch_size), labels].mean()

        return (loss_a2t + loss_t2a) / 2

    def retrieval_loss(self, query_emb, codebook_emb, target_indices, temperature=0.1):
        """
        Compute codebook retrieval loss.

        Args:
            query_emb: (batch, joint_dim) query embeddings
            codebook_emb: (codebook_size, joint_dim) codebook embeddings
            target_indices: (batch,) correct codebook indices
            temperature: softmax temperature

        Returns:
            loss: scalar retrieval loss
        """
        # Similarity to all codebook entries
        sim = query_emb @ codebook_emb.T / temperature  # (batch, codebook_size)

        # Cross-entropy with target indices
        log_probs = sim - np.log(np.exp(sim).sum(axis=1, keepdims=True))
        loss = -log_probs[np.arange(len(target_indices)), target_indices].mean()

        return loss

    def save(self, path):
        """Save projector weights."""
        state = {
            "audio": self.audio_projector.state_dict(),
            "visual": self.visual_projector.state_dict(),
            "text": self.text_projector.state_dict(),
            "weights": {
                "audio": self.audio_weight,
                "visual": self.visual_weight,
                "text": self.text_weight,
            },
        }
        np.savez(path, **{
            f"{mod}_{key}": val
            for mod, proj_state in [("audio", state["audio"]), ("visual", state["visual"]), ("text", state["text"])]
            for key, val in proj_state.items()
        })

    def load(self, path):
        """Load projector weights."""
        data = np.load(path)
        self.audio_projector.weight = data["audio_weight"]
        self.audio_projector.bias = data["audio_bias"]
        self.visual_projector.weight = data["visual_weight"]
        self.visual_projector.bias = data["visual_bias"]
        self.text_projector.weight = data["text_weight"]
        self.text_projector.bias = data["text_bias"]


if __name__ == "__main__":
    print("=== Joint Embedding Space Smoke Test ===")

    space = JointEmbeddingSpace(audio_dim=512, visual_dim=512, text_dim=512)
    print(f"Joint dim: {space.joint_dim}")

    # Test projections
    batch = 4
    audio_feat = np.random.randn(batch, 512).astype(np.float32)
    visual_feat = np.random.randn(2, 512).astype(np.float32)
    text_feat = np.random.randn(3, 512).astype(np.float32)

    audio_emb = space.project_audio(audio_feat)
    visual_emb = space.project_visual(visual_feat)
    text_emb = space.project_text(text_feat)

    print(f"Audio projected: {audio_emb.shape}, norm: {np.linalg.norm(audio_emb[0]):.4f}")
    print(f"Visual projected: {visual_emb.shape}, norm: {np.linalg.norm(visual_emb[0]):.4f}")
    print(f"Text projected: {text_emb.shape}, norm: {np.linalg.norm(text_emb[0]):.4f}")

    # Test fusion
    fused = space.fuse_multimodal(audio_emb, visual_emb, text_emb)
    print(f"Fused: {fused.shape}, norm: {np.linalg.norm(fused[0]):.4f}")

    # Test contrastive loss
    text_paired = np.random.randn(batch, 512).astype(np.float32)
    text_paired_emb = space.project_text(text_paired)
    loss = space.contrastive_loss(audio_emb, text_paired_emb)
    print(f"Contrastive loss: {loss:.4f}")

    # Test retrieval loss
    codebook = np.random.randn(100, 512).astype(np.float32)
    codebook = codebook / np.linalg.norm(codebook, axis=-1, keepdims=True)
    targets = np.array([0, 5, 10, 15])
    r_loss = space.retrieval_loss(audio_emb, codebook, targets)
    print(f"Retrieval loss: {r_loss:.4f}")

    print("\nSmoke test passed.")
