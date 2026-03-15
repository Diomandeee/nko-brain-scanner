#!/usr/bin/env python3
"""
Multi-loss Training Loop for N'Ko Retrieval-Centric ASR.

Trains the joint embedding space by:
1. Encoding audio segments with Whisper encoder
2. Encoding corresponding N'Ko text with the text projector
3. Optimizing contrastive loss (audio ↔ text alignment)
4. Optimizing retrieval loss (audio → correct syllable in codebook)

The syllable codebook embeddings and modality projectors are the
trainable parameters. Whisper encoder stays frozen.

Usage:
    # Train on VAD-segmented audio with N'Ko transcriptions
    python3 asr/train_asr.py --manifest data/djoko-audio/manifest.jsonl \
                              --codebook data/syllable_codebook.json \
                              --epochs 50 --batch-size 16

    # Evaluate round-trip accuracy
    python3 asr/train_asr.py --eval --checkpoint checkpoints/asr_best.npz
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr.joint_embedding import JointEmbeddingSpace, EMBED_DIM
from asr.syllable_retriever import SyllableRetriever

PROJECT_ROOT = Path(__file__).parent.parent


class ASRTrainer:
    """
    Training loop for the N'Ko retrieval-centric ASR system.

    Jointly optimizes:
    - Audio → text contrastive alignment
    - Audio → syllable retrieval accuracy
    - (Optional) Visual → text grounding
    """

    def __init__(
        self,
        codebook_path=None,
        audio_dim=512,
        joint_dim=EMBED_DIM,
        learning_rate=1e-4,
        temperature=0.07,
        retrieval_weight=0.5,
        contrastive_weight=0.5,
    ):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.retrieval_weight = retrieval_weight
        self.contrastive_weight = contrastive_weight

        # Joint embedding space
        self.space = JointEmbeddingSpace(
            audio_dim=audio_dim,
            visual_dim=audio_dim,
            text_dim=audio_dim,
            joint_dim=joint_dim,
        )

        # Syllable retriever with codebook
        if codebook_path is None:
            codebook_path = str(PROJECT_ROOT / "data" / "syllable_codebook.json")
        self.retriever = SyllableRetriever(codebook_path, embed_dim=joint_dim)

        self.epoch = 0
        self.best_loss = float("inf")
        self.history = []

    def compute_loss(self, audio_features, text_features, target_syllable_indices=None):
        """
        Compute combined training loss.

        Args:
            audio_features: (batch, audio_dim) raw audio encoder features
            text_features: (batch, text_dim) raw text features
            target_syllable_indices: (batch,) codebook indices, or None

        Returns:
            total_loss, contrastive_loss, retrieval_loss
        """
        # Project into joint space
        audio_emb = self.space.project_audio(audio_features)
        text_emb = self.space.project_text(text_features)

        # Contrastive loss
        c_loss = self.space.contrastive_loss(audio_emb, text_emb, self.temperature)

        # Retrieval loss
        r_loss = 0.0
        if target_syllable_indices is not None:
            # Get codebook embeddings
            codebook_emb = self.retriever.normed_embeddings
            if hasattr(codebook_emb, "tolist"):
                # Convert from MLX to numpy
                codebook_np = np.array(codebook_emb.tolist())
            else:
                codebook_np = codebook_emb

            r_loss = self.space.retrieval_loss(
                audio_emb, codebook_np, target_syllable_indices, self.temperature
            )

        total = self.contrastive_weight * c_loss + self.retrieval_weight * r_loss
        return total, c_loss, r_loss

    def train_step(self, audio_features, text_features, target_indices=None):
        """
        Single training step with gradient estimation via finite differences.

        For production training, use torch/MLX autograd instead.
        This implementation uses numerical gradients for framework independence.

        Args:
            audio_features: (batch, audio_dim)
            text_features: (batch, text_dim)
            target_indices: (batch,) codebook indices

        Returns:
            loss values
        """
        total, c_loss, r_loss = self.compute_loss(
            audio_features, text_features, target_indices
        )

        # Numerical gradient update for projector weights
        eps = 1e-4
        for projector in [self.space.audio_projector, self.space.text_projector]:
            grad_w = np.zeros_like(projector.weight)

            # Sample a subset of weight indices for efficiency
            num_samples = min(100, projector.weight.size)
            flat_indices = np.random.choice(projector.weight.size, num_samples, replace=False)

            for idx in flat_indices:
                multi_idx = np.unravel_index(idx, projector.weight.shape)

                # f(w + eps)
                projector.weight[multi_idx] += eps
                loss_plus, _, _ = self.compute_loss(
                    audio_features, text_features, target_indices
                )

                # f(w - eps)
                projector.weight[multi_idx] -= 2 * eps
                loss_minus, _, _ = self.compute_loss(
                    audio_features, text_features, target_indices
                )

                # Restore and compute gradient
                projector.weight[multi_idx] += eps
                grad_w[multi_idx] = (loss_plus - loss_minus) / (2 * eps)

            # Scale gradient by sampling ratio
            scale = projector.weight.size / num_samples
            projector.weight -= self.learning_rate * grad_w * scale

        return total, c_loss, r_loss

    def train_epoch(self, dataloader, verbose=True):
        """
        Train for one epoch.

        Args:
            dataloader: iterable of (audio_features, text_features, target_indices) batches
            verbose: print progress

        Returns:
            average loss for the epoch
        """
        self.epoch += 1
        total_loss = 0
        total_c_loss = 0
        total_r_loss = 0
        num_batches = 0

        for batch_idx, (audio_feat, text_feat, target_idx) in enumerate(dataloader):
            loss, c_loss, r_loss = self.train_step(audio_feat, text_feat, target_idx)
            total_loss += loss
            total_c_loss += c_loss
            total_r_loss += r_loss
            num_batches += 1

            if verbose and batch_idx % 10 == 0:
                print(
                    f"  Epoch {self.epoch} | Batch {batch_idx} | "
                    f"Loss: {loss:.4f} (C: {c_loss:.4f}, R: {r_loss:.4f})"
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_c = total_c_loss / max(num_batches, 1)
        avg_r = total_r_loss / max(num_batches, 1)

        self.history.append({
            "epoch": self.epoch,
            "loss": float(avg_loss),
            "contrastive_loss": float(avg_c),
            "retrieval_loss": float(avg_r),
        })

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        return avg_loss

    def evaluate_retrieval(self, audio_features, ground_truth_syllables):
        """
        Evaluate syllable retrieval accuracy.

        Args:
            audio_features: (num_samples, audio_dim)
            ground_truth_syllables: list of expected N'Ko syllable strings

        Returns:
            top1_accuracy, top5_accuracy
        """
        audio_emb = self.space.project_audio(audio_features)

        top1_correct = 0
        top5_correct = 0

        import mlx.core as mx

        for i, gt_syllable in enumerate(ground_truth_syllables):
            query = mx.array(audio_emb[i])
            indices, scores = self.retriever.retrieve_top_k(query, k=5)
            idx_list = indices.tolist()

            retrieved_syllables = [self.retriever.codebook_nko[j] for j in idx_list]

            if retrieved_syllables[0] == gt_syllable:
                top1_correct += 1
            if gt_syllable in retrieved_syllables:
                top5_correct += 1

        n = len(ground_truth_syllables)
        return top1_correct / n if n > 0 else 0, top5_correct / n if n > 0 else 0

    def save_checkpoint(self, path):
        """Save training checkpoint."""
        self.space.save(path)
        # Also save training history
        history_path = path.replace(".npz", "_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "epoch": self.epoch,
                "best_loss": float(self.best_loss),
                "history": self.history,
            }, f, indent=2)

    def load_checkpoint(self, path):
        """Load training checkpoint."""
        self.space.load(path)
        history_path = path.replace(".npz", "_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                data = json.load(f)
                self.epoch = data.get("epoch", 0)
                self.best_loss = data.get("best_loss", float("inf"))
                self.history = data.get("history", [])


def create_synthetic_dataloader(codebook_path, batch_size=16, num_batches=50):
    """
    Create synthetic training data for development/testing.

    Uses codebook embeddings as "perfect" audio features to verify
    the training loop can learn the identity mapping.
    """
    retriever = SyllableRetriever(codebook_path, embed_dim=EMBED_DIM)

    import mlx.core as mx

    codebook_np = np.array(retriever.embeddings.tolist())
    codebook_size = len(codebook_np)

    def generate_batch():
        indices = np.random.randint(0, codebook_size, size=batch_size)

        # Audio = codebook embedding + noise
        audio = codebook_np[indices] + np.random.randn(batch_size, EMBED_DIM).astype(np.float32) * 0.1

        # Text = codebook embedding + different noise (paired)
        text = codebook_np[indices] + np.random.randn(batch_size, EMBED_DIM).astype(np.float32) * 0.1

        return audio, text, indices

    for _ in range(num_batches):
        yield generate_batch()


def main():
    parser = argparse.ArgumentParser(description="Train N'Ko retrieval-centric ASR")
    parser.add_argument("--manifest", help="Training manifest JSONL")
    parser.add_argument("--codebook", default=str(PROJECT_ROOT / "data" / "syllable_codebook.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--checkpoint", help="Checkpoint to load for eval")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    trainer = ASRTrainer(
        codebook_path=args.codebook,
        learning_rate=args.lr,
    )

    if args.eval:
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        print("Evaluation mode not yet connected to real audio data.")
        return

    print(f"=== N'Ko ASR Training ===")
    print(f"Codebook: {trainer.retriever.codebook_size} syllables")
    print(f"Embed dim: {EMBED_DIM}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    if args.synthetic or not args.manifest:
        print("\nUsing synthetic data (codebook + noise)...")
        for epoch in range(args.epochs):
            dataloader = create_synthetic_dataloader(
                args.codebook, args.batch_size, num_batches=20
            )
            avg_loss = trainer.train_epoch(dataloader)
            print(f"Epoch {trainer.epoch}: avg_loss={avg_loss:.4f} (best={trainer.best_loss:.4f})")

            if trainer.epoch % 10 == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"asr_epoch{trainer.epoch}.npz")
                trainer.save_checkpoint(ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
    else:
        print(f"\nManifest: {args.manifest}")
        print("Real audio training requires manifest with VAD segments + transcriptions.")
        print("See asr/audio_pipeline.py for manifest generation.")

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "asr_final.npz")
    trainer.save_checkpoint(final_path)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Best loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
