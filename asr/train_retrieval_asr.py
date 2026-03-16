#!/usr/bin/env python3
"""
Train Retrieval ASR — Full Pipeline on GPU
=============================================
1. Load training pairs (audio_path + nko_text)
2. Extract Whisper features (frozen encoder)
3. Train retrieval model (AudioProjector + CTC head)
4. Evaluate on held-out test set

Usage (on Vast.ai):
    python3 train_retrieval_asr.py --data /workspace/data/unified_training_pairs.jsonl \
                                    --codebook /workspace/data/syllable_codebook.json \
                                    --epochs 20 --batch-size 8 --lr 1e-4
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class NKoASRDataset(Dataset):
    """Dataset of (whisper_features, nko_syllable_indices) pairs."""

    def __init__(self, features_dir, pairs, codebook, max_audio_len=100, max_text_len=50):
        self.features_dir = Path(features_dir)
        self.pairs = pairs
        self.codebook = codebook
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # Build char → syllable index mapping
        self.char_to_idx = {}
        for i, syl in enumerate(codebook["syllables"]):
            self.char_to_idx[syl["nko"]] = i

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load pre-extracted Whisper features
        feat_path = self.features_dir / f"{pair['feat_id']}.pt"
        if feat_path.exists():
            features = torch.load(feat_path, weights_only=True)
        else:
            # Fallback: zero features
            features = torch.zeros(self.max_audio_len, 1280)

        # Pad/truncate audio features
        if features.shape[0] > self.max_audio_len:
            features = features[:self.max_audio_len]
        audio_len = features.shape[0]

        padded = torch.zeros(self.max_audio_len, 1280)
        padded[:audio_len] = features

        # Encode N'Ko text as syllable indices
        nko_text = pair.get("nko", pair.get("nko_text", ""))
        indices = []
        for char in nko_text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
        if not indices:
            indices = [0]  # blank

        # Pad/truncate text
        text_len = min(len(indices), self.max_text_len)
        text_indices = torch.zeros(self.max_text_len, dtype=torch.long)
        text_indices[:text_len] = torch.tensor(indices[:text_len])

        return {
            "features": padded,
            "audio_len": audio_len,
            "targets": text_indices,
            "target_len": text_len,
        }


class AudioProjector(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=768, output_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class RetrievalASR(nn.Module):
    def __init__(self, num_syllables, embed_dim=512, whisper_dim=1280):
        super().__init__()
        self.audio_proj = AudioProjector(whisper_dim, 768, embed_dim)
        self.syllable_embeddings = nn.Embedding(num_syllables, embed_dim)
        self.ctc_head = nn.Linear(embed_dim, num_syllables + 1)  # +1 blank
        self.num_syllables = num_syllables

    def forward(self, features):
        projected = self.audio_proj(features)
        ctc_logits = self.ctc_head(projected)
        return ctc_logits

    def retrieval_scores(self, features):
        projected = self.audio_proj(features)
        proj_norm = F.normalize(projected, dim=-1)
        emb_norm = F.normalize(self.syllable_embeddings.weight, dim=-1)
        return torch.matmul(proj_norm, emb_norm.T) / 0.07


def extract_whisper_features(audio_paths, output_dir, device="cuda", batch_size=16):
    """Extract features from frozen Whisper encoder."""
    import whisper

    print("Loading Whisper large-v3 for feature extraction...")
    model = whisper.load_model("large-v3", device=device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for i, audio_path in enumerate(audio_paths):
        feat_id = Path(audio_path).stem
        out_path = output_dir / f"{feat_id}.pt"

        if out_path.exists():
            extracted += 1
            continue

        try:
            # Load and pad audio to 30s
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Extract mel spectrogram
            mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(device)

            # Run encoder
            with torch.no_grad():
                features = model.encoder(mel.unsqueeze(0))  # (1, T, 1280)

            # Save features
            torch.save(features.squeeze(0).cpu(), out_path)
            extracted += 1

        except Exception as e:
            if i < 5:
                print(f"  Skip {feat_id}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(audio_paths)}] {extracted} extracted")

    print(f"Feature extraction done: {extracted}/{len(audio_paths)}")
    return extracted


def train(model, train_loader, val_loader, epochs, lr, device, save_dir):
    """Train the retrieval ASR model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_steps = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            audio_lens = batch["audio_len"]
            target_lens = batch["target_len"]

            ctc_logits = model(features)
            log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)

            loss = F.ctc_loss(
                log_probs, targets, audio_lens, target_lens,
                blank=model.num_syllables, zero_infinity=True,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                audio_lens = batch["audio_len"]
                target_lens = batch["target_len"]

                ctc_logits = model(features)
                log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)

                loss = F.ctc_loss(
                    log_probs, targets, audio_lens, target_lens,
                    blank=model.num_syllables, zero_infinity=True,
                )
                val_loss += loss.item()
                val_steps += 1

        avg_train = train_loss / max(train_steps, 1)
        avg_val = val_loss / max(val_steps, 1)

        print(f"Epoch {epoch+1}/{epochs} | train_loss: {avg_train:.4f} | val_loss: {avg_val:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")

    # Save final
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Unified training pairs JSONL")
    parser.add_argument("--codebook", required=True, help="Syllable codebook JSON")
    parser.add_argument("--audio-dir", default="", help="Dir with audio WAVs (for feature extraction)")
    parser.add_argument("--features-dir", default="/workspace/features", help="Pre-extracted features dir")
    parser.add_argument("--save-dir", default="/workspace/model", help="Model save dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    with open(args.data) as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(pairs)} training pairs")

    with open(args.codebook) as f:
        codebook = json.load(f)
    print(f"Codebook: {len(codebook['syllables'])} syllables")

    # Feature extraction (if not skipped)
    if not args.skip_extraction and args.audio_dir:
        audio_paths = []
        for p in pairs:
            audio = p.get("audio_path", p.get("file", ""))
            if audio and os.path.exists(audio):
                audio_paths.append(audio)

        if audio_paths:
            # Assign feat_id to each pair
            for p in pairs:
                audio = p.get("audio_path", p.get("file", ""))
                p["feat_id"] = Path(audio).stem if audio else "unknown"

            extract_whisper_features(audio_paths, args.features_dir, device)
        else:
            print("No audio files found locally. Using pre-extracted features.")
    else:
        # Assign feat_ids from filenames in pairs
        for p in pairs:
            audio = p.get("audio_path", p.get("file", p.get("segment_file", "")))
            p["feat_id"] = Path(audio).stem if audio else f"pair_{pairs.index(p)}"

    # Split train/val
    random.shuffle(pairs)
    val_size = int(len(pairs) * args.val_split)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create datasets
    train_ds = NKoASRDataset(args.features_dir, train_pairs, codebook)
    val_ds = NKoASRDataset(args.features_dir, val_pairs, codebook)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    # Create model
    model = RetrievalASR(
        num_syllables=len(codebook["syllables"]),
        embed_dim=512,
        whisper_dim=1280,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # Train
    best_loss = train(model, train_loader, val_loader, args.epochs, args.lr, device, args.save_dir)
    print(f"\nDone. Best val loss: {best_loss:.4f}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
