#!/usr/bin/env python3
"""
N'Ko CTC Training: Character-level ASR with N'Ko output vocabulary.

This is the N'Ko arm of Experiment B. Trains a CTC decoder on top of frozen
Whisper encoder features, outputting individual N'Ko characters (65 classes + blank).

Architecture matches the V3 CharASR from asr/train_v3_fullpower.py:
  - Input: Whisper Large V3 encoder features (1280-dim)
  - 4x downsampling (conv stride 2, twice)
  - Transformer: d_model=768, 6 layers, 12 heads
  - CTC output: 65 N'Ko character classes + blank

Usage:
    python3 train_nko_ctc.py \
        --features-dir /path/to/whisper_features/ \
        --pairs /path/to/nko_pairs.jsonl \
        --epochs 50 \
        --batch-size 16 \
        --checkpoint-dir checkpoints/nko/
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------- N'Ko Character Vocabulary ----------

def build_nko_char_vocab():
    """Build vocabulary of individual N'Ko characters.

    Covers the full N'Ko Unicode block U+07C0-U+07FF (64 code points)
    plus space. CTC blank is index 0 by convention.

    Returns:
        char_to_idx: dict mapping character -> index
        num_classes: total number of classes including blank
    """
    chars = {"<blank>": 0}
    idx = 1
    for cp in range(0x07C0, 0x07FF + 1):
        c = chr(cp)
        chars[c] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ---------- Dataset ----------

class NkoCTCDataset(Dataset):
    """Dataset pairing Whisper features with N'Ko character sequences."""

    def __init__(self, features_dir, pairs, char_vocab, max_audio_len=375, max_text_len=200):
        self.features_dir = Path(features_dir)
        self.char_vocab = char_vocab
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # Match pairs with available feature files
        available = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in pairs if p.get("feat_id", "") in available]
        print(f"NkoCTCDataset: {len(self.pairs)} usable pairs (of {len(pairs)} total)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feat_path = self.features_dir / f"{p['feat_id']}.pt"

        try:
            features = torch.load(feat_path, weights_only=True).float()
        except Exception:
            features = torch.zeros(self.max_audio_len, 1280)

        # Truncate/pad audio
        if features.shape[0] > self.max_audio_len:
            features = features[:self.max_audio_len]
        audio_len = features.shape[0]
        if features.shape[0] < self.max_audio_len:
            pad = torch.zeros(self.max_audio_len - features.shape[0], features.shape[1])
            features = torch.cat([features, pad])

        # Encode N'Ko text as character indices
        nko_text = p.get("nko", "")
        target = []
        for ch in nko_text:
            if ch in self.char_vocab:
                target.append(self.char_vocab[ch])
            # Skip unknown characters
        target = target[:self.max_text_len]
        text_len = len(target)

        # Pad target
        while len(target) < self.max_text_len:
            target.append(0)  # blank padding

        return (
            features,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(audio_len, dtype=torch.long),
            torch.tensor(text_len, dtype=torch.long),
        )


# ---------- Model ----------

class CharASR_NKo(nn.Module):
    """CTC-based character ASR model with N'Ko output.

    Matches V3 architecture from asr/train_v3_fullpower.py.
    """

    def __init__(self, input_dim=1280, d_model=768, nhead=12, num_layers=6, num_classes=66):
        super().__init__()
        self.num_classes = num_classes

        # Downsampling: 2x conv stride
        self.downsample = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CTC output head
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, x, input_lengths=None):
        """
        Args:
            x: (batch, time, input_dim) Whisper encoder features
            input_lengths: (batch,) original lengths before padding

        Returns:
            log_probs: (time, batch, num_classes) for CTC loss
            output_lengths: (batch,) lengths after downsampling
        """
        # Conv expects (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)  # back to (batch, time, d_model)

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_enc[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # CTC output
        logits = self.output(x)
        log_probs = F.log_softmax(logits, dim=-1)

        # Output is (time, batch, classes) for CTC
        log_probs = log_probs.transpose(0, 1)

        # Compute output lengths (4x downsampled from input)
        if input_lengths is not None:
            output_lengths = (input_lengths + 3) // 4
        else:
            output_lengths = torch.full((x.shape[0],), seq_len, dtype=torch.long)

        return log_probs, output_lengths


# ---------- Training ----------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    for batch_idx, (features, targets, audio_lens, text_lens) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        audio_lens = audio_lens.to(device)
        text_lens = text_lens.to(device)

        optimizer.zero_grad()
        log_probs, output_lens = model(features, audio_lens)

        # CTC loss expects (T, N, C), targets as 1D concatenation
        # Filter out zero-length targets
        valid = text_lens > 0
        if not valid.any():
            continue

        loss = ctc_loss_fn(
            log_probs[:, valid],
            targets[valid],
            output_lens[valid],
            text_lens[valid],
        )

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train N'Ko CTC ASR")
    parser.add_argument("--features-dir", required=True, help="Directory with Whisper .pt features")
    parser.add_argument("--pairs", required=True, help="JSONL with feat_id + nko fields")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/nko/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Build vocabulary
    char_vocab, num_classes = build_nko_char_vocab()
    print(f"N'Ko vocab: {num_classes} classes (including blank)")

    # Load pairs
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    print(f"Loaded {len(pairs)} training pairs")

    # Split train/val (90/10)
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    # Datasets
    train_ds = NkoCTCDataset(args.features_dir, train_pairs, char_vocab)
    val_ds = NkoCTCDataset(args.features_dir, val_pairs, char_vocab)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    device = torch.device(args.device)
    model = CharASR_NKo(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M params, {num_classes} output classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_dl) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        with torch.no_grad():
            for features, targets, audio_lens, text_lens in val_dl:
                features = features.to(device)
                targets = targets.to(device)
                audio_lens = audio_lens.to(device)
                text_lens = text_lens.to(device)

                log_probs, output_lens = model(features, audio_lens)
                valid = text_lens > 0
                if valid.any():
                    loss = ctc_loss_fn(log_probs[:, valid], targets[valid],
                                       output_lens[valid], text_lens[valid])
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  New best! Saved checkpoint.")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))

    # Save final
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "final.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
