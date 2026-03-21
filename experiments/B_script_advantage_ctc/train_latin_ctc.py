#!/usr/bin/env python3
"""
Latin Bambara CTC Training: Character-level ASR with Latin output vocabulary.

This is the Latin arm of Experiment B. Identical architecture to train_nko_ctc.py
but with Latin Bambara characters as the output vocabulary (~40 classes + blank).

The key difference: Latin Bambara has digraphs (ny, ng, gb), ambiguous mappings,
and no tone marking. The CTC decoder must learn to navigate this ambiguity.

Usage:
    python3 train_latin_ctc.py \
        --features-dir /path/to/whisper_features/ \
        --pairs /path/to/latin_pairs.jsonl \
        --epochs 50 \
        --batch-size 16 \
        --checkpoint-dir checkpoints/latin/
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


# ---------- Latin Bambara Character Vocabulary ----------

def build_latin_char_vocab():
    """Build vocabulary of Latin Bambara characters.

    Standard Bambara Latin orthography:
    - 26 base Latin letters (a-z)
    - Special characters: open-e (ɛ), open-o (ɔ), eng (ŋ), palatal-n (ɲ)
    - Tone marks are NOT written in standard Latin Bambara
    - Digraphs (ny, ng, gb, ch, sh) are sequences of base characters

    CTC blank is index 0.

    Returns:
        char_to_idx: dict mapping character -> index
        num_classes: total number of classes including blank
    """
    chars = {"<blank>": 0}
    idx = 1

    # Standard Latin letters
    for c in "abcdefghijklmnopqrstuvwxyz":
        chars[c] = idx
        idx += 1

    # Bambara-specific characters
    for c in ["ɛ", "ɔ", "ŋ", "ɲ"]:
        chars[c] = idx
        idx += 1

    # Accented vowels (sometimes used in Bambara texts)
    for c in ["à", "è", "é", "ì", "ò", "ù"]:
        chars[c] = idx
        idx += 1

    # Space and punctuation
    chars[" "] = idx
    idx += 1
    chars["'"] = idx  # apostrophe (common in Bambara)
    idx += 1
    chars["-"] = idx  # hyphen
    idx += 1

    return chars, idx


# ---------- Dataset ----------

class LatinCTCDataset(Dataset):
    """Dataset pairing Whisper features with Latin Bambara character sequences."""

    def __init__(self, features_dir, pairs, char_vocab, max_audio_len=375, max_text_len=200):
        self.features_dir = Path(features_dir)
        self.char_vocab = char_vocab
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        available = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in pairs if p.get("feat_id", "") in available]
        print(f"LatinCTCDataset: {len(self.pairs)} usable pairs (of {len(pairs)} total)")

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

        # Encode Latin text as character indices
        latin_text = p.get("latin", p.get("text", "")).lower()
        target = []
        for ch in latin_text:
            if ch in self.char_vocab:
                target.append(self.char_vocab[ch])
        target = target[:self.max_text_len]
        text_len = len(target)

        while len(target) < self.max_text_len:
            target.append(0)

        return (
            features,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(audio_len, dtype=torch.long),
            torch.tensor(text_len, dtype=torch.long),
        )


# ---------- Model (identical architecture to N'Ko version) ----------

class CharASR_Latin(nn.Module):
    """CTC-based character ASR model with Latin Bambara output.

    Architecture is IDENTICAL to CharASR_NKo, only num_classes differs.
    """

    def __init__(self, input_dim=1280, d_model=768, nhead=12, num_layers=6, num_classes=40):
        super().__init__()
        self.num_classes = num_classes

        self.downsample = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, num_classes)

    def forward(self, x, input_lengths=None):
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)

        seq_len = x.shape[1]
        x = x + self.pos_enc[:, :seq_len, :]
        x = self.transformer(x)

        logits = self.output(x)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)

        if input_lengths is not None:
            output_lengths = (input_lengths + 3) // 4
        else:
            output_lengths = torch.full((x.shape[0],), seq_len, dtype=torch.long)

        return log_probs, output_lengths


# ---------- Training (identical loop to N'Ko version) ----------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
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

        valid = text_lens > 0
        if not valid.any():
            continue

        loss = ctc_loss_fn(
            log_probs[:, valid], targets[valid],
            output_lens[valid], text_lens[valid],
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
    parser = argparse.ArgumentParser(description="Train Latin Bambara CTC ASR")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--pairs", required=True, help="JSONL with feat_id + latin/text fields")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/latin/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    char_vocab, num_classes = build_latin_char_vocab()
    print(f"Latin Bambara vocab: {num_classes} classes (including blank)")

    pairs = []
    with open(args.pairs) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    print(f"Loaded {len(pairs)} training pairs")

    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = LatinCTCDataset(args.features_dir, train_pairs, char_vocab)
    val_ds = LatinCTCDataset(args.features_dir, val_pairs, char_vocab)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = CharASR_Latin(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M params, {num_classes} output classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, epoch)

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

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "final.pt"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
