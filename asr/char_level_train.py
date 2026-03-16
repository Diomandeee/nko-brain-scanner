#!/usr/bin/env python3
"""
Character-Level CTC Training — Fix for the codebook targeting problem
======================================================================
Instead of 3,024 syllable classes (99.2% never seen), use ~30 base
N'Ko characters as CTC targets. The FSM assembles valid syllables
post-decoding.

This is the #1 critical fix identified by research analysis.
"""

import argparse
import json
import os
import time
import glob
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Build character vocabulary from N'Ko Unicode range
def build_nko_char_vocab():
    """Build vocab of individual N'Ko characters (not syllables)."""
    chars = {}
    idx = 0

    # N'Ko consonants (U+07CA - U+07E6)
    # N'Ko vowels, digits, tone marks — all individual characters
    for cp in range(0x07C0, 0x07FF + 1):
        c = chr(cp)
        if c not in chars:
            chars[c] = idx
            idx += 1

    # Add space
    chars[" "] = idx
    idx += 1

    return chars, idx  # ~64 chars + space


class CharDataset(Dataset):
    """Dataset with character-level targets instead of syllable indices."""

    def __init__(self, features_dir, pairs, char_vocab, max_audio_len=1500, max_text_len=200):
        self.features_dir = Path(features_dir)
        self.char_vocab = char_vocab
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # Match pairs with available features
        available = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in pairs if p.get("feat_id", "") in available]

    def refresh(self):
        available = set(p.stem for p in self.features_dir.glob("*.pt"))
        all_pairs = self.pairs  # keep existing
        self.pairs = [p for p in all_pairs if p.get("feat_id", "") in available]
        return len(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feat_path = self.features_dir / f"{p['feat_id']}.pt"

        try:
            features = torch.load(feat_path, weights_only=True)
        except Exception:
            features = torch.zeros(self.max_audio_len, 1280)

        # Downsample features by 4x (1500 → 375 frames)
        # This helps CTC when input >> output length
        if features.shape[0] > 4:
            features = features[::4]

        if features.shape[0] > self.max_audio_len // 4:
            features = features[:self.max_audio_len // 4]
        audio_len = features.shape[0]

        padded = torch.zeros(self.max_audio_len // 4, 1280)
        padded[:audio_len] = features

        # Character-level encoding
        nko = p.get("nko", "")
        indices = []
        for c in nko:
            if c in self.char_vocab:
                indices.append(self.char_vocab[c])
        if not indices:
            indices = [0]

        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])

        return padded, audio_len, text, text_len


class CharASR(nn.Module):
    """Character-level ASR with temporal modeling."""

    def __init__(self, num_chars, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.num_chars = num_chars

        # Project Whisper features
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Temporal modeling (BiLSTM — simpler than Conformer, works well for CTC)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=3, batch_first=True,
            bidirectional=True, dropout=0.1,
        )

        # CTC output
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)  # +1 blank

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)


def train_epoch(model, loader, optimizer, device, num_chars):
    model.train()
    total_loss = 0
    steps = 0
    for features, audio_lens, targets, target_lens in loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        loss = F.ctc_loss(
            log_probs, targets, audio_lens, target_lens,
            blank=num_chars, zero_infinity=True,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def eval_epoch(model, loader, device, num_chars):
    model.eval()
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for features, audio_lens, targets, target_lens in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = F.ctc_loss(
                log_probs, targets, audio_lens, target_lens,
                blank=num_chars, zero_infinity=True,
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                steps += 1
    return total_loss / max(steps, 1)


def decode(logits, idx_to_char, num_chars):
    """CTC greedy decode → string."""
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    prev = -1
    for idx in preds:
        if idx != num_chars and idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="/workspace/features")
    parser.add_argument("--pairs", default="/workspace/results/feature_pairs_djoko.jsonl")
    parser.add_argument("--save-dir", default="/workspace/char_model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-features", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build character vocab
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}
    print(f"Character vocab: {num_chars} chars (vs 3,024 syllables)")

    # Wait for features
    while True:
        n = len(list(Path(args.features_dir).glob("*.pt")))
        if n >= args.min_features:
            break
        print(f"Waiting: {n}/{args.min_features} features...")
        time.sleep(30)

    # Load pairs
    with open(args.pairs) as f:
        all_pairs = [json.loads(l) for l in f if l.strip()]

    # Also try loading babamamadidiane pairs
    baba_path = args.pairs.replace("djoko", "babamamadidiane")
    if os.path.exists(baba_path):
        with open(baba_path) as f:
            all_pairs.extend([json.loads(l) for l in f if l.strip()])
        print(f"Added babamamadidiane pairs. Total: {len(all_pairs)}")

    # Create dataset
    dataset = CharDataset(args.features_dir, all_pairs, char_vocab)
    print(f"Matched pairs: {len(dataset)}")

    # Model
    model = CharASR(num_chars).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params ({num_chars} char classes, BiLSTM encoder)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    Path(args.save_dir).mkdir(exist_ok=True)

    best_val = float("inf")

    for epoch in range(args.epochs):
        # Refresh dataset
        n = dataset.refresh()
        if n < 10:
            time.sleep(30)
            continue

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        val_size = max(int(len(indices) * 0.1), 1)

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, indices[val_size:]),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, indices[:val_size]),
            batch_size=args.batch_size, num_workers=0,
        )

        train_loss = train_epoch(model, train_loader, optimizer, device, num_chars)
        val_loss = eval_epoch(model, val_loader, device, num_chars)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best_char_asr.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Decode a sample
            sample = dataset[0]
            feat = sample[0].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(feat)
            pred = decode(logits[0], idx_to_char, num_chars)
            gold = all_pairs[0].get("nko", "")[:50]
            print(f"E{epoch+1}/{args.epochs} n={n} t={train_loss:.4f} v={val_loss:.4f} "
                  f"best={best_val:.4f} | P:{pred[:40]} G:{gold[:40]}")
        else:
            print(f"E{epoch+1}/{args.epochs} n={n} t={train_loss:.4f} v={val_loss:.4f} "
                  f"best={best_val:.4f}")

    torch.save(model.state_dict(), f"{args.save_dir}/final_char_asr.pt")
    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
