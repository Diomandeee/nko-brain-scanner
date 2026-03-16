#!/usr/bin/env python3
"""
Concurrent Training — Train while features keep arriving
==========================================================
Runs alongside stream_with_features.py on the same GPU.
Watches the features/ directory for new .pt files and trains
incrementally. Each epoch uses whatever features exist at that moment.

Usage (on Vast.ai, while stream_with_features.py is running):
    python3 concurrent_train.py --features-dir /workspace/features \
                                 --pairs /workspace/results/feature_pairs_djoko.jsonl \
                                 --codebook /workspace/syllable_codebook.json
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


class StreamingDataset(Dataset):
    """Dataset that re-scans features dir each epoch."""

    def __init__(self, features_dir, pairs_file, codebook, max_audio_len=1500, max_text_len=100):
        self.features_dir = Path(features_dir)
        self.pairs_file = pairs_file
        self.codebook = codebook
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # Build syllable index
        self.char_to_idx = {}
        for i, s in enumerate(codebook["syllables"]):
            self.char_to_idx[s["nko"]] = i
        self.num_syllables = len(codebook["syllables"])

        # Will be refreshed each epoch
        self.pairs = []
        self.refresh()

    def refresh(self):
        """Re-read pairs file and match with available features."""
        # Read all pairs
        all_pairs = []
        try:
            with open(self.pairs_file) as f:
                for line in f:
                    if line.strip():
                        all_pairs.append(json.loads(line))
        except Exception:
            pass

        # Match with available features
        available_features = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in all_pairs if p.get("feat_id", "") in available_features]
        return len(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feat_path = self.features_dir / f"{p['feat_id']}.pt"

        # Load feature
        try:
            features = torch.load(feat_path, weights_only=True)
        except Exception:
            features = torch.zeros(self.max_audio_len, 1280)

        # Truncate/pad audio
        if features.shape[0] > self.max_audio_len:
            features = features[:self.max_audio_len]
        audio_len = features.shape[0]
        padded = torch.zeros(self.max_audio_len, 1280)
        padded[:audio_len] = features

        # Encode N'Ko text
        nko = p.get("nko", "")
        indices = [self.char_to_idx[c] for c in nko if c in self.char_to_idx]
        if not indices:
            indices = [0]
        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])

        return padded, audio_len, text, text_len


class RetrievalASR(nn.Module):
    def __init__(self, num_syllables, embed_dim=512, input_dim=1280):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 768), nn.GELU(), nn.LayerNorm(768),
            nn.Linear(768, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.ctc_head = nn.Linear(embed_dim, num_syllables + 1)
        self.num_syllables = num_syllables

    def forward(self, x):
        projected = self.proj(x)
        return self.ctc_head(projected)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    steps = 0
    for features, audio_lens, targets, target_lens in loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                          blank=model.num_syllables, zero_infinity=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for features, audio_lens, targets, target_lens in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                              blank=model.num_syllables, zero_infinity=True)
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="/workspace/features")
    parser.add_argument("--pairs", default="/workspace/results/feature_pairs_djoko.jsonl")
    parser.add_argument("--codebook", default="/workspace/syllable_codebook.json")
    parser.add_argument("--save-dir", default="/workspace/model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-features", type=int, default=100,
                        help="Wait until this many features exist before starting")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.codebook) as f:
        codebook = json.load(f)
    num_syllables = len(codebook["syllables"])
    print(f"Codebook: {num_syllables} syllables")

    # Wait for minimum features
    while True:
        n = len(list(Path(args.features_dir).glob("*.pt")))
        if n >= args.min_features:
            print(f"Found {n} features. Starting training.")
            break
        print(f"Waiting for features: {n}/{args.min_features}. Checking in 30s...")
        time.sleep(30)

    # Create model
    model = RetrievalASR(num_syllables).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.save_dir).mkdir(exist_ok=True)

    dataset = StreamingDataset(args.features_dir, args.pairs, codebook)
    best_val = float("inf")

    for epoch in range(args.epochs):
        # Refresh dataset (picks up new features)
        n_pairs = dataset.refresh()
        if n_pairs < 10:
            print(f"Epoch {epoch+1}: Only {n_pairs} pairs, waiting 60s...")
            time.sleep(60)
            continue

        # Split 90/10
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        val_size = max(int(len(indices) * 0.1), 1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, num_workers=0)

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best_retrieval_asr.pt")

        print(f"Epoch {epoch+1}/{args.epochs} | pairs: {n_pairs} | "
              f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"best: {best_val:.4f} | {elapsed:.0f}s")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
                "n_pairs": n_pairs,
            }, f"{args.save_dir}/checkpoint_e{epoch+1}.pt")

    torch.save(model.state_dict(), f"{args.save_dir}/final_retrieval_asr.pt")
    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
