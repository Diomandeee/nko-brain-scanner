#!/usr/bin/env python3
"""
Concurrent Training V2 — Fixed for vowel-collapse bug
========================================================
Fixes from V1:
  1. Syllable-level tokenization (greedy longest-match) instead of char-by-char
  2. Uses RetrievalASRv2 with temporal modeling + downsampling
  3. Focal CTC loss for class imbalance
  4. Learning rate warmup + cosine decay
  5. Proper output length computation after downsampling

Usage (on Vast.ai, while stream_with_features.py is running):
    python3 concurrent_train_v2.py --features-dir /workspace/features \
                                    --pairs /workspace/results/feature_pairs_djoko.jsonl \
                                    --codebook /workspace/syllable_codebook.json
"""

import argparse
import json
import math
import os
import sys
import time
import random
from pathlib import Path

# Ensure retrieval_asr_v2 is importable when both files live in the same directory
# (e.g. /workspace/ on Vast.ai) regardless of the working directory.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from retrieval_asr_v2 import (
    RetrievalASRv2,
    NKoSyllableTokenizer,
    FocalCTCLoss,
)


class StreamingDataset(Dataset):
    """Dataset that re-scans features dir each epoch.

    V2 fix: uses syllable-level tokenizer instead of char-by-char.
    """

    def __init__(self, features_dir, pairs_file, codebook,
                 max_audio_len=1500, max_text_len=200):
        self.features_dir = Path(features_dir)
        self.pairs_file = pairs_file
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # V2: syllable-level tokenizer (NOT char-by-char)
        self.tokenizer = NKoSyllableTokenizer(codebook)
        self.num_syllables = self.tokenizer.num_syllables

        self.pairs = []
        self.refresh()

    def refresh(self):
        """Re-read pairs file and match with available features."""
        all_pairs = []
        try:
            with open(self.pairs_file) as f:
                for line in f:
                    if line.strip():
                        all_pairs.append(json.loads(line))
        except Exception:
            pass

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

        # V2 FIX: Syllable-level tokenization
        nko = p.get("nko", "")
        indices = self.tokenizer.encode(nko)
        if not indices:
            indices = [0]
        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])

        return padded, audio_len, text, text_len


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    steps = 0
    total_target_tokens = 0
    total_blank_predicted = 0
    total_frames = 0

    for features, audio_lens, targets, target_lens in loader:
        features = features.to(device)
        targets = targets.to(device)

        output = model(features)
        ctc_logits = output["ctc_logits"]

        # Compute downsampled output lengths
        output_lens = model.compute_output_lengths(audio_lens)

        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)

        loss = criterion(log_probs, targets, output_lens, target_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

        # Track blank ratio for monitoring
        with torch.no_grad():
            preds = ctc_logits.argmax(dim=-1)
            total_blank_predicted += (preds == model.num_syllables).sum().item()
            total_frames += preds.numel()
            total_target_tokens += target_lens.sum().item()

    blank_ratio = total_blank_predicted / max(total_frames, 1)
    avg_target_len = total_target_tokens / max(len(loader.dataset), 1)

    return {
        "loss": total_loss / max(steps, 1),
        "blank_ratio": blank_ratio,
        "avg_target_len": avg_target_len,
        "lr": scheduler.get_last_lr()[0],
    }


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    steps = 0
    total_correct_syllables = 0
    total_syllables = 0

    with torch.no_grad():
        for features, audio_lens, targets, target_lens in loader:
            features = features.to(device)
            targets = targets.to(device)

            output = model(features)
            ctc_logits = output["ctc_logits"]
            output_lens = model.compute_output_lengths(audio_lens)

            log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
            loss = F.ctc_loss(log_probs, targets, output_lens, target_lens,
                              blank=model.num_syllables, zero_infinity=True)
            total_loss += loss.item()
            steps += 1

            # CTC greedy decode and compare
            decoded_batch = model.ctc_decode(ctc_logits)
            for i, decoded in enumerate(decoded_batch):
                tgt_len = target_lens[i].item()
                tgt = targets[i, :tgt_len].tolist()
                # Simple token-level accuracy (not edit distance, but useful for monitoring)
                min_len = min(len(decoded), len(tgt))
                correct = sum(1 for a, b in zip(decoded[:min_len], tgt[:min_len]) if a == b)
                total_correct_syllables += correct
                total_syllables += len(tgt)

    return {
        "loss": total_loss / max(steps, 1),
        "syllable_accuracy": total_correct_syllables / max(total_syllables, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="/workspace/features")
    parser.add_argument("--pairs", default="/workspace/results/feature_pairs_djoko.jsonl")
    parser.add_argument("--codebook", default="/workspace/syllable_codebook.json")
    parser.add_argument("--save-dir", default="/workspace/model_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-features", type=int, default=100)
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0=standard CTC, 2=focal)")
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

    # Create V2 model
    model = RetrievalASRv2(args.codebook).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model V2: {params:,} params")

    # Focal CTC loss
    criterion = FocalCTCLoss(
        blank=model.num_syllables,
        gamma=args.focal_gamma,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.save_dir).mkdir(exist_ok=True)

    dataset = StreamingDataset(args.features_dir, args.pairs, codebook)
    best_val = float("inf")

    for epoch in range(args.epochs):
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

        # Compute schedule params
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = steps_per_epoch * args.warmup_epochs
        scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)

        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        val_loss = val_metrics["loss"]
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best_retrieval_asr_v2.pt")

        print(f"Epoch {epoch+1}/{args.epochs} | pairs: {n_pairs} | "
              f"train: {train_metrics['loss']:.4f} | val: {val_loss:.4f} | "
              f"best: {best_val:.4f} | blank%: {train_metrics['blank_ratio']:.1%} | "
              f"syl_acc: {val_metrics['syllable_accuracy']:.1%} | "
              f"avg_tgt: {train_metrics['avg_target_len']:.0f} | "
              f"lr: {train_metrics['lr']:.2e} | {elapsed:.0f}s")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
                "n_pairs": n_pairs,
            }, f"{args.save_dir}/checkpoint_v2_e{epoch+1}.pt")

    torch.save(model.state_dict(), f"{args.save_dir}/final_retrieval_asr_v2.pt")
    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
