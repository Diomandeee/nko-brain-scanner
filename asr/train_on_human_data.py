#!/usr/bin/env python3
"""
Train on Human-Labeled bam-asr-early Dataset
==============================================
Loads bam-asr-early from HuggingFace cache, extracts Whisper features
on-the-fly, trains char-level CTC with N'Ko targets.

Usage (on Vast.ai with bam-asr-early already downloaded):
    python3 train_on_human_data.py --epochs 100 --batch-size 4
"""

import argparse
import json
import os
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import whisper


# N'Ko character vocabulary
def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# Latin → N'Ko bridge (inline, no external deps)
LATIN_TO_NKO = {
    "a": "\u07ca", "b": "\u07d3", "c": "\u07d7", "d": "\u07d8", "e": "\u07cd",
    "f": "\u07dd", "g": "\u07dc", "h": "\u07e4", "i": "\u07cc", "j": "\u07d6",
    "k": "\u07de", "l": "\u07df", "m": "\u07e1", "n": "\u07e3", "o": "\u07cf",
    "p": "\u07d4", "r": "\u07d9", "s": "\u07db", "t": "\u07d5", "u": "\u07ce",
    "w": "\u07e5", "y": "\u07e6", "z": "\u07d6", "v": "\u07dd",
    "\u0254": "\u07cf", "\u025b": "\u07d0", "\u0272": "\u07e2", "\u014b": "\u07d2",
}

def bridge_to_nko(text):
    return "".join(LATIN_TO_NKO.get(c, " " if c == " " else "") for c in text.lower())


class HumanLabeledDataset(Dataset):
    """Dataset that loads audio from HF dataset and extracts Whisper features on-the-fly."""

    def __init__(self, hf_dataset, whisper_model, char_vocab, device, max_text_len=200):
        self.dataset = hf_dataset
        self.whisper_model = whisper_model
        self.char_vocab = char_vocab
        self.device = device
        self.max_text_len = max_text_len
        # Pre-compute N'Ko targets
        self.nko_targets = []
        for i in range(len(hf_dataset)):
            bam = hf_dataset[i]["bam"]
            nko = bridge_to_nko(bam)
            indices = [char_vocab[c] for c in nko if c in char_vocab]
            self.nko_targets.append(indices if indices else [0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            import torch.nn.functional as F_
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            audio_array = F_.interpolate(
                audio_tensor.unsqueeze(0).unsqueeze(0),
                size=int(len(audio_array) * 16000 / sr),
                mode="linear", align_corners=False
            ).squeeze().numpy()

        # Extract Whisper features
        audio = whisper.pad_or_trim(torch.tensor(audio_array, dtype=torch.float32))
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.device)
        with torch.no_grad():
            features = self.whisper_model.encoder(mel.unsqueeze(0)).squeeze(0).cpu()

        # Downsample 4x
        features = features[::4]
        audio_len = features.shape[0]

        # Pad to fixed length
        max_len = 375  # 1500 / 4
        padded = torch.zeros(max_len, 1280)
        actual_len = min(audio_len, max_len)
        padded[:actual_len] = features[:actual_len]

        # N'Ko targets
        indices = self.nko_targets[idx][:self.max_text_len]
        text_len = len(indices)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices)

        return padded, actual_len, text, text_len


class CharASR(nn.Module):
    def __init__(self, num_chars, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.num_chars = num_chars
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=3,
                               batch_first=True, bidirectional=True, dropout=0.1)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)


def decode(logits, idx_to_char, num_chars):
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="/workspace/human_model")
    parser.add_argument("--cache-dir", default="/workspace/hf_data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Whisper for feature extraction
    print("Loading Whisper large-v3...")
    whisper_model = whisper.load_model("large-v3", device=device)
    whisper_model.eval()
    print("Whisper loaded.")

    # Load dataset
    from datasets import load_dataset
    print("Loading bam-asr-early...")
    ds = load_dataset("RobotsMali/bam-asr-early", cache_dir=args.cache_dir)
    print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    # Build char vocab
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}
    print(f"Char vocab: {num_chars}")

    # Create datasets
    print("Building datasets (pre-computing N'Ko targets)...")
    train_ds = HumanLabeledDataset(ds["train"], whisper_model, char_vocab, device)
    test_ds = HumanLabeledDataset(ds["test"], whisper_model, char_vocab, device)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    # Model
    model = CharASR(num_chars).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    Path(args.save_dir).mkdir(exist_ok=True)

    best_val = float("inf")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_steps = 0
        for features, audio_lens, targets, target_lens in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                              blank=num_chars, zero_infinity=True)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        # Eval
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for features, audio_lens, targets, target_lens in test_loader:
                features = features.to(device)
                targets = targets.to(device)
                logits = model(features)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                                  blank=num_chars, zero_infinity=True)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_steps += 1

        avg_train = train_loss / max(train_steps, 1)
        avg_val = val_loss / max(val_steps, 1)
        scheduler.step()

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"{args.save_dir}/best_human_asr.pt")

        # Sample decode every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            sample = train_ds[0]
            with torch.no_grad():
                logits = model(sample[0].unsqueeze(0).to(device))
            pred = decode(logits[0], idx_to_char, num_chars)
            gold_nko = bridge_to_nko(ds["train"][0]["bam"])
            print(f"E{epoch+1}/{args.epochs} t={avg_train:.4f} v={avg_val:.4f} "
                  f"best={best_val:.4f} | P:{pred[:40]} G:{gold_nko[:40]}", flush=True)
        else:
            print(f"E{epoch+1}/{args.epochs} t={avg_train:.4f} v={avg_val:.4f} "
                  f"best={best_val:.4f}", flush=True)

    torch.save(model.state_dict(), f"{args.save_dir}/final_human_asr.pt")
    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
