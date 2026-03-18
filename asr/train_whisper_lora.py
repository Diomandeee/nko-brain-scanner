#!/usr/bin/env python3
"""
Whisper LoRA Fine-Tune + Transformer CTC Head
===============================================
Instead of freezing Whisper, fine-tune the top encoder layers with LoRA
so it learns Bambara-specific acoustic patterns. Then train the
Transformer CTC head on the adapted features.

Architecture:
  Whisper large-v3 encoder (LoRA on top 8 layers)
  → Transformer h768 L6 CTC head (same as V3)
  → 65 N'Ko characters

This is end-to-end: audio waveform → N'Ko text.
No pre-extracted features needed.

Requires A100 40GB+ (Whisper encoder + head + gradients).
"""

import argparse, json, time, random, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import whisper


def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ── LoRA Adapter ──────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a linear layer."""
    def __init__(self, original: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return original_out + lora_out


def apply_lora_to_whisper(model, num_layers=8, rank=16, alpha=32):
    """Apply LoRA to the top N encoder layers of Whisper."""
    total_layers = len(model.encoder.blocks)
    start_layer = total_layers - num_layers
    lora_params = []

    for i in range(start_layer, total_layers):
        block = model.encoder.blocks[i]

        # Apply LoRA to attention Q, K, V projections
        for name in ['query', 'key', 'value', 'out']:
            attr = getattr(block.attn, name, None)
            if attr is not None and isinstance(attr, nn.Linear):
                lora = LoRALinear(attr, rank=rank, alpha=alpha)
                setattr(block.attn, name, lora)
                lora_params.extend([lora.lora_A, lora.lora_B])

        # Apply LoRA to MLP layers
        for name in ['0', '2']:
            if hasattr(block.mlp, name):
                layer = getattr(block.mlp, name)
                if isinstance(layer, nn.Linear):
                    lora = LoRALinear(layer, rank=rank, alpha=alpha)
                    setattr(block.mlp, name, lora)
                    lora_params.extend([lora.lora_A, lora.lora_B])

    # Freeze everything except LoRA params
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Whisper LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    print(f"  Adapted layers: {start_layer}-{total_layers-1} ({num_layers} layers)")

    return lora_params


# ── Dataset ──────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """Loads raw audio from HF dataset, returns mel spectrograms."""
    def __init__(self, audio_data, pairs, char_vocab, max_text_len=200):
        self.audio_data = audio_data
        self.char_vocab = char_vocab
        self.max_text_len = max_text_len
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        sample_idx = p["_audio_idx"]

        audio = self.audio_data[sample_idx]["audio"]
        audio_array = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            ratio = 16000 / sr
            audio_array = F.interpolate(
                audio_array.unsqueeze(0).unsqueeze(0),
                size=int(len(audio_array) * ratio),
                mode="linear", align_corners=False
            ).squeeze()

        # Pad/trim to 30s and compute mel
        audio_padded = whisper.pad_or_trim(audio_array)
        mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128)  # (128, 3000)

        # N'Ko text encoding
        nko = p.get("nko", "")
        indices = [self.char_vocab[c] for c in nko if c in self.char_vocab]
        if not indices:
            indices = [0]
        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])

        return mel, text, text_len


# ── CTC Head ─────────────────────────────────────────────────────────

class TransformerCTCHead(nn.Module):
    """Same as V3 but takes Whisper encoder output directly."""
    def __init__(self, num_chars, input_dim=1280, hidden_dim=768, num_layers=6, nhead=12, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal_ds = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2), nn.GELU())
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.temporal_ds(x.permute(0, 2, 1)).permute(0, 2, 1)
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        x = self.encoder(x)
        x = self.ln(x)
        return self.output_proj(x)


# ── Training ─────────────────────────────────────────────────────────

def train_epoch(whisper_model, ctc_head, loader, optimizer, device, num_chars, scaler):
    whisper_model.train()
    ctc_head.train()
    total_loss, steps = 0.0, 0
    total_batches = len(loader)
    t0 = time.time()

    for mel, targets, target_lens in loader:
        mel = mel.to(device)
        targets = targets.to(device)

        with torch.amp.autocast('cuda'):
            # Forward through Whisper encoder (with LoRA)
            encoder_out = whisper_model.encoder(mel)  # (B, 1500, 1280)

            # Forward through CTC head
            logits = ctc_head(encoder_out)  # (B, ~375, num_chars+1)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            T_out = log_probs.shape[0]
            audio_lens = torch.full((mel.shape[0],), T_out, dtype=torch.long)

            loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                              blank=num_chars, zero_infinity=True)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(whisper_model.parameters()) + list(ctc_head.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
        if steps % 50 == 0:
            elapsed = time.time() - t0
            rate = steps / elapsed
            eta = (total_batches - steps) / rate / 60
            print(f"  batch {steps}/{total_batches} loss={loss.item():.3f} ({rate:.1f} b/s, ETA {eta:.0f}m)", flush=True)

    return total_loss / max(steps, 1)


def eval_epoch(whisper_model, ctc_head, loader, device, num_chars):
    whisper_model.eval()
    ctc_head.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for mel, targets, target_lens in loader:
            mel, targets = mel.to(device), targets.to(device)
            encoder_out = whisper_model.encoder(mel)
            logits = ctc_head(encoder_out)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            T_out = log_probs.shape[0]
            audio_lens = torch.full((mel.shape[0],), T_out, dtype=torch.long)
            loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                              blank=num_chars, zero_infinity=True)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                steps += 1
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="/workspace/whisper_lora")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-whisper", type=float, default=1e-5, help="LR for Whisper LoRA")
    parser.add_argument("--lr-head", type=float, default=3e-4, help="LR for CTC head")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--v3-checkpoint", default=None, help="Initialize CTC head from V3")
    args = parser.parse_args()

    device = "cuda"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    # Load Whisper
    print("Loading Whisper large-v3...")
    whisper_model = whisper.load_model("large-v3", device=device)
    print("Whisper loaded.")

    # Apply LoRA
    print(f"Applying LoRA (rank={args.lora_rank}, layers={args.lora_layers})...")
    lora_params = apply_lora_to_whisper(
        whisper_model, num_layers=args.lora_layers,
        rank=args.lora_rank, alpha=args.lora_rank * 2)

    # CTC Head
    ctc_head = TransformerCTCHead(num_chars).to(device)
    head_params = sum(p.numel() for p in ctc_head.parameters())
    print(f"CTC Head: {head_params:,} params")

    # Optionally load V3 weights into head
    if args.v3_checkpoint:
        print(f"Loading V3 checkpoint: {args.v3_checkpoint}")
        state = torch.load(args.v3_checkpoint, weights_only=True, map_location=device)
        ctc_head.load_state_dict(state, strict=False)
        print("V3 weights loaded (transfer learning)")

    # Load dataset
    print("Loading bam-asr-early...")
    from datasets import load_dataset
    ds = load_dataset("RobotsMali/bam-asr-early", cache_dir="/workspace/hf_data")
    print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    # Load pairs with N'Ko labels
    pairs_path = Path("/workspace/bam_train_nko_fixed.jsonl")
    if pairs_path.exists():
        with open(pairs_path) as f:
            pairs = [json.loads(l) for l in f if l.strip()]
    else:
        print("ERROR: Need bam_train_nko_fixed.jsonl")
        return

    # Match pairs to audio indices
    for p in pairs:
        p["_audio_idx"] = p["id"]

    train_dataset = AudioDataset(ds["train"], pairs, char_vocab)
    print(f"Dataset: {len(train_dataset)} samples")

    # Optimizer with different LR for Whisper LoRA vs head
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_whisper},
        {"params": ctc_head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)

    scaler = torch.amp.GradScaler('cuda')
    Path(args.save_dir).mkdir(exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        val_size = max(int(len(indices) * 0.1), 1)

        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, indices[val_size:]),
            batch_size=args.batch_size, shuffle=True, num_workers=2,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, indices[:val_size]),
            batch_size=args.batch_size, num_workers=2,
        )

        train_loss = train_epoch(whisper_model, ctc_head, train_loader, optimizer, device, num_chars, scaler)
        val_loss = eval_epoch(whisper_model, ctc_head, val_loader, device, num_chars)

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            # Save LoRA weights + head
            torch.save({
                "lora_state": {n: p.data for n, p in whisper_model.named_parameters() if p.requires_grad},
                "head_state": ctc_head.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, f"{args.save_dir}/best_whisper_lora.pt")
            improved = " *BEST*"

        print(f"E{epoch+1}/{args.epochs} t={train_loss:.4f} v={val_loss:.4f} best={best_val:.4f}{improved}", flush=True)

    print(f"\nDone. Best val: {best_val:.4f}")
    print(f"Saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
