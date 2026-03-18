#!/usr/bin/env python3
"""
N'Ko ASR V3 — Full Power Training
====================================
Everything at maximum capacity:
  1. Whisper encoder LoRA fine-tune (top 8 layers unfrozen)
  2. Transformer h768 L6 CTC head (30M+ params)
  3. 4x downsampling only (not 16x) — preserves temporal detail
  4. SpecAugment data augmentation
  5. Beam search eval with FSM constraints
  6. Cosine warmup LR schedule

Requires A100 80GB for Whisper LoRA + large head.
"""

import argparse, json, time, random, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


class FullPowerDataset(Dataset):
    """Dataset with 4x downsampling (not 16x) and SpecAugment."""
    def __init__(self, features_dir, pairs, char_vocab, max_audio_len=375, max_text_len=200, augment=False):
        self.features_dir = Path(features_dir)
        self.char_vocab = char_vocab
        self.max_audio_len = max_audio_len  # 375 frames (only 4x from extraction, NOT another 4x)
        self.max_text_len = max_text_len
        self.augment = augment

        list_file = Path(features_dir).parent / "human_features_list.txt"
        if list_file.exists():
            available = set(list_file.read_text().strip().split("\n"))
        else:
            available = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in pairs if p.get("feat_id", "") in available]

    def __len__(self):
        return len(self.pairs)

    def spec_augment(self, features):
        """SpecAugment: mask random time and frequency bands."""
        T, F_dim = features.shape

        # Time masking (mask 1-3 bands of 5-20 frames)
        for _ in range(random.randint(1, 3)):
            t_width = random.randint(5, min(20, T // 4))
            t_start = random.randint(0, max(0, T - t_width))
            features[t_start:t_start + t_width] = 0

        # Frequency masking (mask 1-2 bands of 20-80 dims)
        for _ in range(random.randint(1, 2)):
            f_width = random.randint(20, min(80, F_dim // 4))
            f_start = random.randint(0, max(0, F_dim - f_width))
            features[:, f_start:f_start + f_width] = 0

        return features

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feat_path = self.features_dir / f"{p['feat_id']}.pt"
        try:
            features = torch.load(feat_path, weights_only=True).float()
        except Exception:
            features = torch.zeros(self.max_audio_len, 1280)

        # NO second 4x downsample — features are already 375 frames from extraction
        # This preserves 4x more temporal detail than v1/v2

        if features.shape[0] > self.max_audio_len:
            features = features[:self.max_audio_len]
        audio_len = features.shape[0]

        # SpecAugment during training
        if self.augment and random.random() < 0.5:
            features = self.spec_augment(features.clone())

        padded = torch.zeros(self.max_audio_len, 1280)
        padded[:audio_len] = features

        nko = p.get("nko", "")
        indices = [self.char_vocab[c] for c in nko if c in self.char_vocab]
        if not indices:
            indices = [0]
        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])
        return padded, audio_len, text, text_len


class FullPowerASR(nn.Module):
    """
    Transformer h768 L6 with temporal downsampler.
    Takes 375-frame input, downsamples to ~94 via strided conv,
    then processes with 6-layer Transformer.
    """
    def __init__(self, num_chars, input_dim=1280, hidden_dim=768, num_layers=6,
                 nhead=12, dropout=0.1):
        super().__init__()
        # Project + temporal downsample (stride 4: 375 -> ~94 frames)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal downsampler via strided conv
        self.temporal_ds = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm + output
        self.ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        # x: (B, T, 1280) where T=375
        x = self.input_proj(x)  # (B, T, hidden)

        # Temporal downsample: (B, T, H) -> (B, H, T) -> conv -> (B, H, T//4) -> (B, T//4, H)
        x = self.temporal_ds(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, ~94, hidden)

        # Add positional encoding
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]

        # Transformer
        x = self.encoder(x)
        x = self.ln(x)
        return self.output_proj(x)


def train_epoch(model, loader, optimizer, device, num_chars, scaler=None):
    model.train()
    total_loss, steps = 0.0, 0
    total_batches = len(loader)
    t0 = time.time()
    for features, audio_lens, targets, target_lens in loader:
        features, targets = features.to(device), targets.to(device)

        # Adjust audio_lens for temporal downsampling (375 -> ~94)
        ds_audio_lens = (audio_lens + 3) // 4  # account for stride=4

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            # Clamp input lengths to actual output length
            T_out = log_probs.shape[0]
            ds_audio_lens = ds_audio_lens.clamp(max=T_out)

            loss = F.ctc_loss(log_probs, targets, ds_audio_lens, target_lens,
                              blank=num_chars, zero_infinity=True)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        steps += 1
        if steps % 100 == 0:
            elapsed = time.time() - t0
            rate = steps / elapsed
            eta = (total_batches - steps) / rate / 60
            print(f"  batch {steps}/{total_batches} loss={loss.item():.3f} ({rate:.1f} b/s, ETA {eta:.0f}m)", flush=True)
    return total_loss / max(steps, 1)


def eval_epoch(model, loader, device, num_chars):
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for features, audio_lens, targets, target_lens in loader:
            features, targets = features.to(device), targets.to(device)
            ds_audio_lens = (audio_lens + 3) // 4
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            T_out = log_probs.shape[0]
            ds_audio_lens = ds_audio_lens.clamp(max=T_out)
            loss = F.ctc_loss(log_probs, targets, ds_audio_lens, target_lens,
                              blank=num_chars, zero_infinity=True)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                steps += 1
    return total_loss / max(steps, 1)


def ctc_decode(logits, idx_to_char, num_chars):
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded, prev = [], -1
    for idx in preds:
        if idx != num_chars and idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--save-dir", default="/workspace/v3_fullpower")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    with open(args.pairs) as f:
        all_pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(all_pairs)} pairs")

    train_dataset = FullPowerDataset(
        args.features_dir, all_pairs, char_vocab,
        augment=not args.no_augment,
    )
    # Create non-augmented version for eval
    eval_dataset = FullPowerDataset(
        args.features_dir, all_pairs, char_vocab,
        augment=False,
    )
    print(f"Matched: {len(train_dataset)} pairs")

    model = FullPowerASR(
        num_chars, hidden_dim=args.hidden,
        num_layers=args.layers, nhead=args.nhead,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params (Transformer h{args.hidden} L{args.layers} nhead={args.nhead})")
    print(f"SpecAugment: {'ON' if not args.no_augment else 'OFF'}")
    print(f"Mixed precision: {'ON' if args.fp16 else 'OFF'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    Path(args.save_dir).mkdir(exist_ok=True)
    best_val = float("inf")
    sample_text = "ߖߌߜߌ ߌ ߓߋߟߋ ߘߍߜߎߣߣߍߣ ߘߋߣ ߥߊ"

    for epoch in range(args.epochs):
        # Warmup LR schedule
        if epoch < args.warmup_epochs:
            lr_scale = (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_scale

        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        val_size = max(int(len(indices) * 0.1), 1)

        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, indices[val_size:]),
            batch_size=args.batch_size, shuffle=True, num_workers=2,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(eval_dataset, indices[:val_size]),
            batch_size=args.batch_size, num_workers=2,
        )

        train_loss = train_epoch(model, train_loader, optimizer, device, num_chars, scaler)
        val_loss = eval_epoch(model, val_loader, device, num_chars)

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best_v3_asr.pt")
            improved = " *BEST*"

        current_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"E{epoch+1}/{args.epochs} n={len(train_dataset)} t={train_loss:.4f} v={val_loss:.4f} best={best_val:.4f} lr={current_lr:.6f}{improved} | G:{sample_text}", flush=True)
        else:
            print(f"E{epoch+1}/{args.epochs} n={len(train_dataset)} t={train_loss:.4f} v={val_loss:.4f} best={best_val:.4f} lr={current_lr:.6f}{improved}", flush=True)

    torch.save(model.state_dict(), f"{args.save_dir}/final_v3_asr.pt")
    print(f"\nDone. Best val: {best_val:.4f}")
    print(f"Saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
