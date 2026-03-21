#!/usr/bin/env python3
"""
Latin Bambara CTC Training: Character-level ASR with Latin output vocabulary.
=============================================================================
Hardened for unattended Vast.ai deployment.

This is the Latin arm of Experiment B. Identical architecture to train_nko_ctc.py
but with Latin Bambara characters as the output vocabulary.

Latin Bambara orthography challenges:
  - Digraphs: ny, ng, gb, sh, ch, dj (2 chars -> 1 phoneme)
  - Ambiguous mappings: c can be /k/ or /s/
  - No tone marking in standard Latin orthography
  - Special characters: open-e, open-o, eng, palatal-n

The vocabulary uses individual characters (NOT digraph tokens) to keep
the comparison fair. Each character is a separate class, and the CTC
decoder must learn to produce multi-character digraph sequences.

Architecture: V3 CharASR (identical to N'Ko version)
  - Input: Whisper Large V3 encoder features (1280-dim)
  - 4x downsampling (conv stride 2, twice)
  - Transformer: d_model=768, 6 layers, 12 heads
  - CTC output: ~40 Latin character classes + blank

Resilience features:
  - Resume from checkpoint (--resume)
  - Mid-epoch checkpoints every N batches (--checkpoint-interval)
  - SIGTERM/SIGINT graceful shutdown with checkpoint save
  - Optimizer + scheduler + scaler state preserved
  - Append-mode logging (safe for restarts)
  - Mixed precision (AMP) for A100 efficiency
  - Progress bars with ETA

Usage:
    python3 train_latin_ctc.py \\
        --features-dir /workspace/features/ \\
        --pairs /workspace/latin_pairs.jsonl \\
        --epochs 50 \\
        --batch-size 16 \\
        --checkpoint-dir /workspace/checkpoints/latin/ \\
        --log /workspace/experiment_b_latin.log \\
        --resume
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Globals for signal handler ────────────────────────────────────────
_SHUTDOWN_REQUESTED = False
_SAVE_CONTEXT = {}


# ── Logging ───────────────────────────────────────────────────────────

_LOG_FILE = None


def log(msg, also_print=True):
    """Log with timestamp to both stdout and logfile."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ── Latin Bambara Character Vocabulary ────────────────────────────────

def build_latin_char_vocab():
    """Build vocabulary of Latin Bambara characters.

    Standard Bambara Latin orthography uses individual characters.
    Digraphs (ny, ng, gb, ch, sh, dj) are sequences of base characters,
    NOT special tokens. This keeps the comparison fair: the CTC decoder
    must learn to output multi-character sequences for single phonemes.

    Character inventory:
      - 26 base Latin letters (a-z)
      - Bambara-specific: open-e (U+025B), open-o (U+0254),
        eng (U+014B), palatal-n (U+0272)
      - Accented vowels used in some orthographies
      - Space, apostrophe, hyphen
      - CTC blank at index 0

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
    for c in [
        "\u025B",  # open-e
        "\u0254",  # open-o
        "\u014B",  # eng (ng as single char)
        "\u0272",  # palatal-n (ny as single char)
    ]:
        chars[c] = idx
        idx += 1

    # Accented vowels (used in some Bambara texts for tone)
    for c in [
        "\u00E0",  # a-grave
        "\u00E8",  # e-grave
        "\u00E9",  # e-acute
        "\u00EC",  # i-grave
        "\u00F2",  # o-grave
        "\u00F9",  # u-grave
    ]:
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


# ── Dataset ───────────────────────────────────────────────────────────

class LatinCTCDataset(Dataset):
    """Dataset pairing Whisper features with Latin Bambara character sequences."""

    def __init__(self, features_dir, pairs, char_vocab, max_audio_len=375, max_text_len=200):
        self.features_dir = Path(features_dir)
        self.char_vocab = char_vocab
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        available = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in pairs if p.get("feat_id", "") in available]
        log(f"LatinCTCDataset: {len(self.pairs)} usable pairs (of {len(pairs)} total, "
            f"{len(available)} features on disk)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feat_path = self.features_dir / f"{p['feat_id']}.pt"

        try:
            features = torch.load(feat_path, weights_only=True).float()
        except Exception as e:
            log(f"  [warn] Failed to load {feat_path}: {e}")
            features = torch.zeros(self.max_audio_len, 1280)

        # Truncate/pad audio
        if features.shape[0] > self.max_audio_len:
            features = features[:self.max_audio_len]
        audio_len = features.shape[0]
        if features.shape[0] < self.max_audio_len:
            pad = torch.zeros(self.max_audio_len - features.shape[0], features.shape[1])
            features = torch.cat([features, pad])

        # Encode Latin text as character indices
        # Try 'latin' field first, fall back to 'text', lowercase
        latin_text = p.get("latin", p.get("text", "")).lower()
        target = []
        skipped = 0
        for ch in latin_text:
            if ch in self.char_vocab:
                target.append(self.char_vocab[ch])
            else:
                skipped += 1
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


# ── Model (identical architecture to N'Ko version) ───────────────────

class CharASR_Latin(nn.Module):
    """CTC-based character ASR model with Latin Bambara output.

    Architecture is IDENTICAL to CharASR_NKo, only num_classes differs.
    h768, L6, nhead=12, Conv1d stride-4.
    """

    def __init__(self, input_dim=1280, d_model=768, nhead=12, num_layers=6, num_classes=40):
        super().__init__()
        self.num_classes = num_classes

        # Downsampling: 4x total (stride 2 twice)
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


# ── Checkpoint ────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, batch_idx,
                    best_val, val_loss=None, reason="periodic"):
    """Save full training state for resume. Atomic write."""
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "best_val": best_val,
        "val_loss": val_loss,
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "script": "train_latin_ctc.py",
    }
    tmp_path = str(path) + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)  # atomic rename
    log(f"  [checkpoint] saved: epoch={epoch} batch={batch_idx} reason={reason} -> {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load training state. Returns (start_epoch, batch_idx, best_val) or None."""
    path = Path(path)
    if not path.exists():
        return None
    log(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    log(f"  Restored model state")

    if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            log(f"  Restored optimizer state")
        except Exception as e:
            log(f"  [warn] optimizer restore failed: {e}")

    if scheduler and "scheduler_state" in ckpt and ckpt["scheduler_state"]:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            log(f"  Restored scheduler state")
        except Exception as e:
            log(f"  [warn] scheduler restore failed: {e}")

    if scaler and "scaler_state" in ckpt and ckpt["scaler_state"]:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
            log(f"  Restored scaler state")
        except Exception as e:
            log(f"  [warn] scaler restore failed: {e}")

    epoch = ckpt.get("epoch", 0)
    batch_idx = ckpt.get("batch_idx", 0)
    best_val = ckpt.get("best_val", float("inf"))
    reason = ckpt.get("reason", "unknown")
    log(f"  Resume point: epoch={epoch}, batch={batch_idx}, best_val={best_val:.4f}, reason={reason}")
    return epoch, batch_idx, best_val


# ── Signal handler ────────────────────────────────────────────────────

def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    log(f"\n[SIGNAL] {sig_name} received. Saving checkpoint and shutting down...")
    _SHUTDOWN_REQUESTED = True

    ctx = _SAVE_CONTEXT
    if ctx.get("model"):
        save_checkpoint(
            ctx["checkpoint_path"],
            ctx["model"], ctx["optimizer"], ctx["scheduler"], ctx["scaler"],
            ctx.get("epoch", 0), ctx.get("batch_idx", 0),
            ctx.get("best_val", float("inf")),
            reason=f"signal_{sig_name}",
        )
    sys.exit(0)


# ── Training ──────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch,
                checkpoint_interval=500, skip_batches=0):
    """Train for one epoch with mid-epoch checkpointing and graceful shutdown."""
    global _SHUTDOWN_REQUESTED
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    t0 = time.time()

    for batch_idx, (features, targets, audio_lens, text_lens) in enumerate(dataloader):
        if _SHUTDOWN_REQUESTED:
            log("  [shutdown] stopping training loop")
            break

        if batch_idx < skip_batches:
            continue

        features = features.to(device)
        targets = targets.to(device)
        audio_lens = audio_lens.to(device)
        text_lens = text_lens.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            log_probs, output_lens = model(features, audio_lens)

            valid = text_lens > 0
            if not valid.any():
                continue

            loss = ctc_loss_fn(
                log_probs[:, valid],
                targets[valid],
                output_lens[valid],
                text_lens[valid],
            )

        if not torch.isfinite(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Update signal handler context
        _SAVE_CONTEXT["batch_idx"] = batch_idx + 1

        # Progress log every 50 batches
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            rate = max(num_batches, 1) / elapsed
            remaining = total_batches - batch_idx - 1
            eta_min = remaining / rate / 60 if rate > 0 else 0
            avg_loss = total_loss / max(num_batches, 1)
            lr = optimizer.param_groups[0]["lr"]
            log(f"  E{epoch} [{batch_idx+1}/{total_batches}] "
                f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                f"lr={lr:.2e} {rate:.1f}b/s ETA={eta_min:.0f}m")

        # Mid-epoch checkpoint
        if checkpoint_interval > 0 and (batch_idx + 1) % checkpoint_interval == 0:
            ctx = _SAVE_CONTEXT
            save_checkpoint(
                ctx["checkpoint_path"],
                model, optimizer, scheduler, scaler,
                ctx.get("epoch", 0), batch_idx + 1,
                ctx.get("best_val", float("inf")),
                reason="mid_epoch",
            )

    return total_loss / max(num_batches, 1)


def eval_epoch(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    with torch.no_grad():
        for features, targets, audio_lens, text_lens in dataloader:
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
                    total_loss += loss.item()
                    num_batches += 1

    return total_loss / max(num_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Latin Bambara CTC ASR (Experiment B)")
    parser.add_argument("--features-dir", required=True,
                        help="Directory with Whisper .pt features")
    parser.add_argument("--pairs", required=True,
                        help="JSONL with feat_id + latin/text fields")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear warmup steps before cosine decay")
    parser.add_argument("--checkpoint-dir", default="checkpoints/latin/")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="Save checkpoint every N batches (0=disabled)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--log", default=None,
                        help="Log file path (append mode)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (AMP)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup logging
    global _LOG_FILE
    _LOG_FILE = args.log

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log("=" * 70)
    log("Experiment B: Latin Bambara CTC Training")
    log("=" * 70)
    log(f"PID: {os.getpid()}")
    log(f"Device: {args.device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name()}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    log(f"Features: {args.features_dir}")
    log(f"Pairs: {args.pairs}")
    log(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    log(f"Warmup: {args.warmup_steps} steps")
    log(f"Checkpoint interval: {args.checkpoint_interval} batches")
    log(f"AMP: {'disabled' if args.no_amp else 'enabled'}")
    log(f"Resume: {args.resume}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Build vocabulary
    char_vocab, num_classes = build_latin_char_vocab()
    log(f"Latin Bambara vocab: {num_classes} classes")
    log(f"  Base letters: a-z (26)")
    log(f"  Bambara special: open-e, open-o, eng, palatal-n (4)")
    log(f"  Accented vowels: 6")
    log(f"  Punctuation: space, apostrophe, hyphen (3)")
    log(f"  CTC blank: 1")
    log(f"  NOTE: Digraphs (ny, ng, gb, sh, ch, dj) are character SEQUENCES, not special tokens")

    # Load pairs
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log(f"Loaded {len(pairs)} training pairs from {args.pairs}")

    # Verify some pairs have Latin text
    latin_count = sum(1 for p in pairs if (p.get("latin", "") or p.get("text", "")).strip())
    log(f"  Pairs with Latin text: {latin_count}/{len(pairs)}")
    if latin_count == 0:
        log("[ERROR] No pairs have Latin text. Check 'latin' or 'text' field in pairs JSONL.")
        sys.exit(1)

    # Split train/val (90/10) with fixed seed
    random.seed(args.seed)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    log(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")

    # Datasets
    train_ds = LatinCTCDataset(args.features_dir, train_pairs, char_vocab)
    val_ds = LatinCTCDataset(args.features_dir, val_pairs, char_vocab)

    if len(train_ds) == 0:
        log("[ERROR] No usable training pairs. Check features-dir and pairs file.")
        sys.exit(1)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    device = torch.device(args.device)
    model = CharASR_Latin(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model: {param_count / 1e6:.1f}M params, {num_classes} output classes")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler: linear warmup + cosine decay
    total_steps = len(train_dl) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Checkpoint paths
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    best_val_loss = float("inf")
    start_epoch = 0
    skip_batches = 0

    # Populate signal handler context
    _SAVE_CONTEXT.update({
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "checkpoint_path": checkpoint_path,
        "best_val": best_val_loss,
        "epoch": 0,
        "batch_idx": 0,
    })

    # Resume from checkpoint
    if args.resume:
        result = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device)
        if result is None:
            result = load_checkpoint(best_path, model, optimizer, scheduler, scaler, device)
        if result:
            start_epoch, skip_batches, best_val_loss = result
            _SAVE_CONTEXT["best_val"] = best_val_loss
            log(f"Resuming from epoch {start_epoch}, batch {skip_batches}")
        else:
            log("No checkpoint found, starting fresh")

    log(f"\nStarting training: epochs {start_epoch+1}-{args.epochs}, "
        f"{len(train_dl)} batches/epoch, {total_steps} total steps")
    log("-" * 70)

    try:
        for epoch in range(start_epoch, args.epochs):
            if _SHUTDOWN_REQUESTED:
                break

            _SAVE_CONTEXT["epoch"] = epoch
            epoch_start = time.time()

            current_skip = skip_batches if epoch == start_epoch else 0

            train_loss = train_epoch(
                model, train_dl, optimizer, scheduler, scaler, device,
                epoch + 1, args.checkpoint_interval, current_skip,
            )

            if _SHUTDOWN_REQUESTED:
                break

            # Validation
            val_loss = eval_epoch(model, val_dl, device)
            elapsed = time.time() - epoch_start

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _SAVE_CONTEXT["best_val"] = best_val_loss
                save_checkpoint(
                    best_path, model, optimizer, scheduler, scaler,
                    epoch + 1, 0, best_val_loss, val_loss, reason="best_model",
                )
                improved = " ** NEW BEST **"

            # Always save epoch checkpoint for resume
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                epoch + 1, 0, best_val_loss, val_loss, reason="epoch_end",
            )

            # Periodic epoch snapshots
            if (epoch + 1) % 10 == 0:
                epoch_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt")
                save_checkpoint(
                    epoch_path, model, optimizer, scheduler, scaler,
                    epoch + 1, 0, best_val_loss, val_loss, reason="periodic_snapshot",
                )

            log(f"E{epoch+1}/{args.epochs} | "
                f"Train={train_loss:.4f} Val={val_loss:.4f} Best={best_val_loss:.4f} | "
                f"{elapsed:.0f}s{improved}")

        # Save final model
        final_path = os.path.join(args.checkpoint_dir, "final.pt")
        torch.save(model.state_dict(), final_path)

        log("=" * 70)
        log(f"Training complete. Best val loss: {best_val_loss:.4f}")
        log(f"Checkpoints: {args.checkpoint_dir}")
        log(f"  best.pt   -> best validation model")
        log(f"  final.pt  -> last epoch model state_dict")
        log(f"  checkpoint.pt -> full resumable state")
        log("=" * 70)

    except Exception as e:
        log(f"\n[CRASH] {type(e).__name__}: {e}")
        save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler,
            _SAVE_CONTEXT.get("epoch", 0), _SAVE_CONTEXT.get("batch_idx", 0),
            best_val_loss, reason=f"crash_{type(e).__name__}",
        )
        raise


if __name__ == "__main__":
    main()
