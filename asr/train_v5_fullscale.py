#!/usr/bin/env python3
"""
N'Ko ASR V5 — Full-Scale Whisper LoRA Training
=================================================
The definitive N'Ko ASR training pipeline. Combines:
  - V4's hardening (resume, signal handling, mid-epoch checkpoints, webhooks)
  - V3's architecture (Transformer h768 L6 CTC head with Conv1d stride-4)
  - Full-scale data (all available Bambara/Manding audio, ~714 hours)
  - Full LoRA coverage (all 32 encoder layers, rank=64, alpha=128)
  - 4 target modules per layer (q_proj, k_proj, v_proj, out_proj)
  - SpecAugment data augmentation
  - Dual learning rates (encoder LoRA + CTC head)
  - Cosine LR schedule with warmup
  - Mixed precision (AMP) with GradScaler
  - W&B logging (optional)

Architecture:
  Whisper large-v3 encoder (frozen + LoRA on all 32 layers)
    -> LoRA: rank=64, alpha=128, target=[q,k,v,out], dropout=0.1
    -> Transformer CTC Head: h768, L6, nhead=12, Conv1d stride-4
    -> CTC loss with N'Ko character vocabulary (65 chars + blank)

Data pipeline:
  Pre-extracted Whisper features (4x downsampled, float16, 375 frames max)
  loaded from manifests produced by prepare_v5_data.py.

Requires:
  - A100 80GB (Whisper encoder + LoRA + CTC head + gradients)
  - /workspace/v5_data/ produced by prepare_v5_data.py
  - torch, whisper, datasets

Usage:
    # Standard training
    python3 asr/train_v5_fullscale.py

    # Resume from checkpoint
    python3 asr/train_v5_fullscale.py --resume

    # Custom config
    python3 asr/train_v5_fullscale.py --epochs 30 --batch-size 32 --lora-rank 64

    # With W&B logging
    WANDB_API_KEY=xxx python3 asr/train_v5_fullscale.py --wandb-project nko-asr-v5

    # With webhook notifications
    python3 asr/train_v5_fullscale.py --webhook https://hooks.example.com/notify
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

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── Globals for signal handler ────────────────────────────────────
_SHUTDOWN_REQUESTED = False
_SAVE_CONTEXT = {}


# ── N'Ko Character Vocabulary ────────────────────────────────────

def build_nko_char_vocab():
    """Build N'Ko character vocabulary.

    Covers U+07C0 to U+07FF (64 chars) + space = 65 chars.
    Blank token index = 65.
    """
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ── LoRA Adapter ──────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a linear layer.

    Implements LoRA: h = W_0 x + (B A) x * (alpha / rank)
    where A is (rank, in_features) and B is (out_features, rank).
    """

    def __init__(self, original: nn.Linear, rank: int = 64, alpha: int = 128, dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.randn(rank, original.in_features, device=original.weight.device) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original.out_features, rank, device=original.weight.device)
        )

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scale
        return original_out + lora_out


def apply_lora_to_whisper(
    model,
    num_layers: int = 32,
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.1,
    target_modules: list = None,
):
    """Apply LoRA to Whisper encoder layers.

    V5: ALL 32 layers, 4 target modules per layer attention
    (q_proj, k_proj, v_proj, out_proj) + MLP layers.

    Args:
        model: Whisper model
        num_layers: number of encoder layers to adapt (32 = all)
        rank: LoRA rank (default 64)
        alpha: LoRA alpha (default 128)
        dropout: dropout on LoRA path (default 0.1)
        target_modules: list of attention module names to adapt

    Returns:
        List of trainable LoRA parameters
    """
    if target_modules is None:
        target_modules = ["query", "key", "value", "out"]

    total_layers = len(model.encoder.blocks)
    start_layer = max(0, total_layers - num_layers)
    lora_params = []

    for i in range(start_layer, total_layers):
        block = model.encoder.blocks[i]

        # Attention modules
        for name in target_modules:
            attr = getattr(block.attn, name, None)
            if attr is not None and isinstance(attr, nn.Linear):
                lora = LoRALinear(attr, rank=rank, alpha=alpha, dropout=dropout)
                setattr(block.attn, name, lora)
                lora_params.extend([lora.lora_A, lora.lora_B])

        # MLP layers (two linear layers: 0 and 2 in the Sequential)
        for mlp_name in ["0", "2"]:
            if hasattr(block.mlp, mlp_name):
                layer = getattr(block.mlp, mlp_name)
                if isinstance(layer, nn.Linear):
                    lora = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(block.mlp, mlp_name, lora)
                    lora_params.extend([lora.lora_A, lora.lora_B])

    # Freeze all then unfreeze LoRA
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora_per_layer = len([n for n in target_modules]) * 2  # A + B per module
    mlp_adapted = sum(1 for i in range(start_layer, total_layers)
                      for n in ["0", "2"]
                      if hasattr(model.encoder.blocks[i].mlp, n)
                      and isinstance(getattr(model.encoder.blocks[i].mlp, n), LoRALinear))

    print(f"  Whisper LoRA applied:")
    print(f"    Layers: {start_layer}-{total_layers-1} ({total_layers - start_layer} layers)")
    print(f"    Rank: {rank}, Alpha: {alpha}, Dropout: {dropout}")
    print(f"    Attention targets: {target_modules}")
    print(f"    MLP layers adapted: {mlp_adapted}")
    print(f"    Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return lora_params


# ── Dataset ──────────────────────────────────────────────────────

class V5MelDataset(Dataset):
    """Dataset for V5 training from cached mel spectrograms.

    IMPORTANT: V5 uses LoRA on the Whisper encoder, so we CANNOT use
    pre-extracted encoder features (those were from the frozen encoder).
    Instead, we cache mel spectrograms and run them through the LoRA-adapted
    encoder at each forward pass.

    If mel spectrograms are not cached, falls back to loading raw audio
    from HF datasets and computing mel on-the-fly.

    Flow:
      Dataset returns: mel spectrogram [128, 3000] (30s padded Whisper input)
      Training loop: mel -> whisper_model.encoder(mel) -> ctc_head -> CTC loss

    SpecAugment is applied to mel spectrograms (not encoder features),
    which is the standard practice for ASR data augmentation.
    """

    def __init__(
        self,
        manifest_path: str,
        mel_dir: str = None,
        features_dir: str = None,
        hf_datasets: dict = None,
        char_vocab: dict = None,
        max_text_len: int = 200,
        augment: bool = False,
    ):
        """
        Args:
            manifest_path: JSONL manifest with entries
            mel_dir: directory with cached mel spectrograms (*.mel.pt)
            features_dir: directory with pre-extracted features (fallback, for
                          use with separate CTC-head-only training)
            hf_datasets: dict mapping source name -> HF dataset split object
                         (for on-the-fly mel computation)
            char_vocab: N'Ko character vocabulary
            max_text_len: max text label length
            augment: whether to apply SpecAugment
        """
        self.mel_dir = Path(mel_dir) if mel_dir else None
        self.features_dir = Path(features_dir) if features_dir else None
        self.hf_datasets = hf_datasets or {}
        self.char_vocab = char_vocab
        self.max_text_len = max_text_len
        self.augment = augment
        self.use_mel = True  # True: load mel for LoRA training; False: pre-extracted features

        # Load manifest
        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Check if mel exists
                    if self.mel_dir:
                        mel_path = self.mel_dir / f"{entry['feat_id']}.mel.pt"
                        if mel_path.exists():
                            entry["_mel_path"] = str(mel_path)
                            self.entries.append(entry)
                            continue
                    # Check if pre-extracted features exist (fallback)
                    if self.features_dir:
                        feat_path = self.features_dir / f"{entry['feat_id']}.pt"
                        if feat_path.exists():
                            entry["_feat_path"] = str(feat_path)
                            self.entries.append(entry)
                            continue
                    # Check if we can load from HF dataset
                    source = entry.get("source", "")
                    if source in self.hf_datasets:
                        self.entries.append(entry)
                        continue

        # Determine mode
        mel_count = sum(1 for e in self.entries if "_mel_path" in e)
        feat_count = sum(1 for e in self.entries if "_feat_path" in e)
        hf_count = len(self.entries) - mel_count - feat_count

        if mel_count > 0:
            self.use_mel = True
            print(f"  Dataset: {len(self.entries):,} samples (mel={mel_count:,}, feat={feat_count:,}, hf={hf_count:,})")
        elif feat_count > 0:
            self.use_mel = False
            print(f"  Dataset: {len(self.entries):,} samples (pre-extracted features, no LoRA encoder pass)")
            print(f"  WARNING: Using pre-extracted features bypasses LoRA encoder. For full V5,")
            print(f"           run mel extraction first: python3 prepare_v5_data.py --extract-mel")
        else:
            print(f"  Dataset: {len(self.entries):,} samples (on-the-fly mel from HF)")

    def __len__(self):
        return len(self.entries)

    def spec_augment_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """SpecAugment on mel spectrogram [128, T] (standard ASR augmentation).

        V5 config:
          - Time masking: 1-3 bands of 10-40 frames
          - Frequency masking: 1-2 bands of 5-20 bins
        """
        F_dim, T = mel.shape

        # Time masking
        for _ in range(random.randint(1, 3)):
            t_width = random.randint(10, min(40, max(T // 8, 10)))
            t_start = random.randint(0, max(0, T - t_width))
            mel[:, t_start:t_start + t_width] = 0

        # Frequency masking
        for _ in range(random.randint(1, 2)):
            f_width = random.randint(5, min(20, max(F_dim // 4, 5)))
            f_start = random.randint(0, max(0, F_dim - f_width))
            mel[f_start:f_start + f_width, :] = 0

        return mel

    def spec_augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """SpecAugment on encoder features [T, 1280] (fallback for pre-extracted)."""
        T, F_dim = features.shape

        for _ in range(random.randint(1, 3)):
            t_width = random.randint(5, min(20, max(T // 4, 5)))
            t_start = random.randint(0, max(0, T - t_width))
            features[t_start:t_start + t_width] = 0

        for _ in range(random.randint(1, 2)):
            f_width = random.randint(20, min(80, max(F_dim // 4, 20)))
            f_start = random.randint(0, max(0, F_dim - f_width))
            features[:, f_start:f_start + f_width] = 0

        return features

    def _load_mel(self, entry):
        """Load or compute mel spectrogram for an entry.

        Returns: mel tensor [128, 3000] (Whisper input format)
        """
        import whisper

        # Option 1: Cached mel
        if "_mel_path" in entry:
            try:
                mel = torch.load(entry["_mel_path"], weights_only=True).float()
                return mel
            except Exception:
                pass

        # Option 2: Compute from HF audio
        source = entry.get("source", "")
        source_idx = entry.get("source_idx", 0)
        ds_split = self.hf_datasets.get(source)
        if ds_split is not None:
            try:
                sample = ds_split[source_idx]
                audio_data = sample.get("audio")
                if isinstance(audio_data, dict):
                    audio_array = audio_data["array"]
                    sr = audio_data.get("sampling_rate", 16000)
                else:
                    audio_array = audio_data
                    sr = 16000
                audio_t = torch.tensor(audio_array, dtype=torch.float32)
                if sr != 16000:
                    ratio = 16000 / sr
                    audio_t = F.interpolate(
                        audio_t.unsqueeze(0).unsqueeze(0),
                        size=int(len(audio_t) * ratio),
                        mode="linear", align_corners=False,
                    ).squeeze()
                audio_padded = whisper.pad_or_trim(audio_t)
                mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128)
                return mel
            except Exception:
                pass

        # Fallback: zero mel
        return torch.zeros(128, 3000)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Encode N'Ko text label
        nko = entry.get("nko", "")
        indices = [self.char_vocab[c] for c in nko if c in self.char_vocab]
        if not indices:
            indices = [0]
        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])

        if self.use_mel or "_mel_path" in entry or "_feat_path" not in entry:
            # LoRA mode: return mel spectrogram for encoder pass
            mel = self._load_mel(entry)

            if self.augment and random.random() < 0.5:
                mel = self.spec_augment_mel(mel.clone())

            # mel shape: [128, 3000] — Whisper expects this
            return mel, text, text_len
        else:
            # Fallback: pre-extracted features (skip encoder, CTC head only)
            try:
                features = torch.load(entry["_feat_path"], weights_only=True).float()
            except Exception:
                features = torch.zeros(375, 1280)

            max_audio_len = 375
            if features.shape[0] > max_audio_len:
                features = features[:max_audio_len]
            audio_len = features.shape[0]

            if self.augment and random.random() < 0.5:
                features = self.spec_augment_features(features.clone())

            padded = torch.zeros(max_audio_len, 1280)
            padded[:audio_len] = features

            return padded, audio_len, text, text_len


# ── CTC Head ─────────────────────────────────────────────────────

class TransformerCTCHead(nn.Module):
    """Transformer CTC Head (CharASR V3 architecture).

    Input: Whisper encoder features [B, T, 1280]
      - Mel mode (LoRA training): T=1500, output T'=375
      - Feature mode (pre-extracted): T=375, output T'=~94
    Output: CTC logits [B, T', num_chars+1]

    Architecture:
      Linear(1280, 768) + GELU + Dropout
      Conv1d(768, 768, k=5, s=4) + GELU    # Temporal downsample 4x
      Positional encoding
      6-layer Transformer encoder (h=768, nhead=12, ff=3072)
      LayerNorm
      Linear(768, num_chars + 1)
    """

    def __init__(
        self,
        num_chars: int,
        input_dim: int = 1280,
        hidden_dim: int = 768,
        num_layers: int = 6,
        nhead: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal downsampler via strided convolution
        self.temporal_ds = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)  # +1 for CTC blank

    def forward(self, x):
        """Forward pass.

        Args:
            x: [B, T, 1280] Whisper encoder features (T=375 or T=1500)

        Returns:
            [B, T', num_chars+1] CTC logits
        """
        x = self.input_proj(x)  # [B, T, hidden]
        # Temporal downsample: [B, T, H] -> [B, H, T] -> conv -> [B, H, T//4] -> [B, T//4, H]
        x = self.temporal_ds(x.permute(0, 2, 1)).permute(0, 2, 1)
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        x = self.encoder(x)
        x = self.ln(x)
        return self.output_proj(x)


# ── Webhook ──────────────────────────────────────────────────────

def send_webhook(url, event, data=None):
    """Fire-and-forget webhook notification."""
    if not url:
        return
    try:
        import urllib.request
        payload = json.dumps({
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "model": "nko-asr-v5",
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
            **(data or {}),
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"  [webhook] failed: {e}", flush=True)


# ── Checkpoint ───────────────────────────────────────────────────

def save_checkpoint(
    path, whisper_model, ctc_head, optimizer, scaler, scheduler,
    epoch, batch_idx, best_val, val_loss=None, reason="periodic",
):
    """Save full training state for resume. Atomic write via tmp+rename."""
    ckpt = {
        "lora_state": {
            n: p.data.clone() for n, p in whisper_model.named_parameters() if p.requires_grad
        },
        "head_state": ctc_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "best_val": best_val,
        "val_loss": val_loss,
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "version": "v5",
    }
    tmp_path = str(path) + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    print(f"  [checkpoint] saved: epoch={epoch} batch={batch_idx} reason={reason}", flush=True)


def load_checkpoint(path, whisper_model, ctc_head, optimizer, scaler, scheduler, device):
    """Load training state. Returns (start_epoch, batch_idx, best_val) or None."""
    if not Path(path).exists():
        return None

    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Restore LoRA weights
    lora_state = ckpt.get("lora_state", {})
    model_params = dict(whisper_model.named_parameters())
    loaded = 0
    for name, data in lora_state.items():
        if name in model_params and model_params[name].requires_grad:
            model_params[name].data.copy_(data)
            loaded += 1
    print(f"  Restored {loaded} LoRA parameters")

    # Restore CTC head
    ctc_head.load_state_dict(ckpt["head_state"])
    print(f"  Restored CTC head")

    # Restore optimizer
    if "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print(f"  Restored optimizer state")
        except Exception as e:
            print(f"  [warn] optimizer restore failed: {e}")

    # Restore scaler
    if "scaler_state" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
            print(f"  Restored scaler state")
        except Exception as e:
            print(f"  [warn] scaler restore failed: {e}")

    # Restore scheduler
    if scheduler and ckpt.get("scheduler_state"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            print(f"  Restored scheduler state")
        except Exception as e:
            print(f"  [warn] scheduler restore failed: {e}")

    epoch = ckpt.get("epoch", 0)
    batch_idx = ckpt.get("batch_idx", 0)
    best_val = ckpt.get("best_val", float("inf"))
    reason = ckpt.get("reason", "unknown")
    version = ckpt.get("version", "unknown")
    print(f"  Resume: epoch={epoch}, batch={batch_idx}, best_val={best_val:.4f}, version={version}, reason={reason}")
    return epoch, batch_idx, best_val


# ── Signal Handler ───────────────────────────────────────────────

def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    print(f"\n[SIGNAL] {sig_name} received. Saving checkpoint and shutting down...", flush=True)
    _SHUTDOWN_REQUESTED = True

    ctx = _SAVE_CONTEXT
    if ctx.get("whisper_model"):
        save_checkpoint(
            ctx["checkpoint_path"],
            ctx["whisper_model"], ctx["ctc_head"],
            ctx["optimizer"], ctx["scaler"], ctx.get("scheduler"),
            ctx.get("epoch", 0), ctx.get("batch_idx", 0),
            ctx.get("best_val", float("inf")),
            reason=f"signal_{sig_name}",
        )
        send_webhook(ctx.get("webhook"), "signal_shutdown", {
            "signal": sig_name, "epoch": ctx.get("epoch", 0),
        })
    sys.exit(0)


# ── Learning Rate Schedule ───────────────────────────────────────

class DualCosineScheduler:
    """Cosine annealing with warmup for dual parameter groups.

    Group 0 (LoRA): base_lr = lr_whisper
    Group 1 (CTC head): base_lr = lr_head
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_whisper, lr_head):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [lr_whisper, lr_head]
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]

            if self.current_epoch < self.warmup_epochs:
                lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            else:
                progress = (self.current_epoch - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs, 1
                )
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

            pg["lr"] = base_lr * max(lr_scale, 1e-7)

    def state_dict(self):
        return {"current_epoch": self.current_epoch}

    def load_state_dict(self, state):
        self.current_epoch = state.get("current_epoch", 0)


# ── Training Loop ────────────────────────────────────────────────

def train_epoch(
    whisper_model, ctc_head, loader, optimizer, device, num_chars, scaler,
    checkpoint_interval=1000, skip_batches=0, wandb_log=False, use_mel=True,
):
    """Train one epoch with mid-epoch checkpointing and graceful shutdown.

    Supports two data modes:
      - mel mode (use_mel=True): batch = (mel, targets, target_lens)
        mel goes through LoRA-adapted encoder then CTC head
      - feature mode (use_mel=False): batch = (features, audio_lens, targets, target_lens)
        features go directly to CTC head (encoder already applied)
    """
    global _SHUTDOWN_REQUESTED
    whisper_model.train()
    ctc_head.train()
    total_loss, steps = 0.0, 0
    total_batches = len(loader)
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        if _SHUTDOWN_REQUESTED:
            print("  [shutdown] stopping training loop", flush=True)
            break

        if batch_idx < skip_batches:
            continue

        if use_mel:
            # Mel mode: (mel [B, 128, 3000], targets, target_lens)
            mel, targets, target_lens = batch
            mel = mel.to(device)
            targets = targets.to(device)

            with torch.amp.autocast('cuda'):
                # Run mel through LoRA-adapted Whisper encoder
                encoder_out = whisper_model.encoder(mel)  # [B, 1500, 1280]
                logits = ctc_head(encoder_out)  # CTC head does 4x temporal DS internally
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # [T, B, C]
                T_out = log_probs.shape[0]

                # All audio is 30s padded, so all have same output length
                audio_lens = torch.full((mel.shape[0],), T_out, dtype=torch.long)

                loss = F.ctc_loss(
                    log_probs, targets, audio_lens, target_lens,
                    blank=num_chars, zero_infinity=True,
                )
        else:
            # Feature mode: (features [B, 375, 1280], audio_lens, targets, target_lens)
            features, audio_lens, targets, target_lens = batch
            features = features.to(device)
            targets = targets.to(device)
            ds_audio_lens = (audio_lens + 3) // 4

            with torch.amp.autocast('cuda'):
                # Skip encoder, go straight to CTC head
                logits = ctc_head(features)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                T_out = log_probs.shape[0]
                ds_audio_lens = ds_audio_lens.clamp(max=T_out)

                loss = F.ctc_loss(
                    log_probs, targets, ds_audio_lens, target_lens,
                    blank=num_chars, zero_infinity=True,
                )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Clip gradients for both LoRA and CTC head
        torch.nn.utils.clip_grad_norm_(
            list(whisper_model.parameters()) + list(ctc_head.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
        _SAVE_CONTEXT["batch_idx"] = batch_idx

        # Logging
        if steps % 50 == 0:
            elapsed = time.time() - t0
            rate = steps / elapsed
            remaining = total_batches - batch_idx - 1
            eta = remaining / rate / 60 if rate > 0 else 0
            current_lr_lora = optimizer.param_groups[0]["lr"]
            current_lr_head = optimizer.param_groups[1]["lr"]
            avg_loss = total_loss / steps
            print(
                f"  batch {batch_idx+1}/{total_batches} "
                f"loss={loss.item():.3f} avg={avg_loss:.3f} "
                f"lr_lora={current_lr_lora:.2e} lr_head={current_lr_head:.2e} "
                f"({rate:.1f} b/s, ETA {eta:.0f}m)",
                flush=True,
            )

            if wandb_log and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/lr_lora": current_lr_lora,
                    "train/lr_head": current_lr_head,
                    "train/batch": batch_idx,
                    "train/rate_bs": rate,
                })

        # Mid-epoch checkpoint
        if checkpoint_interval > 0 and (batch_idx + 1) % checkpoint_interval == 0:
            ctx = _SAVE_CONTEXT
            save_checkpoint(
                ctx["checkpoint_path"],
                whisper_model, ctc_head, optimizer, scaler, ctx.get("scheduler"),
                ctx.get("epoch", 0), batch_idx + 1,
                ctx.get("best_val", float("inf")),
                reason="mid_epoch",
            )

    return total_loss / max(steps, 1)


def eval_epoch(whisper_model, ctc_head, loader, device, num_chars, use_mel=True):
    """Evaluate one epoch."""
    whisper_model.eval()
    ctc_head.eval()
    total_loss, steps = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            if use_mel:
                mel, targets, target_lens = batch
                mel = mel.to(device)
                targets = targets.to(device)

                encoder_out = whisper_model.encoder(mel)
                logits = ctc_head(encoder_out)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                T_out = log_probs.shape[0]
                audio_lens = torch.full((mel.shape[0],), T_out, dtype=torch.long)

                loss = F.ctc_loss(
                    log_probs, targets, audio_lens, target_lens,
                    blank=num_chars, zero_infinity=True,
                )
            else:
                features, audio_lens, targets, target_lens = batch
                features = features.to(device)
                targets = targets.to(device)
                ds_audio_lens = (audio_lens + 3) // 4

                logits = ctc_head(features)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                T_out = log_probs.shape[0]
                ds_audio_lens = ds_audio_lens.clamp(max=T_out)

                loss = F.ctc_loss(
                    log_probs, targets, ds_audio_lens, target_lens,
                    blank=num_chars, zero_infinity=True,
                )

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                steps += 1

    return total_loss / max(steps, 1)


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="N'Ko ASR V5 Full-Scale Training")

    # Paths
    parser.add_argument("--data-dir", default="/workspace/v5_data",
                        help="V5 data directory (from prepare_v5_data.py)")
    parser.add_argument("--save-dir", default="/workspace/v5_model",
                        help="Output directory for checkpoints")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-whisper", type=float, default=5e-6,
                        help="Learning rate for Whisper LoRA parameters")
    parser.add_argument("--lr-head", type=float, default=1e-4,
                        help="Learning rate for CTC head parameters")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs for cosine schedule")
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-layers", type=int, default=32,
                        help="Number of encoder layers to adapt (32 = all)")

    # CTC head config
    parser.add_argument("--head-hidden", type=int, default=768)
    parser.add_argument("--head-layers", type=int, default=6)
    parser.add_argument("--head-nhead", type=int, default=12)
    parser.add_argument("--head-dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable SpecAugment")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")

    # Resilience
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save checkpoint every N batches (0=disabled)")

    # Monitoring
    parser.add_argument("--webhook", default=None,
                        help="Webhook URL for notifications")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (requires WANDB_API_KEY)")

    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Device
    device = "cuda"
    print(f"=" * 70)
    print(f"N'Ko ASR V5 — Full-Scale Training")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"PID: {os.getpid()}")
    print(f"Start: {datetime.utcnow().isoformat()}")
    print(f"Resume: {args.resume}")
    print()

    # W&B init
    wandb_log = False
    if args.wandb_project and WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project=args.wandb_project,
            name=f"v5-r{args.lora_rank}-e{args.epochs}",
            config=vars(args),
        )
        wandb_log = True
        print(f"W&B logging enabled: {args.wandb_project}")
    elif args.wandb_project:
        print("W&B requested but not available (install wandb + set WANDB_API_KEY)")

    # ── 1. Character vocabulary ────────────────────────────────
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}
    print(f"Vocabulary: {num_chars} chars (N'Ko U+07C0-U+07FF + space)")

    # ── 2. Load Whisper + apply LoRA ──────────────────────────
    print("\nLoading Whisper large-v3...")
    import whisper
    whisper_model = whisper.load_model("large-v3", device=device)
    print("Whisper loaded.")

    print(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, layers={args.lora_layers})...")
    lora_params = apply_lora_to_whisper(
        whisper_model,
        num_layers=args.lora_layers,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=["query", "key", "value", "out"],
    )

    # ── 3. CTC Head (fresh, trained from scratch) ─────────────
    ctc_head = TransformerCTCHead(
        num_chars,
        hidden_dim=args.head_hidden,
        num_layers=args.head_layers,
        nhead=args.head_nhead,
        dropout=args.head_dropout,
    ).to(device)
    head_params = sum(p.numel() for p in ctc_head.parameters())
    print(f"\nCTC Head: {head_params:,} params (h{args.head_hidden} L{args.head_layers} nhead={args.head_nhead})")
    print(f"  Trained from scratch (V5 does not carry V4 head weights)")

    # ── 4. Load datasets ──────────────────────────────────────
    data_dir = Path(args.data_dir)
    features_dir = data_dir / "features"
    mel_dir = data_dir / "mels"
    train_manifest = data_dir / "manifests" / "train.jsonl"
    val_manifest = data_dir / "manifests" / "val.jsonl"

    if not train_manifest.exists():
        print(f"\nERROR: Train manifest not found: {train_manifest}")
        print(f"Run prepare_v5_data.py first to create manifests and features.")
        sys.exit(1)

    print(f"\nLoading datasets...")

    # Determine data mode: mel spectrograms (preferred) or pre-extracted features
    mel_dir_resolved = mel_dir if mel_dir.exists() else None
    feat_dir_resolved = features_dir if features_dir.exists() else None

    train_dataset = V5MelDataset(
        str(train_manifest),
        mel_dir=str(mel_dir_resolved) if mel_dir_resolved else None,
        features_dir=str(feat_dir_resolved) if feat_dir_resolved else None,
        char_vocab=char_vocab,
        augment=not args.no_augment,
    )

    use_mel = train_dataset.use_mel

    eval_dataset = None
    if val_manifest.exists():
        eval_dataset = V5MelDataset(
            str(val_manifest),
            mel_dir=str(mel_dir_resolved) if mel_dir_resolved else None,
            features_dir=str(feat_dir_resolved) if feat_dir_resolved else None,
            char_vocab=char_vocab,
            augment=False,
        )

    if len(train_dataset) == 0:
        print("ERROR: No training samples loaded. Check feature/mel extraction.")
        sys.exit(1)

    print(f"SpecAugment: {'ON' if not args.no_augment else 'OFF'}")
    print(f"Data mode: {'mel (LoRA encoder active)' if use_mel else 'pre-extracted features (encoder frozen)'}")

    # ── 5. Optimizer + Scheduler ──────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_whisper, "weight_decay": args.weight_decay},
        {"params": ctc_head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
    ])

    scaler = torch.amp.GradScaler('cuda')

    scheduler = DualCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        lr_whisper=args.lr_whisper,
        lr_head=args.lr_head,
    )

    # ── 6. Setup directories + state ──────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    start_epoch = 0
    skip_batches = 0

    checkpoint_path = save_dir / "checkpoint.pt"
    best_path = save_dir / "best_v5.pt"

    # Populate signal handler context
    _SAVE_CONTEXT.update({
        "whisper_model": whisper_model,
        "ctc_head": ctc_head,
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "checkpoint_path": str(checkpoint_path),
        "best_val": best_val,
        "epoch": 0,
        "batch_idx": 0,
        "webhook": args.webhook,
    })

    # Resume from checkpoint
    if args.resume:
        result = load_checkpoint(
            str(checkpoint_path), whisper_model, ctc_head, optimizer, scaler, scheduler, device
        )
        if result is None:
            result = load_checkpoint(
                str(best_path), whisper_model, ctc_head, optimizer, scaler, scheduler, device
            )
        if result:
            start_epoch, skip_batches, best_val = result
            _SAVE_CONTEXT["best_val"] = best_val
            print(f"Resuming from epoch {start_epoch}, batch {skip_batches}")
        else:
            print("No checkpoint found, starting fresh")

    # Save training config
    config_path = save_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 7. Training config summary ────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR (LoRA): {args.lr_whisper}")
    print(f"  LR (CTC head): {args.lr_head}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LoRA: rank={args.lora_rank} alpha={args.lora_alpha} layers={args.lora_layers}")
    print(f"  CTC head: h{args.head_hidden} L{args.head_layers} nhead={args.head_nhead}")
    print(f"  Checkpoint interval: {args.checkpoint_interval} batches")
    print(f"  Train samples: {len(train_dataset):,}")
    if eval_dataset:
        print(f"  Val samples: {len(eval_dataset):,}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*70}\n")

    send_webhook(args.webhook, "training_start", {
        "version": "v5",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": len(train_dataset),
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "resume_epoch": start_epoch,
    })

    # ── 8. Training Loop ──────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs):
            if _SHUTDOWN_REQUESTED:
                break

            _SAVE_CONTEXT["epoch"] = epoch
            epoch_start = time.time()

            # Update learning rate
            scheduler.step(epoch)
            current_lr_lora = optimizer.param_groups[0]["lr"]
            current_lr_head = optimizer.param_groups[1]["lr"]

            print(f"\nEpoch {epoch+1}/{args.epochs} "
                  f"(lr_lora={current_lr_lora:.2e}, lr_head={current_lr_head:.2e})")

            # Create data loaders (re-shuffled each epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            # Skip batches only on first resumed epoch
            current_skip = skip_batches if epoch == start_epoch else 0

            train_loss = train_epoch(
                whisper_model, ctc_head, train_loader, optimizer, device,
                num_chars, scaler, args.checkpoint_interval, current_skip,
                wandb_log, use_mel=use_mel,
            )

            if _SHUTDOWN_REQUESTED:
                break

            # Validation
            val_loss = float("inf")
            if eval_dataset and len(eval_dataset) > 0:
                val_loader = DataLoader(
                    eval_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                val_loss = eval_epoch(whisper_model, ctc_head, val_loader, device, num_chars, use_mel=use_mel)

            # Best model
            improved = ""
            if val_loss < best_val:
                best_val = val_loss
                _SAVE_CONTEXT["best_val"] = best_val
                save_checkpoint(
                    str(best_path), whisper_model, ctc_head, optimizer, scaler, scheduler,
                    epoch, 0, best_val, val_loss, reason="best_model",
                )
                improved = " *BEST*"

            # Epoch checkpoint
            save_checkpoint(
                str(checkpoint_path), whisper_model, ctc_head, optimizer, scaler, scheduler,
                epoch + 1, 0, best_val, val_loss, reason="epoch_end",
            )

            # Also save epoch-specific checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                epoch_ckpt = save_dir / f"epoch_{epoch+1:03d}.pt"
                save_checkpoint(
                    str(epoch_ckpt), whisper_model, ctc_head, optimizer, scaler, scheduler,
                    epoch + 1, 0, best_val, val_loss, reason=f"epoch_{epoch+1}",
                )

            elapsed = time.time() - epoch_start
            print(
                f"E{epoch+1}/{args.epochs} "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"best={best_val:.4f}{improved} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

            if wandb_log and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "val/best_loss": best_val,
                    "lr/lora": current_lr_lora,
                    "lr/head": current_lr_head,
                })

            send_webhook(args.webhook, "epoch_complete", {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "best_val": round(best_val, 4),
                "improved": bool(improved),
                "elapsed_s": round(elapsed, 1),
            })

        # ── 9. Training complete ──────────────────────────────
        print(f"\n{'='*70}")
        print(f"Training Complete")
        print(f"{'='*70}")
        print(f"  Best val loss: {best_val:.4f}")
        print(f"  Checkpoints: {save_dir}/")
        print(f"  Best model: {best_path}")

        # Save final LoRA weights separately (for merging)
        lora_only_path = save_dir / "lora_weights_v5.pt"
        lora_state = {n: p.data.cpu() for n, p in whisper_model.named_parameters() if p.requires_grad}
        torch.save(lora_state, str(lora_only_path))
        print(f"  LoRA weights: {lora_only_path}")

        # Save CTC head separately
        head_path = save_dir / "ctc_head_v5.pt"
        torch.save(ctc_head.state_dict(), str(head_path))
        print(f"  CTC head: {head_path}")

        send_webhook(args.webhook, "training_complete", {
            "best_val": round(best_val, 4),
            "epochs": args.epochs,
        })

        if wandb_log and WANDB_AVAILABLE:
            wandb.finish()

    except Exception as e:
        print(f"\n[CRASH] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

        save_checkpoint(
            str(checkpoint_path), whisper_model, ctc_head, optimizer, scaler, scheduler,
            _SAVE_CONTEXT.get("epoch", 0), _SAVE_CONTEXT.get("batch_idx", 0),
            best_val, reason=f"crash_{type(e).__name__}",
        )
        send_webhook(args.webhook, "training_crash", {
            "error": str(e),
            "epoch": _SAVE_CONTEXT.get("epoch", 0),
        })

        if wandb_log and WANDB_AVAILABLE:
            wandb.finish(exit_code=1)

        raise


if __name__ == "__main__":
    main()
