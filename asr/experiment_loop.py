#!/usr/bin/env python3
"""
Fixed-Budget Experiment Loop for N'Ko ASR
==========================================
Inspired by The Karpathy Loop, but targeted: systematic grid search
over architecture and training hyperparameters with fixed epoch budget.

Each config trains for N epochs (default 10), evaluates CER on test set,
and logs to results.tsv. Run after baseline training completes.

Usage:
    python3 experiment_loop.py \
        --features-dir /workspace/human_features \
        --train-pairs /workspace/bam_train_nko_fixed.jsonl \
        --test-pairs /workspace/bam_test_nko.jsonl \
        --epochs-per-run 10 \
        --budget-hours 12

Cost estimate: ~27 configs x 25 min each = ~11h @ $0.26/hr = $2.86
"""

import argparse
import json
import time
import random
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Vocab ──────────────────────────────────────────────────────────────

def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ── Dataset ────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    def __init__(self, features_dir, pairs, char_vocab, max_audio_len=1500, max_text_len=200):
        self.features_dir = Path(features_dir)
        self.char_vocab = char_vocab
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        list_file = Path(features_dir).parent / "human_features_list.txt"
        if list_file.exists():
            available = set(list_file.read_text().strip().split("\n"))
        else:
            available = set(p.stem for p in self.features_dir.glob("*.pt"))
        self.pairs = [p for p in pairs if p.get("feat_id", "") in available]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        feat_path = self.features_dir / f"{p['feat_id']}.pt"
        try:
            features = torch.load(feat_path, weights_only=True)
        except Exception:
            features = torch.zeros(self.max_audio_len, 1280)

        if features.shape[0] > 4:
            features = features[::4]
        if features.shape[0] > self.max_audio_len // 4:
            features = features[:self.max_audio_len // 4]
        audio_len = features.shape[0]

        padded = torch.zeros(self.max_audio_len // 4, 1280)
        padded[:audio_len] = features

        nko = p.get("nko", "")
        indices = [self.char_vocab[c] for c in nko if c in self.char_vocab]
        if not indices:
            indices = [0]

        text_len = min(len(indices), self.max_text_len)
        text = torch.zeros(self.max_text_len, dtype=torch.long)
        text[:text_len] = torch.tensor(indices[:text_len])

        return padded, audio_len, text, text_len


# ── Models ─────────────────────────────────────────────────────────────

class BiLSTM_CTC(nn.Module):
    """Baseline BiLSTM architecture."""
    def __init__(self, num_chars, input_dim=1280, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)


class Conformer_CTC(nn.Module):
    """Conformer-style: Conv1d + Transformer encoder."""
    def __init__(self, num_chars, input_dim=1280, hidden_dim=512, num_layers=4,
                 nhead=8, dropout=0.1, kernel_size=31):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.encoder(x)
        return self.output_proj(x)


class Transformer_CTC(nn.Module):
    """Pure Transformer encoder."""
    def __init__(self, num_chars, input_dim=1280, hidden_dim=512, num_layers=4,
                 nhead=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.output_proj(x)


# ── Training / Eval ───────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, num_chars):
    model.train()
    total_loss, steps = 0.0, 0
    for features, audio_lens, targets, target_lens in loader:
        features, targets = features.to(device), targets.to(device)
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
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def eval_epoch(model, loader, device, num_chars):
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for features, audio_lens, targets, target_lens in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = F.ctc_loss(log_probs, targets, audio_lens, target_lens,
                              blank=num_chars, zero_infinity=True)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                steps += 1
    return total_loss / max(steps, 1)


def ctc_decode(logits, idx_to_char, num_chars):
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    prev = -1
    for idx in preds:
        if idx != num_chars and idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = temp
    return dp[n]


def compute_cer(model, test_dataset, device, idx_to_char, num_chars, max_samples=300):
    """Compute Character Error Rate on test set."""
    model.eval()
    total_ed, total_len = 0, 0
    n = min(len(test_dataset), max_samples)

    with torch.no_grad():
        for i in range(n):
            features, audio_len, targets, target_len = test_dataset[i]
            feat = features.unsqueeze(0).to(device)
            logits = model(feat)
            pred = ctc_decode(logits[0], idx_to_char, num_chars)

            # Reconstruct gold text
            gold_indices = targets[:target_len].tolist()
            gold = "".join(idx_to_char.get(idx, "") for idx in gold_indices)

            gc = list(gold.replace(" ", ""))
            pc = list(pred.replace(" ", ""))
            if gc:
                total_ed += edit_distance(gc, pc)
                total_len += len(gc)

    return total_ed / max(total_len, 1) * 100


# ── Experiment Configs ─────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    arch: str           # bilstm, transformer, conformer
    hidden_dim: int
    num_layers: int
    lr: float
    batch_size: int
    dropout: float
    nhead: int = 8      # for transformer/conformer


def build_experiment_grid() -> List[ExperimentConfig]:
    """Build the search grid. ~30 configs total."""
    configs = []

    # BiLSTM variations (baseline architecture, vary size and depth)
    for hidden, layers in [(256, 2), (256, 3), (512, 2), (512, 3), (512, 4), (768, 3), (768, 4)]:
        for lr in [1e-4, 3e-4]:
            configs.append(ExperimentConfig(
                name=f"bilstm_h{hidden}_L{layers}_lr{lr}",
                arch="bilstm", hidden_dim=hidden, num_layers=layers,
                lr=lr, batch_size=16, dropout=0.1,
            ))

    # Transformer variations
    for hidden, layers in [(256, 4), (512, 4), (512, 6)]:
        for lr in [1e-4, 3e-4]:
            nhead = 4 if hidden == 256 else 8
            configs.append(ExperimentConfig(
                name=f"transformer_h{hidden}_L{layers}_lr{lr}",
                arch="transformer", hidden_dim=hidden, num_layers=layers,
                lr=lr, batch_size=16, dropout=0.1, nhead=nhead,
            ))

    # Conformer variations
    for hidden, layers in [(256, 4), (512, 4)]:
        for lr in [1e-4, 3e-4]:
            nhead = 4 if hidden == 256 else 8
            configs.append(ExperimentConfig(
                name=f"conformer_h{hidden}_L{layers}_lr{lr}",
                arch="conformer", hidden_dim=hidden, num_layers=layers,
                lr=lr, batch_size=16, dropout=0.1, nhead=nhead,
            ))

    # Dropout sweep on best baseline
    for dropout in [0.0, 0.05, 0.2, 0.3]:
        configs.append(ExperimentConfig(
            name=f"bilstm_h512_L3_drop{dropout}",
            arch="bilstm", hidden_dim=512, num_layers=3,
            lr=3e-4, batch_size=16, dropout=dropout,
        ))

    return configs


def build_model(config: ExperimentConfig, num_chars: int) -> nn.Module:
    if config.arch == "bilstm":
        return BiLSTM_CTC(num_chars, hidden_dim=config.hidden_dim,
                          num_layers=config.num_layers, dropout=config.dropout)
    elif config.arch == "transformer":
        return Transformer_CTC(num_chars, hidden_dim=config.hidden_dim,
                               num_layers=config.num_layers, nhead=config.nhead,
                               dropout=config.dropout)
    elif config.arch == "conformer":
        return Conformer_CTC(num_chars, hidden_dim=config.hidden_dim,
                             num_layers=config.num_layers, nhead=config.nhead,
                             dropout=config.dropout)
    else:
        raise ValueError(f"Unknown arch: {config.arch}")


# ── Main Loop ──────────────────────────────────────────────────────────

def run_experiment(config, train_dataset, test_dataset, char_vocab, num_chars,
                   idx_to_char, device, epochs, save_dir):
    """Run a single experiment and return results."""
    model = build_model(config, num_chars).to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    val_size = max(int(len(indices) * 0.1), 1)

    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, indices[val_size:]),
        batch_size=config.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, indices[:val_size]),
        batch_size=config.batch_size, num_workers=2,
    )

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, num_chars)
        val_loss = eval_epoch(model, val_loader, device, num_chars)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / f"{config.name}_best.pt")

        elapsed = time.time() - t0
        print(f"  E{epoch+1}/{epochs} t={train_loss:.4f} v={val_loss:.4f} "
              f"best={best_val:.4f} ({elapsed/60:.1f}m)", flush=True)

    # Load best and compute CER
    model.load_state_dict(torch.load(save_dir / f"{config.name}_best.pt", weights_only=True))
    cer = compute_cer(model, test_dataset, device, idx_to_char, num_chars)

    elapsed = time.time() - t0
    return {
        "name": config.name,
        "arch": config.arch,
        "hidden": config.hidden_dim,
        "layers": config.num_layers,
        "lr": config.lr,
        "dropout": config.dropout,
        "params": params,
        "best_val_loss": best_val,
        "cer": cer,
        "time_min": elapsed / 60,
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed-Budget N'Ko ASR Experiment Loop")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--epochs-per-run", type=int, default=10)
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--save-dir", default="/workspace/experiments")
    parser.add_argument("--results-file", default="/workspace/experiment_results.tsv")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip configs that already have results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}
    print(f"Vocab: {num_chars} chars")

    with open(args.train_pairs) as f:
        train_pairs = [json.loads(l) for l in f if l.strip()]
    with open(args.test_pairs) as f:
        test_pairs = [json.loads(l) for l in f if l.strip()]

    train_dataset = CharDataset(args.features_dir, train_pairs, char_vocab)
    test_dataset = CharDataset(args.features_dir, test_pairs, char_vocab)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load existing results to skip
    existing = set()
    results_path = Path(args.results_file)
    if args.skip_existing and results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.startswith("name\t"):
                    continue
                existing.add(line.split("\t")[0])
        print(f"Skipping {len(existing)} existing experiments")

    configs = build_experiment_grid()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT LOOP: {len(configs)} configs x {args.epochs_per_run} epochs")
    print(f"Budget: {args.budget_hours}h @ $0.26/hr = ${args.budget_hours * 0.26:.2f}")
    print(f"{'='*60}\n")

    # Write header if new file
    if not results_path.exists():
        with open(results_path, "w") as f:
            f.write("name\tarch\thidden\tlayers\tlr\tdropout\tparams\tbest_val\tcer\ttime_min\n")

    start_time = time.time()
    best_cer = float("inf")
    best_name = ""

    for i, config in enumerate(configs):
        if config.name in existing:
            print(f"[{i+1}/{len(configs)}] SKIP {config.name} (already done)")
            continue

        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= args.budget_hours:
            print(f"\nBudget exhausted ({elapsed_hours:.1f}h >= {args.budget_hours}h). Stopping.")
            break

        print(f"\n[{i+1}/{len(configs)}] {config.name}")
        print(f"  arch={config.arch} hidden={config.hidden_dim} layers={config.num_layers} "
              f"lr={config.lr} dropout={config.dropout}")

        try:
            result = run_experiment(
                config, train_dataset, test_dataset, char_vocab, num_chars,
                idx_to_char, device, args.epochs_per_run, save_dir,
            )

            # Log result
            with open(results_path, "a") as f:
                f.write(f"{result['name']}\t{result['arch']}\t{result['hidden']}\t"
                        f"{result['layers']}\t{result['lr']}\t{result['dropout']}\t"
                        f"{result['params']}\t{result['best_val_loss']:.4f}\t"
                        f"{result['cer']:.1f}\t{result['time_min']:.1f}\n")

            if result["cer"] < best_cer:
                best_cer = result["cer"]
                best_name = result["name"]

            print(f"  >> CER={result['cer']:.1f}% val={result['best_val_loss']:.4f} "
                  f"params={result['params']:,} time={result['time_min']:.1f}m")
            print(f"  >> Best so far: {best_name} CER={best_cer:.1f}%")

            # Clean up non-best model checkpoints to save disk
            if result["name"] != best_name:
                ckpt = save_dir / f"{config.name}_best.pt"
                if ckpt.exists():
                    ckpt.unlink()

        except Exception as e:
            print(f"  FAILED: {e}")
            with open(results_path, "a") as f:
                f.write(f"{config.name}\t{config.arch}\t{config.hidden_dim}\t"
                        f"{config.num_layers}\t{config.lr}\t{config.dropout}\t"
                        f"0\t999.0\t999.0\t0.0\n")

        # GPU cleanup between experiments
        torch.cuda.empty_cache()

    # Final summary
    total_hours = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print(f"EXPERIMENT LOOP COMPLETE")
    print(f"{'='*60}")
    print(f"Configs tested: {len(configs) - len(existing)}")
    print(f"Total time: {total_hours:.1f}h (${total_hours * 0.26:.2f})")
    print(f"Best: {best_name} with CER={best_cer:.1f}%")
    print(f"Results: {args.results_file}")
    print(f"{'='*60}")

    # Print sorted leaderboard
    print(f"\nLEADERBOARD:")
    results = []
    with open(results_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 9 and parts[8] != "999.0":
                results.append((parts[0], float(parts[8]), int(parts[6]), parts[1]))
    results.sort(key=lambda x: x[1])
    for rank, (name, cer, params, arch) in enumerate(results[:10], 1):
        print(f"  #{rank} CER={cer:.1f}% {name} ({params:,} params)")


if __name__ == "__main__":
    main()
