#!/usr/bin/env python3
"""
N'Ko ASR V5 Evaluation Harness
================================
Comprehensive evaluation across all model versions (V3, V4, V5)
with beam search decoding, statistical significance tests, and
publication-ready comparison tables.

Metrics:
  - WER: Word Error Rate (on Latin-transliterated output)
  - CER: Character Error Rate
  - SVR: Syllable Validity Rate (N'Ko phonotactic compliance)
  - Confidence: Mean CTC output confidence
  - Per-phoneme accuracy: broken down by vowel/consonant/tone/nasal
  - Length correlation: Pearson r between utterance length and CER
  - Confidence calibration: Spearman rho between confidence and CER

Statistical tests:
  - Paired t-test on CER (V5 vs V3, V5 vs V4)
  - Sign test: per-sample win/loss/tie counts
  - Bootstrap 95% CI on CER improvement

Outputs:
  - Console summary table
  - JSON results file (all per-sample + aggregate metrics)
  - W&B logging (optional)
  - LaTeX/Markdown comparison table for the paper

Usage:
    # Full 500-sample eval, all models
    python3 asr/eval_v5.py \\
        --v3-model /workspace/best_v3_asr.pt \\
        --v4-model /workspace/whisper_lora/best_whisper_lora.pt \\
        --v5-model /workspace/v5_model/best_v5.pt \\
        --test-dir /workspace/v5_data/test_audio/ \\
        --output /workspace/v5_eval_results.json

    # V5 only
    python3 asr/eval_v5.py \\
        --v5-model /workspace/v5_model/best_v5.pt \\
        --test-manifest /workspace/v5_data/manifests/test.jsonl \\
        --features-dir /workspace/v5_data/features/

    # With beam search
    python3 asr/eval_v5.py \\
        --v5-model /workspace/v5_model/best_v5.pt \\
        --test-dir /workspace/test/ \\
        --beam-width 5
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from constrained.nko_fsm import NKoSyllableFSM, FSMState
    FSM_AVAILABLE = True
except ImportError:
    FSM_AVAILABLE = False

try:
    from asr.beam_search_decoder import BeamSearchDecoder, build_decoder
    BEAM_AVAILABLE = True
except ImportError:
    BEAM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Character Vocab ──────────────────────────────────────────────

def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ── N'Ko Character Classification ──────────────────────────────

NKO_VOWELS = {'\u07CA', '\u07CB', '\u07CC', '\u07CD', '\u07CE', '\u07CF', '\u07D0'}
NKO_CONSONANTS = {
    '\u07D1', '\u07D2', '\u07D3', '\u07D4', '\u07D5', '\u07D6', '\u07D7', '\u07D8',
    '\u07D9', '\u07DA', '\u07DB', '\u07DC', '\u07DD', '\u07DE', '\u07DF', '\u07E0',
    '\u07E1', '\u07E2', '\u07E3', '\u07E4', '\u07E5', '\u07E6', '\u07E7',
}
NKO_TONES = {'\u07EB', '\u07EC', '\u07ED', '\u07EE', '\u07EF'}
NKO_NASALS = {'\u07F2', '\u07F3'}
NKO_DIGITS = set(chr(cp) for cp in range(0x07C0, 0x07CA))

NKO_TO_LATIN = {
    '\u07ca': 'a', '\u07cb': 'ee', '\u07cc': 'i', '\u07cd': 'e', '\u07ce': 'u',
    '\u07cf': 'o', '\u07d0': '\u025b',
    '\u07d2': 'ng', '\u07d3': 'b', '\u07d4': 'p', '\u07d5': 't', '\u07d6': 'j',
    '\u07d7': 'c', '\u07d8': 'd', '\u07d9': 'r', '\u07db': 's', '\u07dc': 'g',
    '\u07dd': 'f', '\u07de': 'k', '\u07df': 'l', '\u07e1': 'm', '\u07e2': 'ny',
    '\u07e3': 'n', '\u07e4': 'h', '\u07e5': 'w', '\u07e6': 'y', '\u07e7': 'ng',
}


def char_class(c: str) -> str:
    if c in NKO_VOWELS:
        return "vowel"
    if c in NKO_CONSONANTS:
        return "consonant"
    if c in NKO_TONES:
        return "tone"
    if c in NKO_NASALS:
        return "nasal"
    if c in NKO_DIGITS:
        return "digit"
    if c == ' ':
        return "space"
    if 0x07C0 <= ord(c) <= 0x07FF:
        return "other_nko"
    return "non_nko"


def is_nko(c: str) -> bool:
    return 0x07C0 <= ord(c) <= 0x07FF


def nko_to_latin(nko_text: str) -> str:
    result = []
    for c in nko_text:
        if c in NKO_TO_LATIN:
            result.append(NKO_TO_LATIN[c])
        elif c == ' ':
            result.append(' ')
    return ''.join(result)


# ── Metrics ─────────────────────────────────────────────────────

def edit_distance(ref, hyp) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]


def word_error_rate(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return min(edit_distance(ref_words, hyp_words) / len(ref_words), 1.0)


def char_error_rate(ref: str, hyp: str) -> float:
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return min(edit_distance(ref_chars, hyp_chars) / len(ref_chars), 1.0)


def validate_syllables(nko_text: str) -> float:
    """Return syllable validity rate for N'Ko text."""
    syllables = []
    current_onset = ""
    current_nucleus = ""

    def flush():
        nonlocal current_onset, current_nucleus
        if current_nucleus:
            syllables.append("valid")
        elif current_onset:
            syllables.append("invalid")
        current_onset = ""
        current_nucleus = ""

    for c in nko_text:
        if c == ' ':
            flush()
            continue
        if not is_nko(c):
            continue
        cls = char_class(c)
        if cls == "consonant":
            if current_nucleus:
                flush()
            current_onset += c
        elif cls == "vowel":
            current_nucleus += c
        elif cls == "nasal":
            current_nucleus += c  # nasal mark on vowel
            flush()

    flush()
    total = len(syllables)
    valid = sum(1 for s in syllables if s == "valid")
    return valid / max(total, 1)


def per_phoneme_errors(ref_nko: str, hyp_nko: str) -> Dict[str, Dict[str, int]]:
    """Break down character errors by N'Ko character class."""
    ref_chars = [c for c in ref_nko if is_nko(c) or c == ' ']
    hyp_chars = [c for c in hyp_nko if is_nko(c) or c == ' ']

    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    errors = {
        cls: {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0}
        for cls in ["vowel", "consonant", "tone", "nasal", "digit", "space"]
    }

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref_chars[i-1] == hyp_chars[j-1] else 1):
            cls = char_class(ref_chars[i-1])
            if cls not in errors:
                cls = "vowel"
            if ref_chars[i-1] == hyp_chars[j-1]:
                errors[cls]["correct"] += 1
            else:
                errors[cls]["substitution"] += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            cls = char_class(ref_chars[i-1])
            if cls not in errors:
                cls = "vowel"
            errors[cls]["deletion"] += 1
            i -= 1
        else:
            cls = char_class(hyp_chars[j-1]) if j > 0 else "vowel"
            if cls not in errors:
                cls = "vowel"
            errors[cls]["insertion"] += 1
            j -= 1

    return errors


# ── Statistical Tests ───────────────────────────────────────────

def pearson_r(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx)**2 for xi in x) / (n - 1)) if n > 1 else 0
    sy = math.sqrt(sum((yi - my)**2 for yi in y) / (n - 1)) if n > 1 else 0
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sx * sy)


def spearman_r(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0

    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j+1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    return pearson_r(rank(x), rank(y))


def paired_t_test(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Two-sided paired t-test. Returns (t_statistic, p_value)."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    diffs = [xi - yi for xi, yi in zip(x, y)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d)**2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) if var_d > 0 else 1e-10
    t_stat = mean_d / se

    # Approximate p-value using normal distribution (large n)
    # For n > 30 this is accurate enough
    p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
    return t_stat, p_value


def sign_test(x: List[float], y: List[float]) -> Dict:
    """Sign test: count per-sample wins/losses/ties."""
    wins = sum(1 for xi, yi in zip(x, y) if xi < yi)
    losses = sum(1 for xi, yi in zip(x, y) if xi > yi)
    ties = sum(1 for xi, yi in zip(x, y) if xi == yi)
    total = len(x)
    return {
        "x_wins": wins,
        "y_wins": losses,
        "ties": ties,
        "total": total,
        "x_win_rate": wins / max(total, 1),
    }


def bootstrap_ci(
    x: List[float], y: List[float],
    n_bootstrap: int = 10000, ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI on mean(x - y). Returns (mean, lower, upper)."""
    rng = np.random.RandomState(seed)
    diffs = np.array(x) - np.array(y)
    n = len(diffs)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lower = boot_means[int(alpha * n_bootstrap)]
    upper = boot_means[int((1 - alpha) * n_bootstrap)]
    mean = np.mean(diffs)

    return float(mean), float(lower), float(upper)


def _norm_cdf(x):
    """Approximate normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ── Model Loading ──────────────────────────────────────────────

class CharASR(nn.Module):
    """CTC Head matching V3/V4/V5 architecture."""
    def __init__(self, num_chars, input_dim=1280, hidden_dim=768, num_layers=6, nhead=12, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.temporal_ds = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2), nn.GELU())
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
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


class LoRALinear(nn.Module):
    """LoRA adapter for loading V4/V5 checkpoints."""
    def __init__(self, original: nn.Linear, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.original(x) + (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scale


def load_v3_model(model_path, device):
    """Load V3 base CTC head (no LoRA)."""
    import whisper
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    whisper_model = whisper.load_model("large-v3", device=device)
    whisper_model.eval()

    ctc_head = CharASR(num_chars).to(device)
    if model_path and Path(model_path).exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "head_state" in state:
            ctc_head.load_state_dict(state["head_state"])
        else:
            ctc_head.load_state_dict(state)
        print(f"  V3 CTC head loaded from {model_path}")
    else:
        print(f"  WARNING: V3 model not found at {model_path}")
    ctc_head.eval()

    return whisper_model, ctc_head, idx_to_char, num_chars


def load_v4_model(model_path, device, lora_rank=16, lora_layers=8):
    """Load V4 model (LoRA checkpoint)."""
    import whisper
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    whisper_model = whisper.load_model("large-v3", device=device)

    # Apply LoRA structure (needed for loading weights)
    total_layers = len(whisper_model.encoder.blocks)
    start = total_layers - lora_layers
    for i in range(start, total_layers):
        block = whisper_model.encoder.blocks[i]
        for name in ['query', 'key', 'value', 'out']:
            attr = getattr(block.attn, name, None)
            if attr is not None and isinstance(attr, nn.Linear):
                setattr(block.attn, name, LoRALinear(attr, rank=lora_rank, alpha=lora_rank*2))

    ctc_head = CharASR(num_chars).to(device)

    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "lora_state" in ckpt:
            model_params = dict(whisper_model.named_parameters())
            for name, data in ckpt["lora_state"].items():
                if name in model_params:
                    model_params[name].data.copy_(data)
        if "head_state" in ckpt:
            ctc_head.load_state_dict(ckpt["head_state"])
        print(f"  V4 model loaded from {model_path}")
    else:
        print(f"  WARNING: V4 model not found at {model_path}")

    whisper_model.eval()
    ctc_head.eval()
    return whisper_model, ctc_head, idx_to_char, num_chars


def load_v5_model(model_path, device, lora_rank=64, lora_layers=32):
    """Load V5 model (full LoRA checkpoint)."""
    import whisper
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    whisper_model = whisper.load_model("large-v3", device=device)

    # Apply LoRA structure matching V5 config
    total_layers = len(whisper_model.encoder.blocks)
    start = max(0, total_layers - lora_layers)
    for i in range(start, total_layers):
        block = whisper_model.encoder.blocks[i]
        for name in ['query', 'key', 'value', 'out']:
            attr = getattr(block.attn, name, None)
            if attr is not None and isinstance(attr, nn.Linear):
                setattr(block.attn, name, LoRALinear(attr, rank=lora_rank, alpha=lora_rank*2, dropout=0.1))
        for mlp_name in ["0", "2"]:
            if hasattr(block.mlp, mlp_name):
                layer = getattr(block.mlp, mlp_name)
                if isinstance(layer, nn.Linear):
                    setattr(block.mlp, mlp_name, LoRALinear(layer, rank=lora_rank, alpha=lora_rank*2, dropout=0.1))

    ctc_head = CharASR(num_chars).to(device)

    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "lora_state" in ckpt:
            model_params = dict(whisper_model.named_parameters())
            loaded = 0
            for name, data in ckpt["lora_state"].items():
                if name in model_params:
                    model_params[name].data.copy_(data)
                    loaded += 1
            print(f"  V5 LoRA: {loaded} parameters loaded")
        if "head_state" in ckpt:
            ctc_head.load_state_dict(ckpt["head_state"])
            print(f"  V5 CTC head loaded")
    else:
        print(f"  WARNING: V5 model not found at {model_path}")

    whisper_model.eval()
    ctc_head.eval()
    return whisper_model, ctc_head, idx_to_char, num_chars


# ── CTC Decode ─────────────────────────────────────────────────

def ctc_greedy_decode(logits, idx_to_char, num_chars):
    """CTC greedy decode. Returns (text, confidence, per_char_confs)."""
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    confidences = []
    prev = -1
    for i, idx in enumerate(pred_ids):
        if idx != num_chars and idx != prev:
            char = idx_to_char.get(idx, "")
            if char:
                decoded.append(char)
                confidences.append(probs[i, idx].item())
        prev = idx
    avg_conf = sum(confidences) / max(len(confidences), 1)
    return "".join(decoded), avg_conf, confidences


# ── Transcription ──────────────────────────────────────────────

def transcribe_audio(whisper_model, ctc_head, audio_path, idx_to_char, num_chars, device, beam_decoder=None):
    """Transcribe a single audio file."""
    import whisper as whisper_lib

    audio = whisper_lib.load_audio(str(audio_path))
    audio = whisper_lib.pad_or_trim(audio)
    mel = whisper_lib.log_mel_spectrogram(audio, n_mels=128).to(device)

    t0 = time.time()
    with torch.no_grad():
        features = whisper_model.encoder(mel.unsqueeze(0))
        logits = ctc_head(features)

        if beam_decoder is not None:
            results = beam_decoder.decode(logits[0], idx_to_char, num_chars)
            nko_text = results[0][0] if results else ""
            confidence = results[0][1] if results else 0.0
            per_char_confs = []
        else:
            nko_text, confidence, per_char_confs = ctc_greedy_decode(logits[0], idx_to_char, num_chars)

    latency = (time.time() - t0) * 1000
    latin_text = nko_to_latin(nko_text)
    return nko_text, latin_text, confidence, latency, per_char_confs


def transcribe_features(ctc_head, features, idx_to_char, num_chars, device, beam_decoder=None):
    """Transcribe from pre-extracted features."""
    features = features.unsqueeze(0).to(device) if features.dim() == 2 else features.to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = ctc_head(features)

        if beam_decoder is not None:
            results = beam_decoder.decode(logits[0], idx_to_char, num_chars)
            nko_text = results[0][0] if results else ""
            confidence = results[0][1] if results else 0.0
            per_char_confs = []
        else:
            nko_text, confidence, per_char_confs = ctc_greedy_decode(logits[0], idx_to_char, num_chars)

    latency = (time.time() - t0) * 1000
    latin_text = nko_to_latin(nko_text)
    return nko_text, latin_text, confidence, latency, per_char_confs


# ── Test Set Loading ────────────────────────────────────────────

def load_test_set(test_dir=None, test_manifest=None, features_dir=None, max_samples=500):
    """Load test samples from directory or manifest.

    Returns list of dicts with keys: audio (path or None), features_path (or None),
    ref_nko, ref_latin.
    """
    samples = []

    if test_manifest and Path(test_manifest).exists():
        print(f"Loading test manifest: {test_manifest}")
        with open(test_manifest) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    sample = {
                        "ref_nko": entry.get("nko", ""),
                        "ref_latin": entry.get("bam", ""),
                        "feat_id": entry.get("feat_id", ""),
                    }
                    if features_dir:
                        feat_path = Path(features_dir) / f"{entry['feat_id']}.pt"
                        if feat_path.exists():
                            sample["features_path"] = str(feat_path)
                    samples.append(sample)
        print(f"  Loaded {len(samples)} from manifest")

    elif test_dir and Path(test_dir).exists():
        test_dir = Path(test_dir)
        # Check for manifest.jsonl
        manifest = test_dir / "manifest.jsonl"
        if manifest.exists():
            with open(manifest) as f:
                for line in f:
                    entry = json.loads(line)
                    audio_path = test_dir / entry.get("audio", entry.get("path", ""))
                    if audio_path.exists():
                        samples.append({
                            "audio": str(audio_path),
                            "ref_nko": entry.get("nko", entry.get("text", "")),
                            "ref_latin": entry.get("latin", entry.get("bambara", "")),
                        })
        else:
            for wav in sorted(test_dir.glob("*.wav")):
                txt = wav.with_suffix(".txt")
                ref = txt.read_text().strip() if txt.exists() else ""
                has_nko_ref = any(0x07C0 <= ord(c) <= 0x07FF for c in ref)
                samples.append({
                    "audio": str(wav),
                    "ref_nko": ref if has_nko_ref else "",
                    "ref_latin": ref if not has_nko_ref else nko_to_latin(ref),
                })
        print(f"  Loaded {len(samples)} from {test_dir}")

    # Cap at max_samples
    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]
        print(f"  Capped at {max_samples} samples")

    return samples


# ── Evaluation ─────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Per-sample evaluation result."""
    sample_id: int = 0
    ref_nko: str = ""
    ref_latin: str = ""
    hyp_nko: str = ""
    hyp_latin: str = ""
    wer: float = 0.0
    cer: float = 0.0
    svr: float = 0.0
    confidence: float = 0.0
    latency_ms: float = 0.0
    ref_len: int = 0


@dataclass
class ModelSummary:
    """Aggregate model evaluation summary."""
    model_name: str = ""
    version: str = ""
    num_samples: int = 0
    avg_wer: float = 0.0
    avg_cer: float = 0.0
    avg_svr: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    median_cer: float = 0.0
    std_cer: float = 0.0
    length_cer_pearson: float = 0.0
    conf_cer_spearman: float = 0.0
    phoneme_errors: Dict = field(default_factory=dict)
    results: List[SampleResult] = field(default_factory=list)


def evaluate_model(
    model_name: str,
    whisper_model,
    ctc_head,
    idx_to_char: dict,
    num_chars: int,
    test_samples: list,
    device: str,
    beam_decoder=None,
    features_dir: str = None,
) -> ModelSummary:
    """Run full evaluation on test samples."""
    summary = ModelSummary(model_name=model_name)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    agg_phoneme = {
        cls: {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0}
        for cls in ["vowel", "consonant", "tone", "nasal", "digit", "space"]
    }

    for i, sample in enumerate(test_samples):
        # Transcribe
        if "features_path" in sample and sample["features_path"]:
            features = torch.load(sample["features_path"], weights_only=True).float()
            nko, latin, conf, latency, _ = transcribe_features(
                ctc_head, features, idx_to_char, num_chars, device, beam_decoder
            )
        elif "audio" in sample and sample["audio"]:
            nko, latin, conf, latency, _ = transcribe_audio(
                whisper_model, ctc_head, sample["audio"], idx_to_char, num_chars, device, beam_decoder
            )
        else:
            continue

        result = SampleResult(
            sample_id=i,
            ref_nko=sample.get("ref_nko", ""),
            ref_latin=sample.get("ref_latin", ""),
            hyp_nko=nko,
            hyp_latin=latin,
            confidence=conf,
            latency_ms=latency,
        )

        # Compute metrics
        if result.ref_latin:
            result.wer = word_error_rate(result.ref_latin, result.hyp_latin)
            result.cer = char_error_rate(result.ref_latin, result.hyp_latin)
            result.ref_len = len(result.ref_latin.strip().split())

        if nko:
            result.svr = validate_syllables(nko)

        # Per-phoneme errors (on N'Ko if ref available)
        if result.ref_nko and result.hyp_nko:
            pe = per_phoneme_errors(result.ref_nko, result.hyp_nko)
            for cls in agg_phoneme:
                for key in agg_phoneme[cls]:
                    agg_phoneme[cls][key] += pe.get(cls, {}).get(key, 0)

        summary.results.append(result)

        # Progress
        if (i + 1) % 50 == 0 or i < 5:
            ref_disp = (result.ref_latin or result.ref_nko or "?")[:40]
            hyp_disp = result.hyp_latin[:40]
            print(
                f"  [{i+1}/{len(test_samples)}] "
                f"CER={result.cer:.2f} WER={result.wer:.2f} SVR={result.svr:.2f} "
                f"conf={result.confidence:.3f}",
                flush=True,
            )

    # Aggregate
    summary.num_samples = len(summary.results)
    if summary.results:
        scored = [r for r in summary.results if r.ref_latin]
        if scored:
            summary.avg_wer = sum(r.wer for r in scored) / len(scored)
            summary.avg_cer = sum(r.cer for r in scored) / len(scored)
            cers = sorted([r.cer for r in scored])
            summary.median_cer = cers[len(cers) // 2]
            summary.std_cer = float(np.std([r.cer for r in scored]))

        summary.avg_confidence = sum(r.confidence for r in summary.results) / len(summary.results)
        summary.avg_latency_ms = sum(r.latency_ms for r in summary.results) / len(summary.results)

        svr_results = [r for r in summary.results if r.hyp_nko]
        if svr_results:
            summary.avg_svr = sum(r.svr for r in svr_results) / len(svr_results)

        if scored and len(scored) >= 3:
            lengths = [float(r.ref_len) for r in scored]
            cers_list = [r.cer for r in scored]
            summary.length_cer_pearson = pearson_r(lengths, cers_list)
            confs = [r.confidence for r in scored]
            summary.conf_cer_spearman = spearman_r(confs, cers_list)

    summary.phoneme_errors = agg_phoneme

    # Print summary
    print(f"\n--- {model_name} ---")
    print(f"  Samples: {summary.num_samples}")
    print(f"  WER:  {summary.avg_wer:.4f}")
    print(f"  CER:  {summary.avg_cer:.4f} (median={summary.median_cer:.4f}, std={summary.std_cer:.4f})")
    print(f"  SVR:  {summary.avg_svr:.4f}")
    print(f"  Conf: {summary.avg_confidence:.4f}")
    print(f"  Latency: {summary.avg_latency_ms:.0f}ms")
    print(f"  Length-CER Pearson: {summary.length_cer_pearson:+.3f}")
    print(f"  Conf-CER Spearman: {summary.conf_cer_spearman:+.3f}")

    # Per-phoneme summary
    print(f"\n  Per-phoneme accuracy:")
    for cls in ["vowel", "consonant", "tone", "nasal"]:
        stats = agg_phoneme[cls]
        total = stats["correct"] + stats["substitution"] + stats["deletion"]
        acc = stats["correct"] / max(total, 1)
        print(f"    {cls:12s}: {acc:.3f} ({stats['correct']}/{total})")

    return summary


# ── Comparison ─────────────────────────────────────────────────

def print_comparison(summaries: List[ModelSummary]):
    """Print comparison table across all models."""
    if len(summaries) < 2:
        return

    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    headers = ["Metric"] + [s.model_name for s in summaries]
    col_width = max(12, max(len(h) for h in headers) + 2)
    header_line = "  " + "".join(h.center(col_width) for h in headers)
    print(header_line)
    print("  " + "-" * (col_width * len(headers)))

    metrics = [
        ("WER", [s.avg_wer for s in summaries]),
        ("CER", [s.avg_cer for s in summaries]),
        ("CER median", [s.median_cer for s in summaries]),
        ("CER std", [s.std_cer for s in summaries]),
        ("SVR", [s.avg_svr for s in summaries]),
        ("Confidence", [s.avg_confidence for s in summaries]),
        ("Latency (ms)", [s.avg_latency_ms for s in summaries]),
        ("Len-CER r", [s.length_cer_pearson for s in summaries]),
        ("Conf-CER rho", [s.conf_cer_spearman for s in summaries]),
    ]

    for name, values in metrics:
        row = f"  {name:<{col_width}}"
        for v in values:
            row += f"{v:>{col_width}.4f}"
        print(row)

    # Pairwise statistical tests (V5 vs each)
    v5 = summaries[-1]  # Assume last is V5
    v5_cers = [r.cer for r in v5.results if r.ref_latin]

    print(f"\n  Statistical Significance (paired tests vs {v5.model_name}):")
    for other in summaries[:-1]:
        other_cers = [r.cer for r in other.results if r.ref_latin]
        if len(v5_cers) != len(other_cers):
            # Try to align by sample count
            min_n = min(len(v5_cers), len(other_cers))
            v5_cers_aligned = v5_cers[:min_n]
            other_cers_aligned = other_cers[:min_n]
        else:
            v5_cers_aligned = v5_cers
            other_cers_aligned = other_cers

        if len(v5_cers_aligned) >= 3:
            t_stat, p_val = paired_t_test(v5_cers_aligned, other_cers_aligned)
            sign = sign_test(v5_cers_aligned, other_cers_aligned)
            mean_diff, ci_lo, ci_hi = bootstrap_ci(v5_cers_aligned, other_cers_aligned)

            print(f"\n    {v5.model_name} vs {other.model_name}:")
            print(f"      Paired t-test: t={t_stat:+.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
            print(f"      Sign test: V5 wins={sign['x_wins']}, Other wins={sign['y_wins']}, ties={sign['ties']}")
            print(f"      CER diff: {mean_diff:+.4f} (95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}])")

    # Per-sample win table (V5 vs best previous)
    if len(summaries) >= 2:
        prev_best = summaries[-2]
        print(f"\n  Per-sample breakdown: {v5.model_name} vs {prev_best.model_name}")
        v5_wins = 0
        prev_wins = 0
        ties = 0
        min_n = min(len(v5.results), len(prev_best.results))
        for i in range(min_n):
            v5_r = v5.results[i]
            prev_r = prev_best.results[i]
            if v5_r.cer < prev_r.cer:
                v5_wins += 1
            elif v5_r.cer > prev_r.cer:
                prev_wins += 1
            else:
                ties += 1
        print(f"    V5 wins: {v5_wins}/{min_n} ({v5_wins/max(min_n,1)*100:.0f}%)")
        print(f"    Prev wins: {prev_wins}/{min_n} ({prev_wins/max(min_n,1)*100:.0f}%)")
        print(f"    Ties: {ties}/{min_n}")


def generate_paper_table(summaries: List[ModelSummary]) -> str:
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{N'Ko ASR Model Comparison}")
    lines.append("\\label{tab:asr-comparison}")

    cols = "l" + "c" * len(summaries)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Metric & " + " & ".join(s.model_name for s in summaries) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    metrics = [
        ("WER $\\downarrow$", [f"{s.avg_wer:.3f}" for s in summaries]),
        ("CER $\\downarrow$", [f"{s.avg_cer:.3f}" for s in summaries]),
        ("SVR $\\uparrow$", [f"{s.avg_svr:.3f}" for s in summaries]),
        ("Confidence $\\uparrow$", [f"{s.avg_confidence:.3f}" for s in summaries]),
        ("Latency (ms) $\\downarrow$", [f"{s.avg_latency_ms:.0f}" for s in summaries]),
        ("Samples", [str(s.num_samples) for s in summaries]),
    ]

    for name, values in metrics:
        row = f"{name} & " + " & ".join(values) + " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_markdown_table(summaries: List[ModelSummary]) -> str:
    """Generate Markdown table for README/paper."""
    lines = []
    headers = ["Metric"] + [s.model_name for s in summaries]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    metrics = [
        ("WER", [f"{s.avg_wer:.3f}" for s in summaries]),
        ("CER", [f"{s.avg_cer:.3f}" for s in summaries]),
        ("SVR", [f"{s.avg_svr:.3f}" for s in summaries]),
        ("Confidence", [f"{s.avg_confidence:.3f}" for s in summaries]),
        ("Latency (ms)", [f"{s.avg_latency_ms:.0f}" for s in summaries]),
        ("Samples", [str(s.num_samples) for s in summaries]),
    ]

    for name, values in metrics:
        lines.append("| " + name + " | " + " | ".join(values) + " |")

    return "\n".join(lines)


# ── JSON Output ────────────────────────────────────────────────

def build_json_output(summaries: List[ModelSummary]) -> dict:
    """Build comprehensive JSON output."""
    output = {
        "timestamp": datetime.utcnow().isoformat() if 'datetime' in dir() else time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {},
    }

    for s in summaries:
        model_data = {
            "name": s.model_name,
            "num_samples": s.num_samples,
            "wer": round(s.avg_wer, 4),
            "cer": round(s.avg_cer, 4),
            "cer_median": round(s.median_cer, 4),
            "cer_std": round(s.std_cer, 4),
            "svr": round(s.avg_svr, 4),
            "confidence": round(s.avg_confidence, 4),
            "latency_ms": round(s.avg_latency_ms, 1),
            "length_cer_pearson": round(s.length_cer_pearson, 4),
            "conf_cer_spearman": round(s.conf_cer_spearman, 4),
            "phoneme_errors": s.phoneme_errors,
            "samples": [
                {
                    "id": r.sample_id,
                    "ref_latin": r.ref_latin,
                    "hyp_latin": r.hyp_latin,
                    "hyp_nko": r.hyp_nko,
                    "wer": round(r.wer, 4),
                    "cer": round(r.cer, 4),
                    "svr": round(r.svr, 4),
                    "confidence": round(r.confidence, 4),
                    "latency_ms": round(r.latency_ms, 1),
                }
                for r in s.results
            ],
        }
        output["models"][s.model_name] = model_data

    # Pairwise comparisons
    if len(summaries) >= 2:
        output["comparisons"] = {}
        v5 = summaries[-1]
        v5_cers = [r.cer for r in v5.results if r.ref_latin]

        for other in summaries[:-1]:
            other_cers = [r.cer for r in other.results if r.ref_latin]
            min_n = min(len(v5_cers), len(other_cers))
            if min_n >= 3:
                v5_a = v5_cers[:min_n]
                other_a = other_cers[:min_n]
                t_stat, p_val = paired_t_test(v5_a, other_a)
                sign = sign_test(v5_a, other_a)
                mean_diff, ci_lo, ci_hi = bootstrap_ci(v5_a, other_a)

                output["comparisons"][f"{v5.model_name}_vs_{other.model_name}"] = {
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_val, 6),
                    "significant": p_val < 0.05,
                    "sign_test": sign,
                    "cer_diff_mean": round(mean_diff, 4),
                    "cer_diff_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                }

    # Tables
    output["latex_table"] = generate_paper_table(summaries)
    output["markdown_table"] = generate_markdown_table(summaries)

    return output


# ── Main ────────────────────────────────────────────────────────

def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(description="N'Ko ASR V5 Evaluation Harness")
    parser.add_argument("--v3-model", default=None, help="V3 base CTC head (.pt)")
    parser.add_argument("--v4-model", default=None, help="V4 LoRA checkpoint (.pt)")
    parser.add_argument("--v5-model", default=None, help="V5 full LoRA checkpoint (.pt)")
    parser.add_argument("--test-dir", default=None, help="Directory with test audio (.wav)")
    parser.add_argument("--test-manifest", default=None, help="Test manifest JSONL")
    parser.add_argument("--features-dir", default=None, help="Pre-extracted features directory")
    parser.add_argument("--max-samples", type=int, default=500, help="Max test samples")
    parser.add_argument("--beam-width", type=int, default=0, help="Beam width (0=greedy)")
    parser.add_argument("--use-fsm", action="store_true", help="Use FSM in beam search")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output", default=None, help="Save results JSON to path")
    parser.add_argument("--wandb-project", default=None, help="W&B project for logging")

    # V4/V5 LoRA config (for loading)
    parser.add_argument("--v4-lora-rank", type=int, default=16)
    parser.add_argument("--v4-lora-layers", type=int, default=8)
    parser.add_argument("--v5-lora-rank", type=int, default=64)
    parser.add_argument("--v5-lora-layers", type=int, default=32)

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"{'='*70}")
    print(f"N'Ko ASR V5 Evaluation Harness")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load test set
    test_samples = load_test_set(
        test_dir=args.test_dir,
        test_manifest=args.test_manifest,
        features_dir=args.features_dir,
        max_samples=args.max_samples,
    )

    if not test_samples:
        print("ERROR: No test samples. Use --test-dir or --test-manifest")
        sys.exit(1)

    print(f"Test set: {len(test_samples)} samples")

    # Beam search decoder
    beam_decoder = None
    if args.beam_width > 0:
        if BEAM_AVAILABLE:
            fsm = NKoSyllableFSM() if args.use_fsm and FSM_AVAILABLE else None
            beam_decoder = BeamSearchDecoder(beam_width=args.beam_width, fsm=fsm)
            print(f"Beam search: width={args.beam_width}, FSM={'ON' if fsm else 'OFF'}")
        else:
            print("WARNING: beam_search_decoder not available, using greedy")

    # W&B
    if args.wandb_project and WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        wandb.init(project=args.wandb_project, name="v5-eval", config=vars(args))

    # Evaluate models
    summaries = []

    # V3
    if args.v3_model:
        print("\nLoading V3 model...")
        whisper_v3, ctc_v3, idx_to_char, num_chars = load_v3_model(args.v3_model, device)
        summary_v3 = evaluate_model(
            "V3 Base", whisper_v3, ctc_v3, idx_to_char, num_chars,
            test_samples, device, beam_decoder, args.features_dir,
        )
        summaries.append(summary_v3)
        del whisper_v3, ctc_v3
        if device == "cuda":
            torch.cuda.empty_cache()

    # V4
    if args.v4_model:
        print("\nLoading V4 model...")
        whisper_v4, ctc_v4, idx_to_char, num_chars = load_v4_model(
            args.v4_model, device, args.v4_lora_rank, args.v4_lora_layers,
        )
        summary_v4 = evaluate_model(
            "V4 LoRA", whisper_v4, ctc_v4, idx_to_char, num_chars,
            test_samples, device, beam_decoder, args.features_dir,
        )
        summaries.append(summary_v4)
        del whisper_v4, ctc_v4
        if device == "cuda":
            torch.cuda.empty_cache()

    # V5
    if args.v5_model:
        print("\nLoading V5 model...")
        whisper_v5, ctc_v5, idx_to_char, num_chars = load_v5_model(
            args.v5_model, device, args.v5_lora_rank, args.v5_lora_layers,
        )
        summary_v5 = evaluate_model(
            "V5 Full-Scale", whisper_v5, ctc_v5, idx_to_char, num_chars,
            test_samples, device, beam_decoder, args.features_dir,
        )
        summaries.append(summary_v5)
        del whisper_v5, ctc_v5
        if device == "cuda":
            torch.cuda.empty_cache()

    if not summaries:
        print("ERROR: No models specified. Use --v3-model, --v4-model, or --v5-model")
        sys.exit(1)

    # Print comparison
    if len(summaries) >= 2:
        print_comparison(summaries)

    # Generate tables
    print(f"\n{'='*60}")
    print("Markdown Table:")
    print(generate_markdown_table(summaries))
    print()
    print("LaTeX Table:")
    print(generate_paper_table(summaries))

    # Save JSON output
    if args.output:
        results = build_json_output(summaries)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    # W&B logging
    if args.wandb_project and WANDB_AVAILABLE:
        for s in summaries:
            wandb.log({
                f"eval/{s.model_name}/wer": s.avg_wer,
                f"eval/{s.model_name}/cer": s.avg_cer,
                f"eval/{s.model_name}/svr": s.avg_svr,
                f"eval/{s.model_name}/confidence": s.avg_confidence,
            })
        wandb.finish()


if __name__ == "__main__":
    main()
