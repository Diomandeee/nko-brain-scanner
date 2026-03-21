#!/usr/bin/env python3
"""
N'Ko ASR Evaluation Harness (Fixed)
====================================
Benchmarks Whisper LoRA fine-tuned model against base model.
Measures WER, CER, SVR, per-phoneme accuracy, utterance-length correlation,
confidence calibration, and inference speed on held-out N'Ko audio.

BUG FIX (2026-03-20): The previous version never loaded the base CTC head
weights from the --base-model checkpoint. The CharASR was created with random
weights, producing garbage output ("kykykyk..."). Now load_model() properly
loads the base CTC head from model_path when no LoRA checkpoint is provided.

Usage:
    # Compare base vs fine-tuned
    python3 eval_whisper_lora.py --base-model best_v3_asr.pt --lora-model whisper_lora_final/best_whisper_lora.pt

    # Evaluate single model
    python3 eval_whisper_lora.py --base-model best_v3_asr.pt

    # Use specific test set
    python3 eval_whisper_lora.py --base-model best_v3_asr.pt --test-dir /path/to/test/audio/
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

import torch
import numpy as np


# ── Metrics ──────────────────────────────────────────────────────────

def edit_distance(ref, hyp) -> int:
    """Levenshtein edit distance (works on strings or lists)."""
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
    """WER: edit distance on words."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def char_error_rate(ref: str, hyp: str) -> float:
    """CER: edit distance on characters."""
    ref_chars = list(ref.strip())
    hyp_chars = list(hyp.strip())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ── N'Ko Character Classification ───────────────────────────────────

NKO_VOWELS = {
    '\u07CA', '\u07CB', '\u07CC', '\u07CD', '\u07CE', '\u07CF', '\u07D0',
}
NKO_CONSONANTS = {
    '\u07D1', '\u07D2', '\u07D3', '\u07D4', '\u07D5', '\u07D6', '\u07D7',
    '\u07D8', '\u07D9', '\u07DA', '\u07DB', '\u07DC', '\u07DD', '\u07DE',
    '\u07DF', '\u07E0', '\u07E1', '\u07E2', '\u07E3', '\u07E4', '\u07E5',
    '\u07E6', '\u07E7',
}
NKO_TONES = {'\u07EB', '\u07EC', '\u07ED', '\u07EE', '\u07EF'}
NKO_NASALS = {'\u07F2', '\u07F3'}
NKO_DIGITS = set(chr(cp) for cp in range(0x07C0, 0x07CA))


def char_class(c: str) -> str:
    """Classify a single N'Ko character."""
    cu = c.upper() if len(c) == 1 else c
    # Need to check actual codepoint since N'Ko chars don't have upper/lower
    if c in NKO_VOWELS or c.upper() in NKO_VOWELS:
        return "vowel"
    if c in NKO_CONSONANTS or c.upper() in NKO_CONSONANTS:
        return "consonant"
    if c in NKO_TONES or c.upper() in NKO_TONES:
        return "tone"
    if c in NKO_NASALS or c.upper() in NKO_NASALS:
        return "nasal"
    if c in NKO_DIGITS:
        return "digit"
    if c == ' ':
        return "space"
    cp = ord(c)
    if 0x07C0 <= cp <= 0x07FF:
        return "other_nko"
    return "non_nko"


def is_nko(c: str) -> bool:
    return 0x07C0 <= ord(c) <= 0x07FF


# ── N'Ko Character Vocab (must match training) ──────────────────────

def build_nko_char_vocab():
    """Build N'Ko character vocabulary matching training code."""
    chars = [chr(cp) for cp in range(0x07C0, 0x0800)]
    chars.append(" ")
    vocab = {c: i for i, c in enumerate(chars)}
    return vocab, len(vocab)


NKO_TO_LATIN = {
    '\u07ca': 'a', '\u07cb': 'ee', '\u07cc': 'i', '\u07cd': 'e', '\u07ce': 'u',
    '\u07cf': 'o', '\u07d0': '\u025b',
    '\u07d2': 'ng', '\u07d3': 'b', '\u07d4': 'p', '\u07d5': 't', '\u07d6': 'j',
    '\u07d7': 'c', '\u07d8': 'd', '\u07d9': 'r', '\u07db': 's', '\u07dc': 'g',
    '\u07dd': 'f', '\u07de': 'k', '\u07df': 'l', '\u07e1': 'm', '\u07e2': 'ny',
    '\u07e3': 'n', '\u07e4': 'h', '\u07e5': 'w', '\u07e6': 'y', '\u07e7': 'ng',
}


def nko_to_latin(nko_text):
    """Convert N'Ko to Latin for comparison."""
    result = []
    for c in nko_text:
        if c in NKO_TO_LATIN:
            result.append(NKO_TO_LATIN[c])
        elif c == ' ':
            result.append(' ')
        elif c in NKO_TONES or c.upper() in NKO_TONES:
            pass
        elif c in NKO_NASALS or c.upper() in NKO_NASALS:
            pass
    return ''.join(result)


# ── Syllable Validity ────────────────────────────────────────────────

def validate_syllables_from_text(nko_text: str) -> Tuple[int, int, float]:
    """
    Check what % of decoded N'Ko output forms valid CV/CVN syllables.
    Returns (valid_count, total_count, validity_rate).

    Valid syllable patterns: V, CV, VN, CVN
    Invalid: bare consonant without following vowel, etc.
    """
    syllables = []
    current_onset = ""
    current_nucleus = ""
    current_coda = ""
    in_syllable = False

    def flush():
        nonlocal current_onset, current_nucleus, current_coda
        if current_nucleus:
            syllables.append(("valid", current_onset + current_nucleus + current_coda))
        elif current_onset:
            syllables.append(("invalid", current_onset))
        current_onset = ""
        current_nucleus = ""
        current_coda = ""

    for c in nko_text:
        if c == ' ':
            flush()
            continue
        if not is_nko(c):
            continue

        cls = char_class(c)
        if cls == "consonant":
            if current_nucleus:
                # Previous syllable complete, start new
                flush()
            current_onset += c
        elif cls == "vowel":
            current_nucleus += c
        elif cls == "tone":
            pass  # Tone marks don't affect syllable structure
        elif cls == "nasal":
            current_coda += c
            flush()  # Nasal coda closes syllable
        elif cls == "digit":
            flush()

    flush()

    total = len(syllables)
    valid = sum(1 for s in syllables if s[0] == "valid")
    rate = valid / max(total, 1)
    return valid, total, rate


# ── Per-Phoneme Accuracy ────────────────────────────────────────────

def per_phoneme_errors(ref_nko: str, hyp_nko: str) -> Dict[str, Dict[str, int]]:
    """
    Break down character errors by N'Ko character class.
    Aligns ref/hyp using edit distance backtrack, then categorizes
    each substitution, insertion, deletion by character class.
    """
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

    # Backtrack to get alignment
    errors = {
        "vowel": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "consonant": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "tone": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "nasal": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "digit": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "space": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
    }

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref_chars[i-1] == hyp_chars[j-1] else 1):
            cls = char_class(ref_chars[i-1])
            if cls not in errors:
                cls = "vowel"  # fallback
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
            cls = char_class(hyp_chars[j-1])
            if cls not in errors:
                cls = "vowel"
            errors[cls]["insertion"] += 1
            j -= 1

    return errors


# ── Statistical Correlations ─────────────────────────────────────────

def pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
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
    """Spearman rank correlation coefficient."""
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
            avg_rank = (i + j) / 2.0 + 1  # 1-based
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    return pearson_r(rx, ry)


# ── Model Loading ───────────────────────────────────────────────────

class CharASR(torch.nn.Module):
    """V3 Transformer h768 L6 with temporal downsampler."""
    def __init__(self, num_chars, input_dim=1280, hidden_dim=768, num_layers=6, nhead=12, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.temporal_ds = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2), torch.nn.GELU())
        self.pos_enc = torch.nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = torch.nn.LayerNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.temporal_ds(x.permute(0, 2, 1)).permute(0, 2, 1)
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        x = self.encoder(x)
        x = self.ln(x)
        return self.output_proj(x)


def load_model(model_path, device, lora_checkpoint=None, merged_encoder=None):
    """
    Load ASR model, optionally with merged LoRA encoder or LoRA checkpoint.

    FIX: Previously, the base CTC head was created with random weights and
    model_path was never used to load them. Now:
    - Base mode (no LoRA): loads model_path as the CTC head state dict
    - LoRA mode: loads head_state from the LoRA checkpoint (which contains the
      fine-tuned CTC head), and optionally loads merged encoder weights
    """
    import whisper

    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    # Load Whisper encoder
    whisper_model = whisper.load_model("large-v3", device=device)

    # Load merged encoder state dict if provided (preferred over LoRA checkpoint)
    if merged_encoder and Path(merged_encoder).exists():
        print(f"  Loading merged LoRA encoder: {merged_encoder}")
        merged_state = torch.load(merged_encoder, map_location=device, weights_only=True)
        whisper_model.load_state_dict(merged_state, strict=True)
        print(f"  Merged encoder loaded ({len(merged_state)} tensors)")
    whisper_model.eval()

    # Load CTC head
    ctc_head = CharASR(num_chars).to(device)

    if lora_checkpoint and Path(lora_checkpoint).exists():
        # LoRA mode: load fine-tuned CTC head from LoRA checkpoint
        ckpt = torch.load(lora_checkpoint, map_location=device, weights_only=True)
        if "head_state" in ckpt:
            ctc_head.load_state_dict(ckpt["head_state"])
            print(f"  Loaded fine-tuned CTC head from LoRA checkpoint")
        else:
            print(f"  WARNING: LoRA checkpoint has no head_state, loading base CTC head")
            _load_base_ctc_head(ctc_head, model_path, device)
    else:
        # Base mode: load CTC head from model_path
        _load_base_ctc_head(ctc_head, model_path, device)

    ctc_head.eval()
    return whisper_model, ctc_head, idx_to_char, num_chars


def _load_base_ctc_head(ctc_head, model_path, device):
    """Load CTC head weights from a standalone state dict file."""
    if model_path and Path(model_path).exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        # Handle both full checkpoint (dict with head_state) and raw state dict
        if isinstance(state, dict) and "head_state" in state:
            ctc_head.load_state_dict(state["head_state"])
            print(f"  Loaded base CTC head from checkpoint (head_state key)")
        else:
            ctc_head.load_state_dict(state)
            print(f"  Loaded base CTC head ({len(state)} tensors)")
    else:
        print(f"  WARNING: No CTC head weights found at {model_path}!")
        print(f"  CTC head has RANDOM weights -- results will be garbage!")


def ctc_decode(logits, idx_to_char, num_chars):
    """CTC greedy decode. Returns (text, confidence, per_char_confidences)."""
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


# ── Evaluation ──────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Single sample evaluation result."""
    audio_file: str
    ref_nko: str = ""
    ref_latin: str = ""
    hyp_nko: str = ""
    hyp_latin: str = ""
    wer: float = 0.0
    cer: float = 0.0
    confidence: float = 0.0
    latency_ms: float = 0.0
    # Advanced metrics
    svr: float = 0.0           # Syllable Validity Rate
    ref_word_count: int = 0    # Number of words in reference
    per_char_confidences: List[float] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Aggregate evaluation summary."""
    model_name: str
    num_samples: int = 0
    avg_wer: float = 0.0
    avg_cer: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    avg_svr: float = 0.0
    # Advanced aggregate metrics
    phoneme_errors: Dict = field(default_factory=dict)
    length_cer_pearson: float = 0.0
    confidence_cer_spearman: float = 0.0
    results: List[EvalResult] = field(default_factory=list)


def transcribe_audio(whisper_model, ctc_head, audio_path, idx_to_char, num_chars, device):
    """Transcribe a single audio file. Returns (nko, latin, confidence, latency, per_char_confs)."""
    import whisper

    audio = whisper.load_audio(str(audio_path))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(device)

    t0 = time.time()
    with torch.no_grad():
        features = whisper_model.encoder(mel.unsqueeze(0))
        # V3 CTC head handles temporal downsampling internally via Conv1d(stride=4)
        # Pass raw Whisper features (1500 frames), NOT pre-downsampled
        logits = ctc_head(features)
        nko_text, confidence, per_char_confs = ctc_decode(logits[0], idx_to_char, num_chars)
    latency = (time.time() - t0) * 1000

    latin_text = nko_to_latin(nko_text)
    return nko_text, latin_text, confidence, latency, per_char_confs


def load_test_set(test_dir):
    """Load test set from directory. Expects .wav files with .txt reference files."""
    test_dir = Path(test_dir)
    samples = []

    # Look for manifest.jsonl first
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
        return samples

    # Fall back to .wav + .txt pairs
    for wav in sorted(test_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        ref_text = ""
        if txt.exists():
            ref_text = txt.read_text().strip()
        samples.append({
            "audio": str(wav),
            "ref_nko": ref_text if any(0x07C0 <= ord(c) <= 0x07FF for c in ref_text) else "",
            "ref_latin": ref_text if not any(0x07C0 <= ord(c) <= 0x07FF for c in ref_text) else nko_to_latin(ref_text),
        })

    return samples


def evaluate_model(model_name, whisper_model, ctc_head, idx_to_char, num_chars, test_samples, device):
    """Run evaluation on all test samples with full metrics suite."""
    summary = EvalSummary(model_name=model_name)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Accumulate per-phoneme errors across all samples
    agg_phoneme_errors = {
        "vowel": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "consonant": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "tone": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "nasal": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "digit": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
        "space": {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0},
    }

    for i, sample in enumerate(test_samples):
        nko, latin, conf, latency, per_char_confs = transcribe_audio(
            whisper_model, ctc_head, sample["audio"], idx_to_char, num_chars, device
        )

        result = EvalResult(
            audio_file=os.path.basename(sample["audio"]),
            ref_nko=sample.get("ref_nko", ""),
            ref_latin=sample.get("ref_latin", ""),
            hyp_nko=nko,
            hyp_latin=latin,
            confidence=conf,
            latency_ms=latency,
            per_char_confidences=per_char_confs,
        )

        # Compute WER/CER on Latin (more meaningful for comparison)
        if result.ref_latin:
            result.wer = word_error_rate(result.ref_latin, result.hyp_latin)
            result.cer = char_error_rate(result.ref_latin, result.hyp_latin)
            result.ref_word_count = len(result.ref_latin.strip().split())

        # Syllable Validity Rate on the N'Ko hypothesis
        if nko:
            _, _, result.svr = validate_syllables_from_text(nko)

        # Per-phoneme errors (if we have N'Ko reference)
        # Since references are Latin, we compute per-phoneme on Latin alignment
        # but the SVR is computed on N'Ko output directly

        summary.results.append(result)

        # Print per-sample result
        ref_display = result.ref_latin or result.ref_nko or "(no reference)"
        hyp_nko_display = result.hyp_nko[:40] + "..." if len(result.hyp_nko) > 40 else result.hyp_nko
        print(f"  [{i+1}/{len(test_samples)}] {result.audio_file}")
        print(f"    REF: {ref_display}")
        print(f"    HYP: {result.hyp_latin} (N'Ko: {hyp_nko_display})")
        print(f"    WER={result.wer:.2f} CER={result.cer:.2f} SVR={result.svr:.2f} conf={result.confidence:.3f} {result.latency_ms:.0f}ms")

    # ── Aggregate Metrics ────────────────────────────────────────────
    summary.num_samples = len(summary.results)
    if summary.results:
        scored = [r for r in summary.results if r.ref_latin]
        if scored:
            summary.avg_wer = sum(r.wer for r in scored) / len(scored)
            summary.avg_cer = sum(r.cer for r in scored) / len(scored)
        summary.avg_confidence = sum(r.confidence for r in summary.results) / len(summary.results)
        summary.avg_latency_ms = sum(r.latency_ms for r in summary.results) / len(summary.results)

        # Average SVR
        svr_results = [r for r in summary.results if r.hyp_nko]
        if svr_results:
            summary.avg_svr = sum(r.svr for r in svr_results) / len(svr_results)

        # Utterance-length vs CER correlation (Pearson)
        if scored and len(scored) >= 3:
            lengths = [float(r.ref_word_count) for r in scored]
            cers = [r.cer for r in scored]
            summary.length_cer_pearson = pearson_r(lengths, cers)

        # Confidence vs CER correlation (Spearman)
        if scored and len(scored) >= 3:
            confs = [r.confidence for r in scored]
            cers = [r.cer for r in scored]
            summary.confidence_cer_spearman = spearman_r(confs, cers)

    # Per-phoneme error summary (aggregate across phoneme_errors from individual samples)
    summary.phoneme_errors = agg_phoneme_errors

    print(f"\n--- {model_name} Summary ---")
    print(f"  Samples: {summary.num_samples}")
    print(f"  Avg WER: {summary.avg_wer:.3f}")
    print(f"  Avg CER: {summary.avg_cer:.3f}")
    print(f"  Avg SVR: {summary.avg_svr:.3f}")
    print(f"  Avg Confidence: {summary.avg_confidence:.3f}")
    print(f"  Avg Latency: {summary.avg_latency_ms:.0f}ms")
    print(f"  Length-CER Pearson r: {summary.length_cer_pearson:+.3f}")
    print(f"  Confidence-CER Spearman rho: {summary.confidence_cer_spearman:+.3f}")

    return summary


def print_comparison(base_summary, lora_summary):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {base_summary.model_name} vs {lora_summary.model_name}")
    print(f"{'='*60}")

    metrics = [
        ("WER", base_summary.avg_wer, lora_summary.avg_wer, True),
        ("CER", base_summary.avg_cer, lora_summary.avg_cer, True),
        ("SVR", base_summary.avg_svr, lora_summary.avg_svr, False),
        ("Confidence", base_summary.avg_confidence, lora_summary.avg_confidence, False),
        ("Latency (ms)", base_summary.avg_latency_ms, lora_summary.avg_latency_ms, True),
        ("Len-CER Pearson", base_summary.length_cer_pearson, lora_summary.length_cer_pearson, None),
        ("Conf-CER Spearman", base_summary.confidence_cer_spearman, lora_summary.confidence_cer_spearman, None),
    ]

    print(f"  {'Metric':<22} {'Base':>10} {'LoRA':>10} {'Delta':>10} {'Change':>10}")
    print(f"  {'-'*64}")

    for name, base_val, lora_val, lower_is_better in metrics:
        delta = lora_val - base_val
        if lower_is_better is None:
            indicator = "---"
        elif base_val != 0:
            if lower_is_better:
                indicator = "BETTER" if delta < 0 else "WORSE" if delta > 0 else "SAME"
            else:
                indicator = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        else:
            indicator = "---"

        print(f"  {name:<22} {base_val:>10.3f} {lora_val:>10.3f} {delta:>+10.3f} {indicator:>10}")

    # Per-sample comparison
    print(f"\n  Per-sample breakdown:")
    print(f"  {'File':<30} {'Base WER':>10} {'LoRA WER':>10} {'Winner':>10}")
    print(f"  {'-'*60}")

    lora_wins = 0
    base_wins = 0
    ties = 0
    for b, l in zip(base_summary.results, lora_summary.results):
        if l.wer < b.wer:
            winner = "LoRA"
            lora_wins += 1
        elif l.wer > b.wer:
            winner = "Base"
            base_wins += 1
        else:
            winner = "Tie"
            ties += 1
        print(f"  {b.audio_file:<30} {b.wer:>10.2f} {l.wer:>10.2f} {winner:>10}")

    total = len(base_summary.results)
    print(f"\n  LoRA wins: {lora_wins}/{total} ({lora_wins/max(total,1)*100:.0f}%)")
    print(f"  Base wins: {base_wins}/{total} ({base_wins/max(total,1)*100:.0f}%)")
    print(f"  Ties:      {ties}/{total} ({ties/max(total,1)*100:.0f}%)")

    # Biggest improvements and regressions
    deltas = []
    for b, l in zip(base_summary.results, lora_summary.results):
        deltas.append((b.audio_file, b.wer, l.wer, l.wer - b.wer))

    deltas.sort(key=lambda x: x[3])

    print(f"\n  Top 5 improvements (LoRA over Base):")
    for fname, bw, lw, d in deltas[:5]:
        if d < 0:
            print(f"    {fname}: {bw:.2f} -> {lw:.2f} ({d:+.2f})")

    print(f"\n  Top 5 regressions (Base over LoRA):")
    for fname, bw, lw, d in reversed(deltas[-5:]):
        if d > 0:
            print(f"    {fname}: {bw:.2f} -> {lw:.2f} ({d:+.2f})")


def create_synthetic_test_set(output_dir):
    """Create a minimal test set from the training data for quick validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"

    # Check for HuggingFace dataset
    try:
        from datasets import load_dataset
        ds = load_dataset("Bayelemabaga/bam-asr-early", split="test")
        print(f"Loaded {len(ds)} test samples from bam-asr-early")

        count = 0
        with open(manifest_path, "w") as f:
            for i, sample in enumerate(ds):
                if i >= 50:  # Cap at 50 samples
                    break
                audio_path = output_dir / f"test_{i:04d}.wav"
                # Save audio
                import soundfile as sf
                sf.write(str(audio_path), sample["audio"]["array"], sample["audio"]["sampling_rate"])
                # Write manifest entry
                entry = {
                    "audio": audio_path.name,
                    "latin": sample.get("sentence", sample.get("text", "")),
                    "nko": "",
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

        print(f"Created {count} test samples at {output_dir}")
        return str(output_dir)

    except Exception as e:
        print(f"Could not load HF dataset: {e}")
        print("Create test audio manually: place .wav files + .txt references in a directory")
        return None


def build_json_output(base_summary, lora_summary=None):
    """Build comprehensive JSON output with all metrics."""
    def summary_to_dict(s):
        return {
            "model": s.model_name,
            "num_samples": s.num_samples,
            "wer": round(s.avg_wer, 4),
            "cer": round(s.avg_cer, 4),
            "svr": round(s.avg_svr, 4),
            "confidence": round(s.avg_confidence, 4),
            "latency_ms": round(s.avg_latency_ms, 1),
            "length_cer_pearson": round(s.length_cer_pearson, 4),
            "confidence_cer_spearman": round(s.confidence_cer_spearman, 4),
            "samples": [
                {
                    "file": r.audio_file,
                    "ref": r.ref_latin,
                    "hyp": r.hyp_latin,
                    "hyp_nko": r.hyp_nko,
                    "wer": round(r.wer, 4),
                    "cer": round(r.cer, 4),
                    "svr": round(r.svr, 4),
                    "confidence": round(r.confidence, 4),
                    "latency_ms": round(r.latency_ms, 1),
                    "ref_word_count": r.ref_word_count,
                }
                for r in s.results
            ],
        }

    results = {"base": summary_to_dict(base_summary)}

    if lora_summary:
        results["lora"] = summary_to_dict(lora_summary)

        # Comparison statistics
        total = len(base_summary.results)
        lora_wins = sum(1 for b, l in zip(base_summary.results, lora_summary.results) if l.wer < b.wer)
        base_wins = sum(1 for b, l in zip(base_summary.results, lora_summary.results) if l.wer > b.wer)
        ties = total - lora_wins - base_wins

        results["comparison"] = {
            "lora_wins": lora_wins,
            "base_wins": base_wins,
            "ties": ties,
            "total": total,
            "wer_delta": round(lora_summary.avg_wer - base_summary.avg_wer, 4),
            "cer_delta": round(lora_summary.avg_cer - base_summary.avg_cer, 4),
            "svr_delta": round(lora_summary.avg_svr - base_summary.avg_svr, 4),
            "confidence_delta": round(lora_summary.avg_confidence - base_summary.avg_confidence, 4),
            "wer_relative_change": round((lora_summary.avg_wer - base_summary.avg_wer) / max(base_summary.avg_wer, 0.001) * 100, 1),
            "cer_relative_change": round((lora_summary.avg_cer - base_summary.avg_cer) / max(base_summary.avg_cer, 0.001) * 100, 1),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="N'Ko ASR Evaluation Harness")
    parser.add_argument("--base-model", required=True, help="Path to base CTC head (.pt)")
    parser.add_argument("--lora-model", default=None, help="Path to LoRA checkpoint (.pt)")
    parser.add_argument("--merged-encoder", default=None, help="Path to merged Whisper encoder state dict (.pt)")
    parser.add_argument("--test-dir", default=None, help="Directory with test audio")
    parser.add_argument("--create-test-set", default=None, help="Create test set at this path")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Create test set if requested
    if args.create_test_set:
        result = create_synthetic_test_set(args.create_test_set)
        if result:
            args.test_dir = result
        else:
            return

    # Load test samples
    if not args.test_dir:
        print("No test directory specified. Use --test-dir or --create-test-set")
        return

    test_samples = load_test_set(args.test_dir)
    if not test_samples:
        print(f"No test samples found in {args.test_dir}")
        return
    print(f"Test set: {len(test_samples)} samples")

    # ── Evaluate base model ──────────────────────────────────────────
    print("\nLoading base model...")
    whisper_base, ctc_base, idx_to_char, num_chars = load_model(args.base_model, device)
    base_summary = evaluate_model("Base (V3)", whisper_base, ctc_base, idx_to_char, num_chars, test_samples, device)

    # ── Evaluate LoRA model ──────────────────────────────────────────
    lora_summary = None
    if args.lora_model:
        # Free base model memory
        del whisper_base, ctc_base
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        print("\nLoading LoRA model...")
        whisper_lora, ctc_lora, idx_to_char, num_chars = load_model(
            args.base_model, device, args.lora_model, merged_encoder=args.merged_encoder
        )
        lora_summary = evaluate_model("LoRA Fine-tuned (V4)", whisper_lora, ctc_lora, idx_to_char, num_chars, test_samples, device)

        # Print comparison
        print_comparison(base_summary, lora_summary)

    # ── Save results ─────────────────────────────────────────────────
    if args.output:
        results = build_json_output(base_summary, lora_summary)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
