#!/usr/bin/env python3
"""
eval_vs_maliba.py — RetrievalASR vs MALIBA-AI Baseline
=======================================================
Compares our CTC-trained retrieval ASR model against the MALIBA-AI
Bambara ASR baseline (WER: 45.73%, reported in their paper).

Pipeline per sample:
    Whisper features (.pt) → RetrievalASR → CTC greedy decode
    → syllable indices → N'Ko string
    → round-trip Latin transliteration (via IPA) → WER vs ground truth

Metrics computed:
    - CER   : Character Error Rate on N'Ko output vs reference N'Ko
    - WER   : Word Error Rate on round-trip Latin vs FarmRadio Whisper Latin
    - SVR   : Syllable Validity Rate (% of decoded indices in codebook)
    - MALIBA baseline WER for comparison: 45.73%

Usage (local):
    python3 asr/eval_vs_maliba.py \\
        --model-path results/best_retrieval_asr.pt \\
        --features-dir /path/to/features \\
        --pairs results/unified_training_pairs.jsonl \\
        --codebook data/syllable_codebook.json

Usage (Vast.ai):
    python3 asr/eval_vs_maliba.py \\
        --model-path /workspace/model/best_retrieval_asr.pt \\
        --features-dir /workspace/features \\
        --pairs /workspace/results/feature_pairs_djoko.jsonl \\
        --codebook /workspace/syllable_codebook.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths — resolve relative to repo root so the script works from any cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_LOCAL = REPO_ROOT / "results" / "best_retrieval_asr.pt"
DEFAULT_MODEL_VASTAI = Path("/workspace/model/best_retrieval_asr.pt")
DEFAULT_CODEBOOK = REPO_ROOT / "data" / "syllable_codebook.json"
DEFAULT_PAIRS_LOCAL = REPO_ROOT / "results" / "unified_training_pairs.jsonl"
DEFAULT_PAIRS_VASTAI = Path("/workspace/results/feature_pairs_djoko.jsonl")
DEFAULT_FEATURES_LOCAL = REPO_ROOT / "results" / "features"
DEFAULT_FEATURES_VASTAI = Path("/workspace/features")

# MALIBA-AI reported baseline (Bambara ASR, FarmRadio corpus)
MALIBA_BASELINE_WER = 45.73  # percent


# ---------------------------------------------------------------------------
# Model — identical architecture to concurrent_train.py
# ---------------------------------------------------------------------------

class RetrievalASR(nn.Module):
    def __init__(self, num_syllables: int, embed_dim: int = 512, input_dim: int = 1280):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 768), nn.GELU(), nn.LayerNorm(768),
            nn.Linear(768, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.ctc_head = nn.Linear(embed_dim, num_syllables + 1)
        self.num_syllables = num_syllables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctc_head(self.proj(x))


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int) -> List[int]:
    """
    CTC greedy decode a single sequence.

    Args:
        logits: (T, num_classes+1) raw logits tensor
        blank_idx: index of the CTC blank token (= num_syllables)

    Returns:
        List of decoded syllable indices (blanks and consecutive
        duplicates removed).
    """
    indices = logits.argmax(dim=-1).tolist()  # (T,)
    decoded: List[int] = []
    prev = None
    for idx in indices:
        if idx == blank_idx:
            prev = None
            continue
        if idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


# ---------------------------------------------------------------------------
# Codebook utilities
# ---------------------------------------------------------------------------

def load_codebook(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_idx_to_syllable(codebook: dict) -> dict:
    """Map integer index → syllable dict."""
    return {s["index"]: s for s in codebook["syllables"]}


def build_nko_to_idx(codebook: dict) -> dict:
    """Map N'Ko string → integer index (for CER reference encoding)."""
    return {s["nko"]: s["index"] for s in codebook["syllables"]}


def indices_to_nko(indices: List[int], idx_to_syl: dict) -> str:
    """Convert decoded indices to N'Ko string (space-separated syllables)."""
    parts = []
    for idx in indices:
        syl = idx_to_syl.get(idx)
        if syl:
            parts.append(syl["nko"])
    return " ".join(parts)


def is_valid_syllable_idx(idx: int, idx_to_syl: dict) -> bool:
    return idx in idx_to_syl


# ---------------------------------------------------------------------------
# N'Ko → Latin round-trip transliteration (IPA-based)
# ---------------------------------------------------------------------------
# We derive a direct N'Ko → Latin mapping from the IPA field in the codebook.
# IPA maps cleanly back to Bambara orthography for WER comparison.

# Tonal diacritics present in N'Ko (used as combining characters)
_NKO_TONES = {"߫", "߬", "߭", "߮", "߯"}

# IPA → Bambara Latin approximation (for WER comparison with Whisper output)
_IPA_TO_LATIN: dict[str, str] = {
    # Vowels
    "a": "a", "o": "o", "i": "i", "e": "e", "u": "u",
    "ɔ": "ɔ", "ə": "ɛ",
    # Nasalised vowels (strip tilde → base vowel + n)
    "ã": "an", "õ": "on", "ĩ": "in", "ẽ": "en", "ũ": "un",
    "ɔ̃": "ɔn", "ə̃": "ɛn",
    # Consonants (IPA → standard Bambara orthography)
    "n": "n", "b": "b", "p": "p", "t": "t",
    "d": "d", "k": "k", "g": "g", "f": "f",
    "s": "s", "l": "l", "r": "r", "w": "w",
    "y": "y", "m": "m", "h": "h",
    "dʒ": "j", "tʃ": "c", "ŋ": "ng",
    "ɲ": "ny", "kp": "kp", "gb": "gb",
    "ɡ": "g",
}


def _ipa_to_latin(ipa: str) -> str:
    """
    Convert an IPA syllable string to approximate Bambara Latin.
    Strips tone markers (߫ ߬ ߭ ߮ ߯) and maps via _IPA_TO_LATIN.
    Falls back to the IPA string itself if no mapping is found.
    """
    # Strip N'Ko tone diacritics that appear in the IPA field
    cleaned = "".join(c for c in ipa if c not in _NKO_TONES)

    # Try direct lookup first
    if cleaned in _IPA_TO_LATIN:
        return _IPA_TO_LATIN[cleaned]

    # Try to decompose: find longest-prefix consonant cluster, then vowel
    result = []
    remaining = cleaned
    # Multi-char consonant clusters (order matters — longest first)
    for cluster in ("dʒ", "tʃ", "ŋ", "ɲ", "kp", "gb", "ɡ"):
        if remaining.startswith(cluster):
            result.append(_IPA_TO_LATIN.get(cluster, cluster))
            remaining = remaining[len(cluster):]
            break
    else:
        if remaining and remaining[0] in _IPA_TO_LATIN:
            result.append(_IPA_TO_LATIN[remaining[0]])
            remaining = remaining[1:]
        elif remaining:
            result.append(remaining[0])
            remaining = remaining[1:]

    # Remaining is the vowel nucleus (possibly nasalised)
    if remaining in _IPA_TO_LATIN:
        result.append(_IPA_TO_LATIN[remaining])
    elif remaining:
        result.append(remaining)

    return "".join(result)


def build_nko_to_latin_map(codebook: dict) -> dict[str, str]:
    """
    Derive N'Ko syllable → Bambara Latin map using the IPA field.
    This is the reverse of the bridge_to_nko direction.
    """
    mapping: dict[str, str] = {}
    for syl in codebook["syllables"]:
        nko = syl["nko"]
        ipa = syl["ipa"]
        mapping[nko] = _ipa_to_latin(ipa)
    return mapping


def nko_to_latin(nko_text: str, nko_to_lat: dict[str, str], idx_to_syl: dict) -> str:
    """
    Convert a space-separated N'Ko syllable string back to approximate Latin.
    Unknown tokens are preserved as-is (should not occur with a codebook decode).
    """
    parts = []
    for token in nko_text.split():
        parts.append(nko_to_lat.get(token, token))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Edit-distance metrics (no external libraries)
# ---------------------------------------------------------------------------

def _edit_distance(seq_a: List, seq_b: List) -> int:
    """
    Standard dynamic-programming Levenshtein edit distance.
    Works on any comparable sequence (characters or words).
    """
    m, n = len(seq_a), len(seq_b)
    # Use two-row DP to keep memory O(n)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate — edit distance on character sequences.
    Returns a value in [0, ∞) where 0 is perfect.
    Expressed as a fraction (multiply by 100 for percent).
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    h_chars = list(hypothesis)
    r_chars = list(reference)
    return _edit_distance(h_chars, r_chars) / len(r_chars)


def wer(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate — edit distance on whitespace-tokenised word sequences.
    Returns a value in [0, ∞). Expressed as a fraction.
    """
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    h_words = hypothesis.split()
    r_words = reference.split()
    return _edit_distance(h_words, r_words) / len(r_words)


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def find_feature_file(feat_id: str, features_dir: Path) -> Optional[Path]:
    """
    Resolve a feature file from a feat_id string.
    Tries: <feat_id>.pt, then stem matching across the dir.
    """
    direct = features_dir / f"{feat_id}.pt"
    if direct.exists():
        return direct
    return None


def load_features(path: Path) -> Optional[torch.Tensor]:
    """Load a .pt Whisper feature tensor. Returns None on failure."""
    try:
        feat = torch.load(path, weights_only=True)
        if not isinstance(feat, torch.Tensor):
            return None
        return feat.float()
    except Exception as e:
        print(f"  [warn] Failed to load {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Pairs loading
# ---------------------------------------------------------------------------

def load_pairs(pairs_path: str) -> List[dict]:
    """Load NKo training pairs from a JSONL file."""
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model: RetrievalASR,
    pairs: List[dict],
    features_dir: Path,
    idx_to_syl: dict,
    nko_to_lat: dict,
    device: str,
    max_samples: int = 0,
    max_audio_len: int = 1500,
    verbose: bool = False,
) -> dict:
    """
    Run evaluation over all pairs that have pre-extracted features.

    Returns a results dict with per-sample records and aggregate metrics.
    """
    model.eval()
    blank_idx = model.num_syllables

    results = []
    skipped = 0
    found = 0

    pairs_to_eval = pairs
    if max_samples > 0:
        pairs_to_eval = pairs[:max_samples]

    print(f"\nRunning inference on up to {len(pairs_to_eval)} samples...")

    for i, pair in enumerate(pairs_to_eval):
        # Resolve feat_id
        feat_id = pair.get("feat_id")
        if not feat_id:
            # Some JSONL formats use segment_file or audio_path as the key
            seg = pair.get("segment_file") or pair.get("audio_path") or ""
            feat_id = Path(seg).stem if seg else None

        if not feat_id:
            skipped += 1
            continue

        feat_path = find_feature_file(feat_id, features_dir)
        if feat_path is None:
            skipped += 1
            continue

        features = load_features(feat_path)
        if features is None:
            skipped += 1
            continue

        found += 1

        # Truncate to max_audio_len
        if features.shape[0] > max_audio_len:
            features = features[:max_audio_len]

        # --- Model inference ---
        with torch.no_grad():
            feat_tensor = features.unsqueeze(0).to(device)  # (1, T, 1280)
            logits = model(feat_tensor)                      # (1, T, num_syl+1)
            logits = logits.squeeze(0)                       # (T, num_syl+1)

        decoded_indices = ctc_greedy_decode(logits, blank_idx)

        # --- N'Ko string ---
        pred_nko = indices_to_nko(decoded_indices, idx_to_syl)

        # --- Reference N'Ko ---
        ref_nko = pair.get("nko") or ""

        # --- Syllable Validity Rate ---
        if decoded_indices:
            valid_count = sum(1 for idx in decoded_indices if is_valid_syllable_idx(idx, idx_to_syl))
            svr = valid_count / len(decoded_indices)
        else:
            svr = 0.0

        # --- CER on N'Ko ---
        # Compare character sequences (strip spaces for pure char-level edit dist)
        cer_val = cer(
            pred_nko.replace(" ", ""),
            ref_nko.replace(" ", ""),
        )

        # --- Round-trip Latin ---
        pred_latin = nko_to_latin(pred_nko, nko_to_lat, idx_to_syl)
        ref_latin = pair.get("latin") or ""

        wer_val = wer(pred_latin, ref_latin)

        rec = {
            "feat_id": feat_id,
            "pred_nko": pred_nko,
            "ref_nko": ref_nko,
            "pred_latin": pred_latin,
            "ref_latin": ref_latin,
            "cer": cer_val,
            "wer": wer_val,
            "svr": svr,
            "decoded_len": len(decoded_indices),
        }
        results.append(rec)

        if verbose and (i + 1) % 50 == 0:
            avg_wer = sum(r["wer"] for r in results) / len(results) * 100
            avg_cer = sum(r["cer"] for r in results) / len(results) * 100
            print(f"  [{found}/{len(pairs_to_eval)}] "
                  f"WER={avg_wer:.1f}%  CER={avg_cer:.1f}%  SVR={sum(r['svr'] for r in results)/len(results)*100:.1f}%")

    print(f"  Evaluated: {found}  Skipped (no features): {skipped}")
    return results


# ---------------------------------------------------------------------------
# Aggregate + report
# ---------------------------------------------------------------------------

def aggregate(results: List[dict]) -> dict:
    if not results:
        return {
            "n": 0,
            "cer_mean": None, "cer_median": None,
            "wer_mean": None, "wer_median": None,
            "svr_mean": None,
        }
    n = len(results)

    def mean(vals):
        return sum(vals) / n

    def median(vals):
        s = sorted(vals)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

    cers = [r["cer"] for r in results]
    wers = [r["wer"] for r in results]
    svrs = [r["svr"] for r in results]

    return {
        "n": n,
        "cer_mean": mean(cers),
        "cer_median": median(cers),
        "wer_mean": mean(wers),
        "wer_median": median(wers),
        "svr_mean": mean(svrs),
    }


def print_comparison_table(agg: dict, verbose_samples: List[dict] = None):
    n = agg["n"]
    if n == 0:
        print("\n[!] No samples evaluated — check that features exist in --features-dir")
        return

    cer_pct = agg["cer_mean"] * 100
    cer_med_pct = agg["cer_median"] * 100
    wer_pct = agg["wer_mean"] * 100
    wer_med_pct = agg["wer_median"] * 100
    svr_pct = agg["svr_mean"] * 100

    # WER delta vs MALIBA
    delta = MALIBA_BASELINE_WER - wer_pct
    delta_str = f"+{abs(delta):.2f}pp worse" if delta < 0 else f"{delta:.2f}pp better"
    arrow = "↑" if delta >= 0 else "↓"

    w = 60  # table width
    sep = "─" * w

    print()
    print("=" * w)
    print(f"  NKo RetrievalASR  vs  MALIBA-AI Baseline")
    print(f"  Evaluated on {n} samples with pre-extracted Whisper features")
    print("=" * w)
    print()
    print(f"{'Metric':<28} {'Ours':>12}   {'MALIBA':>12}")
    print(sep)
    print(f"{'WER (mean)':<28} {wer_pct:>11.2f}%   {MALIBA_BASELINE_WER:>11.2f}%")
    print(f"{'WER (median)':<28} {wer_med_pct:>11.2f}%   {'—':>12}")
    print(sep)
    print(f"{'CER on N\u02bcKo (mean)':<28} {cer_pct:>11.2f}%   {'—':>12}")
    print(f"{'CER on N\u02bcKo (median)':<28} {cer_med_pct:>11.2f}%   {'—':>12}")
    print(sep)
    print(f"{'Syllable Validity Rate':<28} {svr_pct:>11.2f}%   {'—':>12}")
    print("=" * w)
    print()
    print(f"  Verdict: Our WER is {wer_pct:.2f}% vs MALIBA baseline {MALIBA_BASELINE_WER:.2f}%")
    print(f"           {arrow} {delta_str} than MALIBA-AI")
    print()
    print("  Notes:")
    print("    • WER is computed on round-trip Latin (N'Ko → IPA → Bambara Latin)")
    print("      so it reflects both ASR quality and transliteration fidelity.")
    print("    • MALIBA WER is from their paper (FarmRadio test set, Bambara).")
    print("    • SVR = fraction of decoded tokens that are valid codebook syllables.")
    print()

    # Sample predictions
    if verbose_samples:
        print(sep)
        print("  Sample predictions:")
        print(sep)
        for r in verbose_samples[:5]:
            print(f"  feat_id : {r['feat_id']}")
            print(f"  ref_nko : {r['ref_nko'][:80]}")
            print(f"  pred_nko: {r['pred_nko'][:80]}")
            print(f"  ref_lat : {r['ref_latin'][:80]}")
            print(f"  pred_lat: {r['pred_latin'][:80]}")
            print(f"  CER={r['cer']*100:.1f}%  WER={r['wer']*100:.1f}%  SVR={r['svr']*100:.1f}%")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_default_path(*candidates: Path) -> Optional[Path]:
    """Return the first candidate path that exists."""
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RetrievalASR against the MALIBA-AI baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to best_retrieval_asr.pt checkpoint. "
             "Auto-detected from results/ or /workspace/model/ if omitted.",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Directory containing pre-extracted Whisper feature .pt files. "
             "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="JSONL file with (feat_id/audio_path, latin, nko) test pairs. "
             "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help="Path to syllable_codebook.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate at most this many samples. 0 = all.",
    )
    parser.add_argument(
        "--max-audio-len",
        type=int,
        default=1500,
        help="Truncate audio features to this many time steps.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="RetrievalASR embedding dimension (must match training).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1280,
        help="Whisper large-v2 feature dimension.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device ('cpu', 'cuda', 'mps'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rolling metrics every 50 samples + show sample predictions.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save per-sample results as JSON.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    model_path = Path(args.model_path) if args.model_path else resolve_default_path(
        DEFAULT_MODEL_LOCAL, DEFAULT_MODEL_VASTAI
    )
    if model_path is None or not model_path.exists():
        print(f"[ERROR] Model checkpoint not found. "
              f"Tried: {DEFAULT_MODEL_LOCAL}, {DEFAULT_MODEL_VASTAI}")
        print("  Train with asr/concurrent_train.py first, or pass --model-path.")
        sys.exit(1)

    codebook_path = Path(args.codebook) if args.codebook else resolve_default_path(
        DEFAULT_CODEBOOK, Path("/workspace/syllable_codebook.json")
    )
    if codebook_path is None or not codebook_path.exists():
        print(f"[ERROR] Codebook not found. Pass --codebook.")
        sys.exit(1)

    pairs_path = Path(args.pairs) if args.pairs else resolve_default_path(
        DEFAULT_PAIRS_LOCAL, DEFAULT_PAIRS_VASTAI,
        REPO_ROOT / "results" / "sweden_4090_nko_pairs.jsonl",
        REPO_ROOT / "results" / "vastai_3090_nko_pairs.jsonl",
    )
    if pairs_path is None or not pairs_path.exists():
        print(f"[ERROR] Pairs JSONL not found. Pass --pairs.")
        sys.exit(1)

    features_dir = Path(args.features_dir) if args.features_dir else resolve_default_path(
        DEFAULT_FEATURES_LOCAL, DEFAULT_FEATURES_VASTAI,
        REPO_ROOT / "results" / "features",
    )
    if features_dir is None or not features_dir.is_dir():
        print(f"[ERROR] Features directory not found. Pass --features-dir.")
        print("  Features are extracted by asr/stream_with_features.py (on GPU).")
        sys.exit(1)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ------------------------------------------------------------------
    # Load codebook
    # ------------------------------------------------------------------
    print(f"Loading codebook from {codebook_path} ...")
    codebook = load_codebook(str(codebook_path))
    num_syllables = len(codebook["syllables"])
    idx_to_syl = build_idx_to_syllable(codebook)
    nko_to_lat = build_nko_to_latin_map(codebook)
    print(f"  {num_syllables} syllables loaded.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path} ...")
    model = RetrievalASR(
        num_syllables=num_syllables,
        embed_dim=args.embed_dim,
        input_dim=args.input_dim,
    )

    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    # Handle both raw state_dict and checkpoint dict
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Device: {device} | Params: {param_count:,}")

    # ------------------------------------------------------------------
    # Load pairs
    # ------------------------------------------------------------------
    print(f"Loading pairs from {pairs_path} ...")
    pairs = load_pairs(str(pairs_path))
    print(f"  {len(pairs)} pairs loaded.")

    # Check feature availability
    n_feat_files = len(list(features_dir.glob("*.pt")))
    print(f"  Feature files in {features_dir}: {n_feat_files}")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    results = run_eval(
        model=model,
        pairs=pairs,
        features_dir=features_dir,
        idx_to_syl=idx_to_syl,
        nko_to_lat=nko_to_lat,
        device=device,
        max_samples=args.max_samples,
        max_audio_len=args.max_audio_len,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # Aggregate and print
    # ------------------------------------------------------------------
    agg = aggregate(results)

    sample_predictions = results[:10] if args.verbose else None
    print_comparison_table(agg, verbose_samples=sample_predictions)

    # ------------------------------------------------------------------
    # Optional output
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aggregate": {k: (round(v * 100, 4) if isinstance(v, float) else v)
                          for k, v in agg.items()},
            "maliba_baseline_wer_pct": MALIBA_BASELINE_WER,
            "model_path": str(model_path),
            "pairs_path": str(pairs_path),
            "features_dir": str(features_dir),
            "samples": results[:100],  # Cap to avoid huge files
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
