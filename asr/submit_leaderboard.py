#!/usr/bin/env python3
"""
submit_leaderboard.py — MALIBA-AI Bambara ASR Leaderboard Submission
======================================================================
Runs our trained CharASR model over the 500 blind evaluation WAV files,
decodes CTC output as N'Ko text, reverse-transliterates to Latin Bambara,
and writes the predictions JSON that the leaderboard expects.

Leaderboard: https://huggingface.co/spaces/MALIBA-AI/bambara-asr-leaderboard

Usage (local):
    python3 asr/submit_leaderboard.py \\
        --model-path results/char_model/best_char_asr.pt \\
        --audio-dir data/common_voice_bm/audio \\
        --output submissions/predictions.json

Usage (Vast.ai):
    python3 asr/submit_leaderboard.py \\
        --model-path /workspace/char_model/best_char_asr.pt \\
        --audio-dir /workspace/eval_audio \\
        --output /workspace/predictions.json \\
        --manifest /workspace/eval_manifest.json

Architecture:
    WAV → ffmpeg 16kHz PCM → Whisper large-v3 encoder (n_mels=128)
        → 4x temporal downsample → CharASR BiLSTM → CTC greedy decode
        → N'Ko char sequence → reverse transliterate → Latin Bambara
        → predictions.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths — repo-relative defaults that work from any cwd
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "common_voice_bm" / "eval_manifest.json"
DEFAULT_AUDIO_DIR = REPO_ROOT / "data" / "common_voice_bm" / "audio"
DEFAULT_MODEL_LOCAL = REPO_ROOT / "results" / "char_model" / "best_char_asr.pt"
DEFAULT_MODEL_VASTAI = Path("/workspace/char_model/best_char_asr.pt")
DEFAULT_OUTPUT = REPO_ROOT / "submissions" / "predictions.json"

# Whisper large-v3 produces 1280-dim encoder features
WHISPER_FEATURE_DIM = 1280
# 4x temporal downsampling (matches CharDataset in char_level_train.py)
DOWNSAMPLE = 4
# Maximum audio frames after downsampling (1500 // 4 = 375)
MAX_FRAMES = 375


# ---------------------------------------------------------------------------
# CharASR — exact architecture from char_level_train.py
# ---------------------------------------------------------------------------

class CharASR(nn.Module):
    """Character-level CTC ASR model (BiLSTM encoder)."""

    def __init__(self, num_chars: int, input_dim: int = 1280, hidden_dim: int = 512):
        super().__init__()
        self.num_chars = num_chars
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=3, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)  # +1 blank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# N'Ko character vocabulary — mirrors build_nko_char_vocab() in char_level_train.py
# ---------------------------------------------------------------------------

def build_nko_char_vocab() -> Tuple[Dict[str, int], int]:
    """
    Build the exact same vocab used during training:
      U+07C0 … U+07FF (64 codepoints) + space = 65 chars total.
    CTC blank = num_chars (index 65).
    """
    chars: Dict[str, int] = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        c = chr(cp)
        if c not in chars:
            chars[c] = idx
            idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx  # 65 chars


# ---------------------------------------------------------------------------
# N'Ko → Latin Bambara reverse bridge (self-contained, no nko.transliterate import)
#
# Strategy: character-level N'Ko → IPA → Latin, derived directly from the
# canonical NKO_TO_IPA and IPA_TO_LATIN tables in nko/transliterate.py.
# Inlined here so the script runs on Vast.ai without the full repo tree.
# ---------------------------------------------------------------------------

# N'Ko character → IPA phoneme
_NKO_TO_IPA: Dict[str, str] = {
    # Vowels (U+07CA – U+07D0)
    "\u07ca": "a",    # ߊ A
    "\u07cb": "o",    # ߋ O
    "\u07cc": "i",    # ߌ I
    "\u07cd": "e",    # ߍ E
    "\u07ce": "u",    # ߎ U
    "\u07cf": "ɔ",   # ߏ Open-O
    "\u07d0": "ɛ",   # ߐ Open-E / Schwa
    "\u07d1": "a",    # ߑ Dagbamma (rare)
    # Consonants (U+07D2 – U+07EA)
    "\u07d2": "n",    # ߒ Syllabic N (THE N in N'Ko)
    "\u07d3": "b",    # ߓ Ba
    "\u07d4": "p",    # ߔ Pa (loan words)
    "\u07d5": "t",    # ߕ Ta
    "\u07d6": "dj",   # ߖ Ja → "dj" in Bambara orthography
    "\u07d7": "c",    # ߗ Cha (tʃ → Bambara "c")
    "\u07d8": "d",    # ߘ Da
    "\u07d9": "r",    # ߙ Ra
    "\u07da": "rr",   # ߚ Rra (trilled)
    "\u07db": "s",    # ߛ Sa
    "\u07dc": "gb",   # ߜ Gba (labial-velar)
    "\u07dd": "f",    # ߝ Fa
    "\u07de": "k",    # ߞ Ka
    "\u07df": "l",    # ߟ La
    "\u07e0": "na",   # ߠ Na Woloso (syllabic)
    "\u07e1": "m",    # ߡ Ma
    "\u07e2": "ny",   # ߢ Nya (palatal nasal)
    "\u07e3": "n",    # ߣ Na
    "\u07e4": "h",    # ߤ Ha
    "\u07e5": "w",    # ߥ Wa
    "\u07e6": "y",    # ߦ Ya
    "\u07e7": "ng",   # ߧ Nga (velar nasal — ŋ → "ng" in Bambara)
    "\u07e8": "p",    # ߨ Pa alternate
    "\u07e9": "r",    # ߩ Ra alternate
    "\u07ea": "s",    # ߪ Sa alternate
    # Tone marks (U+07EB – U+07EF) — strip in Latin output
    "\u07eb": "",     # ߫ High tone
    "\u07ec": "",     # ߬ Low tone
    "\u07ed": "",     # ߭ Falling tone
    "\u07ee": "",     # ߮ Rising tone
    "\u07ef": "",     # ߯ Long vowel
    # N'Ko digits (U+07C0 – U+07C9)
    "\u07c0": "0",    # ߀
    "\u07c1": "1",    # ߁
    "\u07c2": "2",    # ߂
    "\u07c3": "3",    # ߃
    "\u07c4": "4",    # ߄
    "\u07c5": "5",    # ߅
    "\u07c6": "6",    # ߆
    "\u07c7": "7",    # ߇
    "\u07c8": "8",    # ߈
    "\u07c9": "9",    # ߉
    # Combining marks (U+07F2 – U+07F3) — strip
    "\u07f2": "",     # ߲ Nasalization
    "\u07f3": "",     # ߳ Nasalization tilde
    # Punctuation (U+07F7 – U+07FA)
    "\u07f8": ",",    # ߸ Comma
    "\u07f9": ".",    # ߹ Full stop
    "\u07f7": "!",    # ߷ Exclamation
    "\u07fa": "-",    # ߺ Lajanyalan
}


def nko_char_to_latin(ch: str) -> str:
    """Map a single N'Ko character to its Bambara Latin equivalent."""
    return _NKO_TO_IPA.get(ch, ch)


def nko_to_latin(nko_text: str) -> str:
    """
    Reverse-transliterate N'Ko character sequence to Latin Bambara.

    Each N'Ko character maps to a Latin string (consonant cluster or vowel).
    Tone marks, combining marks, and undefined codepoints are dropped or
    passed through as-is. Spaces are preserved.

    This is a character-by-character mapping, not syllable-based, which
    matches the CharASR decoder output (individual N'Ko chars, not syllables).
    """
    parts: List[str] = []
    for ch in nko_text:
        if ch == " ":
            parts.append(" ")
        else:
            parts.append(nko_char_to_latin(ch))
    result = "".join(parts)
    # Collapse multiple spaces introduced by empty tone-mark mappings
    import re
    result = re.sub(r" {2,}", " ", result).strip()
    return result


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int, idx_to_char: Dict[int, str]) -> str:
    """
    CTC greedy decode: argmax → remove blank → remove consecutive duplicates → string.

    Args:
        logits:     (T, num_chars+1) raw model output (pre-softmax).
        blank_idx:  CTC blank token index (= num_chars).
        idx_to_char: Reverse vocabulary map.

    Returns:
        Decoded N'Ko character string.
    """
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded: List[str] = []
    prev = -1
    for idx in preds:
        if idx == blank_idx:
            prev = -1
            continue
        if idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


# ---------------------------------------------------------------------------
# Whisper feature extraction
# ---------------------------------------------------------------------------

def load_whisper_model(device: str):
    """Load Whisper large-v3 for encoder feature extraction only."""
    try:
        import whisper
    except ImportError:
        print("[ERROR] openai-whisper not installed. Run: pip install openai-whisper")
        sys.exit(1)

    print("Loading Whisper large-v3 encoder ...")
    model = whisper.load_model("large-v3", device=device)
    model.eval()
    print(f"  Whisper loaded on {device}.")
    return model


def extract_whisper_features(
    audio_path: Path,
    whisper_model,
    device: str,
    n_mels: int = 128,
) -> Optional[torch.Tensor]:
    """
    Load audio, compute log-mel spectrogram, run Whisper encoder.

    Args:
        audio_path: Path to WAV (or any ffmpeg-compatible format).
        whisper_model: Loaded whisper model.
        n_mels: Mel filterbank channels (128 for large-v3).
        device: Torch device string.

    Returns:
        Feature tensor of shape (T, 1280) on CPU, or None on failure.
    """
    try:
        import whisper

        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(device)

        with torch.no_grad():
            # encoder returns (1, T, 1280) — squeeze batch dim
            features = whisper_model.encoder(mel.unsqueeze(0))

        return features.squeeze(0).cpu()  # (T, 1280)

    except Exception as exc:
        return None


def downsample_and_truncate(features: torch.Tensor) -> torch.Tensor:
    """
    Apply 4x temporal downsampling and truncate to MAX_FRAMES.
    Mirrors the CharDataset preprocessing in char_level_train.py.
    """
    if features.shape[0] > 4:
        features = features[::DOWNSAMPLE]
    if features.shape[0] > MAX_FRAMES:
        features = features[:MAX_FRAMES]
    return features


# ---------------------------------------------------------------------------
# Manifest / audio discovery
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> List[dict]:
    """Load eval_manifest.json — list of {id, audio_path, duration, text}."""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_audio_path(entry: dict, audio_dir: Path, manifest_path: Path) -> Optional[Path]:
    """
    Resolve the audio file for a manifest entry.

    Tries, in order:
      1. audio_dir / basename(entry["audio_path"])
      2. manifest_dir / entry["audio_path"]  (relative path as stored)
      3. audio_dir / f"au{entry['id']}.wav"
    """
    raw = entry.get("audio_path", "")
    basename = Path(raw).name

    candidates = [
        audio_dir / basename,
        manifest_path.parent / raw,
        audio_dir / f"au{entry['id']}.wav",
        audio_dir / f"AU{entry['id']}.wav",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model: CharASR,
    whisper_model,
    manifest: List[dict],
    audio_dir: Path,
    manifest_path: Path,
    idx_to_char: Dict[int, str],
    num_chars: int,
    device: str,
) -> List[dict]:
    """
    Run the full pipeline over all manifest entries.

    Returns a list of prediction dicts:
        {id, audio_path, prediction_nko, prediction_latin, duration_ms}
    """
    model.eval()
    predictions: List[dict] = []

    total = len(manifest)
    failed = 0
    t0 = time.time()

    print(f"\nRunning inference on {total} audio files ...")
    print(f"  Model: CharASR ({num_chars} chars + blank, BiLSTM 3-layer)")
    print(f"  Features: Whisper large-v3, n_mels=128, 4x downsample")
    print()

    for i, entry in enumerate(manifest):
        sample_id = entry.get("id", i + 1)
        audio_path = resolve_audio_path(entry, audio_dir, manifest_path)

        if audio_path is None:
            # Emit empty prediction so every entry has a row in the output
            predictions.append({
                "id": sample_id,
                "audio_path": entry.get("audio_path", ""),
                "prediction_nko": "",
                "prediction_latin": "",
                "error": "audio file not found",
            })
            failed += 1
            print(f"  [{i+1:3d}/{total}] MISSING  {entry.get('audio_path', '?')}")
            continue

        # --- Feature extraction ---
        features = extract_whisper_features(audio_path, whisper_model, device)
        if features is None:
            predictions.append({
                "id": sample_id,
                "audio_path": entry.get("audio_path", ""),
                "prediction_nko": "",
                "prediction_latin": "",
                "error": "whisper feature extraction failed",
            })
            failed += 1
            print(f"  [{i+1:3d}/{total}] FEAT_ERR {audio_path.name}")
            continue

        # --- Downsample + truncate ---
        features = downsample_and_truncate(features)   # (T', 1280)
        feat_tensor = features.unsqueeze(0).to(device)  # (1, T', 1280)

        # --- CharASR inference ---
        with torch.no_grad():
            logits = model(feat_tensor)   # (1, T', num_chars+1)
            logits = logits.squeeze(0)    # (T', num_chars+1)

        # --- CTC greedy decode → N'Ko ---
        nko_pred = ctc_greedy_decode(logits, num_chars, idx_to_char)

        # --- N'Ko → Latin Bambara ---
        latin_pred = nko_to_latin(nko_pred)

        predictions.append({
            "id": sample_id,
            "audio_path": entry.get("audio_path", ""),
            "prediction_nko": nko_pred,
            "prediction_latin": latin_pred,
        })

        # Progress log every 25 samples
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:3d}/{total}] {audio_path.name:<20}  "
                f"nko={nko_pred[:30]!r:<35}  "
                f"lat={latin_pred[:25]!r:<28}  "
                f"ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  |  {total - failed}/{total} successful  |  {failed} failed")
    return predictions


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def build_submission_json(predictions: List[dict], model_path: Path) -> dict:
    """
    Build the leaderboard submission payload.

    The MALIBA-AI leaderboard accepts a JSON file with this schema:
        {
          "model_name": str,
          "predictions": [
            {"id": int, "transcription": str},
            ...
          ]
        }

    We include extra metadata fields (nko, audio_path) for our own records;
    they are ignored by the leaderboard evaluator.
    """
    return {
        "model_name": "NKo-CharASR-BiLSTM-CTC",
        "description": (
            "Character-level CTC ASR trained on Bambara speech. "
            "Architecture: Whisper large-v3 encoder (frozen) → 4x temporal downsample "
            "→ 3-layer BiLSTM (512 hidden) → CTC over 65 N'Ko chars. "
            "Decoding: CTC greedy → N'Ko chars → reverse transliterate to Latin Bambara."
        ),
        "model_checkpoint": str(model_path),
        "predictions": [
            {
                "id": p["id"],
                "transcription": p.get("prediction_latin", ""),
                # Extra fields for our own auditing (leaderboard ignores these)
                "_prediction_nko": p.get("prediction_nko", ""),
                "_audio_path": p.get("audio_path", ""),
                **({"_error": p["error"]} if "error" in p else {}),
            }
            for p in predictions
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_path(*candidates: Path) -> Optional[Path]:
    """Return the first existing candidate path, or None."""
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate MALIBA-AI leaderboard submission from CharASR model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to CharASR checkpoint (.pt). "
            "Auto-detected from results/char_model/ or /workspace/char_model/ if omitted."
        ),
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help=(
            "Directory containing the 500 blind evaluation WAV files. "
            f"Default: {DEFAULT_AUDIO_DIR}"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "eval_manifest.json with {id, audio_path} entries. "
            f"Default: {DEFAULT_MANIFEST}"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the predictions JSON file. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device ('cuda', 'mps', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="CharASR hidden dimension (must match training).",
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help=(
            "Skip Whisper feature extraction and load pre-extracted .pt files "
            "from --features-dir instead. Useful when features are already cached."
        ),
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help=(
            "Directory of pre-extracted .pt feature files (used with --no-whisper). "
            "Feature file names must match audio file stems."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    model_path = (
        Path(args.model_path) if args.model_path
        else resolve_path(DEFAULT_MODEL_LOCAL, DEFAULT_MODEL_VASTAI)
    )
    if model_path is None or not model_path.exists():
        print(
            f"[ERROR] CharASR checkpoint not found.\n"
            f"  Tried: {DEFAULT_MODEL_LOCAL}\n"
            f"         {DEFAULT_MODEL_VASTAI}\n"
            f"  Train with:  python3 asr/char_level_train.py\n"
            f"  Or pass:     --model-path /path/to/best_char_asr.pt"
        )
        sys.exit(1)

    manifest_path = (
        Path(args.manifest) if args.manifest
        else resolve_path(DEFAULT_MANIFEST, Path("/workspace/eval_manifest.json"))
    )
    if manifest_path is None or not manifest_path.exists():
        print(
            f"[ERROR] eval_manifest.json not found.\n"
            f"  Expected at: {DEFAULT_MANIFEST}\n"
            f"  Or pass:     --manifest /path/to/eval_manifest.json"
        )
        sys.exit(1)

    audio_dir = (
        Path(args.audio_dir) if args.audio_dir
        else resolve_path(DEFAULT_AUDIO_DIR, Path("/workspace/eval_audio"))
    )
    if audio_dir is None or not audio_dir.is_dir():
        print(
            f"[ERROR] Audio directory not found.\n"
            f"  Expected at: {DEFAULT_AUDIO_DIR}\n"
            f"  Or pass:     --audio-dir /path/to/audio"
        )
        sys.exit(1)

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Character vocabulary (must match training exactly)
    # ------------------------------------------------------------------
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}
    print(f"Vocab: {num_chars} N'Ko characters (U+07C0–U+07FF + space) + CTC blank = {num_chars + 1} classes")

    # ------------------------------------------------------------------
    # Load CharASR model
    # ------------------------------------------------------------------
    print(f"Loading CharASR from {model_path} ...")
    model = CharASR(num_chars=num_chars, input_dim=WHISPER_FEATURE_DIM, hidden_dim=args.hidden_dim)

    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    # Handle checkpoints saved as full dict vs raw state_dict
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Params: {param_count:,}  |  Hidden: {args.hidden_dim}  |  Device: {device}")

    # ------------------------------------------------------------------
    # Load Whisper (unless using pre-extracted features)
    # ------------------------------------------------------------------
    if args.no_whisper:
        if not args.features_dir:
            print("[ERROR] --no-whisper requires --features-dir")
            sys.exit(1)
        features_dir = Path(args.features_dir)
        if not features_dir.is_dir():
            print(f"[ERROR] Features directory not found: {features_dir}")
            sys.exit(1)
        whisper_model = None
        print(f"Using pre-extracted features from {features_dir}")
    else:
        whisper_model = load_whisper_model(device)

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest = load_manifest(manifest_path)
    print(f"Manifest: {len(manifest)} entries from {manifest_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    if args.no_whisper:
        predictions = run_inference_precomputed(
            model=model,
            manifest=manifest,
            features_dir=Path(args.features_dir),
            idx_to_char=idx_to_char,
            num_chars=num_chars,
            device=device,
        )
    else:
        predictions = run_inference(
            model=model,
            whisper_model=whisper_model,
            manifest=manifest,
            audio_dir=audio_dir,
            manifest_path=manifest_path,
            idx_to_char=idx_to_char,
            num_chars=num_chars,
            device=device,
        )

    # ------------------------------------------------------------------
    # Build submission payload and save
    # ------------------------------------------------------------------
    submission = build_submission_json(predictions, model_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    n_non_empty = sum(1 for p in predictions if p.get("prediction_latin", "").strip())
    print(f"\nPredictions saved to: {output_path}")
    print(f"  Total entries:   {len(predictions)}")
    print(f"  Non-empty preds: {n_non_empty}")
    print(f"  Empty / failed:  {len(predictions) - n_non_empty}")

    # ------------------------------------------------------------------
    # Submission instructions
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  HOW TO SUBMIT TO THE MALIBA-AI BAMBARA ASR LEADERBOARD")
    print("=" * 70)
    print()
    print("  Leaderboard: https://huggingface.co/spaces/MALIBA-AI/bambara-asr-leaderboard")
    print()
    print("  Option 1 — Upload via the Hugging Face Space UI:")
    print(f"    1. Go to the leaderboard URL above.")
    print(f"    2. Click 'Submit' or 'Upload predictions'.")
    print(f"    3. Upload:  {output_path}")
    print(f"    4. Fill in model name, description, and paper link if prompted.")
    print()
    print("  Option 2 — Submit via huggingface_hub Python API:")
    print()
    print("    pip install huggingface_hub")
    print("    python3 - <<'EOF'")
    print("    from huggingface_hub import HfApi")
    print("    api = HfApi()")
    print("    api.upload_file(")
    print(f"        path_or_fileobj='{output_path}',")
    print("        path_in_repo='submissions/nko_char_asr.json',")
    print("        repo_id='MALIBA-AI/bambara-asr-leaderboard',")
    print("        repo_type='space',")
    print("    )")
    print("    EOF")
    print()
    print("  Option 3 — Email submission (check leaderboard README for contact):")
    print(f"    Attach: {output_path}")
    print()
    print("  Submission file format summary:")
    print("    {")
    print('      "model_name": "NKo-CharASR-BiLSTM-CTC",')
    print('      "predictions": [')
    print('        {"id": 1, "transcription": "n ka kalo"},')
    print("        ...")
    print("      ]")
    print("    }")
    print()
    print("  Notes:")
    print("    - The leaderboard evaluates WER against blind Latin Bambara references.")
    print("    - 'transcription' must be Latin Bambara (not N'Ko).")
    print("    - All 500 IDs must be present (empty string for failed samples).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Pre-extracted features mode (--no-whisper)
# ---------------------------------------------------------------------------

def run_inference_precomputed(
    model: CharASR,
    manifest: List[dict],
    features_dir: Path,
    idx_to_char: Dict[int, str],
    num_chars: int,
    device: str,
) -> List[dict]:
    """
    Run inference using pre-extracted .pt feature files.

    Feature file naming convention: <audio_stem>.pt
    (e.g. au1.pt for au1.wav)
    """
    model.eval()
    predictions: List[dict] = []
    total = len(manifest)
    failed = 0

    print(f"\nRunning inference on {total} pre-extracted feature files ...")

    for i, entry in enumerate(manifest):
        sample_id = entry.get("id", i + 1)
        raw_path = entry.get("audio_path", "")
        stem = Path(raw_path).stem

        feat_path = features_dir / f"{stem}.pt"
        if not feat_path.exists():
            predictions.append({
                "id": sample_id,
                "audio_path": raw_path,
                "prediction_nko": "",
                "prediction_latin": "",
                "error": f"feature file not found: {feat_path}",
            })
            failed += 1
            continue

        try:
            features = torch.load(str(feat_path), weights_only=True)
            if not isinstance(features, torch.Tensor):
                raise ValueError("not a tensor")
            features = features.float()
        except Exception as exc:
            predictions.append({
                "id": sample_id,
                "audio_path": raw_path,
                "prediction_nko": "",
                "prediction_latin": "",
                "error": f"feature load failed: {exc}",
            })
            failed += 1
            continue

        features = downsample_and_truncate(features)
        feat_tensor = features.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(feat_tensor).squeeze(0)

        nko_pred = ctc_greedy_decode(logits, num_chars, idx_to_char)
        latin_pred = nko_to_latin(nko_pred)

        predictions.append({
            "id": sample_id,
            "audio_path": raw_path,
            "prediction_nko": nko_pred,
            "prediction_latin": latin_pred,
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:3d}/{total}]  {stem:<20}  lat={latin_pred[:30]!r}")

    print(f"\n  Done. {total - failed}/{total} successful, {failed} failed.")
    return predictions


if __name__ == "__main__":
    main()
