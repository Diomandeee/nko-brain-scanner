#!/usr/bin/env python3
"""
N'Ko ASR Inference — nko-asr-v1
================================
Transcribes audio to N'Ko script using frozen Whisper features + BiLSTM CTC head.

Requirements:
    pip install torch openai-whisper

Usage:
    python3 inference.py audio_file.wav
    python3 inference.py audio_file.wav --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ── N'Ko Character Vocabulary ─────────────────────────────────────────

def build_nko_char_vocab():
    """Build vocab of individual N'Ko characters (U+07C0-U+07FF + space)."""
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ── Model Architecture ────────────────────────────────────────────────

class CharASR(nn.Module):
    """Character-level CTC ASR head for N'Ko.

    Takes 1280-dim Whisper encoder features, processes through a 3-layer
    BiLSTM, and outputs logits over 65 N'Ko character classes + CTC blank.
    """

    def __init__(self, num_chars=65, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.num_chars = num_chars
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=3, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)  # +1 for CTC blank

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)


# ── CTC Decoding ──────────────────────────────────────────────────────

def ctc_decode(logits, idx_to_char, num_chars):
    """CTC greedy decode: argmax -> collapse repeats -> remove blanks."""
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    prev = -1
    for idx in preds:
        if idx != num_chars and idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


# ── Inference Pipeline ────────────────────────────────────────────────

def transcribe(audio_path, whisper_model, asr_model, idx_to_char, num_chars, device):
    """Transcribe an audio file to N'Ko text."""
    import whisper as whisper_lib

    # Load audio
    audio = whisper_lib.load_audio(audio_path)
    audio = whisper_lib.pad_or_trim(torch.tensor(audio, dtype=torch.float32))

    # Whisper mel spectrogram
    mel = whisper_lib.log_mel_spectrogram(audio, n_mels=128).to(device)

    # Extract frozen Whisper features
    with torch.no_grad():
        features = whisper_model.encoder(mel.unsqueeze(0)).squeeze(0)

    # 4x temporal downsample
    features = features[::4].unsqueeze(0)

    # BiLSTM CTC head
    with torch.no_grad():
        logits = asr_model(features)

    # Decode
    nko_text = ctc_decode(logits[0], idx_to_char, num_chars)
    return nko_text


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to N'Ko script")
    parser.add_argument("audio", help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--model", default=None,
                        help="Path to best_char_asr.pt (default: same directory as this script)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda, cpu, or mps (auto-detected if not specified)")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Model path
    model_path = args.model or str(Path(__file__).parent / "best_char_asr.pt")
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Build vocabulary
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    # Load Whisper
    print("Loading Whisper large-v3 (this takes a moment on first run)...")
    import whisper as whisper_lib
    whisper_model = whisper_lib.load_model("large-v3", device=device)
    whisper_model.eval()

    # Load N'Ko ASR head
    print(f"Loading N'Ko ASR head from {model_path}...")
    asr_model = CharASR(num_chars).to(device)
    asr_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    asr_model.eval()
    print(f"Model loaded: {sum(p.numel() for p in asr_model.parameters()):,} parameters")

    # Transcribe
    nko_text = transcribe(args.audio, whisper_model, asr_model, idx_to_char, num_chars, device)
    print(f"\nN'Ko transcription:\n{nko_text}")


if __name__ == "__main__":
    main()
