#!/usr/bin/env python3
"""
Real-Time N'Ko ASR Demo
========================
Speaks Bambara → see N'Ko text appear in real time.
Runs on Mac4/Mac5 via Tailscale, accessible from any machine.

Usage:
    # On Mac4 or Mac5:
    python3 realtime_asr.py --model ~/Desktop/nko-brain-scanner/models/best_char_asr.pt

    # Then open in browser from any Tailscale machine:
    http://mac4:8899  or  http://100.91.231.93:8899

Features:
    - Record audio from browser microphone
    - Whisper feature extraction (on-device)
    - CTC decode → N'Ko text
    - Reverse bridge: N'Ko → Latin Bambara
    - Optional: N'Ko → English/French via Ollama translation
"""

import argparse
import io
import json
import subprocess
import sys
import os
import tempfile
import wave
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model ──────────────────────────────────────────────────────────────

def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


class CharASR(nn.Module):
    """V3 Transformer h768 L6 with temporal downsampler."""
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


def ctc_decode(logits, idx_to_char, num_chars):
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
    return "".join(decoded), sum(confidences) / max(len(confidences), 1)


# ── Reverse Bridge ────────────────────────────────────────────────────

NKO_TO_LATIN = {
    '\u07ca': 'a', '\u07cb': 'ee', '\u07cc': 'i', '\u07cd': 'e', '\u07ce': 'u',
    '\u07cf': 'o', '\u07d0': 'ɛ',
    '\u07d2': 'ng', '\u07d3': 'b', '\u07d4': 'p', '\u07d5': 't', '\u07d6': 'j',
    '\u07d7': 'c', '\u07d8': 'd', '\u07d9': 'r', '\u07db': 's', '\u07dc': 'g',
    '\u07dd': 'f', '\u07de': 'k', '\u07df': 'l', '\u07e1': 'm', '\u07e2': 'ny',
    '\u07e3': 'n', '\u07e4': 'h', '\u07e5': 'w', '\u07e6': 'y', '\u07e7': 'ng',
}

NKO_TONE_MARKS = {'\u07eb', '\u07ec', '\u07ed', '\u07ee', '\u07ef'}
NKO_NASALS = {'\u07f2', '\u07f3'}


def nko_to_latin(nko_text):
    result = []
    for c in nko_text:
        if c in NKO_TO_LATIN:
            result.append(NKO_TO_LATIN[c])
        elif c == ' ':
            result.append(' ')
        elif c in NKO_TONE_MARKS or c in NKO_NASALS:
            pass
    return ''.join(result)


# ── Ollama Translation ───────────────────────────────────────────────

def translate_via_ollama(text, source="Bambara", target="English", ollama_url="http://localhost:11434"):
    """Translate using local Ollama model."""
    try:
        import urllib.request
        prompt = f"Translate this {source} text to {target}. Only output the translation, nothing else.\n\n{text}"
        data = json.dumps({
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get("response", "").strip()
    except Exception as e:
        return f"[Translation unavailable: {e}]"


# ── Global State ──────────────────────────────────────────────────────

class ASREngine:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.char_vocab, self.num_chars = build_nko_char_vocab()
        self.idx_to_char = {v: k for k, v in self.char_vocab.items()}

        self.model = CharASR(self.num_chars).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()
        print(f"ASR model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

        # Load Whisper
        import whisper
        self.whisper = whisper.load_model("large-v3", device=device)
        self.whisper.eval()
        print("Whisper loaded.")

    def transcribe(self, audio_bytes, translate_to=None):
        """Process raw audio bytes → N'Ko + Latin + optional translation."""
        import whisper

        # Save to temp WAV and load
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.device)

            with torch.no_grad():
                features = self.whisper.encoder(mel.unsqueeze(0)).squeeze(0)

            # V3 uses 375 frames (4x from Whisper), temporal_ds handles the rest
            features = features[::4]  # 1500 → 375

            feat = features.unsqueeze(0)
            with torch.no_grad():
                logits = self.model(feat)

            nko_text, confidence = ctc_decode(logits[0], self.idx_to_char, self.num_chars)
            latin_text = nko_to_latin(nko_text)

            result = {
                "nko": nko_text,
                "latin": latin_text,
                "confidence": round(confidence, 3),
            }

            if translate_to:
                for lang in translate_to:
                    result[lang.lower()] = translate_via_ollama(latin_text, "Bambara", lang)

            return result
        finally:
            os.unlink(tmp_path)


# ── Web UI ────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>N'Ko ASR — Live Demo</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+NKo:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Space Grotesk', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
  }
  h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f59e0b, #ef4444, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }
  .subtitle { color: #888; margin-bottom: 2rem; font-size: 0.95rem; }

  .record-btn {
    width: 120px; height: 120px;
    border-radius: 50%;
    border: 3px solid #333;
    background: radial-gradient(circle at 30% 30%, #1a1a2e, #0a0a15);
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 1rem 0 2rem;
    position: relative;
  }
  .record-btn:hover { border-color: #f59e0b; transform: scale(1.05); }
  .record-btn.recording {
    border-color: #ef4444;
    animation: pulse 1.5s infinite;
    background: radial-gradient(circle at 30% 30%, #2a1015, #150a0f);
  }
  .record-btn .icon { font-size: 2.5rem; }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50% { box-shadow: 0 0 0 20px rgba(239,68,68,0); }
  }

  .results {
    width: 100%;
    max-width: 600px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .result-card {
    background: #111118;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1.2rem;
    transition: border-color 0.3s;
  }
  .result-card:hover { border-color: #333; }
  .result-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #666;
    margin-bottom: 0.5rem;
  }
  .result-nko {
    font-family: 'Noto Sans NKo', sans-serif;
    font-size: 2rem;
    direction: rtl;
    color: #f59e0b;
    min-height: 3rem;
    line-height: 1.4;
  }
  .result-latin {
    font-size: 1.2rem;
    color: #8b5cf6;
    min-height: 1.5rem;
  }
  .result-translation {
    font-size: 1.1rem;
    color: #10b981;
    min-height: 1.5rem;
  }
  .confidence {
    font-size: 0.8rem;
    color: #444;
    margin-top: 0.5rem;
  }

  .options {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  .option-btn {
    padding: 0.4rem 1rem;
    border-radius: 20px;
    border: 1px solid #333;
    background: transparent;
    color: #888;
    cursor: pointer;
    font-size: 0.85rem;
    font-family: inherit;
    transition: all 0.2s;
  }
  .option-btn.active { border-color: #f59e0b; color: #f59e0b; }
  .option-btn:hover { border-color: #555; }

  .status { color: #555; font-size: 0.8rem; margin-top: 1rem; }
  .error { color: #ef4444; }
</style>
</head>
<body>
  <h1>N'Ko ASR</h1>
  <p class="subtitle">Speak Bambara. See N'Ko.</p>

  <div class="options">
    <button class="option-btn active" onclick="toggleLang(this, 'english')">+ English</button>
    <button class="option-btn" onclick="toggleLang(this, 'french')">+ French</button>
  </div>

  <button class="record-btn" id="recordBtn" onclick="toggleRecord()">
    <span class="icon" id="btnIcon">🎙️</span>
  </button>

  <div class="results">
    <div class="result-card">
      <div class="result-label">N'Ko</div>
      <div class="result-nko" id="nkoOutput">—</div>
    </div>
    <div class="result-card">
      <div class="result-label">Latin Bambara</div>
      <div class="result-latin" id="latinOutput">—</div>
    </div>
    <div class="result-card" id="englishCard" style="display:block">
      <div class="result-label">English</div>
      <div class="result-translation" id="englishOutput">—</div>
    </div>
    <div class="result-card" id="frenchCard" style="display:none">
      <div class="result-label">French</div>
      <div class="result-translation" id="frenchOutput">—</div>
    </div>
    <div class="confidence" id="confidence"></div>
  </div>
  <div class="status" id="status">Ready. Click the microphone to record.</div>

<script>
let recording = false;
let mediaRecorder = null;
let chunks = [];
let translateLangs = ['english'];

function toggleLang(btn, lang) {
  btn.classList.toggle('active');
  if (translateLangs.includes(lang)) {
    translateLangs = translateLangs.filter(l => l !== lang);
    document.getElementById(lang + 'Card').style.display = 'none';
  } else {
    translateLangs.push(lang);
    document.getElementById(lang + 'Card').style.display = 'block';
  }
}

async function toggleRecord() {
  if (!recording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunks = [];
      mediaRecorder.ondataavailable = e => chunks.push(e.data);
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        await sendAudio(blob);
      };
      mediaRecorder.start();
      recording = true;
      document.getElementById('recordBtn').classList.add('recording');
      document.getElementById('btnIcon').textContent = '⏹️';
      document.getElementById('status').textContent = 'Recording... Click to stop.';
    } catch(e) {
      document.getElementById('status').textContent = 'Microphone access denied.';
      document.getElementById('status').classList.add('error');
    }
  } else {
    mediaRecorder.stop();
    recording = false;
    document.getElementById('recordBtn').classList.remove('recording');
    document.getElementById('btnIcon').textContent = '🎙️';
    document.getElementById('status').textContent = 'Processing...';
  }
}

async function sendAudio(blob) {
  try {
    const formData = new FormData();
    formData.append('audio', blob);
    formData.append('translate', JSON.stringify(translateLangs));

    const resp = await fetch('/transcribe', { method: 'POST', body: formData });
    const data = await resp.json();

    document.getElementById('nkoOutput').textContent = data.nko || '—';
    document.getElementById('latinOutput').textContent = data.latin || '—';
    document.getElementById('englishOutput').textContent = data.english || '—';
    document.getElementById('frenchOutput').textContent = data.french || '—';
    document.getElementById('confidence').textContent =
      data.confidence ? `Confidence: ${(data.confidence * 100).toFixed(1)}%` : '';
    document.getElementById('status').textContent = 'Ready. Click to record again.';
  } catch(e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
    document.getElementById('status').classList.add('error');
  }
}
</script>
</body>
</html>"""


class ASRHandler(BaseHTTPRequestHandler):
    engine = None

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        if self.path == "/transcribe":
            MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_UPLOAD_BYTES:
                self.send_response(413)
                self.end_headers()
                self.wfile.write(b"Upload too large")
                return
            body = self.rfile.read(content_length)

            # Parse multipart form data (simple extraction)
            boundary = self.headers.get("Content-Type", "").split("boundary=")[-1].encode()
            parts = body.split(b"--" + boundary)

            audio_data = None
            translate_langs = []

            for part in parts:
                if b'name="audio"' in part:
                    audio_data = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                elif b'name="translate"' in part:
                    try:
                        val = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                        translate_langs = json.loads(val)
                    except Exception:
                        pass

            if audio_data:
                # Convert webm to wav using ffmpeg
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                    f.write(audio_data)
                    webm_path = f.name
                wav_path = webm_path.replace(".webm", ".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True,
                )

                try:
                    with open(wav_path, "rb") as f:
                        wav_bytes = f.read()
                    result = self.engine.transcribe(
                        wav_bytes,
                        translate_to=translate_langs if translate_langs else None,
                    )
                except Exception as e:
                    result = {"error": str(e)}
                finally:
                    os.unlink(webm_path)
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "No audio data"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="N'Ko ASR Real-Time Demo")
    parser.add_argument("--model", default="models/best_char_asr.pt")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cpu", help="cpu or cuda or mps")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ssl-cert", default=None, help="Path to SSL cert file")
    parser.add_argument("--ssl-key", default=None, help="Path to SSL key file")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"Loading ASR engine on {device}...")
    engine = ASREngine(args.model, device=device)
    ASRHandler.engine = engine

    server = HTTPServer((args.host, args.port), ASRHandler)

    # Enable HTTPS if certs provided
    if args.ssl_cert and args.ssl_key:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.ssl_cert, args.ssl_key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"
    else:
        proto = "http"

    print(f"\n{'='*50}")
    print(f"N'Ko ASR Demo running at {proto}://{args.host}:{args.port}")
    print(f"{'='*50}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
