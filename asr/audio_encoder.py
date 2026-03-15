#!/usr/bin/env python3
"""
Whisper Audio Encoder — frozen feature extraction for N'Ko ASR.

Extracts frame-level audio embeddings from Whisper's encoder,
without running the decoder. These embeddings feed into the
joint embedding space for syllable retrieval.

Usage:
    encoder = WhisperAudioEncoder(model_size="base")
    embeddings = encoder.encode("path/to/audio.wav")
    # embeddings shape: (num_frames, embed_dim)
"""

import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

# Cross-directory imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WHISPER_MODELS = {
    "tiny": {"dim": 384, "url": "openai/whisper-tiny"},
    "base": {"dim": 512, "url": "openai/whisper-base"},
    "small": {"dim": 768, "url": "openai/whisper-small"},
    "medium": {"dim": 1024, "url": "openai/whisper-medium"},
}

SAMPLE_RATE = 16000
CHUNK_LENGTH_SEC = 30  # Whisper's native chunk length
HOP_LENGTH = 160       # 10ms at 16kHz
N_FFT = 400
N_MELS = 80


def load_audio(path, sr=SAMPLE_RATE):
    """Load audio file and resample to target rate."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio
    except ImportError:
        import soundfile as sf
        audio, orig_sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            # Simple resampling via numpy interpolation
            duration = len(audio) / orig_sr
            target_len = int(duration * sr)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)
        return audio.astype(np.float32)


def compute_log_mel(audio, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute log-mel spectrogram matching Whisper's preprocessing."""
    # Pad or trim to 30s
    target_len = CHUNK_LENGTH_SEC * SAMPLE_RATE
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # STFT
    audio_tensor = torch.from_numpy(audio).float()
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio_tensor, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2

    # Mel filterbank
    mel_filters = _mel_filters(SAMPLE_RATE, n_fft, n_mels)
    mel_spec = mel_filters @ magnitudes

    # Log scale
    log_mel = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    return log_mel


def _mel_filters(sr, n_fft, n_mels):
    """Create mel filterbank matrix."""
    try:
        import librosa
        filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        return torch.from_numpy(filters).float()
    except ImportError:
        # Fallback: compute mel filterbank manually
        fmin, fmax = 0.0, sr / 2
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        freqs = 700 * (10 ** (mels / 2595) - 1)
        bins = np.floor((n_fft + 1) * freqs / sr).astype(int)

        filters = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            start, center, end = bins[i], bins[i + 1], bins[i + 2]
            for j in range(start, center):
                if center != start:
                    filters[i, j] = (j - start) / (center - start)
            for j in range(center, end):
                if end != center:
                    filters[i, j] = (end - j) / (end - center)
        return torch.from_numpy(filters).float()


class WhisperAudioEncoder:
    """
    Frozen Whisper encoder for audio feature extraction.

    Extracts encoder hidden states without running the decoder,
    producing frame-level embeddings for the joint embedding space.
    """

    def __init__(self, model_size="base", device=None):
        if model_size not in WHISPER_MODELS:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(WHISPER_MODELS.keys())}")

        self.model_size = model_size
        self.embed_dim = WHISPER_MODELS[model_size]["dim"]
        self.model_name = WHISPER_MODELS[model_size]["url"]
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load Whisper model and processor."""
        if self._model is not None:
            return

        from transformers import WhisperModel, WhisperProcessor

        print(f"[audio_encoder] Loading {self.model_name}...")
        self._processor = WhisperProcessor.from_pretrained(self.model_name)
        self._model = WhisperModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False

        print(f"[audio_encoder] Loaded. Embed dim: {self.embed_dim}, device: {self.device}")

    def encode(self, audio_path, return_numpy=True):
        """
        Encode audio file to frame-level embeddings.

        Args:
            audio_path: Path to audio file (wav, opus, mp3)
            return_numpy: If True, return numpy array; else torch tensor

        Returns:
            embeddings: (num_frames, embed_dim) array of encoder hidden states
        """
        self._load_model()

        audio = load_audio(audio_path)
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            encoder_outputs = self._model.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state  # (1, num_frames, embed_dim)

        embeddings = hidden_states.squeeze(0)  # (num_frames, embed_dim)

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    def encode_segments(self, audio_path, segments, return_numpy=True):
        """
        Encode specific audio segments to embeddings.

        Args:
            audio_path: Path to audio file
            segments: List of (start_sec, end_sec) tuples
            return_numpy: If True, return numpy arrays

        Returns:
            List of (num_frames, embed_dim) arrays, one per segment
        """
        self._load_model()
        audio = load_audio(audio_path)

        results = []
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < SAMPLE_RATE * 0.1:  # Skip < 100ms
                continue

            inputs = self._processor(segment_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            with torch.no_grad():
                encoder_outputs = self._model.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state.squeeze(0)

            if return_numpy:
                hidden_states = hidden_states.cpu().numpy()
            results.append(hidden_states)

        return results

    def get_embed_dim(self):
        """Return the embedding dimension."""
        return self.embed_dim


if __name__ == "__main__":
    print("=== Whisper Audio Encoder Smoke Test ===")
    encoder = WhisperAudioEncoder(model_size="base")
    print(f"Model: {encoder.model_name}")
    print(f"Embed dim: {encoder.embed_dim}")
    print(f"Device: {encoder.device}")

    # Test with synthetic audio
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Generate 2 seconds of sine wave
        t = np.linspace(0, 2, SAMPLE_RATE * 2)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(f.name, audio, SAMPLE_RATE)

        embeddings = encoder.encode(f.name)
        print(f"Output shape: {embeddings.shape}")
        print(f"Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
        os.unlink(f.name)

    print("Smoke test passed.")
