"""Tests for sound_sigils.audio."""

import struct
import wave
import io

import pytest

from sound_sigils.audio import (
    SAMPLE_RATE,
    generate_sequence_wav,
    generate_wav,
    render_samples,
    samples_to_wav,
)
from sound_sigils.models import SigilDefinition, SigilSound


def _sine_gen(t: float, duration: float, freq: float) -> tuple[float, float]:
    """Simple sine wave for testing."""
    import math
    val = 0.5 * math.sin(2 * math.pi * freq * t)
    return (val, val)


@pytest.fixture
def sine_sigil():
    defn = SigilDefinition("T", "test", "test", "test", 440.0, 0.5)
    return SigilSound(definition=defn, generator=_sine_gen)


class TestRenderSamples:
    def test_correct_length(self, sine_sigil):
        samples = render_samples(sine_sigil)
        expected = int(0.5 * SAMPLE_RATE)
        assert len(samples) == expected

    def test_duration_override(self, sine_sigil):
        samples = render_samples(sine_sigil, duration_override=1.0)
        expected = int(1.0 * SAMPLE_RATE)
        assert len(samples) == expected

    def test_samples_are_clipped(self):
        """Generators that exceed [-1, 1] should be clipped."""
        def loud_gen(t, d, f):
            return (5.0, -5.0)
        defn = SigilDefinition("X", "loud", "d", "s", 440.0, 0.1)
        sigil = SigilSound(definition=defn, generator=loud_gen)
        samples = render_samples(sigil)
        for l, r in samples:
            assert -1.0 <= l <= 1.0
            assert -1.0 <= r <= 1.0

    def test_custom_sample_rate(self, sine_sigil):
        samples = render_samples(sine_sigil, sample_rate=22050)
        expected = int(0.5 * 22050)
        assert len(samples) == expected


class TestSamplesToWav:
    def test_valid_wav(self):
        samples = [(0.5, -0.5)] * 1000
        wav_bytes = samples_to_wav(samples)
        # Should be parseable as WAV
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            assert w.getnchannels() == 2
            assert w.getsampwidth() == 2
            assert w.getframerate() == SAMPLE_RATE
            assert w.getnframes() == 1000

    def test_empty_samples(self):
        wav_bytes = samples_to_wav([])
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            assert w.getnframes() == 0


class TestGenerateWav:
    def test_returns_bytes(self, sine_sigil):
        result = generate_wav(sine_sigil)
        assert isinstance(result, bytes)
        assert len(result) > 44  # WAV header is 44 bytes

    def test_valid_wav_header(self, sine_sigil):
        result = generate_wav(sine_sigil)
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_freq_override(self, sine_sigil):
        # Should not crash with different freq
        result = generate_wav(sine_sigil, freq_override=880.0)
        assert isinstance(result, bytes)


class TestGenerateSequenceWav:
    def test_sequence_longer_than_single(self, sine_sigil):
        single = generate_wav(sine_sigil)
        sequence = generate_sequence_wav([sine_sigil, sine_sigil])
        assert len(sequence) > len(single)

    def test_gap_adds_silence(self, sine_sigil):
        no_gap = generate_sequence_wav([sine_sigil, sine_sigil], gap=0.0)
        with_gap = generate_sequence_wav([sine_sigil, sine_sigil], gap=0.5)
        assert len(with_gap) > len(no_gap)

    def test_single_item_sequence(self, sine_sigil):
        single = generate_wav(sine_sigil)
        seq = generate_sequence_wav([sine_sigil])
        # Should be same size (no trailing gap)
        assert len(seq) == len(single)
