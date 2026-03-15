"""Tests for sound_sigils.models."""

import pytest

from sound_sigils.models import SigilDefinition, SigilSound


def _dummy_gen(t: float, dur: float, freq: float) -> tuple[float, float]:
    return (0.0, 0.0)


class TestSigilDefinition:
    def test_frozen(self):
        d = SigilDefinition("ߛ", "test", "desc", "sound", 440.0, 1.0)
        with pytest.raises(AttributeError):
            d.name = "other"  # type: ignore[misc]

    def test_fields(self):
        d = SigilDefinition("ߛ", "stabilization", "Desc", "Sound", 440.0, 1.5)
        assert d.char == "ߛ"
        assert d.name == "stabilization"
        assert d.base_freq == 440.0
        assert d.duration == 1.5


class TestSigilSound:
    def test_convenience_accessors(self):
        d = SigilDefinition("ߛ", "stabilization", "Desc", "Sound", 440.0, 1.5)
        s = SigilSound(definition=d, generator=_dummy_gen)
        assert s.char == "ߛ"
        assert s.name == "stabilization"
        assert s.description == "Desc"
        assert s.sound_description == "Sound"
        assert s.base_freq == 440.0
        assert s.duration == 1.5

    def test_generator_callable(self):
        d = SigilDefinition("ߛ", "test", "d", "s", 440.0, 1.0)
        s = SigilSound(definition=d, generator=_dummy_gen)
        left, right = s.generator(0.0, 1.0, 440.0)
        assert left == 0.0
        assert right == 0.0
