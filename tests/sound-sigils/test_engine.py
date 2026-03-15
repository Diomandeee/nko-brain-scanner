"""Tests for sound_sigils.engine (SoundSigils API)."""

import os
import tempfile

import pytest

from sound_sigils import SoundSigils


@pytest.fixture
def ss():
    return SoundSigils()


class TestInit:
    def test_loads_all_ten(self, ss):
        assert len(ss) == 10

    def test_has_all_chars(self, ss):
        expected_chars = {"ߛ", "ߜ", "ߕ", "ߙ", "ߡ", "ߚ", "ߞ", "ߣ", "ߠ", "ߥ"}
        assert set(ss.chars) == expected_chars

    def test_has_all_names(self, ss):
        expected = {
            "stabilization", "dispersion", "transition", "return",
            "dwell", "oscillation", "recovery", "novelty",
            "place_shift", "echo",
        }
        assert set(ss.names) == expected


class TestLookup:
    def test_by_char(self, ss):
        s = ss.lookup("ߛ")
        assert s is not None
        assert s.name == "stabilization"

    def test_by_name(self, ss):
        s = ss.lookup("stabilization")
        assert s is not None
        assert s.char == "ߛ"

    def test_by_name_with_hyphen(self, ss):
        s = ss.lookup("place-shift")
        assert s is not None
        assert s.name == "place_shift"

    def test_by_name_case_insensitive(self, ss):
        s = ss.lookup("Stabilization")
        assert s is not None

    def test_unknown_returns_none(self, ss):
        assert ss.lookup("xyz") is None
        assert ss.lookup("Z") is None

    def test_contains(self, ss):
        assert "ߛ" in ss
        assert "stabilization" in ss
        assert "xyz" not in ss


class TestGenerate:
    def test_returns_wav_bytes(self, ss):
        data = ss.generate("ߛ")
        assert isinstance(data, bytes)
        assert data[:4] == b"RIFF"

    def test_by_name(self, ss):
        data = ss.generate("stabilization")
        assert isinstance(data, bytes)
        assert len(data) > 44

    def test_unknown_raises(self, ss):
        with pytest.raises(ValueError, match="Unknown sigil"):
            ss.generate("xyz")

    def test_all_sigils_generate(self, ss):
        """Every registered sigil must produce valid WAV data."""
        for char in ss.chars:
            data = ss.generate(char)
            assert data[:4] == b"RIFF", f"Failed for {char}"

    def test_generate_sequence(self, ss):
        data = ss.generate_sequence(["ߛ", "ߕ", "ߙ"])
        assert isinstance(data, bytes)
        assert data[:4] == b"RIFF"


class TestExport:
    def test_export_creates_file(self, ss):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            result = ss.export("ߛ", path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 44

    def test_export_creates_dirs(self, ss):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "test.wav")
            result = ss.export("ߛ", path)
            assert os.path.exists(result)

    def test_export_all(self, ss):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ss.export_all(tmpdir)
            assert len(paths) == 10
            for p in paths:
                assert os.path.exists(p)
                assert p.endswith(".wav")


class TestDisplay:
    def test_list_sigils(self, ss):
        table = ss.list_sigils()
        assert "stabilization" in table
        assert "ߛ" in table
        assert "440" in table

    def test_info(self, ss):
        info = ss.info("ߛ")
        assert "stabilization" in info
        assert "440" in info
        assert "ߛ" in info

    def test_info_unknown_raises(self, ss):
        with pytest.raises(ValueError):
            ss.info("xyz")


class TestIteration:
    def test_iter(self, ss):
        sigils = list(ss)
        assert len(sigils) == 10

    def test_all_sigils(self, ss):
        assert len(ss.all_sigils) == 10

    def test_definitions(self, ss):
        defs = ss.definitions
        assert len(defs) == 10
        assert all(hasattr(d, "char") for d in defs)
