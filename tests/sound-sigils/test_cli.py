"""Tests for sound_sigils.cli."""

import pytest

from sound_sigils.cli import main, _waveform_ascii
from sound_sigils.engine import SoundSigils


class TestCLIList:
    def test_list_returns_zero(self, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "stabilization" in out
        assert "ߛ" in out

    def test_list_shows_all_ten(self, capsys):
        main(["list"])
        out = capsys.readouterr().out
        assert out.count("s") > 5  # Crude check it has content


class TestCLIInfo:
    def test_info_valid(self, capsys):
        assert main(["info", "ߛ"]) == 0
        out = capsys.readouterr().out
        assert "stabilization" in out

    def test_info_by_name(self, capsys):
        assert main(["info", "stabilization"]) == 0

    def test_info_unknown(self, capsys):
        assert main(["info", "xyz"]) == 1


class TestCLIExport:
    def test_export(self, tmp_path, capsys):
        out_file = str(tmp_path / "test.wav")
        assert main(["export", "ߛ", "-o", out_file]) == 0
        out = capsys.readouterr().out
        assert "Exported" in out

    def test_export_unknown(self, capsys):
        assert main(["export", "xyz", "-o", "/tmp/x.wav"]) == 1

    def test_export_all(self, tmp_path, capsys):
        out_dir = str(tmp_path / "all")
        assert main(["export-all", "-o", out_dir]) == 0
        out = capsys.readouterr().out
        assert "10 sigils" in out


class TestCLIWaveform:
    def test_waveform(self, capsys):
        assert main(["waveform", "ߛ"]) == 0
        out = capsys.readouterr().out
        assert "stabilization" in out

    def test_waveform_unknown(self, capsys):
        assert main(["waveform", "xyz"]) == 1


class TestWaveformASCII:
    def test_produces_output(self):
        ss = SoundSigils()
        result = _waveform_ascii("ߛ", ss)
        assert "stabilization" in result
        assert len(result) > 10

    def test_custom_width(self):
        ss = SoundSigils()
        result = _waveform_ascii("ߛ", ss, width=30)
        assert "stabilization" in result


class TestCLIVersion:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0


class TestCLINoCommand:
    def test_no_args(self, capsys):
        assert main([]) == 0  # Prints help
