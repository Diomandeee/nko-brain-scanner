"""Tests for sound_sigils.definitions."""

import pytest

from sound_sigils.definitions import (
    SIGIL_DEFINITIONS,
    get_definition_by_char,
    get_definition_by_name,
)


class TestDefinitions:
    def test_ten_definitions(self):
        assert len(SIGIL_DEFINITIONS) == 10

    def test_unique_chars(self):
        chars = [d.char for d in SIGIL_DEFINITIONS]
        assert len(chars) == len(set(chars))

    def test_unique_names(self):
        names = [d.name for d in SIGIL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_all_have_positive_freq(self):
        for d in SIGIL_DEFINITIONS:
            assert d.base_freq > 0, f"{d.name} has non-positive freq"

    def test_all_have_positive_duration(self):
        for d in SIGIL_DEFINITIONS:
            assert d.duration > 0, f"{d.name} has non-positive duration"

    def test_all_nko_unicode(self):
        """All chars should be in the N'Ko Unicode block (U+07C0–U+07FF)."""
        for d in SIGIL_DEFINITIONS:
            code = ord(d.char)
            assert 0x07C0 <= code <= 0x07FF, (
                f"{d.name} char U+{code:04X} is not in N'Ko block"
            )


class TestLookup:
    def test_by_char(self):
        d = get_definition_by_char("ߛ")
        assert d is not None
        assert d.name == "stabilization"

    def test_by_char_miss(self):
        assert get_definition_by_char("A") is None

    def test_by_name(self):
        d = get_definition_by_name("stabilization")
        assert d is not None
        assert d.char == "ߛ"

    def test_by_name_hyphen(self):
        d = get_definition_by_name("place-shift")
        assert d is not None
        assert d.name == "place_shift"

    def test_by_name_miss(self):
        assert get_definition_by_name("nonexistent") is None
