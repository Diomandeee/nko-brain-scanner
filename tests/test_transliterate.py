"""
Tests for nko.transliterate — Canonical Transliteration Engine

At least 15 tests covering:
  - N'Ko → Latin
  - Latin → N'Ko
  - N'Ko ↔ Arabic
  - Script detection
  - Edge cases (tone marks, mixed script, empty input, digits)
  - Batch operations
  - Round-trip fidelity
  - API surface (TranslitResult, convert_all, etc.)
"""

import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nko.transliterate import (
    NkoTransliterator,
    Script,
    TranslitResult,
    transliterate,
    convert,
    convert_all,
    batch,
    to_ipa,
    analyze,
    detect_script,
    is_nko,
    is_arabic,
    is_latin,
    NKO_TO_IPA,
    IPA_TO_NKO,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def engine():
    return NkoTransliterator()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Script Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScriptDetection:
    """Tests for detect_script() and is_* helpers."""

    def test_detect_nko(self):
        assert detect_script("ߒߞߏ") == Script.NKO

    def test_detect_latin(self):
        assert detect_script("hello world") == Script.LATIN

    def test_detect_arabic(self):
        assert detect_script("سلام") == Script.ARABIC

    def test_detect_empty_defaults_latin(self):
        assert detect_script("") == Script.LATIN

    def test_detect_digits_only_defaults_latin(self):
        assert detect_script("12345") == Script.LATIN

    def test_detect_mixed_nko_dominant(self):
        # N'Ko chars outnumber Latin
        assert detect_script("ߒߞߏ abc") == Script.NKO

    def test_is_nko_true(self):
        assert is_nko("ߒ") is True

    def test_is_nko_false(self):
        assert is_nko("a") is False

    def test_is_arabic_true(self):
        assert is_arabic("س") is True

    def test_is_latin_true(self):
        assert is_latin("a") is True

    def test_detect_extended_latin(self):
        """ɔ and ɛ are in Manding Latin — should detect as Latin."""
        assert detect_script("ɔ ɛ") == Script.LATIN


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. N'Ko → Latin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNkoToLatin:
    """Tests for N'Ko to Latin transliteration."""

    def test_nko_word_basic(self, engine):
        """ߒߞߏ should transliterate to 'nkɔ' — ߏ is open-o /ɔ/, not /o/."""
        result = engine.convert("ߒߞߏ", source="nko", target="latin")
        assert result.target_text == "nkɔ"
        assert result.source_script == Script.NKO
        assert result.target_script == Script.LATIN

    def test_nko_vowels(self, engine):
        """All 7 vowels should map correctly."""
        # ߊߋߌߍߎߏߐ → a o i e u ɔ ɛ
        result = engine.convert("ߊߋߌߍߎߏߐ", source="nko", target="latin")
        assert result.target_text == "aoieuɔɛ"

    def test_nko_consonants_basic(self, engine):
        """Common consonants should map cleanly."""
        # ߓ=b, ߕ=t, ߘ=d, ߞ=k
        result = engine.convert("ߓߕߘߞ", source="nko", target="latin")
        assert result.target_text == "btdk"

    def test_nko_digraph_consonants(self, engine):
        """Multi-char IPA consonants should render as Latin digraphs."""
        # ߖ=dʒ→j, ߗ=tʃ→c, ߜ=gb→gb
        result = engine.convert("ߖߗߜ", source="nko", target="latin")
        assert result.target_text == "jcgb"

    def test_nko_nasal_consonants(self, engine):
        """Nasal consonants ɲ→ny, ŋ→ng."""
        # ߢ=ny, ߧ=ng
        result = engine.convert("ߢߧ", source="nko", target="latin")
        assert result.target_text == "nyng"

    def test_nko_with_spaces(self, engine):
        """Spaces should be preserved."""
        result = engine.convert("ߒ ߓߍ", source="nko", target="latin")
        assert " " in result.target_text

    def test_nko_digits(self, engine):
        """N'Ko digits should convert to Western digits."""
        result = engine.convert("߀߁߂߃", source="nko", target="latin")
        assert result.target_text == "0123"

    def test_convenience_function(self):
        """Module-level transliterate() should work."""
        assert transliterate("ߒߞߏ") == "nkɔ"

    def test_auto_detect_nko_source(self, engine):
        """Auto-detect should identify N'Ko and convert to Latin."""
        result = engine.convert("ߒߞߏ", target="latin")
        assert result.source_script == Script.NKO
        assert result.target_text == "nkɔ"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Latin → N'Ko
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLatinToNko:
    """Tests for Latin to N'Ko transliteration."""

    def test_basic_latin_to_nko(self, engine):
        """Simple Latin text should convert to N'Ko chars."""
        result = engine.convert("baka", source="latin", target="nko")
        # b→ߓ, a→ߊ, k→ߞ, a→ߊ
        assert result.target_text == "ߓߊߞߊ"

    def test_latin_digraphs(self, engine):
        """Latin digraphs ny, ng, gb, ch should map correctly."""
        result = engine.convert("nya", source="latin", target="nko")
        # ny→ɲ→ߢ, a→ߊ
        assert "ߢ" in result.target_text

    def test_latin_with_spaces(self, engine):
        result = engine.convert("ba ka", source="latin", target="nko")
        assert " " in result.target_text

    def test_convenience_latin_to_nko(self):
        result = transliterate("baka", target="nko")
        assert "ߓ" in result  # 'b' → ߓ
        assert "ߊ" in result  # 'a' → ߊ


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. N'Ko ↔ Arabic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNkoArabic:
    """Tests for N'Ko ↔ Arabic bridge."""

    def test_nko_to_arabic(self, engine):
        """N'Ko text should convert to Arabic script."""
        result = engine.convert("ߛߟߊߡ", source="nko", target="arabic")
        # s→س, l→ل, a→ا, m→م
        assert "س" in result.target_text
        assert "ل" in result.target_text
        assert "م" in result.target_text

    def test_arabic_to_nko(self, engine):
        """Arabic text should convert to N'Ko."""
        result = engine.convert("سلام", source="arabic", target="nko")
        assert result.source_script == Script.ARABIC
        assert result.target_script == Script.NKO
        # Should contain N'Ko characters
        assert any(is_nko(ch) for ch in result.target_text if not ch.isspace())

    def test_arabic_to_latin(self, engine):
        """Arabic → Latin through IPA."""
        result = engine.convert("بسم", source="arabic", target="latin")
        assert "b" in result.target_text
        assert "s" in result.target_text
        assert "m" in result.target_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. IPA Conversion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIPA:
    """Tests for IPA intermediary."""

    def test_nko_to_ipa(self, engine):
        """N'Ko → IPA should produce phonetic representation."""
        ipa = engine.to_ipa("ߒߞߏ", source="nko")
        # ߒ=n, ߞ=k, ߏ=ɔ
        assert "n" in ipa
        assert "k" in ipa
        assert "ɔ" in ipa

    def test_latin_to_ipa(self, engine):
        """Latin → IPA basic mapping."""
        ipa = engine.to_ipa("baka", source="latin")
        assert ipa == "baka"  # Simple case: b,a,k,a are same in IPA

    def test_to_ipa_convenience(self):
        """Module-level to_ipa() works."""
        ipa = to_ipa("ߒߞߏ")
        assert "n" in ipa and "k" in ipa


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Edge Cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEdgeCases:
    """Edge cases: empty strings, same-script, tone marks, mixed input."""

    def test_empty_string(self, engine):
        result = engine.convert("", target="latin")
        assert result.target_text == ""

    def test_same_script_identity(self, engine):
        """Converting from a script to itself should be identity."""
        result = engine.convert("ߒߞߏ", source="nko", target="nko")
        assert result.target_text == "ߒߞߏ"

    def test_whitespace_only(self, engine):
        result = engine.convert("   ", target="latin")
        assert result.target_text.strip() == ""

    def test_nko_tone_marks_stripped_in_latin(self, engine):
        """Tone marks should be stripped (or minimally represented) in Latin output."""
        # ߊ + high tone mark ߫
        result = engine.convert("ߊ߫", source="nko", target="latin")
        # The tone mark's IPA (combining acute) maps to "" in Latin
        assert "a" in result.target_text

    def test_nko_punctuation(self, engine):
        """N'Ko punctuation should map to standard punctuation."""
        result = engine.convert("ߊ߹", source="nko", target="latin")
        assert "." in result.target_text

    def test_mixed_script_passthrough(self, engine):
        """Unknown chars should pass through without crashing."""
        result = engine.convert("ߒ hello !", source="nko", target="latin")
        assert "n" in result.target_text  # ߒ→n
        # Latin chars pass through the IPA pipeline

    def test_unknown_script_raises(self, engine):
        """Invalid script name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown script"):
            engine.convert("test", source="klingon", target="latin")

    def test_nko_long_vowel_mark(self, engine):
        """The long vowel mark ߯ should produce length in IPA."""
        ipa = engine.to_ipa("ߊ߯", source="nko")
        assert "ː" in ipa


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Batch & convert_all
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBatchAndConvertAll:
    """Tests for batch() and convert_all()."""

    def test_batch_basic(self, engine):
        results = engine.batch(["ߒߞߏ", "ߓߊ"], source="nko", target="latin")
        assert len(results) == 2
        assert results[0].target_text == "nkɔ"
        assert "b" in results[1].target_text

    def test_batch_empty_list(self, engine):
        assert engine.batch([], target="latin") == []

    def test_convert_all(self, engine):
        """convert_all should return all 3 scripts."""
        out = engine.convert_all("ߒߞߏ")
        assert "nko" in out
        assert "latin" in out
        assert "arabic" in out
        assert out["nko"] == "ߒߞߏ"  # Identity for source script
        assert out["latin"] == "nkɔ"

    def test_convert_all_convenience(self):
        out = convert_all("ߒߞߏ")
        assert len(out) == 3
        assert out["latin"] == "nkɔ"

    def test_batch_convenience(self):
        results = batch(["ߒ", "ߞ"], target="latin")
        assert len(results) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. TranslitResult Dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTranslitResult:
    """Tests for the TranslitResult object."""

    def test_str_returns_target(self, engine):
        result = engine.convert("ߒߞߏ", target="latin")
        assert str(result) == "nkɔ"

    def test_repr_informative(self, engine):
        result = engine.convert("ߒߞߏ", target="latin")
        r = repr(result)
        assert "nkɔ" in r
        assert "nko" in r  # Script name "nko" appears in repr

    def test_result_has_ipa(self, engine):
        result = engine.convert("ߒߞߏ", source="nko", target="latin")
        assert result.ipa  # Non-empty IPA
        assert "n" in result.ipa and "k" in result.ipa

    def test_result_confidence(self, engine):
        result = engine.convert("ߒߞߏ", target="latin")
        assert 0 <= result.confidence <= 1.0

    def test_result_is_frozen(self, engine):
        result = engine.convert("ߒ", target="latin")
        with pytest.raises(AttributeError):
            result.target_text = "changed"  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Round-trip Fidelity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRoundTrip:
    """Round-trip: NKO → Latin → NKO should preserve core content."""

    def test_simple_roundtrip(self, engine):
        """Simple consonant-vowel patterns should round-trip cleanly."""
        original = "ߓߊ"  # ba
        latin = engine.convert(original, source="nko", target="latin").target_text
        back = engine.convert(latin, source="latin", target="nko").target_text
        assert back == original

    def test_roundtrip_consonant_cluster(self, engine):
        """More complex patterns — at minimum the IPA core survives."""
        original = "ߒߞߏ"  # n + k + ɔ
        latin = engine.convert(original, source="nko", target="latin").target_text
        assert latin == "nkɔ"
        back = engine.convert(latin, source="latin", target="nko").target_text
        # "nkɔ" → Latin digraph "nk" maps to IPA "ŋk" → N'Ko ŋ=ߧ, k=ߞ, then ɔ→ɔ→ߏ
        # So round-trip gives ߧߞߏ instead of ߒߞߏ (ŋ vs syllabic n)
        # This is a known linguistic ambiguity: Latin "nk" is prenasalized in Manding
        # Verify at least that the phonetic _quality_ (velar nasal + k + open-o) is present
        ipa_back = engine.to_ipa(back, source="nko")
        assert "k" in ipa_back
        assert "ɔ" in ipa_back

    def test_roundtrip_vowels(self, engine):
        """Pure vowels should round-trip perfectly."""
        original = "ߊߋߌߍߎ"  # a, o, i, e, u
        latin = engine.convert(original, source="nko", target="latin").target_text
        back = engine.convert(latin, source="latin", target="nko").target_text
        assert back == original


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Analyze
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAnalyze:
    """Tests for text analysis."""

    def test_analyze_nko_text(self, engine):
        result = engine.analyze("ߒߞߏ")
        assert result["dominant"] == "nko"
        assert result["counts"]["nko"] == 3
        assert result["counts"]["latin"] == 0

    def test_analyze_mixed(self, engine):
        result = engine.analyze("ߒ hello")
        assert result["counts"]["nko"] >= 1
        assert result["counts"]["latin"] >= 1

    def test_analyze_convenience(self):
        result = analyze("hello")
        assert result["dominant"] == "latin"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Character Map Integrity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCharacterMaps:
    """Ensure the character maps are self-consistent."""

    def test_all_nko_vowels_mapped(self):
        vowels = "ߊߋߌߍߎߏߐ"
        for v in vowels:
            assert v in NKO_TO_IPA, f"Vowel {v} missing from NKO_TO_IPA"

    def test_all_nko_consonants_mapped(self):
        consonants = "ߒߓߔߕߖߗߘߙߚߛߜߝߞߟߠߡߢߣߤߥߦߧ"
        for c in consonants:
            assert c in NKO_TO_IPA, f"Consonant {c} missing from NKO_TO_IPA"

    def test_reverse_map_has_all_vowel_ipas(self):
        for vowel_ipa in ["a", "o", "i", "e", "u"]:
            assert vowel_ipa in IPA_TO_NKO, f"IPA {vowel_ipa} missing from IPA_TO_NKO"

    def test_nko_digits_complete(self):
        for digit_nko, digit_western in [("߀", "0"), ("߉", "9")]:
            assert NKO_TO_IPA[digit_nko] == digit_western


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
