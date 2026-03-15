"""
tests/test_phonetics.py — Comprehensive tests for nko.phonetics
═══════════════════════════════════════════════════════════════════

At least 25 tests covering:
  - Unicode range utilities
  - Character classification
  - IPA conversion (forward and reverse)
  - Tone mark detection and handling
  - Phoneme generation
  - Syllabification
  - Script detection
  - Digit utilities
  - Data loading
  - Edge cases
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nko.phonetics import (
    NKoPhonetics,
    CharInfo,
    Phoneme,
    ToneType,
    CharCategory,
    NKO_BLOCK_START,
    NKO_BLOCK_END,
    UNICODE_RANGE,
    VOWEL_CHARS,
    CONSONANT_CHARS,
    LETTER_CHARS,
    DIGIT_CHARS,
    TONE_MARK_CHARS,
    COMBINING_CHARS,
    PUNCTUATION_CHARS,
    ALL_NKO_CHARS,
    IPA_VOWELS,
    IPA,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def ph():
    """Full NKoPhonetics instance (with JSON data)."""
    return NKoPhonetics()


@pytest.fixture
def ph_light():
    """Lightweight NKoPhonetics instance (tables only, no JSON)."""
    return NKoPhonetics(load_json=False)


# ══════════════════════════════════════════════════════════════
# 1. Unicode Range Constants
# ══════════════════════════════════════════════════════════════

class TestUnicodeRange:
    def test_block_start(self):
        assert NKO_BLOCK_START == 0x07C0

    def test_block_end(self):
        assert NKO_BLOCK_END == 0x07FF

    def test_unicode_range_tuple(self):
        assert UNICODE_RANGE == (0x07C0, 0x07FF)


# ══════════════════════════════════════════════════════════════
# 2. is_nko_char / is_nko_text
# ══════════════════════════════════════════════════════════════

class TestIsNkoChar:
    def test_nko_letter(self, ph_light):
        assert ph_light.is_nko_char("ߞ") is True

    def test_nko_digit(self, ph_light):
        assert ph_light.is_nko_char("߁") is True

    def test_nko_tone_mark(self, ph_light):
        assert ph_light.is_nko_char("߫") is True

    def test_latin_char(self, ph_light):
        assert ph_light.is_nko_char("A") is False

    def test_arabic_char(self, ph_light):
        assert ph_light.is_nko_char("ب") is False

    def test_empty_string(self, ph_light):
        assert ph_light.is_nko_char("") is False

    def test_multi_char_string(self, ph_light):
        assert ph_light.is_nko_char("ߞߏ") is False

    def test_is_nko_text_pure(self, ph_light):
        assert ph_light.is_nko_text("ߒߞߏ") is True

    def test_is_nko_text_with_spaces(self, ph_light):
        assert ph_light.is_nko_text("ߒ ߞ ߏ") is True

    def test_is_nko_text_mixed(self, ph_light):
        assert ph_light.is_nko_text("ߒko") is False


# ══════════════════════════════════════════════════════════════
# 3. Character Classification
# ══════════════════════════════════════════════════════════════

class TestClassification:
    def test_classify_vowel(self, ph_light):
        assert ph_light.classify("ߊ") == CharCategory.VOWEL

    def test_classify_consonant(self, ph_light):
        assert ph_light.classify("ߞ") == CharCategory.CONSONANT

    def test_classify_digit(self, ph_light):
        assert ph_light.classify("߃") == CharCategory.DIGIT

    def test_classify_tone_mark(self, ph_light):
        assert ph_light.classify("߫") == CharCategory.TONE_MARK

    def test_classify_combining_nasal(self, ph_light):
        assert ph_light.classify("߲") == CharCategory.COMBINING

    def test_classify_punctuation(self, ph_light):
        assert ph_light.classify("߹") == CharCategory.PUNCTUATION

    def test_classify_non_nko(self, ph_light):
        assert ph_light.classify("X") == CharCategory.OTHER

    def test_is_vowel(self, ph_light):
        for ch in "ߊߋߌߍߎߏߐ":
            assert ph_light.is_vowel(ch), f"{ch} should be vowel"

    def test_is_consonant(self, ph_light):
        for ch in "ߓߕߞߡߣ":
            assert ph_light.is_consonant(ch), f"{ch} should be consonant"

    def test_is_letter(self, ph_light):
        assert ph_light.is_letter("ߊ") is True
        assert ph_light.is_letter("ߞ") is True
        assert ph_light.is_letter("߁") is False

    def test_all_vowels_count(self):
        assert len(VOWEL_CHARS) == 7

    def test_all_consonants_count(self):
        assert len(CONSONANT_CHARS) == 26  # includes Dagbamma and variants

    def test_digit_chars_count(self):
        assert len(DIGIT_CHARS) == 10

    def test_tone_mark_chars_count(self):
        assert len(TONE_MARK_CHARS) == 5

    def test_letter_chars_is_union(self):
        assert LETTER_CHARS == VOWEL_CHARS | CONSONANT_CHARS


# ══════════════════════════════════════════════════════════════
# 4. Character Info Lookup
# ══════════════════════════════════════════════════════════════

class TestCharInfo:
    def test_get_char_info_vowel(self, ph_light):
        info = ph_light.get_char_info("ߊ")
        assert info is not None
        assert info.char == "ߊ"
        assert info.code == "U+07CA"
        assert info.ipa == "a"
        assert info.category == CharCategory.VOWEL

    def test_get_char_info_consonant(self, ph_light):
        info = ph_light.get_char_info("ߞ")
        assert info is not None
        assert info.name == "Ka"
        assert info.ipa == "k"
        assert info.category == CharCategory.CONSONANT

    def test_get_char_info_digit(self, ph_light):
        info = ph_light.get_char_info("߃")
        assert info is not None
        assert info.digit_value == 3
        assert info.category == CharCategory.DIGIT

    def test_get_char_info_tone(self, ph_light):
        info = ph_light.get_char_info("߫")
        assert info is not None
        assert info.tone_type == ToneType.HIGH
        assert info.category == CharCategory.TONE_MARK

    def test_get_char_info_punctuation(self, ph_light):
        info = ph_light.get_char_info("߹")
        assert info is not None
        assert info.punctuation_eq == "."

    def test_get_char_info_unknown(self, ph_light):
        assert ph_light.get_char_info("Z") is None

    def test_get_all_chars(self, ph_light):
        all_chars = ph_light.get_all_chars()
        assert len(all_chars) > 50
        assert "ߊ" in all_chars
        assert "ߞ" in all_chars

    def test_char_info_frozen(self, ph_light):
        """CharInfo instances are immutable."""
        info = ph_light.get_char_info("ߊ")
        with pytest.raises(AttributeError):
            info.ipa = "changed"  # type: ignore


# ══════════════════════════════════════════════════════════════
# 5. IPA Conversion
# ══════════════════════════════════════════════════════════════

class TestIPAConversion:
    def test_to_ipa_nko(self, ph_light):
        """The word 'N'Ko' (ߒߞߏ) → 'nkɔ'"""
        assert ph_light.to_ipa("ߒߞߏ") == "nkɔ"

    def test_to_ipa_single_vowel(self, ph_light):
        assert ph_light.to_ipa("ߊ") == "a"

    def test_to_ipa_single_consonant(self, ph_light):
        assert ph_light.to_ipa("ߖ") == "dʒ"

    def test_to_ipa_with_space(self, ph_light):
        result = ph_light.to_ipa("ߒ ߏ")
        assert result == "n ɔ"

    def test_to_ipa_strips_tone_by_default(self, ph_light):
        # ߊ + high tone mark
        result = ph_light.to_ipa("ߊ߫")
        assert result == "a"  # tone stripped

    def test_to_ipa_include_tones(self, ph_light):
        result = ph_light.to_ipa("ߊ߫", include_tones=True)
        assert "\u0301" in result  # combining acute accent

    def test_to_ipa_all_vowels(self, ph_light):
        assert ph_light.to_ipa("ߊߋߌߍߎߏߐ") == "aoieuɔə"

    def test_to_ipa_digraph_consonants(self, ph_light):
        """Ja (dʒ) and Cha (tʃ) produce multi-char IPA."""
        assert ph_light.to_ipa("ߖ") == "dʒ"
        assert ph_light.to_ipa("ߗ") == "tʃ"
        assert ph_light.to_ipa("ߜ") == "gb"

    def test_to_ipa_empty(self, ph_light):
        assert ph_light.to_ipa("") == ""

    def test_to_ipa_strips_punctuation(self, ph_light):
        """N'Ko punctuation is silently dropped."""
        assert ph_light.to_ipa("ߊ߹") == "a"

    def test_char_to_ipa_known(self, ph_light):
        assert ph_light.char_to_ipa("ߞ") == "k"

    def test_char_to_ipa_unknown(self, ph_light):
        # Non-N'Ko char returned as-is
        assert ph_light.char_to_ipa("Z") == "Z"


# ══════════════════════════════════════════════════════════════
# 6. IPA → N'Ko (Reverse Mapping)
# ══════════════════════════════════════════════════════════════

class TestIPAReverse:
    def test_ipa_to_nko_simple(self, ph_light):
        result = ph_light.ipa_to_nko("nkɔ")
        assert result == "ߣߞߏ"

    def test_ipa_to_nko_digraphs(self, ph_light):
        result = ph_light.ipa_to_nko("dʒ")
        assert result == "ߖ"

    def test_ipa_to_nko_round_trip_vowels(self, ph_light):
        """IPA round-trip for all vowels (note: primary forms used)."""
        for ch in "ߊߋߌߍߎߏߐ":
            ipa = ph_light.to_ipa(ch)
            back = ph_light.ipa_to_nko(ipa)
            assert back == ch, f"Round-trip failed for {ch} (ipa={ipa}, back={back})"

    def test_ipa_to_nko_with_space(self, ph_light):
        result = ph_light.ipa_to_nko("a b")
        assert result == "ߊ ߓ"


# ══════════════════════════════════════════════════════════════
# 7. Tone Handling
# ══════════════════════════════════════════════════════════════

class TestToneHandling:
    def test_get_tone_high(self, ph_light):
        assert ph_light.get_tone("߫") == ToneType.HIGH

    def test_get_tone_low(self, ph_light):
        assert ph_light.get_tone("߬") == ToneType.LOW

    def test_get_tone_rising(self, ph_light):
        assert ph_light.get_tone("߭") == ToneType.RISING

    def test_get_tone_long(self, ph_light):
        assert ph_light.get_tone("߮") == ToneType.LONG

    def test_get_tone_nasal(self, ph_light):
        assert ph_light.get_tone("߲") == ToneType.NASAL

    def test_get_tone_non_tone(self, ph_light):
        assert ph_light.get_tone("ߊ") is None

    def test_strip_tones(self, ph_light):
        text = "ߊ߫ߓ߬ߌ"
        stripped = ph_light.strip_tones(text)
        assert stripped == "ߊߓߌ"

    def test_strip_tones_no_change(self, ph_light):
        text = "ߒߞߏ"
        assert ph_light.strip_tones(text) == text

    def test_extract_tones(self, ph_light):
        text = "ߊ߫ߓ߬"
        tones = ph_light.extract_tones(text)
        assert len(tones) == 2
        assert tones[0] == (1, ToneType.HIGH)
        assert tones[1] == (3, ToneType.LOW)

    def test_has_tone_marks_true(self, ph_light):
        assert ph_light.has_tone_marks("ߊ߫") is True

    def test_has_tone_marks_false(self, ph_light):
        assert ph_light.has_tone_marks("ߒߞߏ") is False


# ══════════════════════════════════════════════════════════════
# 8. Phoneme Generation
# ══════════════════════════════════════════════════════════════

class TestPhonemeGeneration:
    def test_to_phonemes_basic(self, ph_light):
        phonemes = ph_light.to_phonemes("ߒߞߏ")
        assert len(phonemes) == 3
        assert phonemes[0].symbol == "n"
        assert phonemes[1].symbol == "k"
        assert phonemes[2].symbol == "ɔ"

    def test_to_phonemes_source_chars(self, ph_light):
        phonemes = ph_light.to_phonemes("ߒߞ")
        assert phonemes[0].source_char == "ߒ"
        assert phonemes[1].source_char == "ߞ"

    def test_to_phonemes_tone_attached(self, ph_light):
        """Tone mark attaches to the preceding phoneme."""
        phonemes = ph_light.to_phonemes("ߊ߫")
        assert len(phonemes) == 1
        assert phonemes[0].symbol == "a"
        assert phonemes[0].tone == ToneType.HIGH

    def test_to_phonemes_duration(self, ph_light):
        phonemes = ph_light.to_phonemes("ߊߞ")
        # Vowel duration > consonant duration
        assert phonemes[0].duration_ms == 120  # vowel
        assert phonemes[1].duration_ms == 80   # consonant

    def test_to_phonemes_empty(self, ph_light):
        assert ph_light.to_phonemes("") == []


# ══════════════════════════════════════════════════════════════
# 9. Syllabification
# ══════════════════════════════════════════════════════════════

class TestSyllabification:
    def test_syllabify_cv(self, ph_light):
        syllables = ph_light.syllabify_ipa("ba")
        assert syllables == ["ba"]

    def test_syllabify_cvcv(self, ph_light):
        syllables = ph_light.syllabify_ipa("baba")
        assert syllables == ["ba", "ba"]

    def test_syllabify_single_vowel(self, ph_light):
        assert ph_light.syllabify_ipa("a") == ["a"]

    def test_syllabify_consonant_only(self, ph_light):
        assert ph_light.syllabify_ipa("nk") == ["nk"]


# ══════════════════════════════════════════════════════════════
# 10. Script Detection
# ══════════════════════════════════════════════════════════════

class TestScriptDetection:
    def test_detect_nko(self, ph_light):
        assert ph_light.detect_script("ߒߞߏ") == "nko"

    def test_detect_latin(self, ph_light):
        assert ph_light.detect_script("hello") == "latin"

    def test_detect_arabic(self, ph_light):
        assert ph_light.detect_script("سلام") == "arabic"

    def test_detect_empty(self, ph_light):
        assert ph_light.detect_script("") == "latin"  # fallback

    def test_detect_mixed(self, ph_light):
        # Should detect as mixed if neither dominates
        result = ph_light.detect_script("ߊ ߋ abc def ghi")
        assert result in ("latin", "mixed")


# ══════════════════════════════════════════════════════════════
# 11. Digit Utilities
# ══════════════════════════════════════════════════════════════

class TestDigitUtilities:
    def test_nko_digit_value(self, ph_light):
        assert ph_light.nko_digit_value("߀") == 0
        assert ph_light.nko_digit_value("߅") == 5
        assert ph_light.nko_digit_value("߉") == 9

    def test_nko_digit_value_non_digit(self, ph_light):
        assert ph_light.nko_digit_value("ߊ") is None

    def test_int_to_nko_digits_zero(self, ph_light):
        assert ph_light.int_to_nko_digits(0) == "߀"

    def test_int_to_nko_digits_multi(self, ph_light):
        result = ph_light.int_to_nko_digits(42)
        assert result == "߄߂"

    def test_int_to_nko_digits_negative(self, ph_light):
        with pytest.raises(ValueError):
            ph_light.int_to_nko_digits(-1)


# ══════════════════════════════════════════════════════════════
# 12. NKO Purity
# ══════════════════════════════════════════════════════════════

class TestPurity:
    def test_purity_pure(self, ph_light):
        assert ph_light.nko_purity("ߒߞߏ") == 1.0

    def test_purity_empty(self, ph_light):
        assert ph_light.nko_purity("") == 0.0

    def test_purity_mixed(self, ph_light):
        # "ߊabc" = 1 nko + 3 latin = 0.25
        assert ph_light.nko_purity("ߊabc") == 0.25


# ══════════════════════════════════════════════════════════════
# 13. JSON Data Loading
# ══════════════════════════════════════════════════════════════

class TestDataLoading:
    def test_json_loaded(self, ph):
        assert ph.json_data is not None
        assert "characters" in ph.json_data
        assert "meta" in ph.json_data

    def test_json_not_loaded(self, ph_light):
        assert ph_light.json_data is None

    def test_vocabulary_access(self, ph):
        # vocabulary can be a dict (keyed by category) or list
        assert isinstance(ph.vocabulary, (list, dict))

    def test_proverbs_access(self, ph):
        # proverbs can be a dict (keyed by language) or list
        assert isinstance(ph.proverbs, (list, dict))


# ══════════════════════════════════════════════════════════════
# 14. Pronunciation Guide
# ══════════════════════════════════════════════════════════════

class TestPronunciationGuide:
    def test_guide_basic(self, ph_light):
        guide = ph_light.pronunciation_guide("ߒߞߏ")
        assert len(guide) == 3
        assert guide[0]["char"] == "ߒ"
        assert guide[0]["ipa"] == "n"
        assert guide[1]["category"] == "consonant"

    def test_guide_with_space(self, ph_light):
        guide = ph_light.pronunciation_guide("ߊ ߌ")
        assert len(guide) == 3
        assert guide[1]["category"] == "space"


# ══════════════════════════════════════════════════════════════
# 15. Singleton / Module-level IPA
# ══════════════════════════════════════════════════════════════

class TestSingleton:
    def test_ipa_singleton_exists(self):
        assert IPA is not None
        assert isinstance(IPA, NKoPhonetics)

    def test_ipa_singleton_works(self):
        assert IPA.to_ipa("ߒߞߏ") == "nkɔ"

    def test_repr(self, ph):
        r = repr(ph)
        assert "NKoPhonetics" in r
        assert "json loaded" in r

    def test_repr_light(self, ph_light):
        r = repr(ph_light)
        assert "tables only" in r


# ══════════════════════════════════════════════════════════════
# 16. Edge Cases
# ══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_all_nko_chars_non_empty(self):
        assert len(ALL_NKO_CHARS) > 50

    def test_ipa_vowels_set(self):
        assert "a" in IPA_VOWELS
        assert "ɔ" in IPA_VOWELS

    def test_nko_n_in_consonants(self):
        """ߒ (the N in N'Ko) is classified as consonant."""
        assert "ߒ" in CONSONANT_CHARS

    def test_classify_all_static_chars(self, ph_light):
        """Every char in our static tables classifies to the right category."""
        for ch in VOWEL_CHARS:
            assert ph_light.classify(ch) == CharCategory.VOWEL
        for ch in CONSONANT_CHARS:
            assert ph_light.classify(ch) == CharCategory.CONSONANT
        for ch in DIGIT_CHARS:
            assert ph_light.classify(ch) == CharCategory.DIGIT
        for ch in TONE_MARK_CHARS:
            assert ph_light.classify(ch) == CharCategory.TONE_MARK
        for ch in PUNCTUATION_CHARS:
            assert ph_light.classify(ch) == CharCategory.PUNCTUATION

    def test_to_ipa_unknown_passthrough(self, ph_light):
        """Non-N'Ko chars pass through to_ipa unchanged."""
        assert ph_light.to_ipa("ߊXߊ") == "aXa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
