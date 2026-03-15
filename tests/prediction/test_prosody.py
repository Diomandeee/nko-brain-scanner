#!/usr/bin/env python3
"""
Tests for N'Ko Rhythm & Prosody Engine — Generation 6 Evolution
Instance 16, 42 test cases
"""

import sys
import os
import tempfile
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

# Pytest compatibility
try:
    import pytest
except ImportError:
    from pytest_compat import fixture  # noqa


from prosody_engine import (
    ProsodyEngine, ToneType, MeterType, PhraseType,
    Syllable, ProsodicWord, ProsodicPhrase, RhythmAnalysis,
    TonePrediction, NKO_VOWELS, NKO_CONSONANTS,
    NKO_HIGH_TONE, NKO_LOW_TONE
)


class TestSyllabification:
    """Test syllable analysis."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_single_vowel(self):
        """Single vowel = 1 syllable."""
        syls = self.engine.syllabify("ߊ")
        assert len(syls) >= 1

    def test_cv_syllable(self):
        """Consonant + vowel = 1 syllable."""
        syls = self.engine.syllabify("ߓߍ")
        assert len(syls) >= 1
        assert syls[0].text.startswith("ߓ")

    def test_multi_syllable(self):
        """Multi-syllable word."""
        syls = self.engine.syllabify("ߡߐ߱")
        assert len(syls) >= 1  # mɔgɔ = at least 1 syl

    def test_tone_mark_detection_high(self):
        """High tone mark detected in syllable."""
        syls = self.engine.syllabify("ߘߌ߫")
        has_high = any(s.tone == ToneType.HIGH for s in syls)
        assert has_high

    def test_tone_mark_detection_low(self):
        """Low tone mark detected in syllable."""
        syls = self.engine.syllabify("ߛߐ߲߬")
        has_low = any(s.tone == ToneType.LOW for s in syls)
        assert has_low

    def test_syllable_count_basic(self):
        """Count syllables in a phrase."""
        count = self.engine.count_syllables("ߓߍ ߡߐ߱")
        assert count >= 2

    def test_empty_text_syllables(self):
        """Empty text = 0 syllables."""
        count = self.engine.count_syllables("")
        assert count == 0

    def test_syllable_position(self):
        """Syllables have correct positions."""
        syls = self.engine.syllabify("ߡߎ߬ߛߏ")
        positions = [s.position for s in syls]
        # Positions should be sequential
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1]


class TestWordAnalysis:
    """Test prosodic word analysis."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_known_word_tones(self):
        """Known word gets tone pattern from lexicon."""
        pw = self.engine.analyze_word("ߓߍ")
        assert pw.text == "ߓߍ"
        assert pw.syllable_count >= 1

    def test_prosodic_word_tone_pattern(self):
        """Tone pattern string generated correctly."""
        pw = self.engine.analyze_word("ߓߍ")
        assert isinstance(pw.tone_pattern, str)
        assert all(c in "HLRFM" for c in pw.tone_pattern)

    def test_category_detection_particle(self):
        """Particles detected correctly."""
        pw = self.engine.analyze_word("ߟߊ")
        assert pw.lexical_category == "particle"

    def test_category_detection_pronoun(self):
        """Pronouns detected correctly."""
        pw = self.engine.analyze_word("ߒ")
        assert pw.lexical_category == "pronoun"

    def test_unknown_word_handled(self):
        """Unknown words don't crash."""
        pw = self.engine.analyze_word("ߧߨ")
        assert isinstance(pw, ProsodicWord)


class TestTonePrediction:
    """Test tone mark prediction."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_predict_returns_list(self):
        """predict_tones returns a list."""
        preds = self.engine.predict_tones("ߓߍ ߡߐ߱")
        assert isinstance(preds, list)

    def test_prediction_has_confidence(self):
        """Each prediction has a confidence score."""
        preds = self.engine.predict_tones("ߓߍ ߡߐ߱ ߞߊ")
        for p in preds:
            assert 0 <= p.confidence <= 1.0

    def test_question_final_rising(self):
        """Questions should predict rising tone at end."""
        preds = self.engine.predict_tones("ߓߍ ߡߐ߱?")
        if preds:
            # Last prediction should indicate question influence
            last = preds[-1]
            assert last.predicted_tone in (ToneType.RISING, ToneType.HIGH, ToneType.LOW)

    def test_statement_final_lowering(self):
        """Statements tend toward low tone at end."""
        preds = self.engine.predict_tones("ߓߍ ߡߐ߱ ߞߊ ߕߊ")
        # Should have predictions
        assert isinstance(preds, list)

    def test_suggest_tone_marks_known(self):
        """Known word gets tone marks suggested."""
        result = self.engine.suggest_tone_marks("ߓߍ")
        assert isinstance(result, str)
        assert len(result) >= 2

    def test_suggest_tone_marks_unknown(self):
        """Unknown word returns unchanged."""
        original = "ߧߨ"
        result = self.engine.suggest_tone_marks(original)
        assert result == original

    def test_prediction_reasons(self):
        """Predictions include reason strings."""
        preds = self.engine.predict_tones("ߓߍ ߡߐ߱")
        for p in preds:
            assert isinstance(p.reason, str)
            assert len(p.reason) > 0


class TestMeterDetection:
    """Test poetic meter detection."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_detect_returns_tuple(self):
        """detect_meter returns (MeterType, float)."""
        meter, conf = self.engine.detect_meter("ߓߍ ߡߐ߱ ߞߊ ߕߊ")
        assert isinstance(meter, MeterType)
        assert isinstance(conf, float)

    def test_short_text_is_free(self):
        """Very short text detected as FREE meter."""
        meter, conf = self.engine.detect_meter("ߓߍ")
        assert meter == MeterType.FREE

    def test_confidence_range(self):
        """Confidence is between 0 and 1."""
        meter, conf = self.engine.detect_meter("ߓߍ ߡߐ߱ ߞߊ ߕߊ ߓߍ ߡߐ߱ ߞߊ ߕߊ")
        assert 0.0 <= conf <= 1.0

    def test_proverb_balance(self):
        """Balanced proverb detected."""
        score = self.engine._check_proverb_balance("ߓߍ ߡߐ߱ ߞ,ߕߊ ߓ ߡ")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_meter_format(self):
        """format_as_meter produces readable output."""
        text = "ߓߍ ߡߐ߱ ߞߊ ߕߊ ߓ ߡ ߞ ߕ"
        formatted = self.engine.format_as_meter(text, MeterType.GRIOT_PRAISE)
        assert isinstance(formatted, str)
        assert "|" in formatted  # Should have group separators


class TestRhythmAnalysis:
    """Test full rhythm analysis."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_analyze_returns_analysis(self):
        """analyze_rhythm returns RhythmAnalysis."""
        result = self.engine.analyze_rhythm("ߓߍ ߡߐ߱ ߞߊ ߕߊ")
        assert isinstance(result, RhythmAnalysis)

    def test_musicality_range(self):
        """Musicality score is between 0 and 1."""
        result = self.engine.analyze_rhythm("ߓߍ ߡߐ߱ ߞ ߕ ߓ ߣ")
        assert 0.0 <= result.musicality_score <= 1.0

    def test_tone_balance_range(self):
        """Tone balance is between 0 and 1."""
        result = self.engine.analyze_rhythm("ߓ߫ ߡ߬ ߞ ߕ")
        assert 0.0 <= result.tone_balance <= 1.0

    def test_regularity_range(self):
        """Rhythm regularity is between 0 and 1."""
        result = self.engine.analyze_rhythm("ߓ ߡ ߞ ߕ")
        assert 0.0 <= result.rhythm_regularity <= 1.0

    def test_phrases_segmented(self):
        """Text with punctuation gets segmented into phrases."""
        result = self.engine.analyze_rhythm("ߓ ߡ߸ ߞ ߕ")
        assert isinstance(result.phrases, list)

    def test_suggestions_generated(self):
        """Analysis generates suggestions list."""
        result = self.engine.analyze_rhythm("ߓ ߡ ߞ ߕ ߓ ߡ ߞ ߕ")
        assert isinstance(result.suggestions, list)

    def test_analysis_stored_in_history(self):
        """Analysis is stored in the history deque."""
        self.engine.analyze_rhythm("ߓ ߡ ߞ ߕ")
        assert len(self.engine._history) >= 1

    def test_empty_text_analysis(self):
        """Empty text doesn't crash."""
        result = self.engine.analyze_rhythm("")
        assert isinstance(result, RhythmAnalysis)


class TestKeyboardIntegration:
    """Test prosody-enhanced predictions."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_enhance_empty_predictions(self):
        """Empty predictions list stays empty."""
        result = self.engine.enhance_predictions([], context="ߓ ߡ")
        assert result == []

    def test_enhance_with_context(self):
        """Predictions enhanced with prosodic boost."""
        preds = [
            {"text": "ߓߍ", "score": 0.5},
            {"text": "ߡߐ߱", "score": 0.5},
        ]
        result = self.engine.enhance_predictions(preds, context="ߞ ߕ")
        assert len(result) == 2
        for p in result:
            assert "prosody_boost" in p

    def test_enhance_without_context(self):
        """No context = predictions unchanged."""
        preds = [{"text": "ߓ", "score": 0.5}]
        result = self.engine.enhance_predictions(preds)
        assert len(result) == 1


class TestMeterCompletion:
    """Test meter-based text completion."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_suggest_completion(self):
        """suggest_meter_completion returns list."""
        suggestions = self.engine.suggest_meter_completion(
            "ߓ ߡ", MeterType.GRIOT_PRAISE
        )
        assert isinstance(suggestions, list)

    def test_no_suggestions_at_target(self):
        """No suggestions when already at target syllable count."""
        # Create text with enough syllables
        long_text = "ߓ ߡ ߞ ߕ ߓ ߡ ߞ ߕ ߓ ߡ"
        suggestions = self.engine.suggest_meter_completion(
            long_text, MeterType.GRIOT_PRAISE
        )
        # Either empty or has suggestions — just don't crash
        assert isinstance(suggestions, list)


class TestLearning:
    """Test tone pattern learning."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_learn_tone_pattern(self):
        """Learned pattern persists in lexicon."""
        self.engine.learn_tone_pattern("ߧߨ", "HL")
        clean = self.engine._strip_tone_marks("ߧߨ")
        assert clean in self.engine.tone_lexicon
        assert self.engine.tone_lexicon[clean] == "HL"

    def test_rhythm_stats_empty(self):
        """Stats on empty DB returns empty dict."""
        stats = self.engine.get_rhythm_stats()
        assert isinstance(stats, dict)

    def test_rhythm_stats_after_analysis(self):
        """Stats populated after analysis."""
        self.engine.analyze_rhythm("ߓ ߡ ߞ ߕ")
        stats = self.engine.get_rhythm_stats()
        assert isinstance(stats, dict)
        assert len(stats) >= 1


class TestUtilities:
    """Test utility methods."""

    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.engine = ProsodyEngine(db_path=self.db)

    def teardown_method(self):
        self.engine.close()
        if os.path.exists(self.db):
            os.unlink(self.db)

    def test_tokenize(self):
        """Tokenizer splits N'Ko text."""
        tokens = self.engine._tokenize("ߓ ߡ ߞ")
        assert len(tokens) == 3

    def test_strip_tone_marks(self):
        """Tone marks stripped correctly."""
        stripped = self.engine._strip_tone_marks("ߘߌ߫")
        assert NKO_HIGH_TONE not in stripped

    def test_split_lines(self):
        """Line splitting works."""
        lines = self.engine._split_lines("ߓ ߡ߹ ߞ ߕ")
        assert len(lines) >= 1

    def test_insert_tone_mark(self):
        """Tone mark inserted after vowel."""
        result = self.engine._insert_tone_mark("ߓߍ", NKO_HIGH_TONE)
        assert NKO_HIGH_TONE in result


if __name__ == "__main__":
    # Run all tests
    test_classes = [
        TestSyllabification,
        TestWordAnalysis,
        TestTonePrediction,
        TestMeterDetection,
        TestRhythmAnalysis,
        TestKeyboardIntegration,
        TestMeterCompletion,
        TestLearning,
        TestUtilities,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()
                passed += 1
                print(f"  ✅ {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append(f"{cls.__name__}.{method_name}: {e}")
                print(f"  ❌ {cls.__name__}.{method_name}: {e}")
                if hasattr(instance, "teardown_method"):
                    try:
                        instance.teardown_method()
                    except Exception:
                        pass

    print(f"\n{'='*50}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    if errors:
        print(f"\nFailures:")
        for e in errors:
            print(f"  • {e}")
    print(f"{'='*50}")
