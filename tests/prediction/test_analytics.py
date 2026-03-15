#!/usr/bin/env python3
"""
Tests for N'Ko Typing Analytics & Adaptive Intelligence Engine.

Comprehensive test coverage for all analytics subsystems:
- Confusion Matrix
- Flow State Detection
- Frustration Detection
- Key Heatmap
- Gamification
- WPM Calculation
- Proficiency Assessment
- Error Classification
- Main Analytics Engine integration
"""

import sys
import os
import json
import tempfile
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from tests.pytest_compat import SimpleTestCase
except ImportError:
    import unittest
    SimpleTestCase = unittest.TestCase

from lib.analytics_engine import (
    AnalyticsEngine,
    ConfusionMatrix,
    FlowStateDetector,
    FrustrationDetector,
    KeyHeatmap,
    GamificationEngine,
    WPMCalculator,
    ProficiencyAssessor,
    ErrorClassifier,
    KeystrokeEvent,
    ErrorRecord,
    SessionStats,
    AdaptiveSettings,
    ProficiencyLevel,
    ErrorType,
    FlowState,
    EventType,
    ADJACENCY_MAP,
    NKO_KEYBOARD_ROWS,
)


class TestConfusionMatrix(SimpleTestCase):
    """Test character confusion tracking."""

    def test_record_correct(self):
        cm = ConfusionMatrix()
        cm.record_correct("ߊ")
        cm.record_correct("ߊ")
        cm.record_correct("ߊ")
        assert cm.get_confusion_rate("ߊ") == 0.0

    def test_record_error(self):
        cm = ConfusionMatrix()
        cm.record_correct("ߊ")
        cm.record_correct("ߊ")
        cm.record_error("ߊ", "ߋ")
        rate = cm.get_confusion_rate("ߊ")
        assert 0.3 < rate < 0.4  # 1/3 ≈ 0.333

    def test_most_confused_with(self):
        cm = ConfusionMatrix()
        cm.record_error("ߊ", "ߋ")
        cm.record_error("ߊ", "ߋ")
        cm.record_error("ߊ", "ߌ")
        confused = cm.get_most_confused_with("ߊ", top_n=2)
        assert len(confused) == 2
        assert confused[0][0] == "ߋ"  # Most confused
        assert confused[0][1] == 2

    def test_most_confused_pairs(self):
        cm = ConfusionMatrix()
        cm.record_error("ߊ", "ߋ")
        cm.record_error("ߊ", "ߋ")
        cm.record_error("ߓ", "ߔ")
        pairs = cm.get_most_confused_pairs(5)
        assert len(pairs) == 2
        assert pairs[0] == ("ߊ", "ߋ", 2)

    def test_problem_characters(self):
        cm = ConfusionMatrix()
        # ߊ: 1 correct, 1 error = 50% error rate
        cm.record_correct("ߊ")
        cm.record_error("ߊ", "ߋ")
        # ߓ: 9 correct, 1 error = 10% error rate
        for _ in range(9):
            cm.record_correct("ߓ")
        cm.record_error("ߓ", "ߔ")

        problems = cm.get_problem_characters(threshold=0.2)
        assert len(problems) == 1
        assert problems[0][0] == "ߊ"

    def test_serialization(self):
        cm = ConfusionMatrix()
        cm.record_correct("ߊ")
        cm.record_error("ߊ", "ߋ")
        data = cm.to_dict()

        cm2 = ConfusionMatrix.from_dict(data)
        assert cm2.get_confusion_rate("ߊ") == cm.get_confusion_rate("ߊ")

    def test_empty_matrix(self):
        cm = ConfusionMatrix()
        assert cm.get_confusion_rate("ߊ") == 0.0
        assert cm.get_most_confused_with("ߊ") == []
        assert cm.get_most_confused_pairs() == []
        assert cm.get_problem_characters() == []


class TestFlowStateDetector(SimpleTestCase):
    """Test typing flow state detection."""

    def test_initial_state(self):
        fsd = FlowStateDetector()
        assert fsd.current_state == FlowState.COMFORTABLE

    def test_struggling_state(self):
        fsd = FlowStateDetector(window_size=10)
        # Many errors and backspaces
        for _ in range(10):
            fsd.record_keystroke(500, True, True)
        assert fsd.current_state == FlowState.STRUGGLING

    def test_flowing_state(self):
        fsd = FlowStateDetector(window_size=10)
        # Fast, accurate, rhythmic
        for _ in range(10):
            fsd.record_keystroke(300, False, False)
        assert fsd.current_state in (FlowState.FLOWING, FlowState.COMFORTABLE)

    def test_zone_state(self):
        fsd = FlowStateDetector(window_size=10)
        # Very fast, no errors, very consistent rhythm
        for _ in range(10):
            fsd.record_keystroke(150, False, False)
        assert fsd.current_state in (FlowState.ZONE, FlowState.FLOWING)

    def test_learning_state(self):
        fsd = FlowStateDetector(window_size=10)
        # Slow with some errors
        for i in range(10):
            fsd.record_keystroke(2000, i % 4 == 0, False)
        state = fsd.current_state
        assert state in (FlowState.LEARNING, FlowState.STRUGGLING, FlowState.COMFORTABLE)

    def test_flow_percentage(self):
        fsd = FlowStateDetector(window_size=5)
        # Start comfortable
        for _ in range(5):
            fsd.record_keystroke(500, False, False)
        pct = fsd.get_flow_percentage()
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 1.0


class TestFrustrationDetector(SimpleTestCase):
    """Test frustration detection."""

    def test_no_frustration(self):
        fd = FrustrationDetector()
        event = KeystrokeEvent(
            event_type=EventType.KEY_DOWN,
            character="ߊ",
            timestamp_ms=1000,
        )
        level = fd.record_event(event)
        assert level < 0.3

    def test_rapid_backspaces(self):
        fd = FrustrationDetector()
        base_time = 1000
        # Rapid backspaces within window
        for i in range(5):
            event = KeystrokeEvent(
                event_type=EventType.BACKSPACE,
                character="\b",
                timestamp_ms=base_time + i * 100,
            )
            fd.record_event(event)
        assert fd.frustration_level > 0.2

    def test_prediction_rejections(self):
        fd = FrustrationDetector()
        base_time = 1000
        for i in range(6):
            event = KeystrokeEvent(
                event_type=EventType.PREDICTION_REJECT,
                timestamp_ms=base_time + i * 500,
            )
            fd.record_event(event)
        assert fd.frustration_level > 0.1

    def test_not_frustrated_initially(self):
        fd = FrustrationDetector()
        assert not fd.is_frustrated

    def test_help_suggestion_none_when_calm(self):
        fd = FrustrationDetector()
        assert fd.get_help_suggestion() is None


class TestKeyHeatmap(SimpleTestCase):
    """Test key usage heatmap generation."""

    def test_record_and_retrieve(self):
        hm = KeyHeatmap()
        hm.record_key("ߊ", speed_ms=200)
        hm.record_key("ߊ", speed_ms=300)
        hm.record_key("ߋ", speed_ms=150)

        usage = hm.get_usage_heatmap()
        assert usage["ߊ"] == 1.0  # Most used = 1.0
        assert usage["ߋ"] == 0.5  # Half as much

    def test_error_heatmap(self):
        hm = KeyHeatmap()
        hm.record_key("ߊ", is_error=True)
        hm.record_key("ߊ", is_error=False)
        hm.record_key("ߋ", is_error=False)

        errors = hm.get_error_heatmap()
        assert errors["ߊ"] == 0.5  # 1/2
        assert errors["ߋ"] == 0.0

    def test_speed_heatmap(self):
        hm = KeyHeatmap()
        hm.record_key("ߊ", speed_ms=200)
        hm.record_key("ߊ", speed_ms=400)
        hm.record_key("ߋ", speed_ms=100)

        speeds = hm.get_speed_heatmap()
        assert speeds["ߊ"] == 300.0  # Average of 200, 400
        assert speeds["ߋ"] == 100.0

    def test_most_used(self):
        hm = KeyHeatmap()
        for _ in range(5):
            hm.record_key("ߊ")
        for _ in range(3):
            hm.record_key("ߋ")
        hm.record_key("ߌ")

        most = hm.get_most_used(2)
        assert most[0] == ("ߊ", 5)
        assert most[1] == ("ߋ", 3)

    def test_serialization(self):
        hm = KeyHeatmap()
        hm.record_key("ߊ", speed_ms=200, is_error=True)
        data = hm.to_dict()

        hm2 = KeyHeatmap.from_dict(data)
        assert hm2.get_usage_heatmap() == hm.get_usage_heatmap()

    def test_empty_heatmap(self):
        hm = KeyHeatmap()
        assert hm.get_usage_heatmap() == {}
        assert hm.get_error_heatmap() == {}
        assert hm.get_speed_heatmap() == {}
        assert hm.get_most_used() == []


class TestGamificationEngine(SimpleTestCase):
    """Test gamification and milestone tracking."""

    def test_first_word_milestone(self):
        ge = GamificationEngine()
        ge.record_word("ߊߟߊ߫")
        unlocked = ge.get_unlocked()
        names = [m.name for m in unlocked]
        assert "first_word" in names

    def test_100_words_milestone(self):
        ge = GamificationEngine()
        for i in range(100):
            ge.record_word(f"word{i}")
        unlocked = ge.get_unlocked()
        names = [m.name for m in unlocked]
        assert "100_words" in names

    def test_streak_tracking(self):
        ge = GamificationEngine()
        ge.record_day_active("2026-02-01")
        ge.record_day_active("2026-02-02")
        ge.record_day_active("2026-02-03")
        assert ge._daily_streak == 3

    def test_streak_broken(self):
        ge = GamificationEngine()
        ge.record_day_active("2026-02-01")
        ge.record_day_active("2026-02-03")  # Skipped a day
        assert ge._daily_streak == 1

    def test_speed_milestone(self):
        ge = GamificationEngine()
        ge.check_speed_milestone(15.0)
        unlocked = ge.get_unlocked()
        names = [m.name for m in unlocked]
        assert "speed_10wpm" in names

    def test_accuracy_milestone(self):
        ge = GamificationEngine()
        ge.check_accuracy_milestone(0.92)
        unlocked = ge.get_unlocked()
        names = [m.name for m in unlocked]
        assert "accuracy_90" in names

    def test_daily_progress(self):
        ge = GamificationEngine()
        for _ in range(25):
            ge.record_word("test")
        progress = ge.get_daily_progress()
        assert progress["words_today"] == 25
        assert progress["progress"] == 0.5  # 25/50

    def test_unique_chars_tracking(self):
        ge = GamificationEngine()
        ge.record_word("ߊߋߌ")
        ge.record_word("ߍߎ")
        assert len(ge._unique_chars_typed) == 5

    def test_flow_state_milestone(self):
        ge = GamificationEngine()
        ge.record_flow_state()
        unlocked = ge.get_unlocked()
        names = [m.name for m in unlocked]
        assert "flow_state" in names

    def test_next_milestone(self):
        ge = GamificationEngine()
        next_m = ge.get_next_milestone()
        assert next_m is not None
        assert not next_m.achieved

    def test_serialization(self):
        ge = GamificationEngine()
        ge.record_word("test")
        ge.record_day_active("2026-02-03")
        data = ge.to_dict()

        ge2 = GamificationEngine.from_dict(data)
        assert ge2._total_words == 1
        assert ge2._daily_streak == 1


class TestWPMCalculator(SimpleTestCase):
    """Test words-per-minute calculation."""

    def test_initial_wpm(self):
        wpm = WPMCalculator()
        assert wpm.get_current_wpm() == 0.0

    def test_wpm_calculation(self):
        wpm = WPMCalculator(window_size=60)
        base = 0
        # Simulate 10 words in 30 seconds = 20 WPM
        for i in range(10):
            wpm.record_word(base + i * 3000)  # 3 seconds per word
        result = wpm.get_current_wpm()
        assert result > 0  # Should be roughly 20 WPM

    def test_char_speed(self):
        wpm = WPMCalculator()
        for i in range(20):
            wpm.record_character(i * 200)  # 200ms between chars = 5 CPS
        speed = wpm.get_char_speed()
        assert speed > 0

    def test_peak_wpm_tracking(self):
        wpm = WPMCalculator(window_size=60)
        for i in range(5):
            wpm.record_word(i * 2000)
        _ = wpm.get_current_wpm()
        assert wpm.peak_wpm >= 0


class TestProficiencyAssessor(SimpleTestCase):
    """Test proficiency assessment."""

    def test_novice_assessment(self):
        pa = ProficiencyAssessor()
        profile = pa.assess(
            wpm=3.0, accuracy=0.5, unique_chars=5,
            flow_percentage=0.0, tone_accuracy=0.3
        )
        assert profile.level in (ProficiencyLevel.NOVICE, ProficiencyLevel.BEGINNER)
        assert profile.score < 40

    def test_intermediate_assessment(self):
        pa = ProficiencyAssessor()
        profile = pa.assess(
            wpm=15.0, accuracy=0.8, unique_chars=20,
            flow_percentage=0.3, tone_accuracy=0.7
        )
        assert profile.level in (ProficiencyLevel.INTERMEDIATE, ProficiencyLevel.ADVANCED)

    def test_master_assessment(self):
        pa = ProficiencyAssessor()
        profile = pa.assess(
            wpm=35.0, accuracy=0.98, unique_chars=27,
            flow_percentage=0.8, tone_accuracy=0.95
        )
        assert profile.level in (ProficiencyLevel.EXPERT, ProficiencyLevel.MASTER)
        assert profile.score > 80

    def test_strengths_weaknesses(self):
        pa = ProficiencyAssessor()
        profile = pa.assess(
            wpm=30.0, accuracy=0.6, unique_chars=15,
            flow_percentage=0.5, tone_accuracy=0.4
        )
        assert "speed" in profile.strengths
        assert "accuracy" in profile.weaknesses or "tone_marks" in profile.weaknesses

    def test_improvement_suggestions(self):
        pa = ProficiencyAssessor()
        pa.assess(wpm=5.0, accuracy=0.5, unique_chars=5, flow_percentage=0.0, tone_accuracy=0.3)
        suggestions = pa.get_improvement_suggestions()
        assert len(suggestions) > 0
        assert all("tip" in s for s in suggestions)
        assert all("tip_nko" in s for s in suggestions)

    def test_trend_new(self):
        pa = ProficiencyAssessor()
        assert pa.get_trend() == "new"

    def test_trend_improving(self):
        pa = ProficiencyAssessor()
        # Simulate improving scores
        for score in [20, 25, 30, 35, 40, 45, 50]:
            pa.assess(wpm=score/2, accuracy=0.7 + score/500, unique_chars=15, flow_percentage=0.3)
        trend = pa.get_trend()
        assert trend in ("improving", "stable")

    def test_trend_declining(self):
        pa = ProficiencyAssessor()
        # Simulate declining scores
        for score in [50, 45, 40, 35, 30, 25, 20]:
            pa.assess(wpm=score/2, accuracy=0.5 + score/500, unique_chars=15, flow_percentage=0.3)
        trend = pa.get_trend()
        assert trend in ("declining", "stable")


class TestErrorClassifier(SimpleTestCase):
    """Test error type classification."""

    def test_omission(self):
        ec = ErrorClassifier()
        result = ec.classify("ߊ", "")
        assert result == ErrorType.OMISSION

    def test_insertion(self):
        ec = ErrorClassifier()
        result = ec.classify("", "ߊ")
        assert result == ErrorType.INSERTION

    def test_tone_error(self):
        ec = ErrorClassifier()
        result = ec.classify("߫", "߬")
        assert result == ErrorType.TONE_ERROR

    def test_slip_adjacent_key(self):
        ec = ErrorClassifier()
        # Find an actual adjacent pair
        if "ߊ" in ADJACENCY_MAP and ADJACENCY_MAP["ߊ"]:
            adjacent = list(ADJACENCY_MAP["ߊ"])[0]
            result = ec.classify("ߊ", adjacent)
            assert result == ErrorType.SLIP

    def test_substitution(self):
        ec = ErrorClassifier()
        # Non-adjacent characters
        result = ec.classify("ߊ", "ߡ")
        # Could be SLIP if they happen to be adjacent, otherwise SUBSTITUTION
        assert result in (ErrorType.SUBSTITUTION, ErrorType.SLIP)

    def test_word_transposition(self):
        ec = ErrorClassifier()
        errors = ec.classify_word_error("ߊߋ", "ߋߊ")
        assert ErrorType.TRANSPOSITION in errors

    def test_word_insertion(self):
        ec = ErrorClassifier()
        errors = ec.classify_word_error("ߊߋ", "ߊߋߌ")
        assert ErrorType.INSERTION in errors

    def test_word_omission(self):
        ec = ErrorClassifier()
        errors = ec.classify_word_error("ߊߋߌ", "ߊߋ")
        assert ErrorType.OMISSION in errors


class TestAdaptiveSettings(SimpleTestCase):
    """Test adaptive settings behavior."""

    def test_default_settings(self):
        settings = AdaptiveSettings()
        assert settings.prediction_aggressiveness == 0.5
        assert settings.autocorrect_confidence == 0.7
        assert settings.suggestion_count == 3
        assert settings.show_tone_hints is True

    def test_frustration_thresholds(self):
        settings = AdaptiveSettings()
        assert settings.frustration_backspace_count == 3
        assert settings.frustration_window_ms == 2000.0


class TestNkoKeyboardLayout(SimpleTestCase):
    """Test N'Ko keyboard layout and adjacency map."""

    def test_keyboard_rows_exist(self):
        assert len(NKO_KEYBOARD_ROWS) == 4

    def test_adjacency_map_populated(self):
        assert len(ADJACENCY_MAP) > 0

    def test_adjacency_symmetric(self):
        """If A is adjacent to B, B should be adjacent to A."""
        for char, neighbors in ADJACENCY_MAP.items():
            for neighbor in neighbors:
                if neighbor in ADJACENCY_MAP:
                    assert char in ADJACENCY_MAP[neighbor], \
                        f"{char} adjacent to {neighbor} but not reverse"

    def test_vowels_in_first_row(self):
        vowels = NKO_KEYBOARD_ROWS[0]
        assert "ߊ" in vowels
        assert "ߋ" in vowels


class TestAnalyticsEngineIntegration(SimpleTestCase):
    """Integration tests for the main AnalyticsEngine."""

    def setUp(self):
        """Create a temporary database for testing."""
        self._tmp = tempfile.mktemp(suffix=".db")
        self.engine = AnalyticsEngine(db_path=self._tmp, user_id="test_user")

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self._tmp):
            os.unlink(self._tmp)

    def test_start_session(self):
        session_id = self.engine.start_session()
        assert session_id is not None
        assert len(session_id) == 12

    def test_record_keystrokes(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_keystroke("ߋ", 1200)
        self.engine.record_keystroke("ߌ", 1400)

        stats = self.engine.get_session_stats()
        assert stats.total_keystrokes == 3
        assert stats.total_characters == 3

    def test_record_error(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߋ", 1000, intended="ߊ")

        stats = self.engine.get_session_stats()
        assert stats.total_errors == 1

    def test_record_backspace(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_keystroke("BACKSPACE", 1200)

        stats = self.engine.get_session_stats()
        assert stats.backspaces == 1

    def test_record_word_completed(self):
        self.engine.start_session()
        self.engine.record_word_completed("ߊߟߊ߫", 1000)

        stats = self.engine.get_session_stats()
        assert stats.total_words == 1

    def test_record_prediction_accepted(self):
        self.engine.start_session()
        self.engine.record_prediction_event(accepted=True, timestamp_ms=1000)

        stats = self.engine.get_session_stats()
        assert stats.predictions_accepted == 1
        assert stats.predictions_shown == 1

    def test_record_prediction_rejected(self):
        self.engine.start_session()
        self.engine.record_prediction_event(accepted=False, timestamp_ms=1000)

        stats = self.engine.get_session_stats()
        assert stats.predictions_rejected == 1

    def test_record_swipe_input(self):
        self.engine.start_session()
        self.engine.record_swipe_input("ߊߟߊ߫", 1000)

        stats = self.engine.get_session_stats()
        assert stats.swipe_inputs == 1
        assert stats.total_words == 1

    def test_record_voice_input(self):
        self.engine.start_session()
        self.engine.record_voice_input("ߊ ߣߌ ߛߐ߲ !", 1000)

        stats = self.engine.get_session_stats()
        assert stats.voice_inputs == 1

    def test_end_session(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_keystroke("ߋ", 1200)
        self.engine.record_word_completed("ߊߋ", 1500)

        session = self.engine.end_session()
        assert session is not None
        assert session.end_time is not None
        assert session.total_keystrokes == 2

    def test_flow_state_tracking(self):
        self.engine.start_session()
        flow = self.engine.get_flow_state()
        assert isinstance(flow, FlowState)

    def test_frustration_tracking(self):
        self.engine.start_session()
        level = self.engine.get_frustration_level()
        assert 0.0 <= level <= 1.0

    def test_wpm_tracking(self):
        self.engine.start_session()
        wpm = self.engine.get_current_wpm()
        assert wpm >= 0.0

    def test_proficiency_tracking(self):
        self.engine.start_session()
        profile = self.engine.get_proficiency()
        assert isinstance(profile.level, ProficiencyLevel)

    def test_heatmap(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        heatmap = self.engine.get_heatmap()
        assert isinstance(heatmap, KeyHeatmap)
        assert heatmap.get_usage_heatmap().get("ߊ", 0) > 0

    def test_confusion_matrix(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_keystroke("ߋ", 1200, intended="ߊ")
        cm = self.engine.get_confusion_matrix()
        assert isinstance(cm, ConfusionMatrix)
        assert cm.get_confusion_rate("ߊ") > 0

    def test_gamification(self):
        self.engine.start_session()
        self.engine.record_word_completed("ߊ", 1000)
        gam = self.engine.get_gamification()
        unlocked = gam.get_unlocked()
        assert any(m.name == "first_word" for m in unlocked)

    def test_daily_summary(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_word_completed("ߊ", 1500)
        self.engine.end_session()

        summary = self.engine.get_daily_summary()
        assert "date" in summary
        assert "words_typed" in summary

    def test_weekly_trend(self):
        trend = self.engine.get_weekly_trend()
        assert isinstance(trend, list)

    def test_problem_areas(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߋ", 1000, intended="ߊ")
        problems = self.engine.get_problem_areas()
        assert "confused_pairs" in problems
        assert "weaknesses" in problems

    def test_cross_engine_insights(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        insights = self.engine.get_cross_engine_insights()

        assert "proficiency_level" in insights
        assert "flow_state" in insights
        assert "frustration" in insights
        assert "for_prediction_engine" in insights
        assert "for_grammar_engine" in insights
        assert "for_learning_engine" in insights
        assert "for_cultural_engine" in insights
        assert "for_voice_engine" in insights

    def test_adaptive_settings(self):
        settings = self.engine.get_adaptive_settings()
        assert isinstance(settings, AdaptiveSettings)

    def test_improvement_tips(self):
        tips = self.engine.get_improvement_tips()
        assert isinstance(tips, list)

    def test_export_analytics(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        export = self.engine.export_analytics()

        assert "user_id" in export
        assert "proficiency" in export
        assert "confusion_matrix" in export
        assert "heatmap" in export
        assert "gamification" in export

    def test_delete_all_data(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_word_completed("ߊ", 1500)
        self.engine.end_session()

        self.engine.delete_all_data()
        summary = self.engine.get_daily_summary()
        assert summary["words_typed"] == 0

    def test_auto_start_session(self):
        """Recording without explicit start should auto-create session."""
        self.engine.record_keystroke("ߊ", 1000)
        stats = self.engine.get_session_stats()
        assert stats is not None

    def test_session_prediction_acceptance_rate(self):
        self.engine.start_session()
        self.engine.record_prediction_event(True, 1000)
        self.engine.record_prediction_event(True, 1200)
        self.engine.record_prediction_event(False, 1400)

        stats = self.engine.get_session_stats()
        rate = stats.prediction_acceptance_rate
        assert 0.6 < rate < 0.7  # 2/3 ≈ 0.667

    def test_input_method_breakdown(self):
        self.engine.start_session()
        self.engine.record_keystroke("ߊ", 1000)
        self.engine.record_swipe_input("ߊ", 1200)
        self.engine.record_voice_input("test", 1400)
        self.engine.record_handwriting_input("ߊ", 1600)

        stats = self.engine.get_session_stats()
        breakdown = stats.input_method_breakdown
        assert breakdown["swipe"] == 1
        assert breakdown["voice"] == 1
        assert breakdown["handwriting"] == 1

    def test_repr(self):
        r = repr(self.engine)
        assert "AnalyticsEngine" in r
        assert "level=" in r


class TestSessionStats(SimpleTestCase):
    """Test SessionStats dataclass."""

    def test_prediction_acceptance_rate_empty(self):
        stats = SessionStats(session_id="test", start_time=datetime.now())
        assert stats.prediction_acceptance_rate == 0.0

    def test_prediction_acceptance_rate(self):
        stats = SessionStats(
            session_id="test",
            start_time=datetime.now(),
            predictions_accepted=3,
            predictions_rejected=1,
        )
        assert stats.prediction_acceptance_rate == 0.75


# ─── Run Tests ──────────────────────────────────────────────

def run_tests():
    """Run all analytics tests."""
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestConfusionMatrix,
        TestFlowStateDetector,
        TestFrustrationDetector,
        TestKeyHeatmap,
        TestGamificationEngine,
        TestWPMCalculator,
        TestProficiencyAssessor,
        TestErrorClassifier,
        TestAdaptiveSettings,
        TestNkoKeyboardLayout,
        TestAnalyticsEngineIntegration,
        TestSessionStats,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
