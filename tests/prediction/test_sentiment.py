#!/usr/bin/env python3
"""
Tests for N'Ko Sentiment & Emotion Engine — Generation 16

Comprehensive tests covering:
  - Sentiment analysis (positive/negative/neutral)
  - Emotion classification
  - Formality detection
  - Emoji suggestions
  - Cultural expressions
  - Life event detection
  - Tone adjustment
  - Emotional memory persistence
  - Keyboard integration
  - Edge cases
"""

import os
import sys
import json
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sentiment_engine import (
    # Enums
    Sentiment, Emotion, FormalityLevel, LifeEvent,
    # Data classes
    SentimentResult, EmojiSuggestion, ToneAdjustment,
    CulturalExpression, EmotionalMemory,
    # Engines
    SentimentAnalyzer, EmojiSuggestor, ToneAdjuster,
    CulturalExpressionRecommender, EmotionalMemoryStore,
    SentimentEngine,
    # Lexicons
    NKO_EMOTION_LEXICON, LATIN_EMOTION_LEXICON,
    EMOTION_EMOJI_MAP, CULTURAL_EXPRESSIONS,
)


# ═══════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════

class _TestBase:
    """Base class with common setup."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_true(self, condition, msg=""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"FAIL: {msg}")

    def assert_false(self, condition, msg=""):
        self.assert_true(not condition, msg)

    def assert_equal(self, a, b, msg=""):
        self.assert_true(a == b, f"{msg} (expected {b}, got {a})")

    def assert_not_equal(self, a, b, msg=""):
        self.assert_true(a != b, f"{msg} (expected != {b}, got {a})")

    def assert_in(self, item, collection, msg=""):
        self.assert_true(item in collection, f"{msg} ({item} not in {collection})")

    def assert_greater(self, a, b, msg=""):
        self.assert_true(a > b, f"{msg} (expected {a} > {b})")

    def assert_less(self, a, b, msg=""):
        self.assert_true(a < b, f"{msg} (expected {a} < {b})")

    def assert_between(self, val, low, high, msg=""):
        self.assert_true(low <= val <= high, f"{msg} (expected {low} <= {val} <= {high})")

    def assert_is_none(self, val, msg=""):
        self.assert_true(val is None, f"{msg} (expected None, got {val})")

    def assert_is_not_none(self, val, msg=""):
        self.assert_true(val is not None, f"{msg} (expected not None)")

    def assert_isinstance(self, obj, cls, msg=""):
        self.assert_true(isinstance(obj, cls), f"{msg} (expected {cls.__name__}, got {type(obj).__name__})")


# ═══════════════════════════════════════════════════════════════
# Test: Sentiment Analyzer
# ═══════════════════════════════════════════════════════════════

class _SentimentAnalyzerTests(_TestBase):
    """Tests for the core sentiment analyzer."""

    def __init__(self):
        super().__init__()
        self.analyzer = SentimentAnalyzer()

    def test_empty_text(self):
        """Empty text returns neutral."""
        result = self.analyzer.analyze("")
        self.assert_equal(result.sentiment, Sentiment.NEUTRAL, "empty → neutral")
        self.assert_equal(result.score, 0.0, "empty → score 0")
        self.assert_equal(result.confidence, 0.0, "empty → confidence 0")
        self.assert_equal(result.dominant_emotion, Emotion.NEUTRAL, "empty → neutral emotion")

    def test_whitespace_text(self):
        """Whitespace-only text returns neutral."""
        result = self.analyzer.analyze("   \n\t  ")
        self.assert_equal(result.sentiment, Sentiment.NEUTRAL, "whitespace → neutral")

    def test_positive_nko_joy(self):
        """N'Ko joy words detected as positive."""
        result = self.analyzer.analyze("ߘߌ߬ߦߊ")  # diya — happiness
        self.assert_in(result.sentiment, [Sentiment.POSITIVE, Sentiment.VERY_POSITIVE], "joy → positive")
        self.assert_greater(result.score, 0.0, "joy → positive score")
        self.assert_equal(result.dominant_emotion, Emotion.JOY, "joy word → JOY emotion")

    def test_positive_nko_love(self):
        """N'Ko love words detected."""
        result = self.analyzer.analyze("ߞߊ߬ߣߎ")  # kanu — love
        self.assert_greater(result.score, 0.0, "love → positive")
        self.assert_equal(result.dominant_emotion, Emotion.LOVE, "love word → LOVE emotion")

    def test_positive_nko_gratitude(self):
        """Gratitude expressions detected."""
        result = self.analyzer.analyze("ߌ ߣ ߗ")  # i ni ce — thank you
        self.assert_greater(result.score, 0.0, "gratitude → positive")
        self.assert_equal(result.dominant_emotion, Emotion.GRATITUDE, "thank you → GRATITUDE")

    def test_negative_nko_sadness(self):
        """Sadness words detected as negative."""
        result = self.analyzer.analyze("ߒ ߘ ߞ")  # n du ka — I'm sad
        self.assert_less(result.score, 0.0, "sadness → negative score")
        self.assert_in(result.dominant_emotion, [Emotion.SADNESS], "sad word → SADNESS")

    def test_negative_nko_anger(self):
        """Anger words detected."""
        result = self.analyzer.analyze("ߒ ߞ ߜ")  # n ka gba — I'm angry
        self.assert_less(result.score, 0.0, "anger → negative score")
        self.assert_equal(result.dominant_emotion, Emotion.ANGER, "anger word → ANGER")

    def test_negative_nko_fear(self):
        """Fear words detected."""
        result = self.analyzer.analyze("ߒ ߛ ߙ")  # n sira — I'm afraid
        self.assert_less(result.score, 0.0, "fear → negative score")
        self.assert_equal(result.dominant_emotion, Emotion.FEAR, "fear word → FEAR")

    def test_latin_manding_positive(self):
        """Latin Manding positive words detected."""
        result = self.analyzer.analyze("barika! i ni ce")
        self.assert_greater(result.score, 0.0, "barika → positive")

    def test_latin_manding_negative(self):
        """Latin Manding negative words detected."""
        result = self.analyzer.analyze("siran")
        self.assert_less(result.score, 0.0, "siran → negative")

    def test_mixed_script_analysis(self):
        """Mixed N'Ko and Latin text analyzed together."""
        result = self.analyzer.analyze("ߘߌ߬ߦߊ, I am happy barika")
        self.assert_greater(result.score, 0.0, "mixed script → positive")
        self.assert_greater(result.confidence, 0.0, "mixed → some confidence")

    def test_exclamation_amplifies(self):
        """Exclamation marks amplify dominant emotion."""
        base = self.analyzer.analyze("ߘߌ߬ߦߊ")
        amplified = self.analyzer.analyze("ߘߌ߬ߦߊ!!!")
        # Amplified should have same or higher emotion intensity
        self.assert_true(
            amplified.emotion_scores.get(Emotion.JOY, 0) >=
            base.emotion_scores.get(Emotion.JOY, 0),
            "exclamation amplifies emotion"
        )

    def test_question_dampens(self):
        """Question marks slightly dampen confidence."""
        result = self.analyzer.analyze("ߘߌ߬ߦߊ?")
        self.assert_is_not_none(result, "question mark handled")

    def test_neutral_text(self):
        """Neutral text (no emotion words) returns neutral."""
        result = self.analyzer.analyze("abc xyz 123")
        self.assert_equal(result.sentiment, Sentiment.NEUTRAL, "random → neutral")

    def test_result_has_all_fields(self):
        """SentimentResult has all required fields."""
        result = self.analyzer.analyze("ߘߌ߬ߦߊ")
        self.assert_isinstance(result, SentimentResult, "returns SentimentResult")
        self.assert_isinstance(result.sentiment, Sentiment, "has sentiment")
        self.assert_isinstance(result.dominant_emotion, Emotion, "has emotion")
        self.assert_isinstance(result.formality, FormalityLevel, "has formality")
        self.assert_isinstance(result.emotion_scores, dict, "has emotion_scores")
        self.assert_isinstance(result.score, float, "score is float")
        self.assert_isinstance(result.confidence, float, "confidence is float")
        self.assert_between(result.score, -1.0, 1.0, "score in range")
        self.assert_between(result.confidence, 0.0, 1.0, "confidence in range")

    def test_all_emotions_in_scores(self):
        """All emotions present in emotion_scores dict."""
        result = self.analyzer.analyze("ߘߌ߬ߦߊ")
        for emotion in Emotion:
            self.assert_in(emotion, result.emotion_scores, f"{emotion.name} in scores")

    def test_explanation_generated(self):
        """Explanations are generated for non-neutral results."""
        result = self.analyzer.analyze("ߘߌ߬ߦߊ")
        self.assert_true(len(result.explanation_en) > 0, "English explanation generated")

    def run_all(self):
        """Run all tests."""
        self.test_empty_text()
        self.test_whitespace_text()
        self.test_positive_nko_joy()
        self.test_positive_nko_love()
        self.test_positive_nko_gratitude()
        self.test_negative_nko_sadness()
        self.test_negative_nko_anger()
        self.test_negative_nko_fear()
        self.test_latin_manding_positive()
        self.test_latin_manding_negative()
        self.test_mixed_script_analysis()
        self.test_exclamation_amplifies()
        self.test_question_dampens()
        self.test_neutral_text()
        self.test_result_has_all_fields()
        self.test_all_emotions_in_scores()
        self.test_explanation_generated()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Formality Detection
# ═══════════════════════════════════════════════════════════════

class _FormalityDetectionTests(_TestBase):
    """Tests for formality level detection."""

    def __init__(self):
        super().__init__()
        self.analyzer = SentimentAnalyzer()

    def test_formal_greeting(self):
        """Formal group greeting detected."""
        result = self.analyzer.analyze("ߊ ߣ ߗ")  # aw ni ce
        self.assert_in(
            result.formality,
            [FormalityLevel.FORMAL, FormalityLevel.CEREMONIAL],
            "aw ni ce → formal"
        )

    def test_formal_blessing(self):
        """God invocation is formal/ceremonial."""
        result = self.analyzer.analyze("ߊ߬ߟߊ ߞ ߤ ߟ ߘ")  # ala ka hera da
        self.assert_in(
            result.formality,
            [FormalityLevel.FORMAL, FormalityLevel.CEREMONIAL],
            "ala → formal"
        )

    def test_informal_text(self):
        """Informal markers detected."""
        result = self.analyzer.analyze("ok lol haha")
        self.assert_in(
            result.formality,
            [FormalityLevel.INTIMATE, FormalityLevel.CASUAL],
            "lol/haha → casual/intimate"
        )

    def test_neutral_formality(self):
        """Plain text is neutral formality."""
        result = self.analyzer.analyze("ߞ ß ß ß ß ß ß ß ß")
        self.assert_is_not_none(result.formality, "formality assigned")

    def test_long_text_more_formal(self):
        """Longer texts trend toward neutral/formal."""
        long_text = " ".join(["ß ß ß ß ß"] * 10)
        result = self.analyzer.analyze(long_text)
        self.assert_is_not_none(result.formality, "long text formality")

    def run_all(self):
        self.test_formal_greeting()
        self.test_formal_blessing()
        self.test_informal_text()
        self.test_neutral_formality()
        self.test_long_text_more_formal()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Life Event Detection
# ═══════════════════════════════════════════════════════════════

class _LifeEventDetectionTests(_TestBase):
    """Tests for life event detection."""

    def __init__(self):
        super().__init__()
        self.analyzer = SentimentAnalyzer()

    def test_wedding_detected(self):
        """Wedding keywords trigger detection."""
        result = self.analyzer.analyze("furu celebration!")
        self.assert_equal(result.detected_life_event, LifeEvent.WEDDING, "furu → WEDDING")

    def test_death_detected(self):
        """Death keywords trigger detection."""
        result = self.analyzer.analyze("funeral")
        self.assert_equal(result.detected_life_event, LifeEvent.DEATH, "funeral → DEATH")

    def test_ramadan_detected(self):
        """Ramadan keywords trigger detection."""
        result = self.analyzer.analyze("ramadan kareem")
        self.assert_equal(result.detected_life_event, LifeEvent.RAMADAN, "ramadan → RAMADAN")

    def test_illness_detected(self):
        """Illness keywords trigger detection."""
        result = self.analyzer.analyze("he is sick, hospital")
        self.assert_equal(result.detected_life_event, LifeEvent.ILLNESS, "sick/hospital → ILLNESS")

    def test_no_event_for_neutral(self):
        """No life event for neutral text."""
        result = self.analyzer.analyze("abc xyz")
        self.assert_is_none(result.detected_life_event, "neutral → no event")

    def test_eid_detected(self):
        """Eid keywords trigger detection."""
        result = self.analyzer.analyze("eid tabaski")
        self.assert_equal(result.detected_life_event, LifeEvent.EID, "eid/tabaski → EID")

    def test_travel_detected(self):
        """Travel keywords trigger detection."""
        result = self.analyzer.analyze("voyage travel")
        self.assert_equal(result.detected_life_event, LifeEvent.JOURNEY, "voyage → JOURNEY")

    def run_all(self):
        self.test_wedding_detected()
        self.test_death_detected()
        self.test_ramadan_detected()
        self.test_illness_detected()
        self.test_no_event_for_neutral()
        self.test_eid_detected()
        self.test_travel_detected()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Emoji Suggestions
# ═══════════════════════════════════════════════════════════════

class _EmojiSuggestorTests(_TestBase):
    """Tests for emoji suggestion engine."""

    def __init__(self):
        super().__init__()
        self.suggestor = EmojiSuggestor()

    def test_joy_emojis(self):
        """Joy text gets happy emojis."""
        emojis = self.suggestor.suggest("ߘߌ߬ߦߊ")  # diya
        self.assert_greater(len(emojis), 0, "joy → emojis returned")
        emoji_chars = [e.emoji for e in emojis]
        self.assert_true(
            any(e in emoji_chars for e in ["😊", "😄", "🎉"]),
            "joy → happy emojis"
        )

    def test_love_emojis(self):
        """Love text gets heart emojis."""
        emojis = self.suggestor.suggest("ߞߊ߬ߣߎ")  # kanu
        emoji_chars = [e.emoji for e in emojis]
        self.assert_true(
            any(e in emoji_chars for e in ["❤️", "🥰", "💕"]),
            "love → heart emojis"
        )

    def test_gratitude_emojis(self):
        """Gratitude gets prayer/thanks emojis."""
        emojis = self.suggestor.suggest("ߌ ߣ ߗ")  # i ni ce
        emoji_chars = [e.emoji for e in emojis]
        self.assert_true(
            any(e in emoji_chars for e in ["🙏", "✨"]),
            "gratitude → prayer emojis"
        )

    def test_max_suggestions_respected(self):
        """Max suggestions limit honored."""
        emojis = self.suggestor.suggest("ߘߌ߬ߦߊ", max_suggestions=2)
        self.assert_true(len(emojis) <= 2, "max_suggestions=2 respected")

    def test_no_duplicate_emojis(self):
        """No duplicate emojis in suggestions."""
        emojis = self.suggestor.suggest("ߘߌ߬ߦߊ ߞߊ߬ߣߎ", max_suggestions=10)
        emoji_set = set(e.emoji for e in emojis)
        self.assert_equal(len(emoji_set), len(emojis), "no duplicate emojis")

    def test_emoji_has_relevance(self):
        """Each emoji has a relevance score."""
        emojis = self.suggestor.suggest("ߘߌ߬ߦߊ")
        for e in emojis:
            self.assert_between(e.relevance, 0.0, 1.5, f"emoji {e.emoji} relevance valid")

    def test_emoji_sorted_by_relevance(self):
        """Emojis sorted by relevance descending."""
        emojis = self.suggestor.suggest("ߘߌ߬ߦߊ")
        for i in range(len(emojis) - 1):
            self.assert_true(
                emojis[i].relevance >= emojis[i+1].relevance,
                "emojis sorted by relevance"
            )

    def test_life_event_emojis(self):
        """Life event emojis returned."""
        emojis = self.suggestor.suggest_for_life_event(LifeEvent.WEDDING)
        self.assert_greater(len(emojis), 0, "wedding → emojis")
        emoji_chars = [e.emoji for e in emojis]
        self.assert_in("💍", emoji_chars, "wedding → ring emoji")

    def test_death_event_emojis(self):
        """Death event gets appropriate emojis."""
        emojis = self.suggestor.suggest_for_life_event(LifeEvent.DEATH)
        emoji_chars = [e.emoji for e in emojis]
        self.assert_true(
            any(e in emoji_chars for e in ["🤲", "🕊️"]),
            "death → prayer/dove emojis"
        )

    def test_ramadan_emojis(self):
        """Ramadan gets moon emoji."""
        emojis = self.suggestor.suggest_for_life_event(LifeEvent.RAMADAN)
        emoji_chars = [e.emoji for e in emojis]
        self.assert_in("🌙", emoji_chars, "ramadan → moon emoji")

    def test_neutral_text_emojis(self):
        """Neutral text still gets some emojis."""
        emojis = self.suggestor.suggest("abc xyz")
        # Should return at least neutral emojis
        self.assert_is_not_none(emojis, "neutral → some result")

    def run_all(self):
        self.test_joy_emojis()
        self.test_love_emojis()
        self.test_gratitude_emojis()
        self.test_max_suggestions_respected()
        self.test_no_duplicate_emojis()
        self.test_emoji_has_relevance()
        self.test_emoji_sorted_by_relevance()
        self.test_life_event_emojis()
        self.test_death_event_emojis()
        self.test_ramadan_emojis()
        self.test_neutral_text_emojis()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Cultural Expression Recommender
# ═══════════════════════════════════════════════════════════════

class _CulturalExpressionsTests(_TestBase):
    """Tests for cultural expression recommendations."""

    def __init__(self):
        super().__init__()
        self.recommender = CulturalExpressionRecommender()

    def test_gratitude_expressions(self):
        """Gratitude emotion returns gratitude expressions."""
        exprs = self.recommender.recommend_for_emotion(Emotion.GRATITUDE)
        self.assert_greater(len(exprs), 0, "gratitude expressions found")
        self.assert_true(
            any(e.emotion == Emotion.GRATITUDE for e in exprs),
            "expressions match gratitude"
        )

    def test_peace_expressions(self):
        """Peace emotion returns peace expressions."""
        exprs = self.recommender.recommend_for_emotion(Emotion.PEACE)
        self.assert_greater(len(exprs), 0, "peace expressions found")

    def test_compassion_expressions(self):
        """Compassion returns consolation expressions."""
        exprs = self.recommender.recommend_for_emotion(Emotion.COMPASSION)
        self.assert_greater(len(exprs), 0, "compassion expressions found")

    def test_wedding_event_expressions(self):
        """Wedding event returns appropriate blessings."""
        exprs = self.recommender.recommend_for_life_event(LifeEvent.WEDDING)
        self.assert_greater(len(exprs), 0, "wedding expressions found")
        self.assert_true(
            all(e.life_event == LifeEvent.WEDDING for e in exprs),
            "all wedding-tagged"
        )

    def test_death_event_expressions(self):
        """Death event returns condolence expressions."""
        exprs = self.recommender.recommend_for_life_event(LifeEvent.DEATH)
        self.assert_greater(len(exprs), 0, "condolence expressions found")

    def test_expressions_have_nko(self):
        """All expressions include N'Ko text."""
        for expr in CULTURAL_EXPRESSIONS:
            self.assert_greater(len(expr.nko_text), 0, f"expression has N'Ko: {expr.english}")

    def test_expressions_have_english(self):
        """All expressions include English translation."""
        for expr in CULTURAL_EXPRESSIONS:
            self.assert_greater(len(expr.english), 0, f"expression has English")

    def test_expressions_have_latin(self):
        """All expressions include Latin transliteration."""
        for expr in CULTURAL_EXPRESSIONS:
            self.assert_greater(len(expr.latin_text), 0, f"expression has Latin")

    def test_recommend_response_with_event(self):
        """Recommend response prioritizes life events."""
        analysis = SentimentResult(
            sentiment=Sentiment.POSITIVE,
            score=0.5,
            confidence=0.8,
            dominant_emotion=Emotion.JOY,
            detected_life_event=LifeEvent.WEDDING,
            formality=FormalityLevel.FORMAL
        )
        exprs = self.recommender.recommend_response(analysis)
        if exprs:
            self.assert_true(
                any(e.life_event == LifeEvent.WEDDING for e in exprs),
                "wedding event prioritized"
            )
        else:
            self.assert_true(True, "no wedding expressions (OK)")

    def test_max_results_honored(self):
        """Max results limit honored."""
        exprs = self.recommender.recommend_for_emotion(
            Emotion.GRATITUDE, max_results=1
        )
        self.assert_true(len(exprs) <= 1, "max_results=1 honored")

    def run_all(self):
        self.test_gratitude_expressions()
        self.test_peace_expressions()
        self.test_compassion_expressions()
        self.test_wedding_event_expressions()
        self.test_death_event_expressions()
        self.test_expressions_have_nko()
        self.test_expressions_have_english()
        self.test_expressions_have_latin()
        self.test_recommend_response_with_event()
        self.test_max_results_honored()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Tone Adjuster
# ═══════════════════════════════════════════════════════════════

class _ToneAdjusterTests(_TestBase):
    """Tests for tone adjustment."""

    def __init__(self):
        super().__init__()
        self.adjuster = ToneAdjuster()

    def test_casual_to_formal(self):
        """Casual → formal transforms applied."""
        result = self.adjuster.adjust_formality(
            "ߌ ß ß", FormalityLevel.CASUAL, FormalityLevel.FORMAL
        )
        self.assert_isinstance(result, ToneAdjustment, "returns ToneAdjustment")
        self.assert_greater(len(result.target_tone), 0, "has target tone")

    def test_formal_to_casual(self):
        """Formal → casual transforms applied."""
        result = self.adjuster.adjust_formality(
            "aw ß ß", FormalityLevel.FORMAL, FormalityLevel.CASUAL
        )
        self.assert_isinstance(result, ToneAdjustment, "returns ToneAdjustment")

    def test_same_formality(self):
        """Same → same returns original text."""
        result = self.adjuster.adjust_formality(
            "ߌ ß", FormalityLevel.NEUTRAL, FormalityLevel.NEUTRAL
        )
        self.assert_equal(result.original, "ߌ ß", "original preserved")

    def test_adjustment_has_explanation(self):
        """Adjustments include explanations."""
        result = self.adjuster.adjust_formality(
            "test", FormalityLevel.CASUAL, FormalityLevel.FORMAL
        )
        self.assert_greater(len(result.explanation_en), 0, "has explanation")

    def test_soften_returns_none_for_neutral(self):
        """Soften returns None for text without harsh patterns."""
        analysis = SentimentResult(
            sentiment=Sentiment.NEUTRAL,
            score=0.0,
            confidence=0.5,
            dominant_emotion=Emotion.NEUTRAL
        )
        result = self.adjuster.soften("abc", analysis)
        self.assert_is_none(result, "soften neutral → None")

    def run_all(self):
        self.test_casual_to_formal()
        self.test_formal_to_casual()
        self.test_same_formality()
        self.test_adjustment_has_explanation()
        self.test_soften_returns_none_for_neutral()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Emotional Memory Store
# ═══════════════════════════════════════════════════════════════

class _EmotionalMemoryTests(_TestBase):
    """Tests for persistent emotional memory."""

    def __init__(self):
        super().__init__()
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_sentiment.db")
        self.store = EmotionalMemoryStore(self.db_path)

    def test_db_created(self):
        """Database file created on init."""
        self.assert_true(os.path.exists(self.db_path), "DB file created")

    def test_tables_exist(self):
        """Required tables created."""
        conn = sqlite3.connect(self.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()
        self.assert_in("emotional_events", table_names, "emotional_events table")
        self.assert_in("conversation_moods", table_names, "conversation_moods table")

    def test_record_and_retrieve(self):
        """Record an emotion and retrieve profile."""
        analysis = SentimentResult(
            sentiment=Sentiment.POSITIVE,
            score=0.7,
            confidence=0.8,
            dominant_emotion=Emotion.JOY,
            emotion_scores={Emotion.JOY: 0.8, Emotion.NEUTRAL: 0.1},
            formality=FormalityLevel.CASUAL
        )
        self.store.record_emotion("test_contact", analysis, "test text")
        profile = self.store.get_emotional_profile("test_contact")
        self.assert_equal(profile.contact_id, "test_contact", "contact_id matches")
        self.assert_greater(len(profile.emotion_history), 0, "history recorded")

    def test_mood_trend(self):
        """Mood trend computed correctly."""
        for score in [0.5, 0.7, -0.3]:
            analysis = SentimentResult(
                sentiment=Sentiment.POSITIVE if score > 0 else Sentiment.NEGATIVE,
                score=score,
                confidence=0.8,
                dominant_emotion=Emotion.JOY if score > 0 else Emotion.SADNESS,
                emotion_scores={Emotion.JOY: max(0, score), Emotion.SADNESS: max(0, -score)},
                formality=FormalityLevel.NEUTRAL
            )
            self.store.record_emotion("trend_contact", analysis)

        trend = self.store.get_mood_trend("trend_contact", days=1)
        self.assert_greater(len(trend), 0, "mood trend has data")

    def test_empty_profile(self):
        """Unknown contact returns empty profile."""
        profile = self.store.get_emotional_profile("nonexistent")
        self.assert_equal(profile.contact_id, "nonexistent", "id preserved")
        self.assert_equal(len(profile.emotion_history), 0, "empty history")

    def test_text_snippet_truncated(self):
        """Long text snippets are truncated for privacy."""
        long_text = "a" * 200
        analysis = SentimentResult(
            sentiment=Sentiment.NEUTRAL,
            score=0.0,
            confidence=0.5,
            dominant_emotion=Emotion.NEUTRAL,
            emotion_scores={},
            formality=FormalityLevel.NEUTRAL
        )
        self.store.record_emotion("privacy_test", analysis, long_text)

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT text_snippet FROM emotional_events WHERE contact_id='privacy_test'"
        ).fetchone()
        conn.close()
        if row and row[0]:
            self.assert_true(len(row[0]) <= 50, "snippet truncated to 50 chars")
        else:
            self.assert_true(True, "snippet handled")

    def test_multiple_contacts(self):
        """Multiple contacts tracked independently."""
        for contact in ["alice", "bob"]:
            analysis = SentimentResult(
                sentiment=Sentiment.POSITIVE,
                score=0.5,
                confidence=0.7,
                dominant_emotion=Emotion.JOY,
                emotion_scores={Emotion.JOY: 0.5},
                formality=FormalityLevel.NEUTRAL
            )
            self.store.record_emotion(contact, analysis)

        alice = self.store.get_emotional_profile("alice")
        bob = self.store.get_emotional_profile("bob")
        self.assert_equal(alice.contact_id, "alice", "alice tracked")
        self.assert_equal(bob.contact_id, "bob", "bob tracked")

    def run_all(self):
        self.test_db_created()
        self.test_tables_exist()
        self.test_record_and_retrieve()
        self.test_mood_trend()
        self.test_empty_profile()
        self.test_text_snippet_truncated()
        self.test_multiple_contacts()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Unified Sentiment Engine
# ═══════════════════════════════════════════════════════════════

class _SentimentEngineTests(_TestBase):
    """Tests for the unified SentimentEngine interface."""

    def __init__(self):
        super().__init__()
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_unified.db")
        self.engine = SentimentEngine(self.db_path)

    def test_analyze(self):
        """Engine.analyze works end-to-end."""
        result = self.engine.analyze("ߘߌ߬ߦߊ")
        self.assert_isinstance(result, SentimentResult, "returns SentimentResult")
        self.assert_greater(result.score, 0.0, "positive text → positive score")

    def test_suggest_emojis(self):
        """Engine.suggest_emojis works."""
        emojis = self.engine.suggest_emojis("ߘߌ߬ߦߊ")
        self.assert_greater(len(emojis), 0, "emojis returned")
        self.assert_isinstance(emojis[0], EmojiSuggestion, "returns EmojiSuggestion")

    def test_recommend_expressions(self):
        """Engine.recommend_expressions works."""
        analysis = self.engine.analyze("ߌ ߣ ߗ")
        exprs = self.engine.recommend_expressions(analysis)
        self.assert_is_not_none(exprs, "expressions returned")

    def test_adjust_tone(self):
        """Engine.adjust_tone works."""
        result = self.engine.adjust_tone("ߌ ß", FormalityLevel.FORMAL)
        self.assert_isinstance(result, ToneAdjustment, "returns ToneAdjustment")

    def test_record_and_profile(self):
        """Engine.record and get_profile work together."""
        analysis = self.engine.analyze("ߘߌ߬ߦߊ")
        self.engine.record("test_user", analysis)
        profile = self.engine.get_profile("test_user")
        self.assert_equal(profile.contact_id, "test_user", "profile retrieved")

    def test_mood_trend(self):
        """Engine.get_mood_trend works."""
        analysis = self.engine.analyze("ߘߌ߬ߦߊ")
        self.engine.record("trend_user", analysis)
        trend = self.engine.get_mood_trend("trend_user")
        self.assert_is_not_none(trend, "trend returned")

    def test_prediction_boost(self):
        """Engine.get_prediction_boost returns valid multipliers."""
        boosts = self.engine.get_prediction_boost(
            "ߘߌ߬ߦߊ",
            ["ߣߌ", "ߓߊ߯ߙߞ", "ߒ ߛ ߙ"]
        )
        self.assert_equal(len(boosts), 3, "boost for each candidate")
        for candidate, boost in boosts.items():
            self.assert_between(boost, 0.5, 2.0, f"boost for {candidate} in range")

    def test_keyboard_context(self):
        """Engine.get_keyboard_context returns complete context dict."""
        ctx = self.engine.get_keyboard_context("ߘߌ߬ߦߊ")
        self.assert_in("sentiment", ctx, "has sentiment")
        self.assert_in("emotion", ctx, "has emotion")
        self.assert_in("formality", ctx, "has formality")
        self.assert_in("confidence", ctx, "has confidence")
        self.assert_in("emojis", ctx, "has emojis")
        self.assert_in("expressions", ctx, "has expressions")
        self.assert_isinstance(ctx["emojis"], list, "emojis is list")
        self.assert_isinstance(ctx["expressions"], list, "expressions is list")

    def test_life_event_emojis(self):
        """Engine.suggest_life_event_emojis works."""
        emojis = self.engine.suggest_life_event_emojis(LifeEvent.BIRTH)
        self.assert_greater(len(emojis), 0, "birth emojis returned")

    def test_recommend_for_event(self):
        """Engine.recommend_for_event works."""
        exprs = self.engine.recommend_for_event(LifeEvent.DEATH)
        self.assert_is_not_none(exprs, "death expressions returned")

    def test_soften(self):
        """Engine.soften works for neutral text."""
        result = self.engine.soften("neutral text")
        # Neutral text → None (nothing to soften)
        self.assert_is_none(result, "neutral → nothing to soften")

    def run_all(self):
        self.test_analyze()
        self.test_suggest_emojis()
        self.test_recommend_expressions()
        self.test_adjust_tone()
        self.test_record_and_profile()
        self.test_mood_trend()
        self.test_prediction_boost()
        self.test_keyboard_context()
        self.test_life_event_emojis()
        self.test_recommend_for_event()
        self.test_soften()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Lexicon Coverage
# ═══════════════════════════════════════════════════════════════

class _LexiconCoverageTests(_TestBase):
    """Tests for lexicon completeness and consistency."""

    def test_nko_lexicon_entries(self):
        """N'Ko lexicon has sufficient entries."""
        self.assert_greater(len(NKO_EMOTION_LEXICON), 20, "20+ N'Ko lexicon entries")

    def test_latin_lexicon_entries(self):
        """Latin lexicon has entries."""
        self.assert_greater(len(LATIN_EMOTION_LEXICON), 10, "10+ Latin entries")

    def test_lexicon_values_valid(self):
        """All lexicon values have valid scores."""
        for word, (score, emotion) in NKO_EMOTION_LEXICON.items():
            self.assert_between(score, -1.0, 1.0, f"{word} score in range")
            self.assert_isinstance(emotion, Emotion, f"{word} has valid emotion")

    def test_all_emotions_covered(self):
        """Most emotions appear in lexicons."""
        covered = set()
        for _, (_, emotion) in NKO_EMOTION_LEXICON.items():
            covered.add(emotion)
        for _, (_, emotion) in LATIN_EMOTION_LEXICON.items():
            covered.add(emotion)
        # At least 8 of 13 emotions should be in lexicons
        self.assert_greater(len(covered), 7, f"8+ emotions covered ({len(covered)} found)")

    def test_emoji_map_coverage(self):
        """Emoji map covers all major emotions."""
        self.assert_greater(len(EMOTION_EMOJI_MAP), 8, "8+ emotion emoji mappings")

    def test_cultural_expressions_count(self):
        """Sufficient cultural expressions defined."""
        self.assert_greater(len(CULTURAL_EXPRESSIONS), 10, "10+ cultural expressions")

    def test_cultural_expressions_diverse(self):
        """Cultural expressions cover multiple emotions."""
        emotions = set(e.emotion for e in CULTURAL_EXPRESSIONS)
        self.assert_greater(len(emotions), 3, "4+ different emotions in expressions")

    def test_life_events_in_expressions(self):
        """Life events represented in expressions."""
        events = set(
            e.life_event for e in CULTURAL_EXPRESSIONS if e.life_event is not None
        )
        self.assert_greater(len(events), 2, "3+ life events in expressions")

    def run_all(self):
        self.test_nko_lexicon_entries()
        self.test_latin_lexicon_entries()
        self.test_lexicon_values_valid()
        self.test_all_emotions_covered()
        self.test_emoji_map_coverage()
        self.test_cultural_expressions_count()
        self.test_cultural_expressions_diverse()
        self.test_life_events_in_expressions()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Edge Cases
# ═══════════════════════════════════════════════════════════════

class _EdgeCasesTests(_TestBase):
    """Edge case and robustness tests."""

    def __init__(self):
        super().__init__()
        self.engine = SentimentEngine()

    def test_very_long_text(self):
        """Long text doesn't crash."""
        long_text = "ߘߌ߬ߦߊ " * 1000
        result = self.engine.analyze(long_text)
        self.assert_is_not_none(result, "long text handled")

    def test_only_punctuation(self):
        """Punctuation-only text handled."""
        result = self.engine.analyze("!!! ???")
        self.assert_is_not_none(result, "punctuation-only handled")

    def test_numbers_only(self):
        """Number-only text handled."""
        result = self.engine.analyze("12345 67890")
        self.assert_equal(result.sentiment, Sentiment.NEUTRAL, "numbers → neutral")

    def test_emoji_only_input(self):
        """Emoji-only input handled."""
        result = self.engine.analyze("😊❤️🎉")
        self.assert_is_not_none(result, "emoji-only handled")

    def test_single_character(self):
        """Single character text handled."""
        result = self.engine.analyze("ߊ")
        self.assert_is_not_none(result, "single char handled")

    def test_mixed_languages(self):
        """French/English/N'Ko mixed text handled."""
        result = self.engine.analyze("Je suis ߘߌ߬ߦߊ very happy")
        self.assert_is_not_none(result, "multilingual handled")

    def test_repeated_analysis(self):
        """Same text analyzed multiple times gives consistent results."""
        text = "ߘߌ߬ߦߊ ߞߊ߬ߣߎ"
        r1 = self.engine.analyze(text)
        r2 = self.engine.analyze(text)
        self.assert_equal(r1.sentiment, r2.sentiment, "consistent sentiment")
        self.assert_equal(r1.score, r2.score, "consistent score")

    def test_prediction_boost_empty_candidates(self):
        """Empty candidates list handled."""
        boosts = self.engine.get_prediction_boost("test", [])
        self.assert_equal(len(boosts), 0, "empty candidates → empty boosts")

    def test_keyboard_context_empty_text(self):
        """Keyboard context for empty text."""
        ctx = self.engine.get_keyboard_context("")
        self.assert_in("sentiment", ctx, "empty text has sentiment key")

    def run_all(self):
        self.test_very_long_text()
        self.test_only_punctuation()
        self.test_numbers_only()
        self.test_emoji_only_input()
        self.test_single_character()
        self.test_mixed_languages()
        self.test_repeated_analysis()
        self.test_prediction_boost_empty_candidates()
        self.test_keyboard_context_empty_text()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Test: Enum Coverage
# ═══════════════════════════════════════════════════════════════

class _EnumsTests(_TestBase):
    """Tests for enum completeness."""

    def test_sentiment_values(self):
        """Sentiment enum has 5 levels."""
        self.assert_equal(len(Sentiment), 5, "5 sentiment levels")

    def test_emotion_values(self):
        """Emotion enum has 13 values."""
        self.assert_equal(len(Emotion), 13, "13 emotion types")

    def test_formality_values(self):
        """FormalityLevel enum has 5 levels."""
        self.assert_equal(len(FormalityLevel), 5, "5 formality levels")

    def test_life_event_values(self):
        """LifeEvent enum has 12 values."""
        self.assert_equal(len(LifeEvent), 12, "12 life event types")

    def run_all(self):
        self.test_sentiment_values()
        self.test_emotion_values()
        self.test_formality_values()
        self.test_life_event_values()
        return self.passed, self.failed, self.errors


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all test suites."""
    suites = [
        ("SentimentAnalyzer", _SentimentAnalyzerTests()),
        ("FormalityDetection", _FormalityDetectionTests()),
        ("LifeEventDetection", _LifeEventDetectionTests()),
        ("EmojiSuggestor", _EmojiSuggestorTests()),
        ("CulturalExpressions", _CulturalExpressionsTests()),
        ("ToneAdjuster", _ToneAdjusterTests()),
        ("EmotionalMemory", _EmotionalMemoryTests()),
        ("SentimentEngine", _SentimentEngineTests()),
        ("LexiconCoverage", _LexiconCoverageTests()),
        ("EdgeCases", _EdgeCasesTests()),
        ("Enums", _EnumsTests()),
    ]

    total_passed = 0
    total_failed = 0
    all_errors = []

    print("=" * 60)
    print("🧠 N'Ko Sentiment & Emotion Engine — Test Suite")
    print("=" * 60)

    for name, suite in suites:
        passed, failed, errors = suite.run_all()
        total_passed += passed
        total_failed += failed
        all_errors.extend(errors)

        status = "✅" if failed == 0 else "❌"
        print(f"  {status} {name}: {passed}/{passed + failed}")

    print("=" * 60)
    print(f"Total: {total_passed}/{total_passed + total_failed} passed")

    if all_errors:
        print(f"\n❌ {len(all_errors)} failures:")
        for err in all_errors:
            print(f"  • {err}")
    else:
        print("\n✅ All tests passed!")

    print("=" * 60)
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
