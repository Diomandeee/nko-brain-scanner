#!/usr/bin/env python3
"""
Tests for N'Ko Learning Engine — Generation 6

Comprehensive test suite for the educational module.
"""

import os
import sys
try:
    import pytest
except ImportError:
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "pytest_compat",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest_compat.py")
        )
        pytest = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(pytest)
    except Exception:
        pytest = None  # pytest optional for unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from learning_engine import (
    NkoLearningEngine,
    LearnerLevel,
    LessonType,
    LearnerProgress,
    LetterMastery,
    Lesson,
    LessonResult,
    NKO_ALPHABET,
    VOCABULARY_BANK,
    PROVERBS,
    ACHIEVEMENTS,
    LEVEL_CURRICULUM,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def engine(temp_db):
    """Create a learning engine with temp database."""
    return NkoLearningEngine(db_path=temp_db)


@pytest.fixture
def user_id():
    """Test user ID."""
    return "test_user_001"


class TestLearnerProgress:
    """Tests for learner progress management."""
    
    def test_create_new_learner(self, engine, user_id):
        """New learners start at ABSOLUTE_BEGINNER."""
        progress = engine.get_progress(user_id)
        
        assert progress.user_id == user_id
        assert progress.level == LearnerLevel.ABSOLUTE_BEGINNER
        assert progress.total_xp == 0
        assert progress.current_streak == 0
        assert len(progress.letters_mastered) == 0
        assert len(progress.words_learned) == 0
        assert len(progress.achievements) == 0
        
    def test_progress_persistence(self, engine, user_id):
        """Progress is persisted to database."""
        progress = engine.get_progress(user_id)
        progress.total_xp = 100
        progress.letters_mastered.add("ߊ")
        engine._save_progress(progress)
        
        # Clear cache and reload
        engine._progress_cache.clear()
        reloaded = engine.get_progress(user_id)
        
        assert reloaded.total_xp == 100
        assert "ߊ" in reloaded.letters_mastered
        
    def test_multiple_users(self, engine):
        """Multiple users have separate progress."""
        user1 = engine.get_progress("user1")
        user2 = engine.get_progress("user2")
        
        user1.total_xp = 500
        engine._save_progress(user1)
        
        user2_check = engine.get_progress("user2")
        assert user2_check.total_xp == 0


class TestSpacedRepetition:
    """Tests for SM-2 spaced repetition algorithm."""
    
    def test_initial_mastery(self, engine, user_id):
        """New letters start with default mastery values."""
        mastery = engine.get_letter_mastery(user_id, "ߊ")
        
        assert mastery.letter == "ߊ"
        assert mastery.easiness == 2.5
        assert mastery.interval == 1
        assert mastery.repetitions == 0
        
    def test_correct_answer_increases_interval(self, engine, user_id):
        """Correct answers increase review interval."""
        initial = engine.get_letter_mastery(user_id, "ߊ")
        assert initial.interval == 1
        
        # Perfect answer (quality 5)
        engine.update_mastery(user_id, "ߊ", 5)
        after1 = engine.get_letter_mastery(user_id, "ߊ")
        assert after1.interval == 1  # First correct: 1 day
        assert after1.repetitions == 1
        
        # Second perfect answer
        engine.update_mastery(user_id, "ߊ", 5)
        after2 = engine.get_letter_mastery(user_id, "ߊ")
        assert after2.interval == 6  # Second correct: 6 days
        assert after2.repetitions == 2
        
    def test_incorrect_answer_resets_progress(self, engine, user_id):
        """Incorrect answers reset repetition count."""
        # Build up some progress
        engine.update_mastery(user_id, "ߊ", 5)
        engine.update_mastery(user_id, "ߊ", 5)
        
        mastery = engine.get_letter_mastery(user_id, "ߊ")
        assert mastery.repetitions == 2
        
        # Fail (quality 2)
        engine.update_mastery(user_id, "ߊ", 2)
        after_fail = engine.get_letter_mastery(user_id, "ߊ")
        
        assert after_fail.repetitions == 0
        assert after_fail.interval == 1
        
    def test_easiness_adjusts(self, engine, user_id):
        """Easiness factor adjusts based on performance."""
        initial_ef = engine.get_letter_mastery(user_id, "ߊ").easiness
        
        # Perfect answers increase EF
        engine.update_mastery(user_id, "ߊ", 5)
        higher_ef = engine.get_letter_mastery(user_id, "ߊ").easiness
        assert higher_ef > initial_ef
        
        # Difficult answer decreases EF
        engine.update_mastery(user_id, "ߓ", 3)
        lower_ef = engine.get_letter_mastery(user_id, "ߓ").easiness
        assert lower_ef < initial_ef
        
    def test_due_reviews(self, engine, user_id):
        """Get letters due for review."""
        # Learn some letters
        engine.update_mastery(user_id, "ߊ", 5)
        engine.update_mastery(user_id, "ߌ", 5)
        
        # Manually set one as due
        mastery = engine.get_letter_mastery(user_id, "ߊ")
        mastery.next_review = datetime.now() - timedelta(days=1)
        
        # Save directly to DB
        import sqlite3
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE letter_mastery SET next_review = ? WHERE user_id = ? AND letter = ?",
            (mastery.next_review.isoformat(), user_id, "ߊ")
        )
        conn.commit()
        conn.close()
        engine._mastery_cache.clear()
        
        due = engine.get_due_reviews(user_id)
        assert "ߊ" in due


class TestLessonGeneration:
    """Tests for lesson generation."""
    
    def test_generate_letter_intro(self, engine, user_id):
        """Generate letter introduction lesson."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        
        assert lesson.lesson_type == LessonType.LETTER_INTRO
        assert lesson.target_letter is not None
        assert lesson.target_letter in NKO_ALPHABET
        assert lesson.xp_reward > 0
        
    def test_generate_sound_match(self, engine, user_id):
        """Generate sound matching lesson."""
        lesson = engine.generate_lesson(user_id, LessonType.SOUND_MATCH)
        
        assert lesson.lesson_type == LessonType.SOUND_MATCH
        assert lesson.target_letter is not None
        assert len(lesson.options) > 0
        assert lesson.correct_answer in lesson.options
        
    def test_generate_word_lesson(self, engine, user_id):
        """Generate vocabulary lesson."""
        # Advance user to word learning level
        progress = engine.get_progress(user_id)
        progress.level = LearnerLevel.WORD_COLLECTOR
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id, LessonType.WORD_MEANING)
        
        assert lesson.lesson_type == LessonType.WORD_MEANING
        assert lesson.target_word is not None
        assert "nko" in lesson.target_word
        assert "english" in lesson.target_word
        
    def test_generate_proverb_lesson(self, engine, user_id):
        """Generate proverb lesson."""
        progress = engine.get_progress(user_id)
        progress.level = LearnerLevel.PROVERB_KEEPER
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id, LessonType.PROVERB_LEARN)
        
        assert lesson.lesson_type == LessonType.PROVERB_LEARN
        assert lesson.target_proverb is not None
        assert "meaning" in lesson.target_proverb
        
    def test_lesson_has_cultural_note(self, engine, user_id):
        """Lessons include cultural context."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        
        # Most lessons should have cultural notes
        assert lesson.cultural_note is not None or lesson.hint is not None


class TestLessonCompletion:
    """Tests for lesson completion and scoring."""
    
    def test_correct_answer_awards_xp(self, engine, user_id):
        """Correct answers award at least the base XP (may include speed bonus)."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        
        result = engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        assert result.correct is True
        assert result.xp_earned >= lesson.xp_reward
        
    def test_incorrect_answer_partial_xp(self, engine, user_id):
        """Incorrect answers award partial XP."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        
        result = engine.complete_lesson(user_id, lesson, "wrong", 5.0)
        
        assert result.correct is False
        assert result.xp_earned == lesson.xp_reward // 4
        
    def test_speed_bonus(self, engine, user_id):
        """Fast correct answers get bonus XP."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        
        # Fast answer (under 10 seconds)
        result = engine.complete_lesson(user_id, lesson, lesson.correct_answer, 3.0)
        
        assert result.xp_earned > lesson.xp_reward  # 20% bonus
        
    def test_letter_mastery_on_correct(self, engine, user_id):
        """Correct letter answers update mastery."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        letter = lesson.target_letter
        
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert letter in progress.letters_mastered
        
    def test_word_learning_tracked(self, engine, user_id):
        """Word lessons track learned vocabulary."""
        progress = engine.get_progress(user_id)
        progress.level = LearnerLevel.WORD_COLLECTOR
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id, LessonType.WORD_MEANING)
        word_nko = lesson.target_word["nko"]
        
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert word_nko in progress.words_learned
        
    def test_encouragement_message(self, engine, user_id):
        """Results include encouragement messages."""
        lesson = engine.generate_lesson(user_id)
        
        result = engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        assert result.encouragement != ""
        assert len(result.encouragement) > 0


class TestLevelProgression:
    """Tests for level advancement."""
    
    def test_level_up_at_threshold(self, engine, user_id):
        """Learner levels up when XP threshold reached."""
        progress = engine.get_progress(user_id)
        progress.total_xp = 95  # Just under level 1 threshold (100)
        engine._save_progress(progress)
        
        # Create lesson that awards enough XP
        lesson = Lesson(
            lesson_id="test",
            lesson_type=LessonType.LETTER_INTRO,
            target_letter="ߊ",
            question="Test",
            correct_answer="ߊ",
            xp_reward=20
        )
        
        result = engine.complete_lesson(user_id, lesson, "ߊ", 5.0)
        
        assert result.new_level == LearnerLevel.LETTER_EXPLORER
        
    def test_level_curriculum_exists(self, engine):
        """All levels have curriculum defined."""
        for level in LearnerLevel:
            assert level in LEVEL_CURRICULUM
            curriculum = LEVEL_CURRICULUM[level]
            assert "focus" in curriculum
            assert "xp_threshold" in curriculum


class TestAchievements:
    """Tests for gamification achievements."""
    
    def test_first_letter_achievement(self, engine, user_id):
        """First letter achievement unlocks correctly."""
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        
        result = engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert "first_letter" in progress.achievements
        
    def test_achievement_awards_bonus_xp(self, engine, user_id):
        """Achievements award bonus XP."""
        progress = engine.get_progress(user_id)
        initial_xp = progress.total_xp
        
        lesson = engine.generate_lesson(user_id, LessonType.LETTER_INTRO)
        result = engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        # XP should include lesson XP + achievement bonus
        expected_min = initial_xp + lesson.xp_reward + ACHIEVEMENTS["first_letter"].xp_bonus
        assert progress.total_xp >= expected_min
        
    def test_streak_achievement(self, engine, user_id):
        """Streak achievements work correctly."""
        progress = engine.get_progress(user_id)
        progress.current_streak = 2
        progress.last_session = datetime.now() - timedelta(days=1)
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id)
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert progress.current_streak == 3
        assert "streak_3" in progress.achievements


class TestStreakTracking:
    """Tests for learning streak functionality."""
    
    def test_streak_starts_at_one(self, engine, user_id):
        """First session starts streak at 1."""
        lesson = engine.generate_lesson(user_id)
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert progress.current_streak == 1
        
    def test_streak_continues(self, engine, user_id):
        """Streak continues on consecutive days."""
        progress = engine.get_progress(user_id)
        progress.current_streak = 5
        progress.last_session = datetime.now() - timedelta(days=1)
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id)
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert progress.current_streak == 6
        
    def test_streak_resets_after_gap(self, engine, user_id):
        """Streak resets after missing a day."""
        progress = engine.get_progress(user_id)
        progress.current_streak = 10
        progress.last_session = datetime.now() - timedelta(days=3)
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id)
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert progress.current_streak == 1
        
    def test_longest_streak_tracked(self, engine, user_id):
        """Longest streak is preserved."""
        progress = engine.get_progress(user_id)
        progress.current_streak = 15
        progress.longest_streak = 15
        progress.last_session = datetime.now() - timedelta(days=5)
        engine._save_progress(progress)
        
        lesson = engine.generate_lesson(user_id)
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        progress = engine.get_progress(user_id)
        assert progress.current_streak == 1
        assert progress.longest_streak == 15  # Still preserved


class TestContextualLearning:
    """Tests for keyboard integration features."""
    
    def test_contextual_tip_for_new_letter(self, engine, user_id):
        """Get tips for unlearned letters in typed text."""
        # User hasn't learned ߊ yet
        tip = engine.get_contextual_tip(user_id, "ߊ ߓߊ")
        
        assert tip is not None
        assert "ߊ" in tip
        assert "a" in tip.lower()  # Sound mentioned
        
    def test_no_tip_for_mastered_letters(self, engine, user_id):
        """No tips for already mastered letters."""
        progress = engine.get_progress(user_id)
        progress.letters_mastered.add("ߊ")
        progress.letters_mastered.add("ߓ")
        engine._save_progress(progress)
        
        tip = engine.get_contextual_tip(user_id, "ߊ ߓ")
        
        # Should be None or a vocabulary tip, not letter tip
        assert tip is None or "📚" in tip
        
    def test_vocabulary_tip(self, engine, user_id):
        """Get contextual tips for words in text."""
        # Find a word from vocabulary bank
        greeting = VOCABULARY_BANK["greetings"][0]["nko"]
        
        tip = engine.get_contextual_tip(user_id, greeting)
        
        assert tip is not None
        assert isinstance(tip, str) and len(tip) > 0
        
    def test_quick_lesson(self, engine, user_id):
        """Quick lessons prioritize reviews."""
        # Set up a due review
        engine.update_mastery(user_id, "ߊ", 5)
        
        import sqlite3
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE letter_mastery SET next_review = ? WHERE user_id = ?",
            ((datetime.now() - timedelta(days=1)).isoformat(), user_id)
        )
        conn.commit()
        conn.close()
        engine._mastery_cache.clear()
        
        lesson = engine.get_quick_lesson(user_id)
        
        assert lesson.lesson_type == LessonType.REVIEW


class TestStatistics:
    """Tests for learning analytics."""
    
    def test_get_statistics(self, engine, user_id):
        """Statistics include all required metrics."""
        # Complete some lessons
        for _ in range(3):
            lesson = engine.generate_lesson(user_id)
            engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
            
        stats = engine.get_statistics(user_id)
        
        assert "level" in stats
        assert "total_xp" in stats
        assert "letters_mastered" in stats
        assert "accuracy_percent" in stats
        assert "current_streak" in stats
        assert "achievements" in stats
        
    def test_accuracy_calculation(self, engine, user_id):
        """Accuracy is calculated correctly."""
        # 2 correct, 1 wrong
        for i in range(3):
            lesson = engine.generate_lesson(user_id)
            answer = lesson.correct_answer if i < 2 else "wrong"
            engine.complete_lesson(user_id, lesson, answer, 5.0)
            
        stats = engine.get_statistics(user_id)
        
        assert stats["accuracy_percent"] == pytest.approx(66.7, abs=0.1)
        
    def test_progress_summary(self, engine, user_id):
        """Progress summary is formatted correctly."""
        lesson = engine.generate_lesson(user_id)
        engine.complete_lesson(user_id, lesson, lesson.correct_answer, 5.0)
        
        summary = engine.get_progress_summary(user_id)
        
        assert "N'Ko Learning" in summary
        assert "Level:" in summary
        assert "XP:" in summary
        assert "Streak:" in summary


class TestDataIntegrity:
    """Tests for data structure completeness."""
    
    def test_all_letters_defined(self):
        """All N'Ko letters have complete data."""
        for letter, info in NKO_ALPHABET.items():
            assert "sound" in info
            assert "name" in info
            assert "strokes" in info
            assert "category" in info
            
    def test_vocabulary_structure(self):
        """All vocabulary entries have required fields."""
        for category, words in VOCABULARY_BANK.items():
            for word in words:
                assert "nko" in word
                assert "latin" in word
                assert "english" in word
                assert "dialect" in word
                
    def test_proverbs_complete(self):
        """All proverbs have meaning and context."""
        for proverb in PROVERBS:
            assert "nko" in proverb
            assert "english" in proverb
            assert "meaning" in proverb
            assert "context" in proverb
            
    def test_achievements_defined(self):
        """All achievements have valid structure."""
        for ach_id, ach in ACHIEVEMENTS.items():
            assert ach.name != ""
            assert ach.description != ""
            assert ach.icon != ""
            assert ach.requirement > 0
            assert ach.xp_bonus >= 0


class TestCurriculumProgression:
    """Tests for curriculum design."""
    
    def test_levels_have_increasing_thresholds(self):
        """XP thresholds increase with level."""
        prev_threshold = -1
        for level in LearnerLevel:
            threshold = LEVEL_CURRICULUM[level]["xp_threshold"]
            assert threshold >= prev_threshold
            prev_threshold = threshold
            
    def test_beginner_levels_focus_on_letters(self):
        """Early levels teach letters before words."""
        beginner_curriculum = LEVEL_CURRICULUM[LearnerLevel.LETTER_EXPLORER]
        assert "letters" in beginner_curriculum
        assert len(beginner_curriculum["letters"]) > 0
        
    def test_advanced_levels_include_culture(self):
        """Advanced levels include cultural content."""
        proverb_curriculum = LEVEL_CURRICULUM[LearnerLevel.PROVERB_KEEPER]
        assert LessonType.PROVERB_LEARN in proverb_curriculum["activities"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
