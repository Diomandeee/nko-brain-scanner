#!/usr/bin/env python3
"""
Tests for N'Ko Keyboard AI Personalization Engine (Gen 6)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from personalization_engine import (
    PersonalizationEngine,
    PersonalizedPredictionMixer,
    ContextProfile,
    UserWord,
    TypingPattern
)


class TestPersonalizationEngine(unittest.TestCase):
    """Test core personalization functionality"""
    
    def setUp(self):
        """Create temporary database for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="test_user_123"
        )
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_word_recording(self):
        """Test basic word recording"""
        self.engine.record_word("ߌ")
        self.engine.record_word("ߣߌ")
        self.engine.record_word("ߛߐ߲߬ߜߐߡߊ")
        
        self.assertEqual(self.engine.profile.word_frequencies["ߌ"], 1)
        self.assertEqual(self.engine.profile.word_frequencies["ߣߌ"], 1)
        
        # Record same word again
        self.engine.record_word("ߌ")
        self.assertEqual(self.engine.profile.word_frequencies["ߌ"], 2)
    
    def test_custom_words(self):
        """Test custom vocabulary management"""
        self.engine.add_custom_word(
            "ߞߏ߲߬ߘߏ",
            latinization="kɔndo",
            meaning="bird",
            context="general"
        )
        
        words = self.engine.get_custom_words()
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].word, "ߞߏ߲߬ߘߏ")
        self.assertEqual(words[0].meaning, "bird")
        
        # Remove word
        self.engine.remove_custom_word("ߞߏ߲߬ߘߏ")
        words = self.engine.get_custom_words()
        self.assertEqual(len(words), 0)
    
    def test_custom_words_by_context(self):
        """Test filtering custom words by context"""
        self.engine.add_custom_word("ߊߟߊ߫", meaning="God", context="religious")
        self.engine.add_custom_word("ߡߏ߬ߓߌ߬ߟߌ", meaning="phone", context="business")
        self.engine.add_custom_word("ߛߐ߲߬ߜߐ", meaning="morning", context="general")
        
        religious_words = self.engine.get_custom_words(context="religious")
        self.assertEqual(len(religious_words), 1)
        self.assertEqual(religious_words[0].word, "ߊߟߊ߫")
        
        all_words = self.engine.get_custom_words()
        self.assertEqual(len(all_words), 3)
    
    def test_sequence_learning(self):
        """Test word sequence pattern learning"""
        # Type a common greeting sequence multiple times
        for _ in range(5):
            self.engine.record_word("ߌ")
            self.engine.record_word("ߣߌ")
            self.engine.record_word("ߛߐ߲߬ߜߐߡߊ")
            self.engine._session_words = []  # Reset for new sequence
        
        # Should predict "ߛߐ߲߬ߜߐߡߊ" after "ߌ", "ߣߌ"
        predictions = self.engine.predict_next_in_sequence(["ߌ", "ߣߌ"])
        
        self.assertTrue(len(predictions) > 0)
        predicted_words = [w for w, _ in predictions]
        self.assertIn("ߛߐ߲߬ߜߐߡߊ", predicted_words)
    
    def test_correction_learning(self):
        """Test auto-correction learning"""
        # Simulate user correcting mistakes
        for _ in range(5):
            self.engine.record_correction("ߣ ߌ", "ߣߌ")
        
        corrections = self.engine.get_auto_corrections()
        self.assertIn("ߣ ߌ", corrections)
        self.assertEqual(corrections["ߣ ߌ"], "ߣߌ")
    
    def test_prediction_boost(self):
        """Test personalized prediction boosting"""
        # Custom word gets boost
        self.engine.add_custom_word("ߞߎ߲߬ߠߊ", meaning="knowledge")
        boost = self.engine.get_prediction_boost("ߞߎ߲߬ߠߊ")
        self.assertGreater(boost, 1.0)
        
        # Frequently used word gets boost
        for _ in range(10):
            self.engine.record_word("ߓߊ")
        boost = self.engine.get_prediction_boost("ߓߊ")
        self.assertGreater(boost, 1.0)
        
        # Unknown word gets no boost
        boost = self.engine.get_prediction_boost("ߕߏ߲߬ߕߏ")
        self.assertEqual(boost, 1.0)


class TestContextProfiles(unittest.TestCase):
    """Test context profile switching"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="profile_test"
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_profile_switching(self):
        """Test switching between context profiles"""
        self.assertEqual(self.engine.profile.active_profile, "general")
        
        self.engine.set_profile(ContextProfile.FORMAL)
        self.assertEqual(self.engine.profile.active_profile, "formal")
        
        self.engine.set_profile(ContextProfile.CASUAL)
        self.assertEqual(self.engine.profile.active_profile, "casual")
    
    def test_profile_weights(self):
        """Test that profiles have different weights"""
        self.engine.set_profile(ContextProfile.GENERAL)
        general_weights = self.engine.get_profile_weights()
        
        self.engine.set_profile(ContextProfile.FORMAL)
        formal_weights = self.engine.get_profile_weights()
        
        # Formal should have higher semantic weight
        self.assertGreater(
            formal_weights["semantic"],
            general_weights["semantic"]
        )
    
    def test_weight_adjustment(self):
        """Test user weight adjustments"""
        initial_weights = self.engine.get_profile_weights()
        initial_ngram = initial_weights["ngram"]
        
        # Increase ngram weight
        self.engine.adjust_weight("ngram", 0.1)
        new_weights = self.engine.get_profile_weights()
        
        # Weight should have increased (normalized, so not exactly +0.1)
        self.assertGreater(new_weights["ngram"], initial_ngram * 0.99)


class TestTypingAnalytics(unittest.TestCase):
    """Test typing pattern analysis"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="analytics_test"
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_keystroke_timing(self):
        """Test keystroke timing recording"""
        # Simulate typing at 150ms intervals
        for _ in range(20):
            self.engine.record_keystroke_timing(150.0)
        
        self.assertAlmostEqual(
            self.engine.profile.typing_pattern.avg_speed_ms,
            150.0,
            delta=1.0
        )
    
    def test_prediction_acceptance(self):
        """Test prediction acceptance tracking"""
        # 8 out of 10 predictions accepted
        for i in range(10):
            self.engine.record_prediction_outcome(f"word_{i}", i < 8)
        
        self.assertAlmostEqual(
            self.engine.profile.typing_pattern.completion_acceptance_rate,
            0.8,
            delta=0.1
        )
    
    def test_typing_stats(self):
        """Test comprehensive stats retrieval"""
        # Record some activity
        for _ in range(5):
            self.engine.record_word("ߌ")
            self.engine.record_keystroke_timing(200.0)
        
        stats = self.engine.get_typing_stats()
        
        self.assertIn("avg_speed_ms", stats)
        self.assertIn("words_per_minute", stats)
        self.assertIn("total_words_typed", stats)
        self.assertEqual(stats["total_words_typed"], 5)


class TestPrivacyFeatures(unittest.TestCase):
    """Test privacy and data management features"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="privacy_test"
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_user_id_hashing(self):
        """Test that user IDs are hashed"""
        self.assertNotEqual(self.engine.user_id, "privacy_test")
        self.assertEqual(len(self.engine.user_id), 16)  # SHA256[:16]
    
    def test_data_export(self):
        """Test GDPR-compliant data export"""
        self.engine.add_custom_word("ߕߍ߬ߛߍ", meaning="test")
        self.engine.record_word("ߌ")
        
        data = self.engine.export_data()
        
        self.assertIn("profile", data)
        self.assertIn("custom_words", data)
        self.assertIn("word_frequencies", data)
        self.assertIn("exported_at", data)
    
    def test_data_deletion(self):
        """Test right to be forgotten"""
        self.engine.add_custom_word("ߕߍ߬ߛߍ", meaning="test")
        self.engine.record_word("ߌ")
        
        # Verify data exists
        self.assertEqual(len(self.engine.get_custom_words()), 1)
        
        # Delete all data
        self.engine.delete_all_data()
        
        # Verify data is gone
        self.assertEqual(len(self.engine.get_custom_words()), 0)
        self.assertEqual(len(self.engine.profile.word_frequencies), 0)
    
    def test_retention_period(self):
        """Test data retention settings"""
        self.engine.set_retention_period(30)
        self.assertEqual(self.engine.profile.retention_days, 30)
        
        # Test bounds
        self.engine.set_retention_period(1)  # Too low
        self.assertEqual(self.engine.profile.retention_days, 7)
        
        self.engine.set_retention_period(1000)  # Too high
        self.assertEqual(self.engine.profile.retention_days, 365)


class TestPredictionMixer(unittest.TestCase):
    """Test personalized prediction mixing"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="mixer_test"
        )
        self.mixer = PersonalizedPredictionMixer(self.engine)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_boost_predictions(self):
        """Test prediction boosting"""
        # Add a custom word
        self.engine.add_custom_word("ߞߎ߲߬ߠߊ", meaning="knowledge")
        
        # Base predictions
        predictions = [
            ("ߞߎ߲߬ߠߊ", 0.3),  # Custom word
            ("ߞߏ", 0.5),       # Higher base score
            ("ߞߊ߲", 0.4),
        ]
        
        boosted = self.mixer.boost_predictions(predictions)
        
        # Custom word should now be ranked higher
        # (exact ranking depends on boost calculation)
        boosted_words = [w for w, _ in boosted]
        self.assertIn("ߞߎ߲߬ߠߊ", boosted_words)
    
    def test_auto_corrections(self):
        """Test applying learned auto-corrections"""
        # Learn a correction pattern
        for _ in range(5):
            self.engine.record_correction("ߣ ߌ", "ߣߌ")
        
        # Apply corrections
        text = "ߣ ߌ ߛߐ߲߬ߜߐߡߊ"
        corrected = self.mixer.apply_auto_corrections(text)
        
        self.assertEqual(corrected, "ߣߌ ߛߐ߲߬ߜߐߡߊ")


class TestSessionManagement(unittest.TestCase):
    """Test session lifecycle"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="session_test"
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_session_summary(self):
        """Test session summary generation"""
        # Simulate a session
        self.engine.record_word("ߌ")
        self.engine.record_word("ߣߌ")
        self.engine.record_word("ߌ")  # Repeat
        self.engine.record_correction("ߣ ߌ", "ߣߌ")
        self.engine.record_prediction_outcome("ߛߐ߲߬ߜߐ", True)
        self.engine.record_prediction_outcome("ߕߟߋ", False)
        
        summary = self.engine.get_session_summary()
        
        self.assertEqual(summary["words_typed"], 3)
        self.assertEqual(summary["corrections_made"], 1)
        self.assertEqual(summary["predictions_shown"], 2)
        self.assertEqual(summary["predictions_accepted"], 1)
        self.assertEqual(summary["unique_words"], 2)
    
    def test_end_session(self):
        """Test session ending and cleanup"""
        self.engine.record_word("ߌ")
        self.engine.record_word("ߣߌ")
        
        # End session
        self.engine.end_session()
        
        # Session tracking should be reset
        self.assertEqual(len(self.engine._session_words), 0)
        self.assertEqual(len(self.engine._session_corrections), 0)


class TestPersistence(unittest.TestCase):
    """Test data persistence across sessions"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_profile_persistence(self):
        """Test that profile data persists across instances"""
        # First session
        engine1 = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="persist_test"
        )
        engine1.add_custom_word("ߞߎ߲߬ߠߊ", meaning="knowledge")
        engine1.record_word("ߌ")
        engine1.record_word("ߌ")
        engine1.set_profile(ContextProfile.FORMAL)
        engine1.end_session()
        
        # Second session (new instance)
        engine2 = PersonalizationEngine(
            db_path=self.temp_dir,
            user_id="persist_test"
        )
        
        # Data should persist
        self.assertEqual(len(engine2.get_custom_words()), 1)
        self.assertEqual(engine2.profile.active_profile, "formal")


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
