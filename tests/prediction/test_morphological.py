#!/usr/bin/env python3
"""
Tests for N'Ko Morphological Engine (Gen 6.3)
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from morphological_engine import (
    MandingMorphologyAnalyzer,
    CodeSwitchingDetector,
    CulturalCalendarEngine,
    Gen63MorphologicalEngine
)


class TestMorphologyAnalyzer(unittest.TestCase):
    """Test Manding morphological analysis"""
    
    def setUp(self):
        self.analyzer = MandingMorphologyAnalyzer()
    
    def test_root_extraction_known_word(self):
        """Test extraction of known roots"""
        analysis = self.analyzer.analyze("ߕߊ߯")
        self.assertEqual(analysis.root, "ߕߊ߯")
        self.assertIn("go", analysis.meaning_components)
    
    def test_suffix_detection(self):
        """Test detection of verbal suffixes"""
        # Word with -ߟߌ suffix (nominalization)
        analysis = self.analyzer.analyze("ߞߍߟߌ")
        self.assertEqual(analysis.derivation_type, "nominal")
        self.assertIn("ߟߌ", analysis.suffixes)
    
    def test_prefix_detection(self):
        """Test detection of prefixes"""
        # Word with ߟߊ߬- prefix (causative)
        analysis = self.analyzer.analyze("ߟߊ߬ߕߊ߯")
        self.assertIn("ߟߊ߬", analysis.prefixes)
    
    def test_related_forms_generation(self):
        """Test generation of related word forms"""
        analysis = self.analyzer.analyze("ߞߍ")  # "do/make"
        self.assertTrue(len(analysis.related_forms) > 0)
    
    def test_morpheme_prediction(self):
        """Test prediction of next morpheme"""
        predictions = self.analyzer.predict_next_morpheme("ߞߍ")
        self.assertTrue(len(predictions) > 0)
        # Should suggest suffixes after a known root
        suffixes_suggested = [p[0] for p in predictions]
        self.assertTrue(any(s in self.analyzer.VERBAL_SUFFIXES for s in suffixes_suggested))


class TestCodeSwitching(unittest.TestCase):
    """Test code-switching detection"""
    
    def setUp(self):
        self.detector = CodeSwitchingDetector()
    
    def test_pure_nko_detection(self):
        """Test detection of pure N'Ko text"""
        result = self.detector.detect("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        self.assertIn("nko", result.detected_languages)
        self.assertGreater(result.detected_languages.get("nko", 0), 0.5)
    
    def test_french_detection(self):
        """Test detection of French words"""
        result = self.detector.detect("mais voilà donc")
        self.assertIn("french", result.detected_languages)
        self.assertGreater(result.detected_languages.get("french", 0), 0.5)
    
    def test_mixed_code_switch(self):
        """Test detection of code-switching"""
        result = self.detector.detect("ߌ ߣߌ ߓߙߍ mais ߞߏ ߡߊ߫")
        # Should detect both languages
        self.assertIn("nko", result.detected_languages)
        self.assertIn("french", result.detected_languages)
        # Should mark switch points
        self.assertTrue(len(result.switch_points) > 0)
    
    def test_english_technical_terms(self):
        """Test detection of English technical terms"""
        result = self.detector.detect("internet wifi email")
        self.assertIn("english", result.detected_languages)
    
    def test_hybrid_completions(self):
        """Test French-Manding hybrid completions"""
        completions = self.detector.get_hybrid_completions("ߟߍ")
        # Should suggest hybrid words starting with ߟߍ
        self.assertTrue(any(c.startswith("ߟߍ") for c in completions) or len(completions) == 0)


class TestCulturalCalendar(unittest.TestCase):
    """Test cultural calendar integration"""
    
    def setUp(self):
        self.calendar = CulturalCalendarEngine()
    
    def test_time_period_morning(self):
        """Test morning time period detection"""
        morning = datetime(2025, 1, 15, 8, 0)  # 8 AM
        context = self.calendar.get_current_context(morning)
        self.assertIn("ߛߐ߲߬ߜߐ", context.relevant_vocabulary)  # Morning vocab
    
    def test_time_period_evening(self):
        """Test evening time period detection"""
        evening = datetime(2025, 1, 15, 19, 0)  # 7 PM
        context = self.calendar.get_current_context(evening)
        self.assertIn("ߛߎ", context.relevant_vocabulary)  # Evening vocab
    
    def test_rainy_season(self):
        """Test rainy season detection"""
        rainy = datetime(2025, 7, 15, 12, 0)  # July, noon
        context = self.calendar.get_current_context(rainy)
        self.assertIn("farming", context.boost_topics)
    
    def test_dry_season(self):
        """Test dry season detection"""
        dry = datetime(2025, 2, 15, 12, 0)  # February
        context = self.calendar.get_current_context(dry)
        self.assertIn("travel", context.boost_topics)
    
    def test_contextual_boosts(self):
        """Test contextual boost generation"""
        boosts = self.calendar.get_contextual_boosts()
        self.assertIsInstance(boosts, dict)
        self.assertTrue(len(boosts) > 0)
        # All boost values should be positive
        self.assertTrue(all(v > 0 for v in boosts.values()))


class TestUnifiedEngine(unittest.TestCase):
    """Test the unified Gen 6.3 engine"""
    
    def setUp(self):
        self.engine = Gen63MorphologicalEngine()
    
    def test_full_analysis(self):
        """Test complete input analysis"""
        analysis = self.engine.analyze_input("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        
        self.assertIn("word_analyses", analysis)
        self.assertIn("code_switch", analysis)
        self.assertIn("cultural_context", analysis)
        self.assertEqual(len(analysis["word_analyses"]), 3)
    
    def test_enhanced_predictions(self):
        """Test enhanced prediction generation"""
        predictions = self.engine.get_enhanced_predictions("ߌ ߣߌ")
        
        self.assertIsInstance(predictions, list)
        # Should return tuples of (word, score)
        if predictions:
            self.assertEqual(len(predictions[0]), 2)
    
    def test_stats(self):
        """Test statistics reporting"""
        stats = self.engine.get_stats()
        
        self.assertEqual(stats["generation"], "6.3")
        self.assertIn("morphological_analysis", stats["features"])
        self.assertIn("code_switching_detection", stats["features"])
        self.assertIn("cultural_calendar_integration", stats["features"])


class TestIntegration(unittest.TestCase):
    """Integration tests for Gen 6.3"""
    
    def setUp(self):
        self.engine = Gen63MorphologicalEngine()
    
    def test_greeting_prediction_morning(self):
        """Test that morning context influences predictions"""
        morning = datetime(2025, 1, 15, 8, 0)
        context = self.engine.cultural.get_current_context(morning)
        
        # Morning greeting should be in greetings
        self.assertTrue(
            any("ߛߐ߲߬ߜߐ" in g for g in context.greetings)
        )
    
    def test_code_switch_resilience(self):
        """Test that engine handles code-switching gracefully"""
        # Mixed Manding-French input
        analysis = self.engine.analyze_input("ߒ ߓߍ߬ mais ߏ߬")
        
        # Should not crash and should provide analysis
        self.assertIn("word_analyses", analysis)
        self.assertTrue(len(analysis["word_analyses"]) > 0)
    
    def test_morphological_chain(self):
        """Test analysis of morphologically complex words"""
        # Build a word: ߟߊ߬ (causative) + ߝߐ (speak) + ߟߌ (nominalization)
        analysis = self.engine.analyze_input("ߟߊ߬ߝߐߟߌ")
        
        self.assertTrue(len(analysis["word_analyses"]) > 0)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
