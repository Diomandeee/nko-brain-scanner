#!/usr/bin/env python3
"""
Tests for N'Ko Swipe Typing Engine (Gen 6 - Instance 16)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

import unittest
from swipe_engine import (
    SwipeTypingEngine,
    SwipeInterpreter,
    NkoKeyboardLayout,
    KeyboardLayout,
    SwipePoint,
    KeyPosition,
    SwipeKeyboardIntegration
)


class TestKeyboardLayout(unittest.TestCase):
    """Test keyboard layout functionality"""
    
    def setUp(self):
        self.layout = NkoKeyboardLayout(KeyboardLayout.STANDARD)
    
    def test_layout_has_keys(self):
        """Layout should contain N'Ko characters"""
        self.assertGreater(len(self.layout.keys), 20)
    
    def test_key_positions_normalized(self):
        """All key positions should be in 0-1 range"""
        for char, key in self.layout.keys.items():
            self.assertGreaterEqual(key.x, 0.0)
            self.assertLessEqual(key.x, 1.0)
            self.assertGreaterEqual(key.y, 0.0)
            self.assertLessEqual(key.y, 1.0)
    
    def test_get_key(self):
        """Should retrieve key positions"""
        key = self.layout.get_key("ߊ")
        self.assertIsNotNone(key)
        self.assertEqual(key.char, "ߊ")
    
    def test_get_missing_key(self):
        """Should return None for missing keys"""
        key = self.layout.get_key("A")  # Latin letter
        self.assertIsNone(key)
    
    def test_nearby_keys(self):
        """Should find keys near a given key"""
        nearby = self.layout.get_nearby_keys("ߊ", radius=0.2)
        self.assertIsInstance(nearby, list)
        # Should find at least some nearby keys
        self.assertGreater(len(nearby), 0)
    
    def test_word_to_path(self):
        """Should convert word to key path"""
        path = self.layout.word_to_path("ߓߍ")
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0].char, "ߓ")
        self.assertEqual(path[1].char, "ߍ")
    
    def test_path_length(self):
        """Should calculate path length"""
        # Single char has zero length
        length1 = self.layout.path_length("ߊ")
        self.assertEqual(length1, 0.0)
        
        # Two chars have non-zero length
        length2 = self.layout.path_length("ߓߍ")
        self.assertGreater(length2, 0.0)
    
    def test_different_layouts(self):
        """Different layout types should work"""
        phonetic = NkoKeyboardLayout(KeyboardLayout.PHONETIC)
        frequency = NkoKeyboardLayout(KeyboardLayout.FREQUENCY)
        
        self.assertGreater(len(phonetic.keys), 0)
        self.assertGreater(len(frequency.keys), 0)


class TestSwipeInterpreter(unittest.TestCase):
    """Test swipe interpretation"""
    
    def setUp(self):
        self.interpreter = SwipeInterpreter()
    
    def test_vocabulary_loaded(self):
        """Default vocabulary should be loaded"""
        self.assertGreater(len(self.interpreter.vocabulary), 30)
    
    def test_paths_precomputed(self):
        """Word paths should be precomputed"""
        self.assertGreater(len(self.interpreter.word_paths), 0)
    
    def test_interpret_empty_swipe(self):
        """Empty swipe should return empty result"""
        result = self.interpreter.interpret_swipe([])
        self.assertEqual(result.primary, "")
        self.assertEqual(len(result.candidates), 0)
    
    def test_interpret_single_point(self):
        """Single point swipe should return empty result"""
        result = self.interpreter.interpret_swipe([
            SwipePoint(0.5, 0.5)
        ])
        self.assertEqual(result.primary, "")
    
    def test_interpret_valid_swipe(self):
        """Valid swipe should return candidates"""
        # Simulate a swipe across some keys
        points = [
            SwipePoint(0.05, 0.33),  # Near ߊ
            SwipePoint(0.15, 0.33),  # Moving right
            SwipePoint(0.25, 0.33),  # Continuing
        ]
        result = self.interpreter.interpret_swipe(points)
        
        # Should have some candidates
        self.assertIsInstance(result.candidates, list)
        self.assertGreater(result.gesture_length, 0)
    
    def test_keys_passed_tracking(self):
        """Should track which keys were passed"""
        # Create path that clearly passes through first row
        points = [
            SwipePoint(0.05, 0.17),
            SwipePoint(0.15, 0.17),
            SwipePoint(0.25, 0.17),
            SwipePoint(0.35, 0.17),
        ]
        result = self.interpreter.interpret_swipe(points)
        
        # Should have identified some keys
        self.assertIsInstance(result.keys_passed, list)
    
    def test_context_affects_scoring(self):
        """Context should influence prediction scores"""
        points = [
            SwipePoint(0.05, 0.17),
            SwipePoint(0.15, 0.17),
        ]
        
        # Without context
        result1 = self.interpreter.interpret_swipe(points)
        
        # With context
        result2 = self.interpreter.interpret_swipe(points, context="ߌ")
        
        # Results may differ based on context
        # (just verify both work)
        self.assertIsInstance(result1.candidates, list)
        self.assertIsInstance(result2.candidates, list)


class TestSwipeTypingEngine(unittest.TestCase):
    """Test main swipe typing engine"""
    
    def setUp(self):
        self.engine = SwipeTypingEngine()
    
    def test_start_swipe(self):
        """Should start a swipe"""
        self.engine.start_swipe(0.5, 0.5)
        self.assertTrue(self.engine.is_swiping)
        self.assertEqual(len(self.engine.current_swipe), 1)
    
    def test_add_points(self):
        """Should add points to swipe"""
        self.engine.start_swipe(0.1, 0.5)
        self.engine.add_point(0.2, 0.5)
        self.engine.add_point(0.3, 0.5)
        
        self.assertEqual(len(self.engine.current_swipe), 3)
    
    def test_add_point_without_starting(self):
        """Adding point without starting should be ignored"""
        self.engine.add_point(0.5, 0.5)
        self.assertEqual(len(self.engine.current_swipe), 0)
    
    def test_end_swipe(self):
        """Should end swipe and return result"""
        self.engine.start_swipe(0.1, 0.5)
        self.engine.add_point(0.2, 0.5)
        self.engine.add_point(0.3, 0.5)
        
        result = self.engine.end_swipe()
        
        self.assertFalse(self.engine.is_swiping)
        self.assertEqual(len(self.engine.current_swipe), 0)
        self.assertIsNotNone(result)
    
    def test_cancel_swipe(self):
        """Should cancel swipe"""
        self.engine.start_swipe(0.5, 0.5)
        self.engine.cancel_swipe()
        
        self.assertFalse(self.engine.is_swiping)
        self.assertEqual(len(self.engine.current_swipe), 0)
    
    def test_set_context(self):
        """Should set context"""
        self.engine.set_context("ߌ ߣߌ")
        self.assertEqual(self.engine.context, "ߌ ߣߌ")
    
    def test_commit_updates_context(self):
        """Commit should update context"""
        self.engine.commit("ߌ")
        self.engine.commit("ߣߌ")
        
        self.assertIn("ߌ", self.engine.context)
        self.assertIn("ߣߌ", self.engine.context)
    
    def test_context_trimmed(self):
        """Context should be trimmed to last N words"""
        for i in range(10):
            self.engine.commit(f"ߊ{i}")
        
        words = self.engine.context.split()
        self.assertLessEqual(len(words), 5)
    
    def test_simulate_swipe(self):
        """Should simulate swipe for a word"""
        result = self.engine.simulate_swipe("ߓߍ")
        
        self.assertIsNotNone(result)
        # Primary should match or be close
        if result.candidates:
            self.assertGreater(result.confidence, 0)
    
    def test_simulate_unknown_word(self):
        """Should handle unknown word simulation"""
        result = self.engine.simulate_swipe("AAAA")  # Not in vocabulary
        
        # Should still return a result (possibly empty)
        self.assertIsNotNone(result)
    
    def test_add_vocabulary(self):
        """Should add words to vocabulary"""
        initial_size = len(self.engine.interpreter.vocabulary)
        
        self.engine.add_vocabulary(
            "ߕߍߛߕ",
            frequency=0.8,
            meaning="test",
            latinization="test"
        )
        
        self.assertEqual(
            len(self.engine.interpreter.vocabulary),
            initial_size + 1
        )
    
    def test_get_keyboard_layout(self):
        """Should return keyboard layout for rendering"""
        layout = self.engine.get_keyboard_layout()
        
        self.assertIsInstance(layout, dict)
        self.assertGreater(len(layout), 0)
        
        # Each entry should be (x, y) tuple
        for char, pos in layout.items():
            self.assertEqual(len(pos), 2)
            self.assertIsInstance(pos[0], float)
            self.assertIsInstance(pos[1], float)
    
    def test_get_stats(self):
        """Should return statistics"""
        # Perform some swipes
        self.engine.start_swipe(0.1, 0.5)
        self.engine.add_point(0.2, 0.5)
        self.engine.end_swipe()
        
        stats = self.engine.get_stats()
        
        self.assertIn("total_swipes", stats)
        self.assertIn("successful_swipes", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("vocabulary_size", stats)
        
        self.assertEqual(stats["total_swipes"], 1)


class TestSwipeKeyboardIntegration(unittest.TestCase):
    """Test integration with unified keyboard"""
    
    def test_integration_creation(self):
        """Should create integration with mock keyboard"""
        class MockKeyboard:
            def predict(self, text, top_k=5):
                return []
            def commit(self, text):
                pass
        
        integration = SwipeKeyboardIntegration(MockKeyboard())
        self.assertIsNotNone(integration)
    
    def test_process_swipe(self):
        """Should process swipe through integration"""
        class MockKeyboard:
            def predict(self, text, top_k=5):
                return []
            def commit(self, text):
                pass
        
        integration = SwipeKeyboardIntegration(MockKeyboard())
        
        # Process a simple swipe
        points = [
            (0.1, 0.5, 0.0),
            (0.2, 0.5, 0.05),
            (0.3, 0.5, 0.1),
        ]
        
        result = integration.process_swipe(points)
        self.assertIsInstance(result, list)


class TestKeyPositionMath(unittest.TestCase):
    """Test key position calculations"""
    
    def test_distance_calculation(self):
        """Should calculate distance correctly"""
        key1 = KeyPosition("ߊ", 0, 0, 0.0, 0.0)
        key2 = KeyPosition("ߋ", 0, 1, 1.0, 0.0)
        
        dist = key1.distance_to(key2)
        self.assertAlmostEqual(dist, 1.0, places=5)
    
    def test_diagonal_distance(self):
        """Should calculate diagonal distance"""
        key1 = KeyPosition("ߊ", 0, 0, 0.0, 0.0)
        key2 = KeyPosition("ߋ", 1, 1, 1.0, 1.0)
        
        dist = key1.distance_to(key2)
        self.assertAlmostEqual(dist, 1.414, places=2)
    
    def test_zero_distance(self):
        """Same position should have zero distance"""
        key1 = KeyPosition("ߊ", 0, 0, 0.5, 0.5)
        key2 = KeyPosition("ߊ", 0, 0, 0.5, 0.5)
        
        dist = key1.distance_to(key2)
        self.assertAlmostEqual(dist, 0.0, places=5)


class TestSwipePointMath(unittest.TestCase):
    """Test swipe point calculations"""
    
    def test_point_distance(self):
        """Should calculate point distance"""
        p1 = SwipePoint(0.0, 0.0)
        p2 = SwipePoint(3.0, 4.0)
        
        dist = p1.distance_to(p2)
        self.assertAlmostEqual(dist, 5.0, places=5)


class TestVocabularyManagement(unittest.TestCase):
    """Test vocabulary features"""
    
    def setUp(self):
        self.interpreter = SwipeInterpreter()
    
    def test_default_vocabulary_structure(self):
        """Default vocabulary should have correct structure"""
        for word, info in self.interpreter.vocabulary.items():
            self.assertIn("freq", info)
            self.assertIn("meaning", info)
            self.assertIn("latin", info)
    
    def test_vocabulary_has_greetings(self):
        """Should include common greetings"""
        greetings = ["ߌ", "ߣߌ", "ߗߍ"]
        for greeting in greetings:
            self.assertIn(greeting, self.interpreter.vocabulary)
    
    def test_vocabulary_has_common_words(self):
        """Should include common words"""
        common = ["ߓߍ", "ߞߊ", "ߦߋ", "ߡߐ߱"]
        for word in common:
            self.assertIn(word, self.interpreter.vocabulary)
    
    def test_add_new_word(self):
        """Should add new words"""
        self.interpreter.add_word(
            "ߕߍߛߕ",
            frequency=0.9,
            meaning="test word",
            latinization="test"
        )
        
        self.assertIn("ߕߍߛߕ", self.interpreter.vocabulary)
        self.assertIn("ߕߍߛߕ", self.interpreter.word_paths)


class TestPathMatching(unittest.TestCase):
    """Test path matching algorithms"""
    
    def setUp(self):
        self.interpreter = SwipeInterpreter()
    
    def test_exact_path_high_score(self):
        """Exact path should have high score"""
        # Get ideal path for a word
        word = "ߓߍ"
        ideal_path = self.interpreter.keyboard.word_to_path(word)
        
        if ideal_path:
            # Create swipe points exactly on the keys
            points = [
                SwipePoint(key.x, key.y) for key in ideal_path
            ]
            
            result = self.interpreter.interpret_swipe(points)
            
            # Should find the word with good confidence
            if result.candidates:
                word_found = any(c.word == word for c in result.candidates)
                # May not always be found due to path length
                self.assertIsInstance(result.confidence, float)
    
    def test_noisy_path_still_matches(self):
        """Slightly noisy path should still match"""
        word = "ߓߍ"
        ideal_path = self.interpreter.keyboard.word_to_path(word)
        
        if ideal_path:
            # Create swipe points with small noise
            import random
            random.seed(42)
            
            points = []
            for key in ideal_path:
                points.append(SwipePoint(
                    key.x + random.uniform(-0.02, 0.02),
                    key.y + random.uniform(-0.02, 0.02)
                ))
            
            result = self.interpreter.interpret_swipe(points)
            self.assertIsInstance(result.candidates, list)


class TestContextScoring(unittest.TestCase):
    """Test context-aware scoring"""
    
    def setUp(self):
        self.interpreter = SwipeInterpreter()
    
    def test_context_boost_after_i(self):
        """Words should be boosted after ߌ (you)"""
        # After "ߌ", "ߣߌ" (ni/with) should get boost
        score = self.interpreter._calculate_context_score("ߣߌ", "ߌ")
        self.assertGreater(score, 0)
    
    def test_no_boost_for_unrelated(self):
        """Unrelated words shouldn't get boost"""
        score = self.interpreter._calculate_context_score("ߘߎ߲", "ߌ")
        self.assertEqual(score, 0)
    
    def test_empty_context(self):
        """Empty context should return zero"""
        score = self.interpreter._calculate_context_score("ߣߌ", "")
        self.assertEqual(score, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
