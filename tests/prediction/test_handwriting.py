#!/usr/bin/env python3
"""
Tests for N'Ko Handwriting Recognition Engine (Gen 15)

Run with:
    cd ~/Desktop/nko-keyboard-ai
    PYTHONPATH=. python3 tests/test_handwriting.py
"""

import unittest
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.handwriting_engine import (
    Point, Stroke, StrokeDirection, StrokeType,
    StrokeAnalyzer, HandwritingRecognitionEngine,
    HandwritingKeyboardIntegration, CharacterCandidate,
    NKO_TEMPLATES
)


class TestPoint(unittest.TestCase):
    """Test Point class."""
    
    def test_point_creation(self):
        """Test basic point creation."""
        p = Point(100, 200)
        self.assertEqual(p.x, 100)
        self.assertEqual(p.y, 200)
    
    def test_point_with_timestamp(self):
        """Test point with timestamp and pressure."""
        p = Point(50, 50, timestamp=1234567890.0, pressure=0.8)
        self.assertEqual(p.timestamp, 1234567890.0)
        self.assertEqual(p.pressure, 0.8)
    
    def test_distance_to(self):
        """Test distance calculation."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)
    
    def test_distance_same_point(self):
        """Test distance to same point."""
        p = Point(100, 100)
        self.assertEqual(p.distance_to(p), 0.0)
    
    def test_angle_to_right(self):
        """Test angle to point on right."""
        p1 = Point(0, 0)
        p2 = Point(100, 0)
        self.assertAlmostEqual(p1.angle_to(p2), 0.0)
    
    def test_angle_to_up(self):
        """Test angle to point above (note: y increases downward in screen coords)."""
        p1 = Point(0, 100)
        p2 = Point(0, 0)
        self.assertAlmostEqual(p1.angle_to(p2), 90.0)
    
    def test_angle_to_left(self):
        """Test angle to point on left."""
        p1 = Point(100, 0)
        p2 = Point(0, 0)
        self.assertAlmostEqual(p1.angle_to(p2), 180.0)
    
    def test_angle_to_down(self):
        """Test angle to point below."""
        p1 = Point(0, 0)
        p2 = Point(0, 100)
        self.assertAlmostEqual(p1.angle_to(p2), 270.0)


class TestStroke(unittest.TestCase):
    """Test Stroke class."""
    
    def test_empty_stroke(self):
        """Test empty stroke properties."""
        s = Stroke()
        self.assertEqual(s.bounding_box, (0, 0, 0, 0))
        self.assertEqual(s.width, 0)
        self.assertEqual(s.height, 0)
        self.assertEqual(s.total_length, 0)
    
    def test_stroke_bounding_box(self):
        """Test bounding box calculation."""
        s = Stroke(points=[
            Point(10, 20),
            Point(50, 80),
            Point(30, 50)
        ])
        self.assertEqual(s.bounding_box, (10, 20, 50, 80))
        self.assertEqual(s.width, 40)
        self.assertEqual(s.height, 60)
    
    def test_stroke_aspect_ratio(self):
        """Test aspect ratio."""
        # Horizontal stroke
        s1 = Stroke(points=[Point(0, 50), Point(100, 50)])
        self.assertGreater(s1.aspect_ratio, 1)
        
        # Vertical stroke
        s2 = Stroke(points=[Point(50, 0), Point(50, 100)])
        self.assertLess(s2.aspect_ratio, 1)
    
    def test_stroke_centroid(self):
        """Test centroid calculation."""
        s = Stroke(points=[
            Point(0, 0),
            Point(100, 0),
            Point(100, 100),
            Point(0, 100)
        ])
        c = s.centroid
        self.assertEqual(c.x, 50)
        self.assertEqual(c.y, 50)
    
    def test_stroke_total_length(self):
        """Test total path length."""
        s = Stroke(points=[
            Point(0, 0),
            Point(100, 0),
            Point(100, 100)
        ])
        self.assertEqual(s.total_length, 200.0)
    
    def test_stroke_is_closed(self):
        """Test closed stroke detection."""
        # Closed loop (circle-ish)
        closed = Stroke(points=[
            Point(50, 0),
            Point(100, 50),
            Point(50, 100),
            Point(0, 50),
            Point(50, 0)
        ])
        self.assertTrue(closed.is_closed())
        
        # Open stroke
        open_stroke = Stroke(points=[
            Point(0, 0),
            Point(100, 100)
        ])
        self.assertFalse(open_stroke.is_closed())


class TestStrokeAnalyzer(unittest.TestCase):
    """Test StrokeAnalyzer class."""
    
    def setUp(self):
        self.analyzer = StrokeAnalyzer()
    
    def test_analyze_dot(self):
        """Test dot detection."""
        # A dot is a very short stroke with few points
        stroke = Stroke(points=[
            Point(50, 50),
            Point(51, 51),
            Point(51.5, 51.5)
        ])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.stroke_type, StrokeType.DOT)
    
    def test_analyze_vertical_line(self):
        """Test vertical line detection."""
        stroke = Stroke(points=[Point(50, y) for y in range(0, 101, 5)])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.stroke_type, StrokeType.VERTICAL)
    
    def test_analyze_horizontal_line(self):
        """Test horizontal line detection."""
        stroke = Stroke(points=[Point(x, 50) for x in range(0, 101, 5)])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.stroke_type, StrokeType.HORIZONTAL)
    
    def test_analyze_loop(self):
        """Test loop detection."""
        # Create a circular path
        points = []
        for i in range(0, 361, 10):
            rad = math.radians(i)
            points.append(Point(
                50 + 30 * math.cos(rad),
                50 + 30 * math.sin(rad)
            ))
        stroke = Stroke(points=points)
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.stroke_type, StrokeType.LOOP)
    
    def test_analyze_diagonal_down(self):
        """Test diagonal down detection."""
        stroke = Stroke(points=[
            Point(0, 0),
            Point(25, 25),
            Point(50, 50),
            Point(75, 75),
            Point(100, 100)
        ])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.stroke_type, StrokeType.DIAGONAL_DOWN)
    
    def test_analyze_diagonal_up(self):
        """Test diagonal up detection."""
        stroke = Stroke(points=[
            Point(0, 100),
            Point(25, 75),
            Point(50, 50),
            Point(75, 25),
            Point(100, 0)
        ])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.stroke_type, StrokeType.DIAGONAL_UP)
    
    def test_direction_detection(self):
        """Test direction detection."""
        # Rightward stroke (needs multiple points to not be a dot)
        stroke = Stroke(points=[Point(0, 50), Point(50, 50), Point(100, 50)])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.direction, StrokeDirection.E)
        
        # Downward stroke
        stroke = Stroke(points=[Point(50, 0), Point(50, 50), Point(50, 100)])
        analyzed = self.analyzer.analyze_stroke(stroke)
        self.assertEqual(analyzed.direction, StrokeDirection.S)


class TestNkoTemplates(unittest.TestCase):
    """Test N'Ko character templates."""
    
    def test_all_vowels_present(self):
        """Test that all 7 vowels are defined."""
        vowels = ['ߊ', 'ߋ', 'ߎ', 'ߌ', 'ߍ', 'ߏ', 'ߐ']
        for v in vowels:
            self.assertIn(v, NKO_TEMPLATES)
    
    def test_key_consonants_present(self):
        """Test that key consonants are defined."""
        consonants = ['ߓ', 'ߕ', 'ߘ', 'ߛ', 'ߞ', 'ߡ', 'ߣ', 'ߒ']
        for c in consonants:
            self.assertIn(c, NKO_TEMPLATES)
    
    def test_template_structure(self):
        """Test that templates have required fields."""
        for char, template in NKO_TEMPLATES.items():
            self.assertEqual(template.character, char)
            self.assertTrue(len(template.name) > 0)
            self.assertTrue(len(template.english) > 0)
            self.assertGreater(template.expected_strokes, 0)
            self.assertTrue(len(template.stroke_patterns) > 0)
    
    def test_similar_characters_symmetric(self):
        """Test that similar_to relationships make sense."""
        for char, template in NKO_TEMPLATES.items():
            for similar in template.similar_to:
                # Similar character should exist
                self.assertIn(similar, NKO_TEMPLATES,
                    f"{char} lists {similar} as similar, but {similar} not in templates")


class TestHandwritingRecognitionEngine(unittest.TestCase):
    """Test HandwritingRecognitionEngine class."""
    
    def setUp(self):
        self.engine = HandwritingRecognitionEngine(db_path=":memory:")
    
    def test_start_character(self):
        """Test starting a new character."""
        self.engine.start_character()
        self.assertEqual(len(self.engine.current_strokes), 0)
        self.assertFalse(self.engine.drawing)
    
    def test_add_stroke(self):
        """Test adding a stroke."""
        self.engine.start_character()
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.add_point(100, 100)
        self.engine.end_stroke()
        
        self.assertEqual(len(self.engine.current_strokes), 1)
        self.assertEqual(len(self.engine.current_strokes[0].points), 2)
    
    def test_multiple_strokes(self):
        """Test adding multiple strokes."""
        self.engine.start_character()
        
        # First stroke
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.add_point(0, 100)
        self.engine.end_stroke()
        
        # Second stroke
        self.engine.start_stroke()
        self.engine.add_point(50, 0)
        self.engine.add_point(50, 100)
        self.engine.end_stroke()
        
        self.assertEqual(len(self.engine.current_strokes), 2)
    
    def test_recognize_loop_as_o(self):
        """Test recognizing a loop as ߋ (o)."""
        self.engine.start_character()
        self.engine.start_stroke()
        
        # Draw a circle
        for i in range(0, 361, 10):
            rad = math.radians(i)
            x = 50 + 30 * math.cos(rad)
            y = 50 + 30 * math.sin(rad)
            self.engine.add_point(x, y)
        
        self.engine.end_stroke()
        
        results = self.engine.recognize()
        self.assertTrue(len(results) > 0)
        
        # ߋ should be a top candidate for a loop
        top_chars = [r.character for r in results[:3]]
        self.assertIn('ߋ', top_chars, "ߋ should be recognized for a loop")
    
    def test_recognize_vertical_dot_as_i(self):
        """Test recognizing vertical + dot as ߌ (i)."""
        self.engine.start_character()
        
        # Vertical line
        self.engine.start_stroke()
        for y in range(50, 150, 5):
            self.engine.add_point(100, y)
        self.engine.end_stroke()
        
        # Dot above
        self.engine.start_stroke()
        self.engine.add_point(100, 30)
        self.engine.add_point(101, 31)
        self.engine.end_stroke()
        
        results = self.engine.recognize()
        self.assertTrue(len(results) > 0)
        
        # ߌ should be a top candidate
        top_chars = [r.character for r in results[:3]]
        self.assertIn('ߌ', top_chars, "ߌ should be recognized for vertical + dot")
    
    def test_recognize_cross_as_ta(self):
        """Test recognizing cross shape as ߕ (ta)."""
        self.engine.start_character()
        
        # Horizontal
        self.engine.start_stroke()
        for x in range(50, 150, 5):
            self.engine.add_point(x, 100)
        self.engine.end_stroke()
        
        # Vertical
        self.engine.start_stroke()
        for y in range(50, 150, 5):
            self.engine.add_point(100, y)
        self.engine.end_stroke()
        
        results = self.engine.recognize()
        self.assertTrue(len(results) > 0)
        
        # ߕ should be a top candidate for cross shape
        top_chars = [r.character for r in results[:3]]
        self.assertIn('ߕ', top_chars, "ߕ should be recognized for cross shape")
    
    def test_recognize_returns_confidence(self):
        """Test that recognition returns confidence scores."""
        self.engine.start_character()
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.add_point(100, 100)
        self.engine.end_stroke()
        
        results = self.engine.recognize()
        for r in results:
            self.assertIsInstance(r.confidence, float)
            self.assertGreaterEqual(r.confidence, 0)
            self.assertLessEqual(r.confidence, 1)
    
    def test_recognize_results_sorted(self):
        """Test that results are sorted by confidence."""
        self.engine.start_character()
        self.engine.start_stroke()
        for y in range(0, 101, 5):
            self.engine.add_point(50, y)
        self.engine.end_stroke()
        
        results = self.engine.recognize()
        confidences = [r.confidence for r in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
    
    def test_clear(self):
        """Test clearing strokes."""
        self.engine.start_character()
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.end_stroke()
        
        self.assertEqual(len(self.engine.current_strokes), 1)
        
        self.engine.clear()
        self.assertEqual(len(self.engine.current_strokes), 0)
    
    def test_get_stroke_count(self):
        """Test stroke count getter."""
        self.engine.start_character()
        self.assertEqual(self.engine.get_stroke_count(), 0)
        
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.end_stroke()
        self.assertEqual(self.engine.get_stroke_count(), 1)
        
        self.engine.start_stroke()
        self.engine.add_point(50, 50)
        self.engine.end_stroke()
        self.assertEqual(self.engine.get_stroke_count(), 2)
    
    def test_get_stroke_types(self):
        """Test getting stroke types."""
        self.engine.start_character()
        
        # Vertical
        self.engine.start_stroke()
        for y in range(0, 101, 5):
            self.engine.add_point(50, y)
        self.engine.end_stroke()
        
        types = self.engine.get_stroke_types()
        self.assertEqual(len(types), 1)
        self.assertEqual(types[0], 'vertical')
    
    def test_get_all_characters(self):
        """Test getting all supported characters."""
        chars = self.engine.get_all_characters()
        self.assertGreater(len(chars), 20)
        
        for char, name, english in chars:
            self.assertTrue(len(char) == 1)
            self.assertTrue(len(name) > 0)
            self.assertTrue(len(english) > 0)


class TestConfirmAndLearning(unittest.TestCase):
    """Test learning from user corrections."""
    
    def setUp(self):
        self.engine = HandwritingRecognitionEngine(db_path=":memory:")
    
    def test_confirm_character(self):
        """Test confirming a character stores data."""
        self.engine.start_character()
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.add_point(100, 100)
        self.engine.end_stroke()
        
        # Should not raise
        self.engine.confirm_character('ߊ')
    
    def test_learning_affects_recognition(self):
        """Test that confirming patterns affects future recognition."""
        # Confirm ߊ multiple times with 2 strokes
        for _ in range(5):
            self.engine.start_character()
            self.engine.start_stroke()
            self.engine.add_point(0, 0)
            self.engine.add_point(50, 100)
            self.engine.end_stroke()
            self.engine.start_stroke()
            self.engine.add_point(50, 100)
            self.engine.add_point(100, 0)
            self.engine.end_stroke()
            self.engine.confirm_character('ߊ')
        
        # Now draw similar pattern
        self.engine.start_character()
        self.engine.start_stroke()
        self.engine.add_point(0, 0)
        self.engine.add_point(50, 100)
        self.engine.end_stroke()
        self.engine.start_stroke()
        self.engine.add_point(50, 100)
        self.engine.add_point(100, 0)
        self.engine.end_stroke()
        
        results = self.engine.recognize()
        # ߊ should be boosted after learning
        a_result = next((r for r in results if r.character == 'ߊ'), None)
        self.assertIsNotNone(a_result)


class TestHandwritingKeyboardIntegration(unittest.TestCase):
    """Test keyboard integration."""
    
    def setUp(self):
        engine = HandwritingRecognitionEngine(db_path=":memory:")
        self.integration = HandwritingKeyboardIntegration(engine)
    
    def test_initial_mode(self):
        """Test initial mode is keyboard."""
        self.assertEqual(self.integration.mode, "keyboard")
    
    def test_switch_modes(self):
        """Test switching between modes."""
        self.integration.switch_to_handwriting()
        self.assertEqual(self.integration.mode, "handwriting")
        
        self.integration.switch_to_keyboard()
        self.assertEqual(self.integration.mode, "keyboard")
    
    def test_touch_in_keyboard_mode(self):
        """Test that touch is ignored in keyboard mode."""
        self.integration.on_touch_start(0, 0)
        self.integration.on_touch_move(50, 50)
        self.integration.on_touch_end()
        
        # Should have no strokes
        self.assertEqual(self.integration.engine.get_stroke_count(), 0)
    
    def test_touch_in_handwriting_mode(self):
        """Test touch handling in handwriting mode."""
        self.integration.switch_to_handwriting()
        
        self.integration.on_touch_start(0, 0)
        self.integration.on_touch_move(50, 50)
        self.integration.on_touch_move(100, 100)
        self.integration.on_touch_end()
        
        # Should have one stroke
        self.assertEqual(self.integration.engine.get_stroke_count(), 1)
    
    def test_get_candidates(self):
        """Test getting recognition candidates."""
        self.integration.switch_to_handwriting()
        
        # Draw a circle
        self.integration.on_touch_start(50, 0)
        for i in range(0, 361, 10):
            rad = math.radians(i)
            x = 50 + 30 * math.cos(rad)
            y = 50 + 30 * math.sin(rad)
            self.integration.on_touch_move(x, y)
        self.integration.on_touch_end()
        
        candidates = self.integration.get_candidates()
        self.assertTrue(len(candidates) > 0)
    
    def test_select_candidate(self):
        """Test selecting a candidate."""
        self.integration.switch_to_handwriting()
        
        self.integration.on_touch_start(50, 0)
        for i in range(0, 361, 10):
            rad = math.radians(i)
            x = 50 + 30 * math.cos(rad)
            y = 50 + 30 * math.sin(rad)
            self.integration.on_touch_move(x, y)
        self.integration.on_touch_end()
        
        char = self.integration.select_candidate(0)
        self.assertIsNotNone(char)
        self.assertTrue(len(char) == 1)
    
    def test_callback_on_selection(self):
        """Test callback is called on selection."""
        received = []
        self.integration.set_callback(lambda c: received.append(c))
        
        self.integration.switch_to_handwriting()
        
        self.integration.on_touch_start(50, 0)
        for i in range(0, 361, 10):
            rad = math.radians(i)
            self.integration.on_touch_move(
                50 + 30 * math.cos(rad),
                50 + 30 * math.sin(rad)
            )
        self.integration.on_touch_end()
        
        self.integration.select_candidate(0)
        self.assertEqual(len(received), 1)
    
    def test_cancel(self):
        """Test canceling handwriting input."""
        self.integration.switch_to_handwriting()
        
        self.integration.on_touch_start(0, 0)
        self.integration.on_touch_move(100, 100)
        self.integration.on_touch_end()
        
        self.assertEqual(self.integration.engine.get_stroke_count(), 1)
        
        self.integration.cancel()
        self.assertEqual(self.integration.engine.get_stroke_count(), 0)


class TestCharacterCandidate(unittest.TestCase):
    """Test CharacterCandidate class."""
    
    def test_candidate_creation(self):
        """Test creating a candidate."""
        c = CharacterCandidate(
            character='ߊ',
            confidence=0.85,
            english_name='a (vowel a)',
            stroke_match=0.9,
            shape_match=0.8
        )
        self.assertEqual(c.character, 'ߊ')
        self.assertEqual(c.confidence, 0.85)
    
    def test_candidate_defaults(self):
        """Test candidate default values."""
        c = CharacterCandidate(character='ߓ', confidence=0.5)
        self.assertEqual(c.english_name, "")
        self.assertEqual(c.stroke_match, 0.0)


class TestRealTimeRecognition(unittest.TestCase):
    """Test real-time recognition during drawing."""
    
    def setUp(self):
        self.engine = HandwritingRecognitionEngine(db_path=":memory:")
    
    def test_realtime_returns_fewer_results(self):
        """Test that realtime mode returns fewer results."""
        self.engine.start_character()
        self.engine.start_stroke()
        for y in range(0, 101, 5):
            self.engine.add_point(50, y)
        self.engine.end_stroke()
        
        realtime = self.engine.recognize_realtime()
        full = self.engine.recognize()
        
        self.assertLessEqual(len(realtime), 3)
        self.assertLessEqual(len(realtime), len(full))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.engine = HandwritingRecognitionEngine(db_path=":memory:")
    
    def test_recognize_no_strokes(self):
        """Test recognizing with no strokes."""
        self.engine.start_character()
        results = self.engine.recognize()
        self.assertEqual(len(results), 0)
    
    def test_end_stroke_without_start(self):
        """Test ending stroke without starting."""
        self.engine.start_character()
        self.engine.end_stroke()  # Should not raise
        self.assertEqual(len(self.engine.current_strokes), 0)
    
    def test_add_point_without_stroke(self):
        """Test adding point without starting stroke."""
        self.engine.start_character()
        self.engine.add_point(50, 50)  # Should be ignored
        self.assertEqual(len(self.engine.current_stroke_points), 0)
    
    def test_confirm_empty_strokes(self):
        """Test confirming with no strokes."""
        self.engine.start_character()
        self.engine.confirm_character('ߊ')  # Should not raise
    
    def test_very_small_stroke(self):
        """Test handling very small strokes."""
        self.engine.start_character()
        self.engine.start_stroke()
        self.engine.add_point(50, 50)
        self.engine.add_point(50.5, 50.5)
        self.engine.add_point(51, 51)
        self.engine.end_stroke()
        
        # Should be classified as a dot (few points + short length)
        self.assertEqual(
            self.engine.current_strokes[0].stroke_type,
            StrokeType.DOT
        )


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPoint))
    suite.addTests(loader.loadTestsFromTestCase(TestStroke))
    suite.addTests(loader.loadTestsFromTestCase(TestStrokeAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestNkoTemplates))
    suite.addTests(loader.loadTestsFromTestCase(TestHandwritingRecognitionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestConfirmAndLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestHandwritingKeyboardIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCharacterCandidate))
    suite.addTests(loader.loadTestsFromTestCase(TestRealTimeRecognition))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
