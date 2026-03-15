#!/usr/bin/env python3
"""
Tests for N'Ko Smart Compose Engine — Generation 14

Comprehensive tests for intelligent sentence completion,
context-aware suggestions, and conversation flow.
"""

import unittest
import sys
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from smart_compose_engine import (
    SmartComposeEngine,
    SmartComposeKeyboardIntegration,
    ComposeSuggestion,
    ComposeIntent,
    ComposeContext,
    ConversationState,
)


class TestSmartComposeEngine(unittest.TestCase):
    """Tests for SmartComposeEngine core functionality."""
    
    def setUp(self):
        """Create engine with in-memory database."""
        self.engine = SmartComposeEngine(":memory:")
        
    def test_init(self):
        """Test engine initializes correctly."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.conversation_state)
        
    def test_time_greetings_exist(self):
        """Test time-based greetings are configured."""
        self.assertIn("morning", self.engine.TIME_GREETINGS)
        self.assertIn("afternoon", self.engine.TIME_GREETINGS)
        self.assertIn("evening", self.engine.TIME_GREETINGS)
        self.assertIn("night", self.engine.TIME_GREETINGS)
        
    def test_proverbs_exist(self):
        """Test proverbs are configured."""
        self.assertGreater(len(self.engine.PROVERBS), 5)
        for start, (end, english) in self.engine.PROVERBS.items():
            self.assertIsInstance(start, str)
            self.assertIsInstance(end, str)
            self.assertIsInstance(english, str)
            
    def test_blessings_exist(self):
        """Test blessings are configured."""
        self.assertGreater(len(self.engine.BLESSINGS), 5)
        for nko, english in self.engine.BLESSINGS:
            self.assertIsInstance(nko, str)
            self.assertIn("ߊߟߊ߫", nko)  # Should contain "Allah"


class TestIntentDetection(unittest.TestCase):
    """Tests for intent detection."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_greeting_intent(self):
        """Test greeting intent detection."""
        intent, conf = self.engine._detect_intent("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        self.assertEqual(intent, ComposeIntent.GREETING)
        self.assertGreater(conf, 0.8)
        
    def test_blessing_intent(self):
        """Test blessing intent detection."""
        intent, conf = self.engine._detect_intent("ߊߟߊ߫ ߦߊ߫ ߌ")
        self.assertEqual(intent, ComposeIntent.BLESSING)
        self.assertGreater(conf, 0.8)
        
    def test_welfare_intent(self):
        """Test welfare inquiry detection."""
        intent, conf = self.engine._detect_intent("ߌ ߖߐ߫ ߓߊ߬")
        self.assertEqual(intent, ComposeIntent.WELFARE_INQUIRY)
        self.assertGreater(conf, 0.8)
        
    def test_gratitude_intent(self):
        """Test gratitude detection."""
        intent, conf = self.engine._detect_intent("ߌ ߣߌ ߓߊ߬ߙߊ")
        self.assertEqual(intent, ComposeIntent.GRATITUDE)
        self.assertGreater(conf, 0.8)
        
    def test_farewell_intent(self):
        """Test farewell detection."""
        intent, conf = self.engine._detect_intent("ߞ ߕ ߏ")
        self.assertEqual(intent, ComposeIntent.FAREWELL)
        self.assertGreater(conf, 0.8)
        
    def test_question_intent(self):
        """Test question detection."""
        intent, conf = self.engine._detect_intent("ߡߎ߲ ߠߋ")
        self.assertEqual(intent, ComposeIntent.QUESTION)
        self.assertGreater(conf, 0.7)
        
    def test_proverb_intent(self):
        """Test proverb detection."""
        intent, conf = self.engine._detect_intent("ߡߐ߱ ߕߍ߫ ߓߐ߫")
        self.assertEqual(intent, ComposeIntent.PROVERB)
        self.assertGreater(conf, 0.8)
        
    def test_default_statement_intent(self):
        """Test unknown text defaults to statement."""
        intent, conf = self.engine._detect_intent("ߞߏ ߡߍ߲")
        self.assertEqual(intent, ComposeIntent.STATEMENT)


class TestSuggestions(unittest.TestCase):
    """Tests for suggestion generation."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_empty_input_suggestions(self):
        """Test suggestions for empty input."""
        suggestions = self.engine.get_suggestions("")
        self.assertGreater(len(suggestions), 0)
        # Should be time-based greetings
        for s in suggestions:
            self.assertEqual(s.intent, ComposeIntent.GREETING)
            
    def test_greeting_suggestions(self):
        """Test greeting-based suggestions."""
        suggestions = self.engine.get_suggestions("ߌ ߣߌ")
        self.assertGreater(len(suggestions), 0)
        
    def test_blessing_suggestions(self):
        """Test blessing suggestions."""
        suggestions = self.engine.get_suggestions("ߊߟߊ߫ ߦߊ߫")
        self.assertGreater(len(suggestions), 0)
        for s in suggestions:
            self.assertEqual(s.intent, ComposeIntent.BLESSING)
            
    def test_proverb_completion(self):
        """Test proverb completion suggestions."""
        suggestions = self.engine.get_suggestions("ߡߐ߱ ߕߍ߫ ߓߐ߫")
        
        # Should find proverb completion
        proverb_suggestions = [s for s in suggestions if s.intent == ComposeIntent.PROVERB]
        self.assertGreater(len(proverb_suggestions), 0)
        
    def test_suggestion_has_english(self):
        """Test suggestions include English translations."""
        suggestions = self.engine.get_suggestions("")
        for s in suggestions:
            # Most built-in suggestions should have English
            if s.source != "learned":
                self.assertTrue(len(s.english) > 0, f"Missing English for: {s.text}")
                
    def test_suggestion_confidence_range(self):
        """Test suggestion confidence is within valid range."""
        suggestions = self.engine.get_suggestions("")
        for s in suggestions:
            self.assertGreaterEqual(s.confidence, 0.0)
            self.assertLessEqual(s.confidence, 1.0)
            
    def test_limit_respected(self):
        """Test suggestion limit is respected."""
        suggestions = self.engine.get_suggestions("", limit=2)
        self.assertLessEqual(len(suggestions), 2)
        
    def test_min_confidence_filter(self):
        """Test minimum confidence filter."""
        suggestions = self.engine.get_suggestions("", min_confidence=0.8)
        for s in suggestions:
            self.assertGreaterEqual(s.confidence, 0.8)


class TestResponseSuggestions(unittest.TestCase):
    """Tests for response suggestion generation."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_greeting_response(self):
        """Test response suggestions for greeting."""
        suggestions = self.engine.get_response_suggestions("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        self.assertGreater(len(suggestions), 0)
        
    def test_welfare_inquiry_response(self):
        """Test response to welfare inquiry."""
        suggestions = self.engine.get_response_suggestions("ߌ ߖߐ߫ ߓߊ߬")
        self.assertGreater(len(suggestions), 0)
        
    def test_blessing_response(self):
        """Test response to blessing."""
        suggestions = self.engine.get_response_suggestions("ߊߟߊ߫ ߦߊ߫ ߌ ߘߍ߬ߡߍ߲")
        self.assertGreater(len(suggestions), 0)
        # Should suggest "Amen" type responses
        
    def test_question_response(self):
        """Test response to question."""
        suggestions = self.engine.get_response_suggestions("ߡߎ߲ ߠߋ ߓߊ߬")
        self.assertGreater(len(suggestions), 0)
        
    def test_updates_conversation_state(self):
        """Test response suggestions update conversation state."""
        self.engine.get_response_suggestions("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        self.assertGreater(len(self.engine.conversation_state.messages), 0)


class TestSentenceCompletion(unittest.TestCase):
    """Tests for sentence completion."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_greeting_completion(self):
        """Test completing greeting patterns."""
        completions = self.engine.complete_sentence("ߌ ߣߌ")
        self.assertGreater(len(completions), 0)
        
    def test_blessing_completion(self):
        """Test completing blessing patterns."""
        completions = self.engine.complete_sentence("ߊߟߊ߫ ߦߊ߫")
        self.assertGreater(len(completions), 0)
        
    def test_proverb_completion(self):
        """Test completing proverbs."""
        completions = self.engine.complete_sentence("ߡߐ߱ ߕߍ߫")
        proverb_completions = [c for c in completions if c.intent == ComposeIntent.PROVERB]
        self.assertGreater(len(proverb_completions), 0)
        
    def test_i_statement_completion(self):
        """Test completing 'I' statements."""
        completions = self.engine.complete_sentence("ߒ")
        self.assertGreater(len(completions), 0)


class TestContextualSuggestions(unittest.TestCase):
    """Tests for context-specific suggestions."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_celebration_context(self):
        """Test celebration context suggestions."""
        suggestions = self.engine.get_suggestions("", context=ComposeContext.CELEBRATION)
        congratulation_found = any(s.intent == ComposeIntent.CONGRATULATION for s in suggestions)
        self.assertTrue(congratulation_found)
        
    def test_mourning_context(self):
        """Test mourning context suggestions."""
        suggestions = self.engine.get_suggestions("", context=ComposeContext.MOURNING)
        condolence_found = any(s.intent == ComposeIntent.CONDOLENCE for s in suggestions)
        self.assertTrue(condolence_found)
        
    def test_formal_letter_context(self):
        """Test formal letter context."""
        suggestions = self.engine.get_suggestions("", context=ComposeContext.LETTER_FORMAL)
        letter_found = any(s.intent == ComposeIntent.LETTER_OPENING for s in suggestions)
        self.assertTrue(letter_found)


class TestLearning(unittest.TestCase):
    """Tests for pattern learning."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_learn_pattern(self):
        """Test learning new pattern."""
        self.engine.learn_pattern("ߒ ߧߴ", "ߌ ߝߏ߫ ߟߊ߫", ComposeIntent.GREETING)
        suggestions = self.engine._get_learned_suggestions("ߒ ߧߴ")
        self.assertGreater(len(suggestions), 0)
        
    def test_pattern_use_count_increases(self):
        """Test pattern use count increases with repeated learning."""
        self.engine.learn_pattern("ߌ ߣ", "ߌ ߛߐ߲߬ߜߐ߫", ComposeIntent.GREETING)
        self.engine.learn_pattern("ߌ ߣ", "ߌ ߛߐ߲߬ߜߐ߫", ComposeIntent.GREETING)
        
        # Get statistics
        stats = self.engine.get_statistics()
        self.assertGreaterEqual(stats["learned_patterns"], 1)
        
    def test_record_feedback(self):
        """Test recording suggestion feedback."""
        self.engine.record_feedback("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ", accepted=True)
        self.engine.record_feedback("ߌ ߣߌ ߕߋ߬ߟߋ", accepted=False)
        
        stats = self.engine.get_statistics()
        self.assertEqual(stats["feedback_count"], 2)


class TestConversationState(unittest.TestCase):
    """Tests for conversation state management."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_initial_state(self):
        """Test initial conversation state."""
        state = self.engine.conversation_state
        self.assertEqual(len(state.messages), 0)
        self.assertEqual(state.current_phase, "opening")
        
    def test_reset_conversation(self):
        """Test conversation reset."""
        # Add some messages
        self.engine.get_response_suggestions("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        self.assertGreater(len(self.engine.conversation_state.messages), 0)
        
        # Reset
        self.engine.reset_conversation()
        self.assertEqual(len(self.engine.conversation_state.messages), 0)
        
    def test_welfare_phase_tracking(self):
        """Test welfare inquiry phase tracking."""
        # Initial phase
        phase = self.engine._get_welfare_phase()
        self.assertEqual(phase, 0)
        
        # After welfare message
        self.engine.get_response_suggestions("ߌ ߖߐ߫ ߓߊ߬")
        phase = self.engine._get_welfare_phase()
        # Phase should increment based on welfare questions in history


class TestTimeOfDay(unittest.TestCase):
    """Tests for time-of-day functionality."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    @patch('smart_compose_engine.datetime')
    def test_morning_detection(self, mock_datetime):
        """Test morning time detection."""
        mock_datetime.now.return_value = MagicMock(hour=9)
        time = self.engine._get_time_of_day()
        self.assertEqual(time, "morning")
        
    @patch('smart_compose_engine.datetime')  
    def test_afternoon_detection(self, mock_datetime):
        """Test afternoon time detection."""
        mock_datetime.now.return_value = MagicMock(hour=14)
        time = self.engine._get_time_of_day()
        self.assertEqual(time, "afternoon")
        
    @patch('smart_compose_engine.datetime')
    def test_evening_detection(self, mock_datetime):
        """Test evening time detection."""
        mock_datetime.now.return_value = MagicMock(hour=19)
        time = self.engine._get_time_of_day()
        self.assertEqual(time, "evening")
        
    @patch('smart_compose_engine.datetime')
    def test_night_detection(self, mock_datetime):
        """Test night time detection."""
        mock_datetime.now.return_value = MagicMock(hour=23)
        time = self.engine._get_time_of_day()
        self.assertEqual(time, "night")


class TestKeyboardIntegration(unittest.TestCase):
    """Tests for keyboard integration layer."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        self.integration = SmartComposeKeyboardIntegration(self.engine)
        
    def test_on_text_changed(self):
        """Test text change handler."""
        suggestions = self.integration.on_text_changed("ߌ ߣߌ")
        self.assertIsInstance(suggestions, list)
        for s in suggestions:
            self.assertIn("text", s)
            self.assertIn("completion", s)
            
    def test_on_suggestion_selected(self):
        """Test suggestion selection handler."""
        self.integration.on_text_changed("ߌ ߣ")
        # Should not raise
        self.integration.on_suggestion_selected("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        
    def test_on_suggestion_dismissed(self):
        """Test suggestion dismissal handler."""
        # Should not raise
        self.integration.on_suggestion_dismissed("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        
    def test_on_message_received(self):
        """Test message received handler."""
        responses = self.integration.on_message_received("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        self.assertIsInstance(responses, list)
        
    def test_set_context(self):
        """Test context setting."""
        self.integration.set_context(ComposeContext.CELEBRATION)
        self.assertEqual(self.integration.current_context, ComposeContext.CELEBRATION)
        
    def test_clear_context(self):
        """Test context clearing."""
        self.integration.set_context(ComposeContext.CELEBRATION)
        self.integration.clear_context()
        self.assertIsNone(self.integration.current_context)
        
    def test_new_conversation(self):
        """Test new conversation initialization."""
        self.integration.on_message_received("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        self.integration.new_conversation()
        self.assertEqual(self.integration.current_context, ComposeContext.CONVERSATION)


class TestTextSimilarity(unittest.TestCase):
    """Tests for text similarity calculation."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_identical_texts(self):
        """Test similarity of identical texts."""
        sim = self.engine._text_similarity("ߌ ߣߌ", "ߌ ߣߌ")
        self.assertEqual(sim, 1.0)
        
    def test_empty_text(self):
        """Test similarity with empty text."""
        sim = self.engine._text_similarity("", "ߌ ߣߌ")
        self.assertEqual(sim, 0.0)
        
    def test_different_texts(self):
        """Test similarity of different texts."""
        # Use texts that share some characters
        sim = self.engine._text_similarity("ߌ ߣߌ ߛߐ߲߬ߜߐ", "ߌ ߣߌ ߕߋ߬ߟߋ")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)


class TestStatistics(unittest.TestCase):
    """Tests for statistics reporting."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_initial_statistics(self):
        """Test initial statistics are empty."""
        stats = self.engine.get_statistics()
        self.assertEqual(stats["learned_patterns"], 0)
        self.assertEqual(stats["feedback_count"], 0)
        
    def test_statistics_after_learning(self):
        """Test statistics update after learning."""
        self.engine.learn_pattern("ߌ", "ߣߌ", ComposeIntent.GREETING)
        self.engine.record_feedback("ߌ ߣߌ", accepted=True)
        
        stats = self.engine.get_statistics()
        self.assertEqual(stats["learned_patterns"], 1)
        self.assertEqual(stats["feedback_count"], 1)


class TestLetterTemplates(unittest.TestCase):
    """Tests for letter template suggestions."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_formal_letter_templates_exist(self):
        """Test formal letter templates are configured."""
        self.assertIn("formal_opening", self.engine.LETTER_TEMPLATES)
        self.assertIn("formal_closing", self.engine.LETTER_TEMPLATES)
        
    def test_informal_letter_templates_exist(self):
        """Test informal letter templates are configured."""
        self.assertIn("informal_opening", self.engine.LETTER_TEMPLATES)
        self.assertIn("informal_closing", self.engine.LETTER_TEMPLATES)
        
    def test_template_has_slots(self):
        """Test templates have slot markers."""
        for templates in self.engine.LETTER_TEMPLATES.values():
            for nko, english in templates:
                if "{" in nko:
                    self.assertIn("}", nko)


class TestWelfareChains(unittest.TestCase):
    """Tests for welfare inquiry chain suggestions."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_welfare_chains_exist(self):
        """Test welfare chains are configured."""
        self.assertGreater(len(self.engine.WELFARE_CHAINS), 5)
        
    def test_welfare_chains_cover_family(self):
        """Test welfare chains cover family members."""
        all_text = " ".join([nko for nko, _ in self.engine.WELFARE_CHAINS])
        # Should mention family members
        self.assertTrue(
            any(word in all_text for word in ["ߘߋ߲", "ߡߎ߬ߛߏ", "ߗߍ", "ߓߊ", "ߝߊ"])
        )


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""
    
    def setUp(self):
        self.engine = SmartComposeEngine(":memory:")
        
    def test_very_long_input(self):
        """Test handling of very long input."""
        long_input = "ߌ ߣߌ " * 100
        suggestions = self.engine.get_suggestions(long_input)
        # Should not raise, may return empty
        self.assertIsInstance(suggestions, list)
        
    def test_special_characters(self):
        """Test handling of special characters."""
        special_input = "ߌ ߣߌ 123!@#"
        suggestions = self.engine.get_suggestions(special_input)
        self.assertIsInstance(suggestions, list)
        
    def test_mixed_scripts(self):
        """Test handling of mixed N'Ko and Latin."""
        mixed_input = "ߌ ߣߌ hello"
        suggestions = self.engine.get_suggestions(mixed_input)
        self.assertIsInstance(suggestions, list)
        
    def test_whitespace_only(self):
        """Test handling of whitespace-only input."""
        suggestions = self.engine.get_suggestions("   ")
        self.assertIsInstance(suggestions, list)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
