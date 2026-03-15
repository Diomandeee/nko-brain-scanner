#!/usr/bin/env python3
"""
Tests for N'Ko Advanced Context Engine (Gen 6.2)
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from context_engine import (
    AdvancedContextEngine,
    TopicDetector,
    GrammarAnalyzer,
    EntityRecognizer,
    EmotionalAnalyzer,
    ConversationMemoryManager
)


class TestTopicDetector(unittest.TestCase):
    """Test topic detection"""
    
    def setUp(self):
        self.detector = TopicDetector()
    
    def test_family_topic(self):
        """Detect family topic from keywords"""
        result = self.detector.detect("ߛߏ ߓߊ ߝߊ ߘߋ߲")  # Using family keywords
        self.assertEqual(result.topic, "family")
        self.assertGreater(result.confidence, 0.0)
    
    def test_religion_topic(self):
        """Detect religion topic"""
        result = self.detector.detect("ߊߟߊ߫ ߦߋ ߓߊ߬ߙߊ߬ߞߊ")  # May God bless
        self.assertEqual(result.topic, "religion")
    
    def test_market_topic(self):
        """Detect market/trade topic"""
        result = self.detector.detect("ߛߊ߲ ߟߐ߯ߕߊ ߝߍ߬ߙߍ ߖߊ")  # Market keywords
        self.assertEqual(result.topic, "market")
    
    def test_casual_fallback(self):
        """Fall back to casual topic when no strong signals"""
        result = self.detector.detect("ߏ ߓߍ ߘߌ")
        self.assertEqual(result.topic, "casual")
    
    def test_topic_with_history(self):
        """Use conversation history for topic detection"""
        history = ["ߛߏ ߓߊ ߝߊ", "ߘߋ߲ ߡߏ߬ߛߏ"]
        result = self.detector.detect("ߊ߬ ߓߍ", history)
        # Should recognize family context from history
        self.assertEqual(result.topic, "family")


class TestGrammarAnalyzer(unittest.TestCase):
    """Test grammar analysis"""
    
    def setUp(self):
        self.analyzer = GrammarAnalyzer()
    
    def test_subject_detection(self):
        """After subject pronoun, expect TAM or verb"""
        result = self.analyzer.analyze("ߒ")
        self.assertEqual(result.expected_next, "tam_or_verb")
    
    def test_tam_detection(self):
        """After TAM marker, expect verb"""
        result = self.analyzer.analyze("ߒ ߓߍ߬")
        self.assertEqual(result.expected_next, "verb")
        self.assertEqual(result.verb_state, "present")
    
    def test_past_tense_detection(self):
        """Detect past tense TAM"""
        result = self.analyzer.analyze("ߊ߬ ߞߊ߬")
        self.assertEqual(result.verb_state, "past")
    
    def test_question_detection(self):
        """Detect question structure"""
        result = self.analyzer.analyze("ߡߎ߲ ߓߍ ߦߋ")
        self.assertEqual(result.structure, "question")
    
    def test_formality_informal(self):
        """Detect informal address"""
        result = self.analyzer.analyze("ߌ ߓߍ ߘߌ")
        self.assertEqual(result.formality, "informal")


class TestEntityRecognizer(unittest.TestCase):
    """Test named entity recognition"""
    
    def setUp(self):
        self.recognizer = EntityRecognizer()
    
    def test_person_entity(self):
        """Recognize person names"""
        result = self.recognizer.recognize("ߡߎ߬ߛߊ ߓߍ ߦߋ")
        self.assertIn("person", result.entities)
        self.assertIn("ߡߎ߬ߛߊ", result.entities["person"])
    
    def test_place_entity(self):
        """Recognize place names"""
        result = self.recognizer.recognize("ߒ ߓߍ ߓߡߊ߬ߞߐ")
        self.assertIn("place", result.entities)
        self.assertIn("ߓߡߊ߬ߞߐ", result.entities["place"])
    
    def test_time_entity(self):
        """Recognize time references"""
        result = self.recognizer.recognize("ߛߐ߲߬ߜߐ ߒ ߕߊ߯")
        self.assertIn("time", result.entities)
    
    def test_greeting_entity(self):
        """Recognize greeting patterns"""
        result = self.recognizer.recognize("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        self.assertIn("greeting", result.entities)


class TestEmotionalAnalyzer(unittest.TestCase):
    """Test emotional context analysis"""
    
    def setUp(self):
        self.analyzer = EmotionalAnalyzer()
    
    def test_positive_sentiment(self):
        """Detect positive sentiment"""
        result = self.analyzer.analyze("ߊ߬ ߘߌ ߓߍ ߘߌ")  # It is good
        self.assertEqual(result.sentiment, "positive")
    
    def test_negative_sentiment(self):
        """Detect negative sentiment"""
        result = self.analyzer.analyze("ߓߊ߬ߣߊ ߓߍ")  # There is sickness
        self.assertEqual(result.sentiment, "negative")
    
    def test_ceremonial_tone(self):
        """Detect ceremonial/religious tone"""
        # Ceremonial markers can also be detected as respectful
        result = self.analyzer.analyze("ߊߟߊ߫ ߓߊ߬ߙߊ߬ߞߊ")
        self.assertIn(result.tone, ["ceremonial", "respectful"])
    
    def test_neutral_sentiment(self):
        """Neutral when no strong signals"""
        result = self.analyzer.analyze("ߏ ߓߍ")
        self.assertEqual(result.sentiment, "neutral")


class TestConversationMemory(unittest.TestCase):
    """Test cross-sentence memory"""
    
    def setUp(self):
        self.memory = ConversationMemoryManager()
    
    def test_add_turn(self):
        """Add conversation turn"""
        self.memory.add_turn("ߌ ߣߌ ߛߐ߲߬ߜߐ", "casual", {})
        self.assertEqual(self.memory.current_session.turn_count, 1)
        self.assertEqual(len(self.memory.current_session.previous_sentences), 1)
    
    def test_entity_tracking(self):
        """Track entity mentions across turns"""
        self.memory.add_turn("ߡߎ߬ߛߊ ߓߍ", "casual", {"person": ["ߡߎ߬ߛߊ"]})
        self.memory.add_turn("ߡߎ߬ߛߊ ߕߊ߯", "casual", {"person": ["ߡߎ߬ߛߊ"]})
        
        self.assertEqual(self.memory.current_session.entity_mentions["ߡߎ߬ߛߊ"], 2)
        
        boost = self.memory.get_context_boost()
        self.assertIn("ߡߎ߬ߛߊ", boost)
    
    def test_topic_history(self):
        """Track topic history"""
        self.memory.add_turn("text1", "family", {})
        self.memory.add_turn("text2", "religion", {})
        
        self.assertEqual(self.memory.current_session.topic_history, ["family", "religion"])
    
    def test_reset_session(self):
        """Reset clears memory"""
        self.memory.add_turn("text", "topic", {})
        self.memory.reset_session()
        
        self.assertEqual(self.memory.current_session.turn_count, 0)
        self.assertEqual(len(self.memory.current_session.previous_sentences), 0)


class TestAdvancedContextEngine(unittest.TestCase):
    """Test full context engine"""
    
    def setUp(self):
        self.engine = AdvancedContextEngine()
    
    def test_full_analysis(self):
        """Full context analysis returns all components"""
        context = self.engine.analyze("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        
        self.assertIsNotNone(context.topic)
        self.assertIsNotNone(context.grammar)
        self.assertIsNotNone(context.entities)
        self.assertIsNotNone(context.emotion)
        self.assertIsNotNone(context.memory)
    
    def test_contextual_boosts(self):
        """Get contextual boost words"""
        boosts = self.engine.get_contextual_boosts("ߒ ߓߊ ߓߍ")
        
        self.assertIsInstance(boosts, dict)
        # Should have some boosts from family topic
        self.assertTrue(len(boosts) > 0)
    
    def test_expected_pattern(self):
        """Get grammar-based expected pattern"""
        pattern = self.engine.get_expected_pattern("ߒ ߓߍ߬")
        self.assertIsNotNone(pattern)
        # Pattern description contains "action" for verb
        self.assertIn("action", pattern.lower())
    
    def test_conversation_continuity(self):
        """Context builds across multiple analyses"""
        self.engine.analyze("ߡߎ߬ߛߊ ߓߍ")
        self.engine.analyze("ߡߎ߬ߛߊ ߕߊ߯")
        
        stats = self.engine.get_stats()
        self.assertEqual(stats["turn_count"], 2)
    
    def test_reset(self):
        """Reset clears all context"""
        self.engine.analyze("text")
        self.engine.reset()
        
        stats = self.engine.get_stats()
        self.assertEqual(stats["turn_count"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests with unified keyboard"""
    
    def test_keyboard_with_context(self):
        """Keyboard uses context engine"""
        from unified_keyboard import UnifiedKeyboard
        
        kb = UnifiedKeyboard(enable_neural=False)
        
        # First prediction builds context (may or may not return predictions)
        preds = kb.predict("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        
        # Context info should be available regardless
        ctx_info = kb.get_context_info()
        self.assertIn("topic", ctx_info)
        self.assertIn("grammar_structure", ctx_info)
    
    def test_commit_learns_patterns(self):
        """Commit teaches context engine"""
        import tempfile
        from pathlib import Path
        from unified_keyboard import UnifiedKeyboard
        
        kb = UnifiedKeyboard(enable_neural=False)
        
        # Use a fresh temp database to avoid constraint issues
        kb.context_engine.memory_manager.db_path = Path(tempfile.mktemp(suffix=".db"))
        kb.context_engine.memory_manager._init_db()
        
        kb.predict("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        kb.commit("ߌ ߣߌ ߛߐ߲߬ߜߐ")
        kb.predict("ߒ ߓߍ")
        kb.commit("ߒ ߓߍ ߖߊ߲")
        
        # Context should reflect learned patterns
        ctx_info = kb.get_context_info()
        self.assertGreater(ctx_info["turn_count"], 0)


if __name__ == "__main__":
    print("🧪 N'Ko Context Engine Tests (Gen 6.2)\n")
    unittest.main(verbosity=2)
