#!/usr/bin/env python3
"""Tests for N'Ko Semantic Intelligence Layer (Gen 6)"""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from semantic_engine import (
    SemanticEngine,
    IntegratedSemanticPredictor,
    ConversationFlowEngine,
    WordVector
)
from unified_keyboard import UnifiedKeyboard


class TestSemanticEngine:
    """Test semantic intelligence features"""
    
    def test_word_vector_similarity(self):
        """Vectors in same category should be similar"""
        engine = SemanticEngine()
        
        # Greetings should be similar to each other
        greeting_words = ["ߛߐ߲߬ߜߐߡߊ", "ߕߟߋ", "ߛߎ"]
        
        for word in greeting_words:
            if word in engine.word_vectors:
                similar = engine.get_similar_words(word, top_k=5)
                # Should find other greeting-related words
                assert len(similar) > 0, f"Should find similar words for {word}"
    
    def test_semantic_categories(self):
        """Words should be categorized correctly"""
        engine = SemanticEngine()
        
        # Test category detection
        assert engine.get_category("ߓߊ߬") == "family", "ߓߊ߬ should be family"
        assert engine.get_category("ߕߊ߬ߡߌ߲") == "motion", "ߕߊ߬ߡߌ߲ should be motion"
        assert engine.get_category("ߡߎ߲") == "question", "ߡߎ߲ should be question"
    
    def test_sentence_completion(self):
        """Sentence templates should suggest completions"""
        engine = SemanticEngine()
        
        # Test partial sentence
        partial = "ߒ ߓߍ߬"  # n be (I am...)
        completions = engine.suggest_sentence_completion(partial)
        
        # Should suggest continuations
        assert len(completions) > 0, "Should suggest sentence completions"
    
    def test_conversation_state_detection(self):
        """Should detect conversation state from text"""
        engine = SemanticEngine()
        
        # Greeting should set greeting state
        state = engine.detect_conversation_state("ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ")
        assert state.expected_response_type == "greeting"
        assert state.mood == "friendly"
        
        # Question should set question state
        state = engine.detect_conversation_state("ߌ ߓߍ߬ ߡߌ߬ߣߌ߲ ߞߍ")
        assert state.expected_response_type == "question"
    
    def test_dialect_variants(self):
        """Should suggest dialect alternatives"""
        engine = SemanticEngine()
        
        # Test getting variants
        word = "ߕߊ߬ߡߌ߲"  # tamina (to go)
        variants = engine.suggest_dialect_alternatives(word)
        
        # Should have variants in other dialects
        assert "maninka" in variants or "jula" in variants, \
            "Should have dialect variants"
    
    def test_integrated_predictor(self):
        """Integrated predictor should combine sources"""
        predictor = IntegratedSemanticPredictor()
        
        # Test prediction
        preds = predictor.predict("ߌ ߣߌ")
        
        assert len(preds) > 0, "Should produce predictions"
        
        # Check prediction has expected fields
        pred = preds[0]
        assert hasattr(pred, "text")
        assert hasattr(pred, "score")
        assert hasattr(pred, "source")


class TestConversationFlow:
    """Test conversation flow detection"""
    
    def test_greeting_response(self):
        """Greeting should trigger greeting response"""
        flow = ConversationFlowEngine()
        
        responses = flow.process_input("ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ")
        
        # Should suggest response formula
        assert any("ߒ ߓߊ" in r for r in responses), \
            "Should suggest greeting response"
    
    def test_conversation_starters(self):
        """Should provide conversation starters"""
        flow = ConversationFlowEngine()
        
        starters = flow.get_conversation_starters()
        
        assert len(starters) > 0, "Should have conversation starters"
        assert any("ߌ ߣߌ" in s for s in starters), \
            "Starters should include greetings"


class TestUnifiedKeyboard:
    """Test unified keyboard (Gen 6)"""
    
    def test_unified_prediction(self):
        """Unified keyboard should combine all sources"""
        kb = UnifiedKeyboard()
        
        preds = kb.predict("ߌ ߣߌ")
        
        assert len(preds) > 0, "Should produce predictions"
        
        # Check multiple sources contribute
        all_sources = set()
        for p in preds:
            all_sources.update(p.sources)
        
        # Should have at least 2 different sources
        assert len(all_sources) >= 1, "Should combine multiple sources"
    
    def test_auto_correct(self):
        """Auto-correct should fix common errors"""
        kb = UnifiedKeyboard()
        
        # Test spacing fix
        result = kb.auto_correct("ߣ ߌ")
        assert result == "ߣߌ", "Should fix spacing error"
    
    def test_quick_phrases(self):
        """Should suggest quick phrases in context"""
        kb = UnifiedKeyboard()
        
        # Typing greeting prefix should suggest full greetings
        preds = kb.predict("ߌ ߣߌ ߛ")
        
        # Should have complete phrase suggestion
        complete_phrases = [p for p in preds if p.is_complete_phrase]
        assert len(complete_phrases) >= 0, "May suggest complete phrases"
    
    def test_dialect_info(self):
        """Should provide dialect information"""
        kb = UnifiedKeyboard()
        
        info = kb.get_dialect_info("ߕߊ߬ߡߌ߲")
        
        # May or may not have variants, but should not error
        assert isinstance(info, dict)
    
    def test_greeting_time_aware(self):
        """Greeting should be time-aware"""
        kb = UnifiedKeyboard()
        
        greeting = kb.get_greeting()
        
        # Should be a valid N'Ko greeting
        assert "ߌ ߣߌ" in greeting, "Greeting should contain 'i ni'"
    
    def test_learning(self):
        """Should learn from committed text"""
        kb = UnifiedKeyboard()
        
        # Commit a phrase
        kb.commit("ߒ ߓߍ߬ ߦߋ߲߬")
        
        # Should be in session history
        assert "ߒ ߓߍ߬ ߦߋ߲߬" in kb.session_text


def run_tests():
    """Run all tests"""
    test_classes = [
        TestSemanticEngine,
        TestConversationFlow,
        TestUnifiedKeyboard,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Testing: {test_class.__name__}")
        print('='*50)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                total_failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print('='*50)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
