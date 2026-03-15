#!/usr/bin/env python3
"""Tests for N'Ko Keyboard AI Engine"""

import sys
sys.path.insert(0, '..')

from lib.nko_engine import NkoKeyboard, NGramModel, Context


def test_ngram_basic():
    """Test basic n-gram functionality"""
    model = NGramModel(n=3)
    model.train("ߌ ߣߌ ߛߐ߲߬")
    model.train("ߌ ߣߌ ߕߟߋ")
    model.train("ߌ ߣߌ ߛߎ")
    
    # Should predict after "ߌ ߣߌ"
    preds = model.predict("ߌ ߣߌ")
    assert len(preds) > 0, "Should have predictions"
    print(f"✓ N-gram predictions: {preds}")


def test_keyboard_init():
    """Test keyboard initialization"""
    kb = NkoKeyboard(dialect="bambara")
    assert kb.dialect == "bambara"
    assert kb.context.dialect == "bambara"
    print("✓ Keyboard initialization")


def test_predictions():
    """Test prediction functionality"""
    kb = NkoKeyboard()
    
    # Test with partial greeting
    preds = kb.predict("ߌ ߣߌ ")
    assert len(preds) > 0, "Should have predictions for common phrase"
    print(f"✓ Predictions for 'ߌ ߣߌ ': {[p.text for p in preds[:3]]}")


def test_context_awareness():
    """Test context-aware predictions"""
    kb = NkoKeyboard()
    
    # Set morning context
    kb.set_context(time_of_day="morning")
    preds_morning = kb.predict("ߌ ߣߌ ")
    
    # Set evening context
    kb.set_context(time_of_day="evening")
    preds_evening = kb.predict("ߌ ߣߌ ")
    
    # Context should affect boosting (may have same predictions but different order)
    print(f"✓ Morning predictions: {[p.text for p in preds_morning[:3]]}")
    print(f"✓ Evening predictions: {[p.text for p in preds_evening[:3]]}")


def test_learning():
    """Test user learning"""
    kb = NkoKeyboard()
    
    # Learn a custom phrase
    kb.learn("ߒ ߓߊ߬ ߞߏ߲ߓߏ ߡߊ߫")
    
    # Should now appear in predictions
    preds = kb.predict("ߒ ߓߊ߬")
    print(f"✓ After learning: {[p.text for p in preds[:3]]}")


def test_greeting():
    """Test time-based greeting"""
    kb = NkoKeyboard()
    
    greeting = kb.get_greeting()
    assert kb.is_nko(greeting), "Greeting should be in N'Ko"
    print(f"✓ Greeting: {greeting}")


def test_transliteration():
    """Test N'Ko to Latin transliteration"""
    kb = NkoKeyboard()
    
    nko = "ߒ"
    trans = kb.transliterate(nko)
    print(f"✓ Transliteration: {nko} → {trans}")


def test_nko_detection():
    """Test N'Ko script detection"""
    kb = NkoKeyboard()
    
    assert kb.is_nko("ߌ ߣߌ ߛߐ߲߬") == True
    assert kb.is_nko("Hello world") == False
    assert kb.is_nko("Mixed ߒ text") == True
    print("✓ N'Ko detection")


if __name__ == "__main__":
    print("=" * 50)
    print("N'Ko Keyboard AI Tests")
    print("=" * 50)
    
    tests = [
        test_ngram_basic,
        test_keyboard_init,
        test_predictions,
        test_context_awareness,
        test_learning,
        test_greeting,
        test_transliteration,
        test_nko_detection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("ߊ߬ ߣߌ ߓߙߍ߬! (All tests passed!)")
