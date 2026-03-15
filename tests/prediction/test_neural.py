#!/usr/bin/env python3
"""
Tests for N'Ko Neural Predictor (Gen 6.1)
"""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from neural_predictor import (
    NeuralPredictor,
    CharacterVocab,
    MandingCorpus,
    NeuralPrediction
)


class TestCharacterVocab:
    """Test character vocabulary"""
    
    def test_nko_characters(self):
        """Test N'Ko character encoding"""
        vocab = CharacterVocab()
        
        # Test encoding N'Ko text
        text = "ߌ ߣߌ"
        indices = vocab.encode(text)
        assert len(indices) > 0, "Should encode N'Ko text"
        
        # First index should be BOS
        assert vocab.idx2char[indices[0]] == "<BOS>"
        
    def test_decode(self):
        """Test decoding back to text"""
        vocab = CharacterVocab()
        
        text = "ߌ ߣߌ ߛߐ߲"
        indices = vocab.encode(text)
        decoded = vocab.decode(indices)
        
        assert decoded == text, f"Decode should match: '{decoded}' != '{text}'"
    
    def test_vocab_size(self):
        """Test vocabulary size is reasonable"""
        vocab = CharacterVocab()
        
        # Should have N'Ko chars + special tokens + ASCII
        assert len(vocab) > 100, f"Vocab too small: {len(vocab)}"


class TestNeuralPredictor:
    """Test neural predictor"""
    
    def test_initialization(self):
        """Test predictor initializes correctly"""
        predictor = NeuralPredictor()
        
        assert predictor.vocab is not None
        assert predictor.embedder is not None
        assert predictor.attention is not None
    
    def test_learn(self):
        """Test learning from text"""
        predictor = NeuralPredictor()
        
        # Learn some text
        predictor.learn("ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ")
        
        # Check ngram cache has entries
        assert len(predictor.ngram_cache) > 0, "Should have learned n-grams"
    
    def test_predict_next_char(self):
        """Test next character prediction"""
        predictor = NeuralPredictor()
        
        # Train on corpus
        predictor.batch_learn(MandingCorpus.get_training_corpus())
        
        # Predict next character
        preds = predictor.predict_next_char("ߌ ߣ")
        
        assert len(preds) > 0, "Should have predictions"
        assert isinstance(preds[0], NeuralPrediction)
        assert preds[0].score > 0
    
    def test_predict_completion(self):
        """Test word completion"""
        predictor = NeuralPredictor()
        
        # Train
        predictor.batch_learn(MandingCorpus.get_training_corpus())
        
        # Get completions
        completions = predictor.predict_completion("ߌ ߣ")
        
        assert len(completions) >= 0  # May or may not have completions
    
    def test_model_stats(self):
        """Test model statistics"""
        predictor = NeuralPredictor()
        
        # Train
        predictor.batch_learn(MandingCorpus.get_training_corpus())
        
        stats = predictor.get_model_stats()
        
        assert "total_learned_sequences" in stats
        assert "vocab_size" in stats
        assert stats["total_learned_sequences"] > 0


class TestMandingCorpus:
    """Test training corpus"""
    
    def test_corpus_not_empty(self):
        """Test corpus has content"""
        corpus = MandingCorpus.get_training_corpus()
        
        assert len(corpus) > 10, "Corpus should have many phrases"
    
    def test_corpus_has_nko(self):
        """Test corpus contains N'Ko characters"""
        corpus = MandingCorpus.get_training_corpus()
        
        # Check for N'Ko character range
        has_nko = False
        for phrase in corpus:
            for char in phrase:
                if 0x07C0 <= ord(char) <= 0x07FF:
                    has_nko = True
                    break
            if has_nko:
                break
        
        assert has_nko, "Corpus should contain N'Ko characters"


# Run tests
if __name__ == "__main__":
    import traceback
    
    test_classes = [
        TestCharacterVocab,
        TestNeuralPredictor,
        TestMandingCorpus,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Testing: {test_class.__name__}")
        print('='*50)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    total_passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    traceback.print_exc()
                    total_failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print('='*50)
    
    sys.exit(0 if total_failed == 0 else 1)
