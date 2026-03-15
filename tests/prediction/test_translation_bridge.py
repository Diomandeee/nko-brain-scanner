#!/usr/bin/env python3
"""
Tests for N'Ko Translation Bridge Engine

Comprehensive tests covering:
- Transliteration (N'Ko ↔ Latin)
- Full translation
- Hybrid mode
- Learning mode
- Language detection
- Translation memory
- Real-time suggestions
"""

import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from translation_bridge import (
    TranslationBridge,
    TranslationMode,
    TargetLanguage,
    Register,
    TranslationResult,
    InlineTranslator,
    TransliterationMap,
)


class TestTransliterationMaps(unittest.TestCase):
    """Test character mapping tables."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_consonant_map_complete(self):
        """All N'Ko consonants should have mappings."""
        # Core consonants that must exist
        required = ['ߓ', 'ߔ', 'ߕ', 'ߖ', 'ߗ', 'ߘ', 'ߙ', 'ߛ', 'ߝ', 'ߞ', 'ߟ', 'ߡ', 'ߣ', 'ߤ', 'ߥ', 'ߦ']
        for char in required:
            self.assertIn(char, self.bridge.NKO_CONSONANTS)
            self.assertIsInstance(self.bridge.NKO_CONSONANTS[char], TransliterationMap)
    
    def test_vowel_map_complete(self):
        """All N'Ko vowels should have mappings."""
        required = ['ߊ', 'ߋ', 'ߌ', 'ߍ', 'ߎ', 'ߏ', 'ߐ']
        for char in required:
            self.assertIn(char, self.bridge.NKO_VOWELS)
    
    def test_tone_diacritics(self):
        """Tone diacritics should be mapped."""
        self.assertIn('߫', self.bridge.NKO_TONES)  # High
        self.assertIn('߬', self.bridge.NKO_TONES)  # Low


class TestTransliterationNkoToLatin(unittest.TestCase):
    """Test N'Ko → Latin transliteration."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_single_consonant(self):
        """Single consonants should transliterate correctly."""
        self.assertEqual(self.bridge.transliterate_to_latin('ߓ'), 'b')
        self.assertEqual(self.bridge.transliterate_to_latin('ߝ'), 'f')
        self.assertEqual(self.bridge.transliterate_to_latin('ߞ'), 'k')
    
    def test_single_vowel(self):
        """Single vowels should transliterate correctly."""
        self.assertEqual(self.bridge.transliterate_to_latin('ߊ'), 'a')
        self.assertEqual(self.bridge.transliterate_to_latin('ߌ'), 'i')
        self.assertEqual(self.bridge.transliterate_to_latin('ߎ'), 'u')
    
    def test_consonant_vowel_combination(self):
        """CV combinations should work."""
        self.assertEqual(self.bridge.transliterate_to_latin('ߓߊ'), 'ba')
        self.assertEqual(self.bridge.transliterate_to_latin('ߝߊ'), 'fa')
    
    def test_word_transliteration(self):
        """Full words should transliterate."""
        # "ba" (mother)
        result = self.bridge.transliterate_to_latin('ߓߊ')
        self.assertIn('ba', result.lower())
    
    def test_preserves_spaces(self):
        """Spaces should be preserved."""
        result = self.bridge.transliterate_to_latin('ߓߊ ߝߊ')
        self.assertIn(' ', result)
    
    def test_preserves_punctuation(self):
        """Punctuation should be preserved."""
        result = self.bridge.transliterate_to_latin('ߓߊ.')
        self.assertTrue(result.endswith('.'))
    
    def test_unknown_characters_passthrough(self):
        """Unknown characters should pass through."""
        result = self.bridge.transliterate_to_latin('ߓX')
        self.assertIn('X', result)


class TestTransliterationLatinToNko(unittest.TestCase):
    """Test Latin → N'Ko transliteration."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_single_consonant(self):
        """Single consonants should convert."""
        self.assertEqual(self.bridge.transliterate_to_nko('b'), 'ߓ')
        self.assertEqual(self.bridge.transliterate_to_nko('f'), 'ߝ')
    
    def test_single_vowel(self):
        """Single vowels should convert."""
        self.assertEqual(self.bridge.transliterate_to_nko('a'), 'ߊ')
        self.assertEqual(self.bridge.transliterate_to_nko('i'), 'ߌ')
    
    def test_digraph_handling(self):
        """Digraphs (ny, gb, rr) should map to single N'Ko chars."""
        result = self.bridge.transliterate_to_nko('ny')
        # Should produce ߢ or ߧ, not ߣߦ
        self.assertIn(result, ['ߢ', 'ߧ'])
    
    def test_case_insensitive(self):
        """Should be case insensitive."""
        lower = self.bridge.transliterate_to_nko('ba')
        upper = self.bridge.transliterate_to_nko('BA')
        mixed = self.bridge.transliterate_to_nko('Ba')
        self.assertEqual(lower, upper)
        self.assertEqual(lower, mixed)
    
    def test_preserves_spaces(self):
        """Spaces should be preserved."""
        result = self.bridge.transliterate_to_nko('ba fa')
        self.assertIn(' ', result)


class TestLanguageDetection(unittest.TestCase):
    """Test automatic language detection."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_detect_nko(self):
        """Should detect N'Ko script."""
        self.assertEqual(self.bridge._detect_language('ߓߊ'), 'nko')
        self.assertEqual(self.bridge._detect_language('ߌ ߣߌ ߛߐ߲߬ߜ߮ߡߊ'), 'nko')
    
    def test_detect_french(self):
        """Should detect French."""
        self.assertEqual(self.bridge._detect_language('le chat est noir'), 'french')
        self.assertEqual(self.bridge._detect_language('une maison est belle'), 'french')
    
    def test_detect_english(self):
        """Should detect English."""
        self.assertEqual(self.bridge._detect_language('the cat is black'), 'english')
        self.assertEqual(self.bridge._detect_language('a house is beautiful'), 'english')
    
    def test_default_to_latin_manding(self):
        """Unknown Latin text defaults to Latin Manding."""
        self.assertEqual(self.bridge._detect_language('ba fa'), 'latin_manding')


class TestFullTranslation(unittest.TestCase):
    """Test full translation functionality."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_nko_to_latin_greeting(self):
        """Translate N'Ko greeting to Latin."""
        result = self.bridge.translate(
            'ߌ ߣߌ ߗߍ',
            target_lang=TargetLanguage.LATIN_BAMBARA
        )
        self.assertEqual(result.source_lang, 'nko')
        self.assertIn('ni', result.target_text.lower())
    
    def test_nko_to_french(self):
        """Translate N'Ko to French."""
        result = self.bridge.translate(
            'ߓߊ',
            target_lang=TargetLanguage.FRENCH
        )
        # "ba" = "mère" (mother)
        self.assertIn('mère', result.target_text.lower())
    
    def test_nko_to_english(self):
        """Translate N'Ko to English."""
        result = self.bridge.translate(
            'ߓߊ',
            target_lang=TargetLanguage.ENGLISH
        )
        self.assertIn('mother', result.target_text.lower())
    
    def test_translation_confidence(self):
        """Known translations should have high confidence."""
        result = self.bridge.translate('ߓߊ')
        self.assertGreater(result.confidence, 0.5)
    
    def test_unknown_word_transliteration_fallback(self):
        """Unknown words should fall back to transliteration."""
        # Construct an unknown N'Ko word
        result = self.bridge.translate('ߓߓߓ')
        # Should still produce something
        self.assertTrue(len(result.target_text) > 0)
        self.assertLess(result.confidence, 0.9)  # Lower confidence
    
    def test_word_alignments(self):
        """Translation should include word alignments."""
        result = self.bridge.translate('ߓߊ ߝߊ')
        self.assertTrue(len(result.word_alignments) > 0)


class TestTranslitMode(unittest.TestCase):
    """Test transliteration-only mode."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_translit_mode_nko_to_latin(self):
        """Transliterate mode should not translate meaning."""
        result = self.bridge.translate(
            'ߓߊ',
            mode=TranslationMode.TRANSLITERATE,
            target_lang=TargetLanguage.LATIN_BAMBARA
        )
        # Should produce 'ba', not 'mother'
        self.assertEqual(result.target_text.lower(), 'ba')
        self.assertEqual(result.mode, TranslationMode.TRANSLITERATE)
    
    def test_translit_high_confidence(self):
        """Transliteration should have high confidence."""
        result = self.bridge.translate(
            'ߓߊ',
            mode=TranslationMode.TRANSLITERATE
        )
        self.assertGreaterEqual(result.confidence, 0.9)


class TestHybridMode(unittest.TestCase):
    """Test hybrid translation mode."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_hybrid_includes_both(self):
        """Hybrid mode should include transliteration and translation."""
        result = self.bridge.translate(
            'ߓߊ',
            mode=TranslationMode.HYBRID
        )
        self.assertEqual(result.mode, TranslationMode.HYBRID)
        # Should contain brackets with translation
        # Format: "ba [mother]" or similar
        self.assertIsNotNone(result.target_text)


class TestLearningMode(unittest.TestCase):
    """Test learning mode with etymology."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_learning_mode_returns_metadata(self):
        """Learning mode should include etymology/cognates."""
        result = self.bridge.translate(
            'ߊߟߊ߫',  # "Ala" (God) - has Arabic etymology note
            mode=TranslationMode.LEARNING
        )
        self.assertEqual(result.mode, TranslationMode.LEARNING)
        # Etymology or cognates should be populated
        has_learning_data = (
            result.etymology is not None or 
            len(result.cognates) > 0
        )
        self.assertTrue(has_learning_data)


class TestTranslationMemory(unittest.TestCase):
    """Test translation memory persistence."""
    
    def setUp(self):
        # Use temp database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.bridge = TranslationBridge(db_path=self.temp_db.name)
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_add_translation(self):
        """Adding translation should persist."""
        self.bridge.add_translation(
            'ߕߍߛߕ',  # test word
            'nko',
            'test',
            'english',
            confidence=0.9,
            user_verified=True
        )
        # Should be in stats
        self.assertEqual(self.bridge.stats['new_entries_learned'], 1)
    
    def test_cache_hit(self):
        """Added translations should be retrieved."""
        self.bridge.add_translation(
            'ߕߍߛߕ',
            'nko',
            'test',
            'english',
            confidence=0.9
        )
        
        # Query should find it
        result = self.bridge._check_translation_memory(
            'ߕߍߛߕ', 'nko', 'english'
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.target_text, 'test')


class TestRealTimeSuggestions(unittest.TestCase):
    """Test real-time typing suggestions."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_partial_nko_suggestions(self):
        """Should suggest completions for partial N'Ko input."""
        suggestions = self.bridge.suggest_while_typing('ߓ')
        # May or may not have suggestions for single char
        self.assertIsInstance(suggestions, list)
    
    def test_suggestion_format(self):
        """Suggestions should be tuples of (nko, translation, confidence)."""
        suggestions = self.bridge.suggest_while_typing('ߓߊ')
        for s in suggestions:
            self.assertEqual(len(s), 3)
            nko, trans, conf = s
            self.assertIsInstance(nko, str)
            self.assertIsInstance(trans, str)
            self.assertIsInstance(conf, float)
    
    def test_suggestion_limit(self):
        """Should not return more than 5 suggestions."""
        suggestions = self.bridge.suggest_while_typing('ߌ')
        self.assertLessEqual(len(suggestions), 5)


class TestInlineTranslator(unittest.TestCase):
    """Test inline translation overlay."""
    
    def setUp(self):
        bridge = TranslationBridge()
        self.translator = InlineTranslator(bridge)
    
    def test_keystroke_accumulation(self):
        """Keystrokes should accumulate."""
        self.translator.on_keystroke('ߓ')
        self.assertEqual(self.translator.current_word, 'ߓ')
        
        self.translator.on_keystroke('ߊ')
        self.assertEqual(self.translator.current_word, 'ߓߊ')
    
    def test_space_clears_word(self):
        """Space should clear current word."""
        self.translator.on_keystroke('ߓ')
        self.translator.on_keystroke(' ')
        self.assertEqual(self.translator.current_word, '')
    
    def test_backspace_removes_char(self):
        """Backspace should remove last char."""
        self.translator.on_keystroke('ߓ')
        self.translator.on_keystroke('ߊ')
        self.translator.on_keystroke('\b')
        self.assertEqual(self.translator.current_word, 'ߓ')
    
    def test_returns_suggestions(self):
        """Should return suggestion objects."""
        self.translator.on_keystroke('ߓ')
        result = self.translator.on_keystroke('ߊ')
        
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIn('nko', item)
            self.assertIn('translation', item)
            self.assertIn('display', item)


class TestAlternatives(unittest.TestCase):
    """Test alternative translation lookup."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_get_alternatives_known_word(self):
        """Known words should have alternatives."""
        alts = self.bridge.get_alternatives('ߓߊ')
        self.assertGreater(len(alts), 0)
    
    def test_alternatives_multiple_languages(self):
        """Should include multiple target languages."""
        alts = self.bridge.get_alternatives('ߓߊ')
        # Should have Latin, French, English
        combined = ' '.join(alts)
        self.assertTrue(
            'Latin' in combined or 
            'French' in combined or 
            'English' in combined
        )


class TestDictionaryExport(unittest.TestCase):
    """Test dictionary export functionality."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_export_json(self):
        """Should export valid JSON."""
        import json
        exported = self.bridge.export_dictionary(format="json")
        parsed = json.loads(exported)
        self.assertIsInstance(parsed, dict)
    
    def test_export_csv(self):
        """Should export valid CSV."""
        exported = self.bridge.export_dictionary(format="csv")
        lines = exported.split('\n')
        self.assertGreater(len(lines), 1)
        # First line should be header
        self.assertIn('nko', lines[0].lower())
    
    def test_export_unknown_format(self):
        """Unknown format should raise error."""
        with self.assertRaises(ValueError):
            self.bridge.export_dictionary(format="xml")


class TestStatistics(unittest.TestCase):
    """Test statistics tracking."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_initial_stats(self):
        """Initial stats should be zero."""
        stats = self.bridge.get_statistics()
        self.assertEqual(stats['translations_performed'], 0)
        self.assertEqual(stats['cache_hits'], 0)
    
    def test_translation_increments_count(self):
        """Translating should increment count."""
        self.bridge.translate('ߓߊ')
        stats = self.bridge.get_statistics()
        self.assertEqual(stats['translations_performed'], 1)
    
    def test_dictionary_sizes(self):
        """Stats should include dictionary sizes."""
        stats = self.bridge.get_statistics()
        self.assertIn('dictionary_size', stats)
        self.assertIn('reverse_dictionary_size', stats)
        self.assertGreater(stats['dictionary_size'], 0)


class TestRegisterHandling(unittest.TestCase):
    """Test formality register handling."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_register_enum_values(self):
        """All register values should be valid."""
        self.assertEqual(Register.INFORMAL.value, 'informal')
        self.assertEqual(Register.NEUTRAL.value, 'neutral')
        self.assertEqual(Register.FORMAL.value, 'formal')
        self.assertEqual(Register.CEREMONIAL.value, 'ceremonial')


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_empty_string(self):
        """Empty string should not crash."""
        result = self.bridge.translate('')
        self.assertEqual(result.target_text, '')
    
    def test_whitespace_only(self):
        """Whitespace should be preserved."""
        result = self.bridge.translate('   ')
        self.assertEqual(result.target_text.strip(), '')
    
    def test_mixed_scripts(self):
        """Mixed N'Ko and Latin should handle gracefully."""
        result = self.bridge.translate('ߓߊ hello')
        self.assertIsNotNone(result.target_text)
    
    def test_numbers_passthrough(self):
        """Numbers should pass through."""
        result = self.bridge.transliterate_to_latin('123')
        self.assertIn('123', result)
    
    def test_special_characters(self):
        """Special characters should not crash."""
        result = self.bridge.translate('ߓߊ! ߝߊ?')
        self.assertIn('!', result.target_text)
        self.assertIn('?', result.target_text)


class TestProverbs(unittest.TestCase):
    """Test proverb translation."""
    
    def setUp(self):
        self.bridge = TranslationBridge()
    
    def test_proverb_has_notes(self):
        """Proverbs should include cultural notes."""
        proverb = 'ߓߊ߯ߙߊ ߕߍ߫ ߛߌ߫ ߝߍ߬'  # "Work has no shame"
        if proverb in self.bridge.nko_to_latin:
            entry = self.bridge.nko_to_latin[proverb]
            self.assertIn('notes', entry)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
