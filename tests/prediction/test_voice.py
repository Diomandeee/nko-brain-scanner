#!/usr/bin/env python3
"""
Tests for N'Ko Voice Input Engine (Gen 7)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.voice_engine import (
    VoiceInputEngine,
    LatinToNkoConverter,
    MandingPhoneticNormalizer,
    VoiceCommandHandler,
    VoiceKeyboardIntegration,
    VoiceMode,
    PHONEME_TO_NKO,
    WORD_PRONUNCIATIONS
)


class TestPhoneticNormalizer(unittest.TestCase):
    """Tests for Latin orthography normalization."""
    
    def setUp(self):
        self.normalizer = MandingPhoneticNormalizer()
    
    def test_basic_normalize(self):
        """Test basic character normalization."""
        self.assertEqual(self.normalizer.normalize("è"), "ɛ")
        self.assertEqual(self.normalizer.normalize("ò"), "ɔ")
        self.assertEqual(self.normalizer.normalize("ñ"), "ɲ")
    
    def test_digraph_normalize(self):
        """Test digraph normalization."""
        self.assertEqual(self.normalizer.normalize("ny"), "ɲ")
        self.assertEqual(self.normalizer.normalize("dj"), "j")
    
    def test_case_insensitive(self):
        """Test case insensitivity."""
        self.assertEqual(self.normalizer.normalize("BAMBARA"), "bambara")
        self.assertEqual(self.normalizer.normalize("Sogoma"), "sogoma")
    
    def test_whitespace(self):
        """Test whitespace handling."""
        self.assertEqual(self.normalizer.normalize("  i ni ce  "), "i ni ce")


class TestLatinToNkoConverter(unittest.TestCase):
    """Tests for Latin to N'Ko conversion."""
    
    def setUp(self):
        self.converter = LatinToNkoConverter()
    
    def test_single_phonemes(self):
        """Test single phoneme conversion."""
        self.assertIn('ߊ', self.converter.convert('a'))
        self.assertIn('ߓ', self.converter.convert('b'))
        self.assertIn('ߞ', self.converter.convert('k'))
    
    def test_known_words(self):
        """Test known word pronunciations."""
        result = self.converter.convert("i ni sogoma")
        self.assertEqual(result, "ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ")
    
    def test_greeting_variations(self):
        """Test greeting conversions."""
        self.assertEqual(self.converter.convert("i ni ce"), "ߌ ߣߌ ߗߍ")
        self.assertEqual(self.converter.convert("i ni tile"), "ߌ ߣߌ ߕߟߋ")
        self.assertEqual(self.converter.convert("i ni su"), "ߌ ߣߌ ߛߎ")
    
    def test_numbers(self):
        """Test number word conversions."""
        self.assertEqual(self.converter.convert("kelen"), "ߞߋߟߋ߲")
        self.assertEqual(self.converter.convert("fila"), "ߝߌ߬ߟߊ")
        self.assertEqual(self.converter.convert("saba"), "ߛߓߊ")
    
    def test_common_phrases(self):
        """Test common phrase conversions."""
        self.assertEqual(self.converter.convert("barika"), "ߓߙߌߞߊ")
    
    def test_unknown_word(self):
        """Test conversion of unknown words."""
        # Should still produce some output via phoneme mapping
        result = self.converter.convert("wasa")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestVoiceCommandHandler(unittest.TestCase):
    """Tests for voice command handling."""
    
    def setUp(self):
        self.handler = VoiceCommandHandler()
    
    def test_punctuation_commands(self):
        """Test punctuation voice commands."""
        self.assertTrue(self.handler.is_command("point"))
        self.assertTrue(self.handler.is_command("virgule"))
        self.assertTrue(self.handler.is_command("question"))
    
    def test_command_execution(self):
        """Test command execution."""
        text, action = self.handler.execute("point")
        self.assertEqual(text, ".")
        self.assertIsNone(action)
        
        text, action = self.handler.execute("virgule")
        self.assertEqual(text, ",")
    
    def test_control_commands(self):
        """Test control commands."""
        text, action = self.handler.execute("effacer")
        self.assertEqual(text, "")
        self.assertEqual(action, "DELETE")
        
        text, action = self.handler.execute("annuler")
        self.assertEqual(text, "")
        self.assertEqual(action, "UNDO")
    
    def test_nko_punctuation(self):
        """Test N'Ko specific punctuation."""
        text, _ = self.handler.execute("laban")
        self.assertEqual(text, "߹")  # N'Ko full stop
    
    def test_custom_command(self):
        """Test adding custom commands."""
        self.handler.add_custom_command("merci", "ߌ ߣߌ ߓߙߊ")
        self.assertTrue(self.handler.is_command("merci"))
        text, _ = self.handler.execute("merci")
        self.assertEqual(text, "ߌ ߣߌ ߓߙߊ")
    
    def test_non_command(self):
        """Test non-command text."""
        self.assertFalse(self.handler.is_command("bonjour"))


class TestVoiceInputEngine(unittest.TestCase):
    """Tests for main voice input engine."""
    
    def setUp(self):
        self.engine = VoiceInputEngine()
    
    def test_basic_speech_processing(self):
        """Test basic speech to N'Ko conversion."""
        result = self.engine.process_speech("i ni ce")
        self.assertEqual(result.text, "ߌ ߣߌ ߗߍ")
        self.assertEqual(result.latin_transcription, "i ni ce")
        self.assertFalse(result.is_command)
    
    def test_confidence_scores(self):
        """Test confidence scoring."""
        result = self.engine.process_speech("barika")
        self.assertGreater(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_command_detection(self):
        """Test voice command detection."""
        result = self.engine.process_speech("point")
        self.assertTrue(result.is_command)
        self.assertEqual(result.text, ".")
    
    def test_alternatives_generation(self):
        """Test alternative transcription generation."""
        result = self.engine.process_speech("kɛlɛ")
        # Should have alternatives for ɛ/e confusion
        self.assertIsInstance(result.alternatives, list)
    
    def test_dialect_switching(self):
        """Test dialect switching."""
        self.engine.set_dialect("maninka")
        self.assertEqual(self.engine.dialect, "maninka")
        self.assertEqual(self.engine.session.dialect, "maninka")
    
    def test_mode_switching(self):
        """Test mode switching."""
        self.engine.set_mode(VoiceMode.COMMAND)
        self.assertEqual(self.engine.session.mode, VoiceMode.COMMAND)
    
    def test_dictation_session(self):
        """Test dictation session management."""
        self.engine.start_dictation(continuous=True)
        self.assertTrue(self.engine.session.is_listening)
        self.assertTrue(self.engine.session.continuous_mode)
        
        self.engine.process_speech("i ni ce")
        self.assertIn("ߌ ߣߌ ߗߍ", self.engine.session.buffer)
        
        result = self.engine.stop_dictation()
        self.assertFalse(self.engine.session.is_listening)
        self.assertEqual(result, "ߌ ߣߌ ߗߍ")
    
    def test_session_info(self):
        """Test session info retrieval."""
        info = self.engine.get_session_info()
        self.assertIn('mode', info)
        self.assertIn('dialect', info)
        self.assertIn('is_listening', info)
    
    def test_custom_pronunciation(self):
        """Test adding custom pronunciations."""
        self.engine.add_pronunciation("karamogo", "ߞߙߡߐ߬ߜߐ")
        result = self.engine.process_speech("karamogo")
        self.assertEqual(result.text, "ߞߙߡߐ߬ߜߐ")


class TestVoiceKeyboardIntegration(unittest.TestCase):
    """Tests for voice-keyboard integration."""
    
    def test_integration_without_keyboard(self):
        """Test integration without keyboard instance."""
        integration = VoiceKeyboardIntegration()
        result = integration.voice_to_nko("i ni ce")
        self.assertEqual(result.text, "ߌ ߣߌ ߗߍ")
    
    def test_callback_wiring(self):
        """Test callback wiring."""
        integration = VoiceKeyboardIntegration()
        self.assertIsNotNone(integration.voice_engine.on_text)


class TestPhonemeMapping(unittest.TestCase):
    """Tests for phoneme to N'Ko mapping completeness."""
    
    def test_vowels_present(self):
        """Test all Manding vowels are mapped."""
        vowels = ['a', 'e', 'ɛ', 'i', 'o', 'ɔ', 'u']
        for v in vowels:
            self.assertIn(v, PHONEME_TO_NKO)
    
    def test_consonants_present(self):
        """Test common consonants are mapped."""
        consonants = ['b', 'p', 'd', 't', 'k', 'g', 'f', 's', 'm', 'n', 'l', 'r', 'w', 'y']
        for c in consonants:
            self.assertIn(c, PHONEME_TO_NKO)
    
    def test_digraphs_present(self):
        """Test Manding digraphs are mapped."""
        digraphs = ['ny', 'ng', 'gb']
        for d in digraphs:
            self.assertIn(d, PHONEME_TO_NKO)


class TestWordPronunciations(unittest.TestCase):
    """Tests for word pronunciation dictionary."""
    
    def test_greetings_present(self):
        """Test greeting pronunciations are present."""
        # Check that at least one variant is present
        self.assertTrue('i ni ce' in WORD_PRONUNCIATIONS or 'ini ce' in WORD_PRONUNCIATIONS)
        self.assertTrue('i ni sogoma' in WORD_PRONUNCIATIONS or 'ini sogoma' in WORD_PRONUNCIATIONS)
        self.assertTrue('i ni tile' in WORD_PRONUNCIATIONS or 'ini tile' in WORD_PRONUNCIATIONS)
        self.assertTrue('i ni su' in WORD_PRONUNCIATIONS or 'ini su' in WORD_PRONUNCIATIONS)
    
    def test_numbers_present(self):
        """Test number pronunciations are present."""
        self.assertIn('kelen', WORD_PRONUNCIATIONS)
        self.assertIn('fila', WORD_PRONUNCIATIONS)
        self.assertIn('saba', WORD_PRONUNCIATIONS)


if __name__ == '__main__':
    print("=" * 60)
    print("N'Ko Voice Engine Tests (Gen 7)")
    print("=" * 60)
    print()
    
    # Run tests with verbosity
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern='test_voice.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 60)
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("ߊ߬ ߣߌ ߓߙߍ߬! (All tests passed!)")
    print("=" * 60)
