#!/usr/bin/env python3
"""
Tests for N'Ko TTS Engine (Generation 10)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

import unittest
from tts_engine import (
    NkoTTSEngine, NkoPhonemeDB, WordPronouncer, ProsodyEngine,
    DialectPronunciation, TTSKeyboardIntegration, VoiceProfile,
    Dialect, SpeechSpeed, TonePattern, EmphasisType
)

# =============================================================================
# PHONEME DATABASE TESTS
# =============================================================================

class TestPhonemeDB(unittest.TestCase):
    """Test N'Ko phoneme database."""
    
    def test_vowels_complete(self):
        """All N'Ko vowels should be mapped."""
        vowels = ["ߊ", "ߋ", "ߎ", "ߍ", "ߌ"]
        for v in vowels:
            self.assertIn(v, NkoPhonemeDB.VOWELS)
            self.assertIsNotNone(NkoPhonemeDB.get_phoneme(v))
    
    def test_consonants_complete(self):
        """All N'Ko consonants should be mapped."""
        consonants = ["ߓ", "ߔ", "ߕ", "ߖ", "ߗ", "ߘ", "ߙ", "ߛ", "ߞ", "ߟ", "ߡ", "ߣ"]
        for c in consonants:
            self.assertIn(c, NkoPhonemeDB.CONSONANTS)
            self.assertIsNotNone(NkoPhonemeDB.get_phoneme(c))
    
    def test_long_vowels(self):
        """Long vowels should have longer duration."""
        short_a = NkoPhonemeDB.VOWELS.get("ߊ")
        long_a = NkoPhonemeDB.VOWELS.get("ߊ߲")
        
        self.assertIsNotNone(short_a)
        self.assertIsNotNone(long_a)
        self.assertGreater(long_a.duration_ms, short_a.duration_ms)
    
    def test_is_vowel(self):
        """Vowel detection should work."""
        self.assertTrue(NkoPhonemeDB.is_vowel("ߊ"))
        self.assertTrue(NkoPhonemeDB.is_vowel("ߌ"))
        self.assertFalse(NkoPhonemeDB.is_vowel("ߓ"))
        self.assertFalse(NkoPhonemeDB.is_vowel("ߞ"))
    
    def test_is_consonant(self):
        """Consonant detection should work."""
        self.assertTrue(NkoPhonemeDB.is_consonant("ߓ"))
        self.assertTrue(NkoPhonemeDB.is_consonant("ߞ"))
        self.assertFalse(NkoPhonemeDB.is_consonant("ߊ"))
    
    def test_punctuation_pauses(self):
        """Punctuation should have defined pause durations."""
        self.assertIn("߸", NkoPhonemeDB.PUNCTUATION)  # comma
        self.assertIn("߹", NkoPhonemeDB.PUNCTUATION)  # period
        self.assertGreater(
            NkoPhonemeDB.PUNCTUATION["߹"],
            NkoPhonemeDB.PUNCTUATION["߸"]
        )
    
    def test_tone_marks(self):
        """Tone marks should be mapped."""
        self.assertIn("߫", NkoPhonemeDB.TONE_MARKS)  # high
        self.assertIn("߬", NkoPhonemeDB.TONE_MARKS)  # low
        self.assertEqual(NkoPhonemeDB.TONE_MARKS["߫"], TonePattern.HIGH)
        self.assertEqual(NkoPhonemeDB.TONE_MARKS["߬"], TonePattern.LOW)

# =============================================================================
# WORD PRONOUNCER TESTS
# =============================================================================

class TestWordPronouncer(unittest.TestCase):
    """Test word pronunciation."""
    
    def setUp(self):
        self.pronouncer = WordPronouncer(dialect=Dialect.BAMBARA)
    
    def test_simple_word(self):
        """Simple words should produce phonemes."""
        result = self.pronouncer.pronounce("ߓߊ")  # "ba"
        
        self.assertEqual(result.nko_text, "ߓߊ")
        self.assertGreater(len(result.phonemes), 0)
        self.assertGreater(result.total_duration_ms, 0)
    
    def test_greeting_special(self):
        """Special pronunciations should be used for greetings."""
        result = self.pronouncer.pronounce("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        
        # Should have custom pronunciation
        self.assertGreater(len(result.syllables), 0)
        self.assertGreater(len(result.stress_pattern), 0)
    
    def test_syllables_generated(self):
        """Words should be split into syllables."""
        result = self.pronouncer.pronounce("ߓߊ߬ߙߌ߬ߞߊ")  # barika
        
        self.assertGreater(len(result.syllables), 0)  # At least one syllable
    
    def test_stress_pattern(self):
        """Stress pattern should be generated."""
        result = self.pronouncer.pronounce("ߓߊ߬ߙߌ߬ߞߊ")
        
        # Stress pattern should match syllables (or be non-empty)
        self.assertGreaterEqual(len(result.stress_pattern), len(result.syllables))
        self.assertTrue(all(s in [0, 1, 2] for s in result.stress_pattern))

# =============================================================================
# DIALECT TESTS
# =============================================================================

class TestDialectPronunciation(unittest.TestCase):
    """Test dialect-specific pronunciations."""
    
    def test_dialect_transforms_exist(self):
        """All dialects should have transform rules."""
        for dialect in Dialect:
            self.assertIn(dialect, DialectPronunciation.DIALECT_TRANSFORMS)
    
    def test_pitch_contours_exist(self):
        """All dialects should have pitch contours."""
        for dialect in Dialect:
            self.assertIn(dialect, DialectPronunciation.PITCH_CONTOURS)
            contours = DialectPronunciation.PITCH_CONTOURS[dialect]
            self.assertIn("statement", contours)
            self.assertIn("question", contours)
    
    def test_bambara_baseline(self):
        """Bambara should not transform standard phonemes."""
        result = DialectPronunciation.transform_phoneme("b", Dialect.BAMBARA)
        self.assertEqual(result, "b")
    
    def test_maninka_transforms(self):
        """Maninka should have specific transforms."""
        # gb → g in Maninka
        result = DialectPronunciation.transform_phoneme("gb", Dialect.MANINKA)
        self.assertEqual(result, "g")
    
    def test_pitch_contour_question(self):
        """Question contour should rise."""
        contour = DialectPronunciation.get_pitch_contour(
            Dialect.BAMBARA, EmphasisType.QUESTION
        )
        
        self.assertGreater(len(contour), 1)
        self.assertGreater(contour[-1], contour[0])  # Rising

# =============================================================================
# PROSODY ENGINE TESTS
# =============================================================================

class TestProsodyEngine(unittest.TestCase):
    """Test prosody/intonation handling."""
    
    def setUp(self):
        self.prosody = ProsodyEngine(dialect=Dialect.BAMBARA)
    
    def test_detect_question(self):
        """Questions should be detected."""
        result = self.prosody.detect_emphasis("ߌ ߕߐ߯ ߓߍ ߖߏ߫?")
        self.assertEqual(result, EmphasisType.QUESTION)
    
    def test_detect_exclamation(self):
        """Exclamations should be detected."""
        result = self.prosody.detect_emphasis("ߏ ߞߊ߫ ߘߌ߫߷")
        self.assertEqual(result, EmphasisType.EXCLAMATION)
    
    def test_detect_greeting(self):
        """Greetings should be detected."""
        result = self.prosody.detect_emphasis("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        self.assertEqual(result, EmphasisType.GREETING)
    
    def test_detect_statement(self):
        """Default should be statement."""
        result = self.prosody.detect_emphasis("ߒ ߓߍ߫")
        self.assertEqual(result, EmphasisType.STATEMENT)
    
    def test_pause_calculation(self):
        """Pauses should vary by punctuation."""
        segments = ["ߊ߹", "ߓ߸", "ߞ"]
        pauses = self.prosody.calculate_pauses(segments)
        
        self.assertEqual(len(pauses), 3)
        self.assertGreater(pauses[0], pauses[1])  # Period > comma

# =============================================================================
# MAIN TTS ENGINE TESTS
# =============================================================================

class TestNkoTTSEngine(unittest.TestCase):
    """Test main TTS engine."""
    
    def setUp(self):
        self.engine = NkoTTSEngine(dialect=Dialect.BAMBARA)
    
    def test_synthesize_basic(self):
        """Basic synthesis should work."""
        result = self.engine.synthesize("ߌ ߣߌ ߗߍ")
        
        self.assertEqual(result.original_text, "ߌ ߣߌ ߗߍ")
        self.assertGreater(len(result.segments), 0)
        self.assertGreater(result.total_duration_ms, 0)
        self.assertEqual(result.dialect, Dialect.BAMBARA)
    
    def test_phonetic_representation(self):
        """IPA should be generated."""
        result = self.engine.synthesize("ߓߊ")
        
        self.assertIsNotNone(result.phonetic_representation)
        self.assertIn("b", result.phonetic_representation)
        self.assertIn("a", result.phonetic_representation)
    
    def test_ssml_generated(self):
        """SSML should be generated."""
        result = self.engine.synthesize("ߌ ߣߌ")
        
        self.assertIn("<speak>", result.ssml)
        self.assertIn("</speak>", result.ssml)
        self.assertIn("<phoneme", result.ssml)
    
    def test_audio_hints(self):
        """Audio hints should be populated."""
        result = self.engine.synthesize("ߓߊ")
        
        self.assertIn("pitch_base", result.audio_hints)
        self.assertIn("rate", result.audio_hints)
        self.assertIn("emphasis", result.audio_hints)
    
    def test_pronunciation_guide(self):
        """Pronunciation guide should be readable."""
        guide = self.engine.get_pronunciation_guide("ߓߊ")
        
        self.assertIsInstance(guide, str)
        self.assertGreater(len(guide), 0)
    
    def test_dialect_change(self):
        """Dialect can be changed."""
        self.engine.set_dialect(Dialect.MANINKA)
        
        self.assertEqual(self.engine.dialect, Dialect.MANINKA)
        
        result = self.engine.synthesize("ߓߊ")
        self.assertEqual(result.dialect, Dialect.MANINKA)
    
    def test_speed_change(self):
        """Speed can be changed."""
        self.engine.set_speed(SpeechSpeed.SLOW)
        
        result_slow = self.engine.synthesize("ߌ ߣߌ")
        
        self.engine.set_speed(SpeechSpeed.FAST)
        result_fast = self.engine.synthesize("ߌ ߣߌ")
        
        # Slow should take longer
        self.assertGreater(result_slow.total_duration_ms, result_fast.total_duration_ms)

# =============================================================================
# SPECIAL MODES TESTS
# =============================================================================

class TestSpecialModes(unittest.TestCase):
    """Test special TTS modes."""
    
    def setUp(self):
        self.engine = NkoTTSEngine(dialect=Dialect.BAMBARA)
    
    def test_spell_out(self):
        """Spell-out mode should letter by letter."""
        result = self.engine.spell_out("ߓߊ")
        
        # Should have separate segments
        self.assertGreater(len(result.segments), 1)
        # Should have longer pauses
        for seg in result.segments:
            self.assertGreater(seg.pause_after_ms, 300)
    
    def test_recite_proverb(self):
        """Proverb mode should have slower cadence."""
        normal = self.engine.synthesize("ߛߓߍ ߦߋ߫")
        proverb = self.engine.recite_proverb("ߛߓߍ ߦߋ߫")
        
        # Proverb should be slower
        self.assertGreater(proverb.total_duration_ms, normal.total_duration_ms)
    
    def test_learning_breakdown(self):
        """Learning breakdown should detail each character."""
        breakdown = self.engine.get_learning_breakdown("ߓߊ")
        
        self.assertGreater(len(breakdown), 0)
        for item in breakdown:
            self.assertIn("character", item)
            self.assertIn("ipa", item)
            self.assertIn("romanized", item)
            self.assertIn("type", item)
    
    def test_describe_text(self):
        """Text description should be accessible."""
        description = self.engine.describe_text("ߌ ߣߌ")
        
        self.assertIn("N'Ko text", description)
        self.assertIn("Pronunciation", description)

# =============================================================================
# KEYBOARD INTEGRATION TESTS
# =============================================================================

class TestKeyboardIntegration(unittest.TestCase):
    """Test TTS keyboard integration."""
    
    def setUp(self):
        engine = NkoTTSEngine()
        self.integration = TTSKeyboardIntegration(engine)
    
    def test_character_typed(self):
        """Character typing should update buffer."""
        self.integration.on_character_typed("ߓ")
        self.integration.on_character_typed("ߊ")
        
        self.assertEqual(self.integration.current_buffer, "ߓߊ")
    
    def test_auto_read_off(self):
        """Auto-read off should not return result."""
        self.integration.auto_read = False
        result = self.integration.on_character_typed("ߓ")
        
        self.assertIsNone(result)
    
    def test_auto_read_on(self):
        """Auto-read on should return result."""
        self.integration.auto_read = True
        result = self.integration.on_character_typed("ߓ")
        
        self.assertIsNotNone(result)
    
    def test_read_buffer(self):
        """Should be able to read current buffer."""
        self.integration.on_character_typed("ߓ")
        self.integration.on_character_typed("ߊ")
        
        result = self.integration.read_current_buffer()
        
        self.assertEqual(result.original_text, "ߓߊ")
    
    def test_clear_buffer(self):
        """Buffer should be clearable."""
        self.integration.on_character_typed("ߓ")
        self.integration.clear_buffer()
        
        self.assertEqual(self.integration.current_buffer, "")

# =============================================================================
# VOICE PROFILE TESTS
# =============================================================================

class TestVoiceProfile(unittest.TestCase):
    """Test voice profile handling."""
    
    def test_default_voice(self):
        """Default voice should be created."""
        engine = NkoTTSEngine()
        
        self.assertIsNotNone(engine.voice)
        self.assertEqual(engine.voice.dialect, Dialect.BAMBARA)
    
    def test_custom_voice(self):
        """Custom voice should be usable."""
        custom = VoiceProfile(
            name="custom_voice",
            dialect=Dialect.MANINKA,
            pitch_base=1.2,
            rate=SpeechSpeed.SLOW
        )
        
        engine = NkoTTSEngine(voice=custom)
        
        self.assertEqual(engine.voice.name, "custom_voice")
        self.assertEqual(engine.voice.pitch_base, 1.2)

# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.engine = NkoTTSEngine()
    
    def test_empty_text(self):
        """Empty text should not crash."""
        result = self.engine.synthesize("")
        
        self.assertEqual(result.original_text, "")
        self.assertEqual(len(result.segments), 0)
    
    def test_punctuation_only(self):
        """Punctuation-only text should work."""
        result = self.engine.synthesize("߸ ߹")
        
        self.assertIsNotNone(result)
    
    def test_mixed_script(self):
        """Mixed N'Ko and Latin should not crash."""
        result = self.engine.synthesize("ߌ ߣߌ hello")
        
        # N'Ko parts should be processed
        self.assertGreater(len(result.segments), 0)
    
    def test_long_text(self):
        """Long text should be handled."""
        long_text = "ߌ ߣߌ ߗߍ " * 20
        result = self.engine.synthesize(long_text)
        
        self.assertGreater(len(result.segments), 10)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
