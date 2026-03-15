"""
Tests for N'Ko Keyboard AI Accessibility Engine (Generation 12)
"""

import os
import sys

try:
    import pytest
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "pytest_compat",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest_compat.py")
    )
    pytest = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(pytest)
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.accessibility_engine import (
    NkoAccessibilityEngine,
    AccessibilityProfile,
    AccessibilityMode,
    ContrastTheme,
    ScanPattern,
    VoiceCommandCategory,
    FeedbackChannel,
    ScreenReaderEngine,
    MotorAccessibilityEngine,
    VoiceControlEngine,
    HapticFeedbackEngine,
    OneHandedLayoutEngine,
    CognitiveAccessibilityEngine,
    NKO_PHONETICS,
    VOICE_COMMANDS,
    create_accessibility_engine,
    get_nko_phonetic,
    get_voice_commands,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db():
    """Create temporary database"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def accessibility_engine(temp_db):
    """Create accessibility engine with temp database"""
    return NkoAccessibilityEngine(db_path=temp_db)


@pytest.fixture
def standard_profile(accessibility_engine):
    """Create standard accessibility profile"""
    return accessibility_engine.create_profile(
        user_id="test_user",
        name="Test Profile",
        mode=AccessibilityMode.STANDARD
    )


@pytest.fixture
def screen_reader_profile(temp_db):
    """Create screen reader profile"""
    engine = NkoAccessibilityEngine(db_path=temp_db)
    engine.create_profile(
        user_id="test_user",
        name="Screen Reader Profile",
        mode=AccessibilityMode.SCREEN_READER
    )
    return engine


@pytest.fixture
def motor_assist_profile(temp_db):
    """Create motor assist profile"""
    engine = NkoAccessibilityEngine(db_path=temp_db)
    engine.create_profile(
        user_id="test_user",
        name="Motor Assist Profile",
        mode=AccessibilityMode.MOTOR_ASSIST
    )
    return engine


# =============================================================================
# PROFILE TESTS
# =============================================================================

class TestAccessibilityProfile:
    """Test accessibility profile creation and management"""
    
    def test_create_standard_profile(self, accessibility_engine):
        """Test creating standard profile"""
        profile = accessibility_engine.create_profile(
            user_id="user1",
            name="Standard Test",
            mode=AccessibilityMode.STANDARD
        )
        
        assert profile is not None
        assert profile.user_id == "user1"
        assert profile.name == "Standard Test"
        assert profile.mode == AccessibilityMode.STANDARD
        
    def test_create_screen_reader_profile(self, accessibility_engine):
        """Test screen reader profile has correct defaults"""
        profile = accessibility_engine.create_profile(
            user_id="user2",
            name="Screen Reader Test",
            mode=AccessibilityMode.SCREEN_READER
        )
        
        assert profile.screen_reader_enabled is True
        assert profile.speak_characters is True
        assert profile.speak_words is True
        assert profile.speak_predictions is True
        
    def test_create_high_contrast_profile(self, accessibility_engine):
        """Test high contrast profile settings"""
        profile = accessibility_engine.create_profile(
            user_id="user3",
            name="High Contrast Test",
            mode=AccessibilityMode.HIGH_CONTRAST
        )
        
        assert profile.contrast_theme == ContrastTheme.YELLOW_ON_BLACK
        assert profile.text_scale > 1.0
        assert profile.animation_enabled is False
        
    def test_create_motor_assist_profile(self, accessibility_engine):
        """Test motor assist profile settings"""
        profile = accessibility_engine.create_profile(
            user_id="user4",
            name="Motor Assist Test",
            mode=AccessibilityMode.MOTOR_ASSIST
        )
        
        assert profile.key_hold_time_ms > 0
        assert profile.touch_tolerance_px > 10
        assert profile.haptic_enabled is True
        
    def test_create_cognitive_ease_profile(self, accessibility_engine):
        """Test cognitive ease profile settings"""
        profile = accessibility_engine.create_profile(
            user_id="user5",
            name="Cognitive Ease Test",
            mode=AccessibilityMode.COGNITIVE_EASE
        )
        
        assert profile.simplified_layout is True
        assert profile.max_predictions_shown <= 3
        assert profile.confirmation_prompts is True
        
    def test_create_one_handed_profile(self, accessibility_engine):
        """Test one-handed profile settings"""
        profile = accessibility_engine.create_profile(
            user_id="user6",
            name="One Handed Test",
            mode=AccessibilityMode.ONE_HANDED
        )
        
        assert profile.one_hand_layout is True
        assert profile.key_size_multiplier > 1.0
        
    def test_profile_persistence(self, temp_db):
        """Test profile save and load"""
        # Create engine and profile
        engine1 = NkoAccessibilityEngine(db_path=temp_db)
        profile1 = engine1.create_profile(
            user_id="persist_user",
            name="Persist Test",
            mode=AccessibilityMode.HIGH_CONTRAST
        )
        profile_id = profile1.profile_id
        
        # Create new engine and load profile
        engine2 = NkoAccessibilityEngine(db_path=temp_db)
        profile2 = engine2.load_profile(profile_id)
        
        assert profile2 is not None
        assert profile2.user_id == "persist_user"
        assert profile2.name == "Persist Test"
        assert profile2.mode == AccessibilityMode.HIGH_CONTRAST
        
    def test_profile_to_dict(self, standard_profile):
        """Test profile serialization"""
        profile_dict = standard_profile.to_dict()
        
        assert "profile_id" in profile_dict
        assert "user_id" in profile_dict
        assert "mode" in profile_dict
        assert profile_dict["mode"] == "STANDARD"


# =============================================================================
# SCREEN READER TESTS
# =============================================================================

class TestScreenReader:
    """Test screen reader functionality"""
    
    def test_announce_character(self):
        """Test character announcement"""
        reader = ScreenReaderEngine(language="bambara")
        output = reader.announce_character("ߊ")
        
        assert output.text == "ߊ"
        assert output.announcement_type == "character"
        assert "a" in output.phonetic.lower()
        
    def test_announce_word(self):
        """Test word announcement"""
        reader = ScreenReaderEngine()
        output = reader.announce_word("ߌ ߣߌ")
        
        assert output.text == "ߌ ߣߌ"
        assert output.announcement_type == "word"
        
    def test_announce_prediction(self):
        """Test prediction announcement"""
        reader = ScreenReaderEngine()
        output = reader.announce_prediction("ߛߐ߲߬ߜߐ߫ߡߊ", 0)
        
        assert output.announcement_type == "prediction"
        assert "first" in output.text.lower()
        
    def test_spell_text(self):
        """Test text spelling"""
        reader = ScreenReaderEngine()
        outputs = reader.spell_text("ߊߓ")
        
        assert len(outputs) == 2
        assert all(o.announcement_type == "character" for o in outputs)
        
    def test_announce_alert(self):
        """Test alert announcement"""
        reader = ScreenReaderEngine()
        output = reader.announce_alert("Error occurred", urgent=True)
        
        assert output.priority == 5
        assert output.interrupt_current is True
        
    def test_get_context_empty(self):
        """Test context with empty text"""
        reader = ScreenReaderEngine()
        context = reader.get_context()
        
        assert "Empty" in context


# =============================================================================
# MOTOR ACCESSIBILITY TESTS
# =============================================================================

class TestMotorAccessibility:
    """Test motor accessibility features"""
    
    def test_dwell_instant(self, motor_assist_profile):
        """Test instant key press (no dwell)"""
        profile = motor_assist_profile.profile
        profile.key_hold_time_ms = 0
        
        motor = MotorAccessibilityEngine(profile)
        result = motor.start_dwell("ߊ")
        
        assert result is True  # Instant selection
        
    def test_dwell_timer(self, motor_assist_profile):
        """Test dwell timer start"""
        profile = motor_assist_profile.profile
        profile.key_hold_time_ms = 500
        
        motor = MotorAccessibilityEngine(profile)
        result = motor.start_dwell("ߊ")
        
        assert result is False  # Waiting for dwell
        assert motor.current_key == "ߊ"
        
    def test_dwell_cancel(self, motor_assist_profile):
        """Test dwell cancellation"""
        profile = motor_assist_profile.profile
        profile.key_hold_time_ms = 500
        
        motor = MotorAccessibilityEngine(profile)
        motor.start_dwell("ߊ")
        motor.cancel_dwell()
        
        assert motor.current_key is None
        
    def test_scan_start(self, motor_assist_profile):
        """Test scanning start"""
        motor = MotorAccessibilityEngine(motor_assist_profile.profile)
        
        selected = []
        motor.start_scanning(lambda k: selected.append(k))
        
        assert motor.is_scanning is True
        
    def test_scan_advance(self, motor_assist_profile):
        """Test scan advancement"""
        motor = MotorAccessibilityEngine(motor_assist_profile.profile)
        motor.start_scanning(lambda k: None)
        
        pos1 = motor.advance_scan()
        pos2 = motor.advance_scan()
        
        assert pos1 != pos2  # Position should change
        
    def test_scan_patterns(self, motor_assist_profile):
        """Test different scan patterns"""
        profile = motor_assist_profile.profile
        
        for pattern in ScanPattern:
            profile.scan_pattern = pattern
            motor = MotorAccessibilityEngine(profile)
            motor.start_scanning(lambda k: None)
            
            # Should not raise error
            result = motor.advance_scan()
            assert result is not None


# =============================================================================
# VOICE CONTROL TESTS
# =============================================================================

class TestVoiceControl:
    """Test voice control functionality"""
    
    def test_parse_navigation_command(self):
        """Test parsing navigation command"""
        voice = VoiceControlEngine()
        command = voice.parse_command("left")
        
        assert command is not None
        assert command.category == VoiceCommandCategory.NAVIGATION
        assert command.action == "left"
        
    def test_parse_input_command(self):
        """Test parsing input command"""
        voice = VoiceControlEngine()
        command = voice.parse_command("type hello")
        
        assert command is not None
        assert command.category == VoiceCommandCategory.INPUT
        assert command.action == "type"
        assert command.parameters.get("text") == "hello"
        
    def test_parse_bambara_command(self):
        """Test parsing Bambara command"""
        voice = VoiceControlEngine(language="bambara")
        command = voice.parse_command("sɛbɛn")  # type
        
        assert command is not None
        assert command.category == VoiceCommandCategory.INPUT
        
    def test_parse_french_command(self):
        """Test parsing French command"""
        voice = VoiceControlEngine()
        command = voice.parse_command("supprimer")  # delete
        
        assert command is not None
        assert command.category == VoiceCommandCategory.EDITING
        
    def test_custom_command(self):
        """Test custom command registration"""
        voice = VoiceControlEngine()
        called = []
        
        voice.register_custom_command("my command", lambda: called.append(True))
        command = voice.parse_command("my command")
        
        assert command is not None
        assert command.action == "custom"
        
    def test_available_commands(self):
        """Test getting available commands"""
        voice = VoiceControlEngine()
        commands = voice.get_available_commands()
        
        assert len(commands) > 0


# =============================================================================
# HAPTIC FEEDBACK TESTS
# =============================================================================

class TestHapticFeedback:
    """Test haptic feedback functionality"""
    
    def test_trigger_keypress(self, standard_profile):
        """Test keypress haptic"""
        haptic = HapticFeedbackEngine(standard_profile)
        
        triggered = []
        haptic.set_haptic_callback(lambda p: triggered.append(p))
        
        haptic.trigger_keypress("ߊ")
        
        assert len(triggered) == 1
        
    def test_trigger_space(self, standard_profile):
        """Test space key haptic"""
        haptic = HapticFeedbackEngine(standard_profile)
        
        triggered = []
        haptic.set_haptic_callback(lambda p: triggered.append(p))
        
        haptic.trigger_keypress(" ")
        
        # Space has distinct pattern
        assert len(triggered) == 1
        
    def test_trigger_error(self, standard_profile):
        """Test error haptic"""
        haptic = HapticFeedbackEngine(standard_profile)
        
        triggered = []
        haptic.set_haptic_callback(lambda p: triggered.append(p))
        
        haptic.trigger_error()
        
        assert len(triggered) == 1
        
    def test_haptic_disabled(self, standard_profile):
        """Test haptic when disabled"""
        standard_profile.haptic_enabled = False
        haptic = HapticFeedbackEngine(standard_profile)
        
        triggered = []
        haptic.set_haptic_callback(lambda p: triggered.append(p))
        
        result = haptic.trigger("keypress")
        
        assert result is False
        assert len(triggered) == 0
        
    def test_intensity_scaling(self, standard_profile):
        """Test intensity scaling"""
        standard_profile.haptic_intensity = 0.5
        haptic = HapticFeedbackEngine(standard_profile)
        
        patterns = []
        haptic.set_haptic_callback(lambda p: patterns.append(p))
        
        haptic.trigger("keypress")
        
        # Check intensity is scaled
        if patterns and patterns[0]:
            _, intensity = patterns[0][0]
            assert intensity <= 50  # Scaled from 80


# =============================================================================
# ONE-HANDED LAYOUT TESTS
# =============================================================================

class TestOneHandedLayout:
    """Test one-handed keyboard layout"""
    
    def test_right_hand_layout(self):
        """Test right hand layout"""
        layout = OneHandedLayoutEngine(preferred_hand="right")
        
        assert layout.preferred_hand == "right"
        assert len(layout.layout) > 0
        
    def test_left_hand_layout(self):
        """Test left hand layout"""
        layout = OneHandedLayoutEngine(preferred_hand="left")
        
        assert layout.preferred_hand == "left"
        
    def test_switch_hand(self):
        """Test switching hands"""
        layout = OneHandedLayoutEngine(preferred_hand="right")
        original_rows = len(layout.layout)
        
        layout.switch_hand("left")
        
        assert layout.preferred_hand == "left"
        assert len(layout.layout) == original_rows
        
    def test_get_key_at(self):
        """Test getting key at position"""
        layout = OneHandedLayoutEngine()
        
        cell = layout.get_key_at(0, 0)
        
        assert cell is not None
        assert cell.char != ""
        
    def test_key_phonetics(self):
        """Test key phonetic data"""
        layout = OneHandedLayoutEngine()
        
        for row in layout.layout:
            for cell in row:
                if cell.char in NKO_PHONETICS:
                    assert cell.phonetic != ""


# =============================================================================
# COGNITIVE ACCESSIBILITY TESTS
# =============================================================================

class TestCognitiveAccessibility:
    """Test cognitive accessibility features"""
    
    def test_simplify_predictions(self, temp_db):
        """Test prediction simplification"""
        engine = NkoAccessibilityEngine(db_path=temp_db)
        profile = engine.create_profile(
            user_id="test",
            name="Cognitive Test",
            mode=AccessibilityMode.COGNITIVE_EASE
        )
        
        predictions = ["word1", "word2", "word3", "word4", "word5"]
        simplified = engine.cognitive.simplify_predictions(predictions)
        
        assert len(simplified) <= profile.max_predictions_shown
        
    def test_confirmation_prompt(self, temp_db):
        """Test confirmation prompt creation"""
        engine = NkoAccessibilityEngine(db_path=temp_db)
        engine.create_profile(
            user_id="test",
            name="Cognitive Test",
            mode=AccessibilityMode.COGNITIVE_EASE
        )
        
        confirmation = engine.cognitive.create_confirmation("delete", "word")
        
        assert "action" in confirmation
        assert "prompt" in confirmation
        assert "options" in confirmation
        
    def test_workflow_steps(self, temp_db):
        """Test workflow guidance"""
        engine = NkoAccessibilityEngine(db_path=temp_db)
        engine.create_profile(
            user_id="test",
            name="Cognitive Test",
            mode=AccessibilityMode.COGNITIVE_EASE
        )
        
        steps = ["Step 1", "Step 2", "Step 3"]
        engine.cognitive.set_workflow(steps)
        
        current = engine.cognitive.get_current_step()
        
        assert current["step_number"] == 1
        assert current["total_steps"] == 3
        
    def test_workflow_advance(self, temp_db):
        """Test workflow advancement"""
        engine = NkoAccessibilityEngine(db_path=temp_db)
        engine.create_profile(
            user_id="test",
            name="Cognitive Test",
            mode=AccessibilityMode.COGNITIVE_EASE
        )
        
        engine.cognitive.set_workflow(["A", "B", "C"])
        engine.cognitive.advance_workflow()
        
        current = engine.cognitive.get_current_step()
        
        assert current["step_number"] == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAccessibilityIntegration:
    """Integration tests for accessibility engine"""
    
    def test_on_key_press_standard(self, accessibility_engine):
        """Test key press handling in standard mode"""
        accessibility_engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.STANDARD
        )
        
        result = accessibility_engine.on_key_press("ߊ")
        
        assert result["char"] == "ߊ"
        
    def test_on_key_press_screen_reader(self, screen_reader_profile):
        """Test key press with screen reader"""
        result = screen_reader_profile.on_key_press("ߊ")
        
        assert result["announced"] is True
        assert "announcement" in result
        
    def test_on_word_complete(self, screen_reader_profile):
        """Test word completion"""
        result = screen_reader_profile.on_word_complete("ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ")
        
        assert "announcement" in result
        
    def test_predictions_with_accessibility(self, accessibility_engine):
        """Test predictions through accessibility layer"""
        accessibility_engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.COGNITIVE_EASE
        )
        
        predictions = ["word1", "word2", "word3", "word4", "word5"]
        result = accessibility_engine.on_prediction_shown(predictions)
        
        # Should be limited by cognitive accessibility
        assert len(result) <= 3
        
    def test_voice_input_handling(self, accessibility_engine):
        """Test voice input processing"""
        accessibility_engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.VOICE_CONTROL
        )
        
        command = accessibility_engine.on_voice_input("delete")
        
        assert command is not None
        assert command.action == "delete"
        
    def test_get_keyboard_layout_standard(self, accessibility_engine):
        """Test getting standard layout"""
        accessibility_engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.STANDARD
        )
        
        layout = accessibility_engine.get_keyboard_layout()
        
        assert layout["layout_type"] == "standard"
        
    def test_get_keyboard_layout_one_handed(self, temp_db):
        """Test getting one-handed layout"""
        engine = NkoAccessibilityEngine(db_path=temp_db)
        engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.ONE_HANDED
        )
        
        layout = engine.get_keyboard_layout()
        
        assert layout["layout_type"] == "one_handed"
        assert len(layout["rows"]) > 0
        
    def test_visual_settings(self, accessibility_engine):
        """Test getting visual settings"""
        accessibility_engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.HIGH_CONTRAST
        )
        
        settings = accessibility_engine.get_visual_settings()
        
        assert settings["contrast_theme"] == "YELLOW_ON_BLACK"
        assert settings["text_scale"] > 1.0
        
    def test_usage_logging(self, accessibility_engine):
        """Test usage analytics logging"""
        accessibility_engine.create_profile(
            user_id="test",
            name="Test",
            mode=AccessibilityMode.STANDARD
        )
        
        # Should not raise error
        accessibility_engine.log_usage("test", "action", success=True, duration_ms=100)


# =============================================================================
# PHONETICS TESTS
# =============================================================================

class TestPhonetics:
    """Test N'Ko phonetics data"""
    
    def test_vowels_present(self):
        """Test vowel phonetics"""
        vowels = ["ߊ", "ߋ", "ߎ", "ߍ", "ߌ", "ߏ", "ߐ"]
        
        for v in vowels:
            assert v in NKO_PHONETICS
            assert "ipa" in NKO_PHONETICS[v]
            
    def test_consonants_present(self):
        """Test consonant phonetics"""
        consonants = ["ߓ", "ߘ", "ߞ", "ߟ", "ߡ", "ߣ", "ߛ"]
        
        for c in consonants:
            assert c in NKO_PHONETICS
            
    def test_numbers_present(self):
        """Test number phonetics"""
        for i in range(10):
            nko_digit = chr(0x07C0 + i)
            assert nko_digit in NKO_PHONETICS
            
    def test_get_phonetic_helper(self):
        """Test get_nko_phonetic helper"""
        phonetic = get_nko_phonetic("ߊ", "bambara")
        
        assert phonetic != ""
        
    def test_unknown_character(self):
        """Test unknown character handling"""
        phonetic = get_nko_phonetic("X")
        
        assert phonetic == "X"  # Returns unchanged


# =============================================================================
# VOICE COMMANDS DATA TESTS
# =============================================================================

class TestVoiceCommandsData:
    """Test voice commands data"""
    
    def test_all_categories_present(self):
        """Test all categories have commands"""
        for category in VoiceCommandCategory:
            assert category.name in VOICE_COMMANDS
            
    def test_multilingual_commands(self):
        """Test commands have multiple languages"""
        nav_commands = VOICE_COMMANDS[VoiceCommandCategory.NAVIGATION.name]
        
        for action, triggers in nav_commands.items():
            assert len(triggers) >= 2  # At least English and one other
            
    def test_get_voice_commands_helper(self):
        """Test get_voice_commands helper"""
        commands = get_voice_commands("english")
        
        assert len(commands) > 0
        assert VoiceCommandCategory.NAVIGATION.name in commands


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_accessibility_engine(self, temp_db):
        """Test engine creation helper"""
        engine = create_accessibility_engine(
            user_id="test",
            mode=AccessibilityMode.SCREEN_READER,
            db_path=temp_db
        )
        
        assert engine.profile is not None
        assert engine.profile.mode == AccessibilityMode.SCREEN_READER
        
    def test_available_modes(self, accessibility_engine):
        """Test getting available modes"""
        modes = accessibility_engine.get_available_modes()
        
        assert len(modes) == len(AccessibilityMode)
        
        for mode_info in modes:
            assert "mode" in mode_info
            assert "name" in mode_info
            assert "description" in mode_info


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
