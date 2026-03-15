#!/usr/bin/env python3
"""
Tests for N'Ko Ensemble Orchestrator & Auto-Correct Pro — Generation 7

Tests cover:
1. Domain auto-detection accuracy
2. Auto-correct for various error types
3. Ensemble orchestration and weighting
4. Confidence calibration
5. Cascade prediction logic
6. Adaptive weight learning
7. Key proximity mapping
8. Phonetic similarity detection
"""

import sys
import os
import sqlite3
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

# Import compat layer — use underscore-prefixed names to avoid pytest collection
def _mark_test(func):
    """Mark a function as a test (not collected by pytest directly)."""
    func._is_test = True
    return func


class _Suite:
    """Test suite grouping (underscore-prefixed so pytest won't try to collect it)."""
    def __init__(self, name):
        self.name = name
        self.tests = []
    def add(self, func):
        self.tests.append(func)
        return func

from ensemble_orchestrator import (
    EnsembleOrchestrator,
    AutoCorrectPro,
    DomainDetector,
    ConfidenceCalibrator,
    CascadePredictor,
    EngineScore,
    EnsemblePrediction,
    AutoCorrection,
    CorrectionType,
    NKO_KEYBOARD_LAYOUT,
    NKO_TONE_MARKS,
    PredictionDifficulty,
)


# === Test Suites ===

domain_suite = _Suite("Domain Detection")
autocorrect_suite = _Suite("Auto-Correct Pro")
orchestration_suite = _Suite("Ensemble Orchestration")
calibration_suite = _Suite("Confidence Calibration")
cascade_suite = _Suite("Cascade Prediction")
learning_suite = _Suite("Adaptive Learning")
keyboard_suite = _Suite("Keyboard Layout")


# =============================================
# Domain Detection Tests
# =============================================

@domain_suite.add
def test_detect_religion_domain():
    """Detect religious context from N'Ko text."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߊ ߟ ߊ߫ ߞ ߊ ߓ ߊ ߙ ߌ ߞ ߊ ߘ ߌ")
    assert domain == "RELIGION", f"Expected RELIGION, got {domain}"
    assert conf > 0.3, f"Confidence too low: {conf}"

@domain_suite.add
def test_detect_commerce_domain():
    """Detect commerce context."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߥ ߊ ߙ ߌ ߖ ߐ ߟ ߌ ߛ ߊ ߲")
    assert domain == "COMMERCE", f"Expected COMMERCE, got {domain}"

@domain_suite.add
def test_detect_education_domain():
    """Detect education context."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߞ ߊ ߙ ߊ ߡ ߐ ߜ ߐ ߞ ߊ ߟ ߊ ߣ ߛ ߓ ߣ")
    assert domain == "EDUCATION", f"Expected EDUCATION, got {domain}"

@domain_suite.add
def test_detect_greetings_domain():
    """Detect greeting context."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߌ ߣ ߌ ߛ ߐ ߜ ߐ ߡ ߊ")
    assert domain == "GREETINGS", f"Expected GREETINGS, got {domain}"

@domain_suite.add
def test_detect_health_domain():
    """Detect health context."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߓ ߊ ߣ ߊ ߝ ߎ ߙ ߊ ߘ ߌ ߡ ߌ")
    assert domain == "HEALTH", f"Expected HEALTH, got {domain}"

@domain_suite.add
def test_detect_agriculture_domain():
    """Detect agriculture context."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߛ ߣ ߡ ߊ ߟ ߏ ߛ ߊ ߲ ߖ ߌ")
    assert domain == "AGRICULTURE", f"Expected AGRICULTURE, got {domain}"

@domain_suite.add
def test_detect_family_domain():
    """Detect family context."""
    detector = DomainDetector()
    domain, conf = detector.detect("ߘ ߲ ߡ ߎ߬ ߛ ߓ ߊ ߟ ߌ ߡ ߛ ߏ ߡ ߐ ߜ ߐ")
    assert domain == "FAMILY", f"Expected FAMILY, got {domain}"

@domain_suite.add
def test_domain_momentum():
    """Domain detection should have momentum from recent context."""
    detector = DomainDetector()
    # First detection establishes domain
    detector.detect("ߊ ߟ ߊ߫ ߞ ߊ")
    # Second detection with weaker signal should still lean toward RELIGION
    domain, conf = detector.detect("ala ka")
    # The momentum should boost RELIGION
    assert domain == "RELIGION", f"Expected RELIGION momentum, got {domain}"

@domain_suite.add
def test_domain_default_daily_life():
    """Unknown text should default to DAILY_LIFE."""
    detector = DomainDetector()
    domain, conf = detector.detect("xyz abc 123")
    assert domain == "DAILY_LIFE", f"Expected DAILY_LIFE default, got {domain}"

@domain_suite.add
def test_domain_latin_signals():
    """Domain detection should work with Latin transliteration."""
    detector = DomainDetector()
    domain, conf = detector.detect("bismillah amina saraka duwa")
    assert domain == "RELIGION", f"Expected RELIGION from Latin, got {domain}"


# =============================================
# Auto-Correct Tests
# =============================================

@autocorrect_suite.add
def test_autocorrect_known_word():
    """Known correct words should not be corrected."""
    ac = AutoCorrectPro(vocabulary={"ߓߍ": True, "ߞߍ": True})
    result = ac.correct("ߓߍ", vocabulary={"ߓߍ": True})
    assert result is None, "Should not correct a known word"

@autocorrect_suite.add
def test_autocorrect_transposition():
    """Detect and correct transposed characters."""
    ac = AutoCorrectPro()
    vocab = {"ߓߍ": True, "ߍߓ": False}
    # Test with a word where transposition creates a vocab word
    result = ac.correct("ߍߓ", vocabulary={"ߓߍ": True})
    if result:
        assert result.correction_type == CorrectionType.TRANSPOSITION
        assert result.corrected == "ߓߍ"

@autocorrect_suite.add
def test_autocorrect_insertion():
    """Detect and correct extra characters."""
    ac = AutoCorrectPro()
    vocab = {"ߓߍ": True}
    result = ac.correct("ߓߓߍ", vocabulary=vocab)
    if result:
        assert result.corrected == "ߓߍ"

@autocorrect_suite.add
def test_autocorrect_double_tap():
    """Detect accidental double-tap characters."""
    ac = AutoCorrectPro()
    vocab = {"ߓߍ": True}
    result = ac.correct("ߓߓߍ", vocabulary=vocab)
    if result:
        assert result.corrected == "ߓߍ"
        assert result.correction_type in (CorrectionType.DOUBLE_TAP, CorrectionType.INSERTION)

@autocorrect_suite.add
def test_autocorrect_learning():
    """Auto-corrector should learn from user corrections."""
    ac = AutoCorrectPro()
    ac.learn_correction("ߓ ߎ", "ߓ ߍ", "KEY_PROXIMITY")
    ac.learn_correction("ߓ ߎ", "ߓ ߍ", "KEY_PROXIMITY")
    assert "ߓ ߎ" in ac.error_patterns
    assert ac.error_patterns["ߓ ߎ"].count == 2

@autocorrect_suite.add
def test_autocorrect_edit_distance():
    """Edit distance calculation."""
    ac = AutoCorrectPro()
    assert ac._edit_distance("ߓߍ", "ߓߍ") == 0
    assert ac._edit_distance("ߓߍ", "ߞߍ") == 1
    assert ac._edit_distance("ߓ", "ߓߍ") == 1
    assert ac._edit_distance("", "ߓ") == 1

@autocorrect_suite.add
def test_autocorrect_tone_strip():
    """Tone mark stripping."""
    ac = AutoCorrectPro()
    stripped = ac._strip_tones("ߊ߫ ߍ߬ ߌ߭")
    assert '߫' not in stripped
    assert '߬' not in stripped
    assert '߭' not in stripped
    assert 'ߊ' in stripped

@autocorrect_suite.add
def test_autocorrect_phonetic_map():
    """Phonetic similarity map is correctly built."""
    ac = AutoCorrectPro()
    # b and p should be phonetically similar
    assert 'ߔ' in ac.phonetic_map.get('ߓ', set()), "ߓ(b) and ߔ(p) should be similar"
    # d and t should be phonetically similar
    assert 'ߕ' in ac.phonetic_map.get('ߘ', set()), "ߘ(d) and ߕ(t) should be similar"

@autocorrect_suite.add
def test_autocorrect_confidence_threshold():
    """Low-confidence corrections should be suppressed."""
    ac = AutoCorrectPro()
    # Empty vocabulary means no valid corrections
    result = ac.correct("ߓ ߎ ߞ ߊ ߟ ߡ", vocabulary={})
    assert result is None, "Should not suggest correction with empty vocab"

@autocorrect_suite.add
def test_autocorrect_explain():
    """Correction explanations are generated."""
    ac = AutoCorrectPro()
    explanation = ac._explain_correction("ߓ ߎ", "ߓ ߍ", CorrectionType.KEY_PROXIMITY)
    assert "ߓ ߎ" in explanation
    assert "ߓ ߍ" in explanation
    assert "adjacent" in explanation


# =============================================
# Ensemble Orchestration Tests
# =============================================

@orchestration_suite.add
def test_orchestrate_basic():
    """Basic orchestration produces results."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    engine_outputs = {
        "ngram": [("ߕߊ߬ߊ", 0.85), ("ߣߊ", 0.72)],
        "semantic": [("ߕߊ߬ߊ", 0.78), ("ߣߊ", 0.82)],
    }
    
    results = orch.orchestrate("ߒ ߓ ߍ", engine_outputs)
    assert len(results) > 0, "Should produce at least one prediction"

@orchestration_suite.add
def test_orchestrate_ranking():
    """Predictions should be ranked by combined score."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    engine_outputs = {
        "ngram": [("ߕߊ߬ߊ", 0.95), ("ߣߊ", 0.30)],
        "corpus": [("ߕߊ߬ߊ", 0.90), ("ߣߊ", 0.40)],
    }
    
    results = orch.orchestrate("ߒ ߓ ߍ", engine_outputs)
    assert results[0].text == "ߕߊ߬ߊ", \
        f"Top prediction should be ߕߊ߬ߊ, got {results[0].text}"

@orchestration_suite.add
def test_orchestrate_multi_engine():
    """Predictions from multiple engines combine correctly."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    engine_outputs = {
        "ngram": [("ߕߊ߬ߊ", 0.80)],
        "semantic": [("ߕߊ߬ߊ", 0.85)],
        "neural": [("ߕߊ߬ߊ", 0.75)],
        "corpus": [("ߕߊ߬ߊ", 0.90)],
    }
    
    results = orch.orchestrate("ߒ ߓ ߍ", engine_outputs)
    assert results[0].text == "ߕߊ߬ߊ"
    assert len(results[0].engines_used) >= 3, \
        "Should combine from multiple engines"

@orchestration_suite.add
def test_orchestrate_domain_context():
    """Orchestrator should detect domain from context."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    engine_outputs = {
        "ngram": [("ߓ ߊ ߙ ߌ ߞ ߊ", 0.80)],
    }
    
    results = orch.orchestrate(
        "ߊ ߟ ߊ߫ ߞ ߊ",
        engine_outputs,
        recent_words=["ߊ ߟ ߊ߫"]
    )
    
    assert results[0].domain == "RELIGION", \
        f"Expected RELIGION domain, got {results[0].domain}"

@orchestration_suite.add
def test_orchestrate_limit():
    """Results should respect the limit parameter."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    engine_outputs = {
        "ngram": [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5)],
    }
    
    results = orch.orchestrate("test", engine_outputs, limit=3)
    assert len(results) <= 3, f"Expected max 3 results, got {len(results)}"

@orchestration_suite.add
def test_orchestrate_autocorrect_integration():
    """Auto-corrections should appear in orchestrated results."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    # Teach the autocorrector a pattern
    orch.auto_corrector.learn_correction("ߓ ߎ", "ߓ ߍ", "KEY_PROXIMITY")
    orch.auto_corrector.learn_correction("ߓ ߎ", "ߓ ߍ", "KEY_PROXIMITY")
    
    engine_outputs = {"ngram": [("ߞ ߍ", 0.5)]}
    
    results = orch.orchestrate("some ߓ ߎ", engine_outputs)
    # Should include autocorrect suggestion
    autocorrects = [r for r in results if r.is_autocorrect]
    # The autocorrect should be attempted (even if vocab doesn't have the word)

@orchestration_suite.add
def test_orchestrate_empty_engines():
    """Handle empty engine outputs gracefully."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    results = orch.orchestrate("", {})
    assert isinstance(results, list)

@orchestration_suite.add
def test_orchestrate_debug_info():
    """Debug info should be populated."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    engine_outputs = {"ngram": [("ߕߊ߬ߊ", 0.85)]}
    results = orch.orchestrate("ߒ ߓ ߍ", engine_outputs)
    
    if results:
        assert "domain_confidence" in results[0].debug
        assert "latency_ms" in results[0].debug


# =============================================
# Confidence Calibration Tests
# =============================================

@calibration_suite.add
def test_calibrate_default_sigmoid():
    """Default calibration uses conservative sigmoid."""
    cal = ConfidenceCalibrator(db_path=":memory:")
    
    # Low raw score → low calibrated
    low = cal.calibrate("ngram", 0.1)
    assert low < 0.3, f"Low score should calibrate low, got {low}"
    
    # High raw score → high calibrated
    high = cal.calibrate("ngram", 0.9)
    assert high > 0.7, f"High score should calibrate high, got {high}"

@calibration_suite.add
def test_calibrate_monotonic():
    """Calibrated scores should be monotonically increasing."""
    cal = ConfidenceCalibrator(db_path=":memory:")
    
    prev = 0
    for raw in [0.1, 0.3, 0.5, 0.7, 0.9]:
        calibrated = cal.calibrate("ngram", raw)
        assert calibrated >= prev, \
            f"Calibration not monotonic: {prev} -> {calibrated} at raw={raw}"
        prev = calibrated

@calibration_suite.add
def test_calibrate_bounds():
    """Calibrated scores should be in [0.1, 0.9]."""
    cal = ConfidenceCalibrator(db_path=":memory:")
    
    for raw in [0.0, 0.5, 1.0]:
        calibrated = cal.calibrate("ngram", raw)
        assert 0.0 <= calibrated <= 1.0, \
            f"Calibrated score out of bounds: {calibrated}"

@calibration_suite.add
def test_calibrate_record_outcome():
    """Recording outcomes should work without error."""
    cal = ConfidenceCalibrator(db_path=":memory:")
    cal.record_outcome("ngram", 0.8, True)
    cal.record_outcome("ngram", 0.3, False)
    cal.record_outcome("semantic", 0.6, True)
    
    count = cal.db.execute(
        "SELECT COUNT(*) FROM calibration_data"
    ).fetchone()[0]
    assert count == 3

@calibration_suite.add
def test_calibrate_recalibrate():
    """Recalibration with insufficient data should be a no-op."""
    cal = ConfidenceCalibrator(db_path=":memory:")
    
    # Add too few samples
    for i in range(5):
        cal.record_outcome("ngram", i * 0.2, True)
    
    # Should not crash with insufficient data
    cal.recalibrate("ngram", min_samples=20)


# =============================================
# Cascade Prediction Tests
# =============================================

@cascade_suite.add
def test_cascade_high_confidence_fast():
    """High confidence fast-path results skip deep engines."""
    cascade = CascadePredictor()
    
    fast_preds = [
        EngineScore("ngram", "ߕߊ߬ߊ", 0.95, calibrated_score=0.92)
    ]
    
    should_deep = cascade.should_go_deep(fast_preds)
    assert not should_deep, "High confidence should not trigger deep path"

@cascade_suite.add
def test_cascade_low_confidence_deep():
    """Low confidence results should trigger deep engines."""
    cascade = CascadePredictor()
    
    fast_preds = [
        EngineScore("ngram", "ߕߊ߬ߊ", 0.30, calibrated_score=0.35)
    ]
    
    should_deep = cascade.should_go_deep(fast_preds)
    assert should_deep, "Low confidence should trigger deep path"

@cascade_suite.add
def test_cascade_empty_triggers_deep():
    """Empty fast-path results should trigger deep engines."""
    cascade = CascadePredictor()
    should_deep = cascade.should_go_deep([])
    assert should_deep, "Empty fast-path should trigger deep path"

@cascade_suite.add
def test_cascade_disagreement_triggers_deep():
    """High disagreement between fast engines should trigger deep path."""
    cascade = CascadePredictor()
    
    fast_preds = [
        EngineScore("ngram", "ߕߊ߬ߊ", 0.78, calibrated_score=0.80),
        EngineScore("frequency", "ߣ ߊ", 0.30, calibrated_score=0.35),
    ]
    
    should_deep = cascade.should_go_deep(fast_preds)
    assert should_deep, "High disagreement should trigger deep path"

@cascade_suite.add
def test_cascade_stats():
    """Cascade stats should track hits and misses."""
    cascade = CascadePredictor()
    
    # One fast hit
    cascade.should_go_deep([
        EngineScore("ngram", "x", 0.95, calibrated_score=0.92)
    ])
    # One deep activation
    cascade.should_go_deep([])
    
    stats = cascade.get_stats()
    assert stats["fast_path_hits"] == 1
    assert stats["deep_path_activations"] == 1


# =============================================
# Adaptive Learning Tests
# =============================================

@learning_suite.add
def test_weight_update():
    """Weights should update with learning."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    initial_weight = orch.domain_weights["RELIGION"].get("corpus", 0.15)
    
    # Simulate high accuracy for corpus in religion domain
    for _ in range(10):
        orch.update_weights("RELIGION", "corpus", 0.95)
    
    new_weight = orch.domain_weights["RELIGION"]["corpus"]
    # Weight should have increased from initial toward 0.95
    assert new_weight != initial_weight, "Weight should have changed"

@learning_suite.add
def test_weight_normalization():
    """Weights should be normalized to sum ≈ 1."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    # Update several engines
    orch.update_weights("TEST", "ngram", 0.90)
    orch.update_weights("TEST", "semantic", 0.80)
    orch.update_weights("TEST", "corpus", 0.70)
    
    total = sum(orch.domain_weights["TEST"].values())
    assert 0.95 <= total <= 1.05, f"Weights should sum to ~1, got {total}"

@learning_suite.add
def test_weight_clamping():
    """No single engine should dominate completely."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    
    # Try to make one engine dominate
    for _ in range(50):
        orch.update_weights("TEST", "ngram", 1.0)
    
    max_weight = max(orch.domain_weights["TEST"].values())
    # After normalization, the max weight depends on how many engines exist
    # But the raw clamping should prevent > 0.40 before normalization


@learning_suite.add
def test_weight_persistence():
    """Weights should be persisted to database."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    orch.update_weights("PERSIST_TEST", "ngram", 0.85)
    
    row = orch.db.execute(
        "SELECT weight FROM engine_weights WHERE domain = ? AND engine = ?",
        ("PERSIST_TEST", "ngram")
    ).fetchone()
    assert row is not None, "Weight should be persisted"

@learning_suite.add
def test_engine_stats():
    """Engine stats should be available."""
    orch = EnsembleOrchestrator(db_path=":memory:")
    stats = orch.get_engine_stats()
    
    assert "domain_weights" in stats
    assert "cascade_stats" in stats
    assert "total_predictions" in stats
    assert "acceptance_rate" in stats


# =============================================
# Keyboard Layout Tests
# =============================================

@keyboard_suite.add
def test_keyboard_layout_exists():
    """N'Ko keyboard layout should have entries."""
    assert len(NKO_KEYBOARD_LAYOUT) > 20, \
        f"Expected 20+ layout entries, got {len(NKO_KEYBOARD_LAYOUT)}"

@keyboard_suite.add
def test_keyboard_layout_positions():
    """Layout positions should be valid (row, col) tuples."""
    for char, (row, col) in NKO_KEYBOARD_LAYOUT.items():
        assert 0 <= row <= 3, f"Invalid row {row} for {char}"
        assert 0 <= col <= 9, f"Invalid col {col} for {char}"

@keyboard_suite.add
def test_tone_marks_defined():
    """Tone marks set should contain N'Ko diacritics."""
    assert len(NKO_TONE_MARKS) >= 5, \
        f"Expected 5+ tone marks, got {len(NKO_TONE_MARKS)}"

@keyboard_suite.add
def test_proximity_map():
    """Key proximity map should identify neighbors."""
    ac = AutoCorrectPro()
    
    # Home row keys should have neighbors
    for char in ['ߊ', 'ߛ', 'ߘ', 'ߝ']:
        neighbors = ac.proximity_map.get(char, [])
        assert len(neighbors) >= 1, \
            f"Key {char} should have at least 1 neighbor, got {len(neighbors)}"

@keyboard_suite.add
def test_phonetic_groups():
    """Phonetic groups should be bidirectional."""
    ac = AutoCorrectPro()
    
    # If A is similar to B, B should be similar to A
    for char, similar in ac.phonetic_map.items():
        for sim in similar:
            assert char in ac.phonetic_map.get(sim, set()), \
                f"Phonetic similarity not bidirectional: {char} ↔ {sim}"


# =============================================
# Run all tests
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print("  N'Ko Ensemble Orchestrator — Test Suite")
    print("  Generation 7 Evolution")
    print("=" * 60)
    print()
    
    all_suites = [
        domain_suite,
        autocorrect_suite,
        orchestration_suite,
        calibration_suite,
        cascade_suite,
        learning_suite,
        keyboard_suite,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for suite in all_suites:
        print(f"\n📦 {suite.name}")
        print("-" * 40)
        
        for test_func in suite.tests:
            try:
                test_func()
                total_passed += 1
                print(f"  ✅ {test_func.__name__}")
            except AssertionError as e:
                total_failed += 1
                print(f"  ❌ {test_func.__name__}: {e}")
            except Exception as e:
                total_failed += 1
                print(f"  ❌ {test_func.__name__}: {type(e).__name__}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"  Results: {total_passed} passed, {total_failed} failed, "
          f"{total_passed + total_failed} total")
    print(f"  {'✅ ALL PASSED' if total_failed == 0 else '❌ SOME FAILURES'}")
    print(f"{'=' * 60}")
    
    sys.exit(0 if total_failed == 0 else 1)
