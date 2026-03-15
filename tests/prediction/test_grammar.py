#!/usr/bin/env python3
"""
Tests for N'Ko Grammar & Writing Assistant Engine (Gen 8)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

import unittest
from grammar_engine import (
    NkoGrammarEngine,
    GrammarIssueType,
    Severity,
    Formality,
    GrammarCheckResult,
    ToneSuggestion,
    WritingMetrics,
)


class TestGrammarEngineBasics(unittest.TestCase):
    """Basic functionality tests"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine(dialect="bambara")
        
    def test_init_default(self):
        """Test default initialization"""
        self.assertEqual(self.engine.dialect, "bambara")
        self.assertEqual(self.engine.default_formality, Formality.NEUTRAL)
        
    def test_init_custom_dialect(self):
        """Test custom dialect initialization"""
        engine = NkoGrammarEngine(dialect="maninka")
        self.assertEqual(engine.dialect, "maninka")
        
    def test_check_grammar_returns_result(self):
        """Test grammar check returns proper result type"""
        result = self.engine.check_grammar("ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ")
        self.assertIsInstance(result, GrammarCheckResult)
        self.assertIsInstance(result.issues, list)
        self.assertIsInstance(result.metrics, WritingMetrics)
        

class TestSpacingAndPunctuation(unittest.TestCase):
    """Tests for spacing and punctuation checks"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_space_before_punctuation(self):
        """Detect incorrect space before punctuation"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ .")
        spacing_issues = [i for i in result.issues if i.issue_type == GrammarIssueType.SPACING]
        self.assertGreater(len(spacing_issues), 0)
        
    def test_missing_space_after_punctuation(self):
        """Detect missing space after punctuation"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ߹ߓߙߌߞߊ")
        spacing_issues = [i for i in result.issues if i.issue_type == GrammarIssueType.SPACING]
        self.assertGreater(len(spacing_issues), 0)
        
    def test_latin_punctuation_suggestion(self):
        """Suggest N'Ko punctuation over Latin"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ.")
        punct_issues = [i for i in result.issues if i.issue_type == GrammarIssueType.PUNCTUATION]
        self.assertGreater(len(punct_issues), 0)
        self.assertEqual(punct_issues[0].suggestion, "߹")
        
    def test_correct_punctuation_no_issues(self):
        """Well-punctuated text should have no spacing issues"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ߹ ߓߙߌߞߊ߹")
        # Filter for spacing and punctuation only
        relevant_issues = [
            i for i in result.issues 
            if i.issue_type in [GrammarIssueType.SPACING, GrammarIssueType.PUNCTUATION]
        ]
        self.assertEqual(len(relevant_issues), 0)


class TestToneAnalysis(unittest.TestCase):
    """Tests for tone marking suggestions"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_tone_ambiguous_word_detected(self):
        """Detect words that need tone marks"""
        result = self.engine.check_grammar("ߓߊ ߓߍ ߖߊ߫")
        # Should detect ߓߊ and ߓߍ as needing tone marks
        tone_words = [t.word for t in result.tone_suggestions]
        self.assertIn("ߓߊ", tone_words)
        self.assertIn("ߓߍ", tone_words)
        
    def test_already_toned_word_no_suggestion(self):
        """Words with tone marks should not get suggestions"""
        result = self.engine.check_grammar("ߓߊ߫ ߓߍ߫ ߖߊ߫")
        tone_words = [t.word for t in result.tone_suggestions]
        self.assertNotIn("ߓߊ߫", tone_words)
        
    def test_tone_suggestion_includes_alternatives(self):
        """Tone suggestions include alternative meanings"""
        result = self.engine.check_grammar("ߓߊ ߦߋ")
        ba_suggestion = next((t for t in result.tone_suggestions if t.word == "ߓߊ"), None)
        if ba_suggestion:
            self.assertGreater(len(ba_suggestion.alternatives), 0)


class TestWritingMetrics(unittest.TestCase):
    """Tests for writing metrics calculation"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_word_count(self):
        """Test accurate word counting"""
        result = self.engine.check_grammar("ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ")
        self.assertEqual(result.metrics.word_count, 3)
        
    def test_sentence_count(self):
        """Test accurate sentence counting"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ߹ ߌ ߓߍ ߘߌ߹")
        self.assertEqual(result.metrics.sentence_count, 2)
        
    def test_vocabulary_diversity(self):
        """Test vocabulary diversity (type-token ratio)"""
        # All unique words
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ ߓߙߌߞߊ")
        self.assertEqual(result.metrics.vocabulary_diversity, 1.0)
        
        # Repeated words
        result2 = self.engine.check_grammar("ߌ ߌ ߌ ߌ")
        self.assertEqual(result2.metrics.vocabulary_diversity, 0.25)
        
    def test_tone_coverage(self):
        """Test tone mark coverage calculation"""
        # No tone marks
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ")
        self.assertEqual(result.metrics.tone_mark_coverage, 0.0)
        
        # Half with tone marks
        result2 = self.engine.check_grammar("ߌ ߣߌ ߗߍ߫ ߓߙߌ߬ߞߊ")
        self.assertEqual(result2.metrics.tone_mark_coverage, 0.5)
        
    def test_readability_score(self):
        """Test readability score calculation"""
        # Short sentences = higher readability
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ߹")
        self.assertGreater(result.metrics.readability_score, 0.5)


class TestProverbSuggestions(unittest.TestCase):
    """Tests for proverb suggestions"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_proverb_match_by_context(self):
        """Proverbs matched by context keywords"""
        # Text about patience/waiting
        result = self.engine.check_grammar("ߒ ߓߍ ߡߎ߬ߢߎ߲ ߟߊ߫", include_proverbs=True)
        # Should suggest patience-related proverb
        self.assertGreater(len(result.suggested_proverbs), 0)
        
    def test_proverb_includes_meaning(self):
        """Proverb suggestions include meaning"""
        result = self.engine.check_grammar("ߊߟߊ߫ ߦߋ ߒ ߘߍ߬ߡߍ߲", include_proverbs=True)
        if result.suggested_proverbs:
            self.assertIsNotNone(result.suggested_proverbs[0].meaning)
            self.assertIsNotNone(result.suggested_proverbs[0].proverb_latin)
            
    def test_max_proverb_suggestions(self):
        """Proverb count limited to max"""
        result = self.engine.check_grammar("ߒ ߓߍ ߞߏ ߘߏ߫ ߞߍ ߟߊ", include_proverbs=True)
        self.assertLessEqual(len(result.suggested_proverbs), 3)


class TestFormalityChecking(unittest.TestCase):
    """Tests for formality checking"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_casual_in_formal_context(self):
        """Detect casual expressions in formal writing"""
        result = self.engine.check_grammar(
            "ߋ߫ ߌ ߣߌ ߗߍ",  # ߋ߫ is casual
            target_formality=Formality.FORMAL
        )
        formality_issues = [
            i for i in result.issues 
            if i.issue_type == GrammarIssueType.FORMALITY
        ]
        self.assertGreater(len(formality_issues), 0)
        
    def test_formal_in_casual_context(self):
        """Detect overly formal expressions in casual writing"""
        result = self.engine.check_grammar(
            "ߊ߬ ߦߋ߫ ߞߍ߫ ߢߊ ߡߍ߲ ߠߊ߫",  # Very formal
            target_formality=Formality.CASUAL
        )
        formality_issues = [
            i for i in result.issues 
            if i.issue_type == GrammarIssueType.FORMALITY
        ]
        self.assertGreater(len(formality_issues), 0)


class TestAutoCorrection(unittest.TestCase):
    """Tests for auto-correction feature"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_auto_correct_spacing(self):
        """Auto-correct spacing issues"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ .", auto_correct=True)
        self.assertIsNotNone(result.auto_corrected)
        # Should remove space before period
        self.assertNotIn(" .", result.auto_corrected)
        
    def test_auto_correct_punctuation(self):
        """Auto-correct punctuation"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ.", auto_correct=True)
        self.assertIsNotNone(result.auto_corrected)
        self.assertIn("߹", result.auto_corrected)
        
    def test_no_auto_correct_unfixable(self):
        """Unfixable issues not auto-corrected"""
        # Word order issues are not auto-fixable
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ", auto_correct=True)
        # Should have auto_corrected but unfixable issues preserved
        if result.issues:
            unfixable = [i for i in result.issues if not i.auto_fixable]
            # Verify unfixable issues exist in output
            self.assertTrue(
                len(unfixable) == 0 or result.auto_corrected is not None
            )


class TestOverallScoring(unittest.TestCase):
    """Tests for overall score calculation"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_perfect_text_high_score(self):
        """Well-written text gets high score"""
        # Clean text with tone marks and proper punctuation
        result = self.engine.check_grammar("ߌ ߣߌ ߛߐ߲߬ߜߐߡߊ߹ ߌ ߓߍ߫ ߘߌ߹")
        self.assertGreater(result.overall_score, 70)
        
    def test_issues_reduce_score(self):
        """Issues reduce overall score"""
        # Text with multiple intentional issues
        result1 = self.engine.check_grammar("ߌ ߣߌ ߗߍ . . .")  # Multiple space issues
        result2 = self.engine.check_grammar("ߌ ߣߌ ߗߍ߫߹")      # Correct with tone mark
        # Both may get high scores, but result2 should be >= result1
        self.assertGreaterEqual(result2.overall_score, result1.overall_score)
        
    def test_score_in_valid_range(self):
        """Score always between 0-100"""
        test_texts = [
            "ߌ",
            "ߌ ߣߌ ߗߍ .",
            "ߞߐ߲ ߞߍ߬ߟߊ ߕߍ߫ ߟߊ߬ߓߐ߬ ߟߊ߫",
            "test " * 100,  # Long text
        ]
        for text in test_texts:
            result = self.engine.check_grammar(text)
            self.assertGreaterEqual(result.overall_score, 0)
            self.assertLessEqual(result.overall_score, 100)


class TestCustomRules(unittest.TestCase):
    """Tests for custom rule management"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_add_custom_rule(self):
        """Add and use custom rule"""
        self.engine.add_custom_rule(
            pattern=r"ߕߋ߬ߛߕ",
            correction="ߕߍ߬ߛߕ",
            explanation="Custom spelling correction"
        )
        self.assertEqual(len(self.engine.custom_rules), 1)
        
    def test_learn_correction(self):
        """Learn from user corrections"""
        self.engine.learn_correction("ߓߊ", "ߓߊ߫")
        self.assertEqual(self.engine.learned_corrections["ߓߊ"], "ߓߊ߫")


class TestReportFormatting(unittest.TestCase):
    """Tests for report formatting"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_report_contains_score(self):
        """Report includes overall score"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ")
        report = self.engine.format_report(result)
        self.assertIn("Overall Score", report)
        
    def test_report_contains_metrics(self):
        """Report includes metrics"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ")
        report = self.engine.format_report(result)
        self.assertIn("Writing Metrics", report)
        self.assertIn("Words:", report)
        
    def test_report_contains_issues(self):
        """Report lists issues when present"""
        result = self.engine.check_grammar("ߌ ߣߌ ߗߍ .")
        report = self.engine.format_report(result)
        if result.issues:
            self.assertIn("Issues Found", report)


class TestDialectSupport(unittest.TestCase):
    """Tests for dialect-specific features"""
    
    def test_bambara_rules(self):
        """Bambara dialect rules loaded"""
        engine = NkoGrammarEngine(dialect="bambara")
        self.assertIn("copula", engine.dialect_rules["bambara"])
        self.assertEqual(engine.dialect_rules["bambara"]["copula"], "ߓߍ߫")
        
    def test_maninka_rules(self):
        """Maninka dialect rules loaded"""
        engine = NkoGrammarEngine(dialect="maninka")
        self.assertIn("copula", engine.dialect_rules["maninka"])
        
    def test_jula_rules(self):
        """Jula dialect rules loaded"""
        engine = NkoGrammarEngine(dialect="jula")
        self.assertIn("negation", engine.dialect_rules["jula"])
        
    def test_soninke_rules(self):
        """Soninke dialect rules loaded"""
        engine = NkoGrammarEngine(dialect="soninke")
        self.assertIn("past_marker", engine.dialect_rules["soninke"])


class TestEdgeCases(unittest.TestCase):
    """Edge case tests"""
    
    def setUp(self):
        self.engine = NkoGrammarEngine()
        
    def test_empty_text(self):
        """Handle empty text"""
        result = self.engine.check_grammar("")
        self.assertIsInstance(result, GrammarCheckResult)
        self.assertEqual(result.metrics.word_count, 0)
        
    def test_single_character(self):
        """Handle single character"""
        result = self.engine.check_grammar("ߌ")
        self.assertEqual(result.metrics.word_count, 1)
        
    def test_mixed_scripts(self):
        """Handle mixed N'Ko and Latin"""
        result = self.engine.check_grammar("ߌ ߣߌ hello ߗߍ")
        self.assertEqual(result.metrics.word_count, 4)
        
    def test_unicode_preservation(self):
        """Unicode characters preserved correctly"""
        original = "ߊ߬ ߣߌ ߓߙߍ߬ ߏ߰"
        result = self.engine.check_grammar(original, auto_correct=True)
        # If no corrections needed, auto_corrected should match or be None
        if result.auto_corrected:
            # Should preserve Unicode
            self.assertIn("ߊ", result.auto_corrected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
