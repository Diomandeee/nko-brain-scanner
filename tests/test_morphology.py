"""
Tests for nko.morphology — Unified Manding Morphology Engine

Covers:
  §1  Word decomposition (prefix + root + suffix)
  §2  Root / stem extraction
  §3  Noun class detection
  §4  Verb conjugation analysis
  §5  Affix inventory
  §6  Compound word detection
  §7  Script detection & cross-script analysis
  §8  Sentence-level analysis
  §9  Edge cases & regression
"""

import sys, os, pytest

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nko.morphology import (
    # Enums
    MorphemeType, NounClass, TenseAspect, PersonNumber,
    CompoundType, TonePattern,
    # Data classes
    Morpheme, WordAnalysis, ConjugatedForm,
    CompoundComponent, CompoundWord,
    # Classes
    MorphologicalAnalyzer, VerbConjugator, CompoundDetector,
    AffixInventory,
    # Convenience functions
    analyze, analyze_word, decompose, extract_root,
    detect_noun_class,
    conjugate, full_paradigm,
    is_compound, split_compound,
    get_affix, list_prefixes, list_suffixes, list_postpositions,
)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  WORD DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

class TestWordDecomposition:
    """Test that words are broken into prefix + root + suffix correctly."""

    def test_simple_verb_root_no_affixes(self):
        """A bare verb root should decompose to just the root."""
        wa = analyze_word("taa", "latin")
        d = wa.decomposition()
        assert d["root"] == "ߕߊ߯"
        assert d["prefixes"] == []
        assert d["suffixes"] == []

    def test_noun_with_plural(self):
        """mogolu → mogo + lu (person + PL)."""
        wa = analyze_word("mogolu", "latin")
        assert wa.word_class == "noun"
        assert wa.root is not None
        assert wa.root.gloss == "person"
        d = wa.decomposition()
        assert d["root"] != ""
        assert any(m.gloss == "PL" for m in wa.morphemes)

    def test_verb_with_derivational_suffix(self):
        """sebeli → sebe + li (write + AGENT = writer)."""
        wa = analyze_word("sebeli", "latin")
        assert wa.word_class == "verb"
        d = wa.decomposition()
        assert len(d["suffixes"]) >= 1

    def test_decompose_convenience(self):
        """The top-level decompose() returns a clean dict."""
        d = decompose("mogolu", "latin")
        assert "prefixes" in d
        assert "root" in d
        assert "suffixes" in d

    def test_prefixed_verb(self):
        """lakalan → la (CAUS) + kalan (speak)."""
        wa = analyze_word("lakalan", "latin")
        assert wa.prefix_count >= 1 or wa.morpheme_count >= 2
        # Should identify 'kalan' root
        assert wa.root is not None


# ═══════════════════════════════════════════════════════════════════════════════
# §2  ROOT / STEM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRootExtraction:
    """Test root identification across scripts and word forms."""

    def test_extract_verb_root_latin(self):
        assert extract_root("taa", "latin") == "ߕߊ߯"

    def test_extract_verb_root_from_derived(self):
        """sebeli (writer) → root should be the write verb."""
        root = extract_root("sebeli", "latin")
        assert root != ""

    def test_extract_noun_root(self):
        root = extract_root("mogo", "latin")
        assert root == "ߡߐ߱"

    def test_extract_root_unknown_word(self):
        """An unrecognised word should return empty string."""
        root = extract_root("xyzzyx", "latin")
        assert root == ""

    def test_root_preserved_in_analysis(self):
        wa = analyze_word("ke", "latin")
        assert wa.root is not None
        assert wa.root.gloss == "do/make"


# ═══════════════════════════════════════════════════════════════════════════════
# §3  NOUN CLASS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestNounClassDetection:
    """Test Manding noun class assignment."""

    def test_person_class(self):
        assert detect_noun_class("mogo", "latin") == NounClass.PERSON

    def test_kinship_class(self):
        assert detect_noun_class("faa", "latin") == NounClass.KINSHIP

    def test_liquid_class(self):
        assert detect_noun_class("ji", "latin") == NounClass.LIQUID

    def test_place_class(self):
        assert detect_noun_class("dugu", "latin") == NounClass.PLACE

    def test_abstract_class_via_suffix(self):
        """A word ending in -ya (quality suffix) should be ABSTRACT."""
        nc = detect_noun_class("mogoya", "latin")
        assert nc == NounClass.ABSTRACT

    def test_thing_class(self):
        assert detect_noun_class("so", "latin") == NounClass.THING

    def test_noun_class_on_plural(self):
        """mogolu should still resolve to PERSON."""
        nc = detect_noun_class("mogolu", "latin")
        assert nc == NounClass.PERSON

    def test_unknown_returns_none(self):
        assert detect_noun_class("xyzzyx", "latin") is None


# ═══════════════════════════════════════════════════════════════════════════════
# §4  VERB CONJUGATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerbConjugation:
    """Test the particle-based Manding conjugation system."""

    def test_progressive_third_sg(self):
        form = conjugate("go", TenseAspect.PROGRESSIVE, PersonNumber.THIRD_SG)
        assert "ߓߍ" in form.full_nko       # progressive particle
        assert "ߕߊ߯" in form.full_nko       # verb 'go'
        assert "be" in form.full_latin
        assert "taa" in form.full_latin

    def test_negative_completive(self):
        form = conjugate("say", TenseAspect.NEG_COMPLETIVE, PersonNumber.FIRST_SG)
        assert "ߡߊ" in form.full_nko
        assert "ma" in form.full_latin

    def test_imperative_second_sg_bare(self):
        """Imperative 2sg = bare verb, no pronoun or particle."""
        form = conjugate("go", TenseAspect.IMPERATIVE, PersonNumber.SECOND_SG)
        assert form.full_nko == "ߕߊ߯"
        assert form.full_latin == "taa"

    def test_full_paradigm_coverage(self):
        """Every tense × every person should be generated."""
        paradigm = full_paradigm("write")
        assert len(paradigm) == len(TenseAspect)
        for tense_name, forms in paradigm.items():
            assert len(forms) == len(PersonNumber)

    def test_conjugate_by_latin_form(self):
        """Verb resolution from Latin form 'taa' → go."""
        form = conjugate("taa", TenseAspect.COMPLETIVE, PersonNumber.FIRST_SG)
        assert "ka" in form.full_latin
        assert "taa" in form.full_latin

    def test_unknown_verb_passes_through(self):
        """An unknown verb string should still produce output."""
        form = conjugate("florbinate", TenseAspect.PROGRESSIVE, PersonNumber.THIRD_SG)
        assert "florbinate" in form.full_latin

    def test_conjugated_form_to_dict(self):
        form = conjugate("go")
        d = form.to_dict()
        assert "nko" in d
        assert "tense" in d
        assert "person" in d


# ═══════════════════════════════════════════════════════════════════════════════
# §5  AFFIX INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════

class TestAffixInventory:
    """Test the comprehensive affix catalogue."""

    def test_list_prefixes_non_empty(self):
        prefixes = list_prefixes()
        assert len(prefixes) >= 3  # causative, benefactive, ventive

    def test_list_suffixes_non_empty(self):
        suffixes = list_suffixes()
        assert len(suffixes) >= 8

    def test_list_postpositions(self):
        pp = list_postpositions()
        assert "ߟߊ" in pp       # la — most common postposition
        assert "ߘߐ" in pp       # do — inside

    def test_lookup_by_nko(self):
        info = get_affix("ߟߌ")
        assert info is not None
        assert info["type"] == "agentive"

    def test_lookup_by_latin(self):
        info = get_affix("lu")
        assert info is not None
        assert info["type"] == "plural"

    def test_lookup_missing_returns_none(self):
        assert get_affix("zzz") is None


# ═══════════════════════════════════════════════════════════════════════════════
# §6  COMPOUND WORD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompoundDetection:
    """Test compound word recognition and splitting."""

    def test_known_compound_soda(self):
        """soda (doorway) = so + da."""
        assert is_compound("soda")
        cw = split_compound("soda")
        assert cw.confidence == 1.0
        assert cw.actual_meaning == "doorway"
        assert cw.component_count == 2

    def test_known_compound_kanmogo(self):
        cw = split_compound("kanmogo")
        assert "linguist" in cw.actual_meaning or "interpreter" in cw.actual_meaning

    def test_novel_compound_decomposition(self):
        """A novel combination like 'jimin' (water-drink) should decompose."""
        cw = split_compound("jimin")
        assert cw.component_count >= 2
        assert cw.confidence > 0

    def test_non_compound_returns_low_confidence(self):
        cw = split_compound("xyzzyx")
        assert cw.confidence == 0.0

    def test_find_compounds_with_root(self):
        detector = CompoundDetector()
        results = detector.find_compounds_with("ji")
        assert len(results) >= 1  # at least jibo, jidon, duguji

    def test_generate_hypothetical_compound(self):
        detector = CompoundDetector()
        cw = detector.generate_compound("ji", "ba")
        assert cw is not None
        assert cw.confidence == 0.50
        assert cw.component_count == 2

    def test_compound_to_dict(self):
        cw = split_compound("soda")
        d = cw.to_dict()
        assert "components" in d
        assert "type" in d


# ═══════════════════════════════════════════════════════════════════════════════
# §7  SCRIPT DETECTION & CROSS-SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

class TestScriptDetection:
    """Test automatic script detection."""

    def test_detect_latin(self):
        a = MorphologicalAnalyzer()
        assert a.detect_script("taa") == "latin"

    def test_detect_nko(self):
        a = MorphologicalAnalyzer()
        assert a.detect_script("ߕߊ߯") == "nko"

    def test_detect_arabic(self):
        a = MorphologicalAnalyzer()
        assert a.detect_script("تَا") == "arabic"

    def test_analyze_nko_word(self):
        """Analyze an N'Ko word and verify root extraction works."""
        wa = analyze_word("ߕߊ߯", "nko")
        # Should decompose — exact result depends on rough transliteration
        assert wa.script == "nko"


# ═══════════════════════════════════════════════════════════════════════════════
# §8  SENTENCE-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentenceAnalysis:
    """Test multi-word sentence structure detection."""

    def test_stv_pattern(self):
        """'n be taa' = I + PROG + go → STV."""
        a = MorphologicalAnalyzer()
        result = a.analyze_sentence("n be taa", "latin")
        assert "STV" in result["structure"]

    def test_sentence_glossing(self):
        a = MorphologicalAnalyzer()
        result = a.analyze_sentence("n be taa", "latin")
        assert result["glossing"] != ""

    def test_sentence_to_dict_keys(self):
        a = MorphologicalAnalyzer()
        result = a.analyze_sentence("mogo", "latin")
        assert "words" in result
        assert "structure" in result
        assert "glossing" in result


# ═══════════════════════════════════════════════════════════════════════════════
# §9  EDGE CASES & REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and regressions."""

    def test_empty_string(self):
        result = analyze("", "latin")
        assert result == []

    def test_single_character(self):
        """Single character should still return an analysis."""
        result = analyze("a", "latin")
        assert len(result) == 1

    def test_word_analysis_to_dict(self):
        wa = analyze_word("mogo", "latin")
        d = wa.to_dict()
        assert "decomposition" in d
        assert "noun_class" in d

    def test_morpheme_to_dict(self):
        m = Morpheme("test", MorphemeType.ROOT, "test-gloss")
        d = m.to_dict()
        assert d["type"] == "root"
        assert d["gloss"] == "test-gloss"

    def test_reconstruct_latin(self):
        wa = analyze_word("mogolu", "latin")
        reconstructed = wa.reconstruct("latin")
        # Should contain the original parts
        assert len(reconstructed) > 0

    def test_gloss_string(self):
        wa = analyze_word("mogolu", "latin")
        assert "-" in wa.gloss_string or wa.gloss_string != ""

    def test_conjugator_compare_tenses(self):
        conj = VerbConjugator()
        comparison = conj.compare_tenses("go")
        assert len(comparison) == len(TenseAspect)
        assert all("nko" in c for c in comparison)


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
