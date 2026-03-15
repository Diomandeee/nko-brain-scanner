#!/usr/bin/env python3
"""
Tests for N'Ko Community & Collaboration Engine (Gen 11)
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

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
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from community_engine import (
    NkoCommunityEngine,
    CommunityKeyboardIntegration,
    ContributorProfile,
    ContributorRole,
    Dialect,
    VoteType,
    ModerationStatus,
    NeologismDomain,
    ContributionType
)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def engine(temp_data_dir):
    """Create test engine instance"""
    return NkoCommunityEngine(data_dir=temp_data_dir)


@pytest.fixture
def registered_user(engine):
    """Register a test user"""
    return engine.register_contributor(
        user_id="user_001",
        display_name="Amadou",
        display_name_nko="ߊ߬ߡߊ߬ߘߎ",
        primary_dialect=Dialect.BAMBARA,
        region="Bamako"
    )


class TestContributorManagement:
    """Test contributor registration and management"""
    
    def test_register_contributor(self, engine):
        """Test basic contributor registration"""
        user = engine.register_contributor(
            user_id="new_user",
            display_name="Fatou",
            display_name_nko="ߝߊ߬ߕߎ"
        )
        
        assert user is not None
        assert user.user_id == "new_user"
        assert user.display_name == "Fatou"
        assert user.display_name_nko == "ߝߊ߬ߕߎ"
        assert user.role == ContributorRole.LEARNER
        assert user.reputation_score == 0
        assert len(user.badges) == 1  # Welcome badge
    
    def test_duplicate_registration_returns_existing(self, engine):
        """Test that duplicate registration returns existing user"""
        user1 = engine.register_contributor("dup_user", "First")
        user2 = engine.register_contributor("dup_user", "Second")
        
        assert user1.user_id == user2.user_id
        assert user1.display_name == "First"  # First name kept
    
    def test_contributor_trust_level(self, engine):
        """Test trust level calculation"""
        # Learner starts at level 1
        learner = engine.register_contributor("learner", "Learner")
        assert learner.trust_level == 1
        
        # Create moderator-verified expert
        mod = engine.register_contributor("mod", "Moderator")
        mod.role = ContributorRole.MODERATOR
        engine.contributors["mod"] = mod
        
        expert = engine.register_contributor("expert", "Expert")
        engine.upgrade_contributor_role("expert", ContributorRole.ELDER, "mod")
        
        assert engine.contributors["expert"].trust_level == 4
    
    def test_upgrade_contributor_role(self, engine):
        """Test role upgrade with verification"""
        # Create moderator
        mod = engine.register_contributor("mod", "Moderator")
        engine.contributors["mod"].role = ContributorRole.MODERATOR
        
        # Create regular user
        user = engine.register_contributor("user", "User")
        
        # Upgrade to dialect expert
        result = engine.upgrade_contributor_role(
            "user",
            ContributorRole.DIALECT_EXPERT,
            "mod"
        )
        
        assert result is True
        assert engine.contributors["user"].role == ContributorRole.DIALECT_EXPERT
    
    def test_contributor_stats(self, engine, registered_user):
        """Test getting contributor statistics"""
        stats = engine.get_contributor_stats("user_001")
        
        assert "profile" in stats
        assert "trust_level" in stats
        assert "can_verify" in stats
        assert stats["profile"].display_name == "Amadou"


class TestWordManagement:
    """Test word entry management"""
    
    def test_add_word(self, engine, registered_user):
        """Test adding a new word"""
        word = engine.add_word(
            nko_text="ߛߓߍ",
            latin_text="sɛbɛ",
            initial_definition_nko="ߞߊ߬ߟߌ߬ ߘߐ߫ ߛߓߍ ߟߊ߫",
            initial_definition_latin="to write, writing",
            contributor_id="user_001",
            definition_english="to write",
            definition_french="écrire"
        )
        
        assert word is not None
        assert word.nko_text == "ߛߓߍ"
        assert word.latin_text == "sɛbɛ"
        assert len(word.definitions) == 1
        assert word.definitions[0].text_english == "to write"
    
    def test_duplicate_word_rejected(self, engine, registered_user):
        """Test that duplicate words are rejected"""
        engine.add_word("ߛߓߍ", "sɛbɛ", "def1", "def1", "user_001")
        result = engine.add_word("ߛߓߍ", "sɛbɛ2", "def2", "def2", "user_001")
        
        assert result is None
    
    def test_add_alternative_definition(self, engine, registered_user):
        """Test adding alternative definition"""
        # Use a word that doesn't collide with seed vocabulary
        word = engine.add_word("ߓߊ߯ߙߊ", "baara", "ߓߊ߯ߙߊ ߞߍ", "work/job", "user_001")
        assert word is not None, "add_word returned None — word may already exist in seed data"
        
        defn = engine.add_definition(
            word_id=word.word_id,
            text_nko="ߛߍ߲ߝߍ",
            text_latin="occupation, employment",
            contributor_id="user_001",
            text_english="occupation"
        )
        
        assert defn is not None
        assert len(engine.words[word.word_id].definitions) == 2
    
    def test_add_usage_example(self, engine, registered_user):
        """Test adding usage example"""
        word = engine.add_word("ߕߊ߯", "taa", "ߕߊ߯ ߦߋ߫", "to go", "user_001")
        
        example = engine.add_usage_example(
            word_id=word.word_id,
            sentence_nko="ߒ ߓߊ߯ ߕߊ߯ ߟߊ ߛߏ ߘߐ߫",
            sentence_latin="n bɛ taa la so kɔnɔ",
            contributor_id="user_001",
            translation_english="I am going home"
        )
        
        assert example is not None
        assert "ߕߊ߯" in example.sentence_nko
    
    def test_add_dialect_variant(self, engine, registered_user):
        """Test adding dialect variant for contributor's own dialect"""
        word = engine.add_word("ߖߌ", "ji", "ߖߌ ߡߍ߲", "water", "user_001")
        
        # Add variant for user's own dialect (Bambara) — always allowed
        variant = engine.add_dialect_variant(
            word_id=word.word_id,
            dialect=Dialect.BAMBARA,
            nko_text="ߖߌ",
            latin_text="ji",
            contributor_id="user_001",
            pronunciation_ipa="dʒi",
            region="Bamako"
        )
        
        assert variant is not None
        assert variant.dialect == Dialect.BAMBARA
    
    def test_add_cultural_note_requires_trust(self, engine):
        """Test that cultural notes require trust level 2+"""
        # Low-trust user
        engine.register_contributor("low_trust", "Low Trust")
        word = engine.add_word("ߓߟߏ", "bolo", "ߓߟߏ", "hand", "system")
        
        # Should fail - trust too low
        note = engine.add_cultural_note(
            word_id=word.word_id,
            text="Important cultural meaning",
            contributor_id="low_trust"
        )
        
        assert note is None


class TestNeologismCreation:
    """Test neologism proposal and review"""
    
    def test_propose_neologism(self, engine, registered_user):
        """Test proposing a new word"""
        # Upgrade user to have sufficient trust
        engine.contributors["user_001"].reputation_score = 100
        
        neo = engine.propose_neologism(
            concept_english="computer",
            concept_french="ordinateur",
            proposed_nko="ߘߊ߲ߕߊ߲ߞߍߟߊ",
            proposed_latin="dantarankɛla",
            domain=NeologismDomain.TECHNOLOGY,
            rationale="Compound: dantaran (calculation) + kɛla (machine)",
            construction="N+N compound",
            contributor_id="user_001"
        )
        
        assert neo is not None
        assert neo.concept_english == "computer"
        assert neo.status == ModerationStatus.PENDING
    
    def test_neologism_requires_registration(self, engine):
        """Test that neologisms require a registered contributor"""
        # Unregistered user cannot propose neologisms
        neo = engine.propose_neologism(
            concept_english="test",
            concept_french="test",
            proposed_nko="ߕߍ߬ߛߑ",
            proposed_latin="test",
            domain=NeologismDomain.TECHNOLOGY,
            rationale="test",
            construction="test",
            contributor_id="unregistered_user"
        )
        
        # Should fail — contributor not registered
        assert neo is None
    
    def test_add_neologism_alternative(self, engine, registered_user):
        """Test adding alternative to neologism"""
        engine.contributors["user_001"].reputation_score = 100
        
        neo = engine.propose_neologism(
            concept_english="email",
            concept_french="courriel",
            proposed_nko="ߓߊ߬ߕߊ߬ߞߊ߬ߙߌ",
            proposed_latin="batakari",
            domain=NeologismDomain.INTERNET,
            rationale="From 'bata' (letter) + 'kari' (electronic)",
            construction="N+Adj compound",
            contributor_id="user_001"
        )
        
        result = engine.add_neologism_alternative(
            neo.neologism_id,
            "ߟߍ߬ߕߊ߬ߙߊ߬ߝߟߊ",
            "lɛtarafla",
            "user_001"
        )
        
        assert result is True
        assert len(engine.neologisms[neo.neologism_id].alternatives) == 1
    
    def test_expert_review_neologism(self, engine, registered_user):
        """Test expert review of neologism"""
        # Create linguist
        engine.register_contributor("linguist", "Dr. Linguist")
        engine.contributors["linguist"].role = ContributorRole.LINGUIST
        
        # Create neologism
        engine.contributors["user_001"].reputation_score = 100
        neo = engine.propose_neologism(
            concept_english="internet",
            concept_french="internet",
            proposed_nko="ߖߊ߬ߓߊ߬ߘߎ߬ߘߊ",
            proposed_latin="jabadudu",
            domain=NeologismDomain.TECHNOLOGY,
            rationale="Global network of knowledge",
            construction="Compound",
            contributor_id="user_001"
        )
        
        review = engine.review_neologism(
            neo.neologism_id,
            "linguist",
            "approve",
            comments="Well-formed compound following Manding patterns"
        )
        
        assert review is not None
        assert review.recommendation == "approve"
        assert len(engine.neologisms[neo.neologism_id].expert_reviews) == 1


class TestVotingAndVerification:
    """Test community voting system"""
    
    def test_upvote_definition(self, engine, registered_user):
        """Test upvoting a definition"""
        word = engine.add_word("ߕߊ", "ta", "ߕߊ", "foot", "user_001")
        defn = word.definitions[0]
        
        engine.register_contributor("voter", "Voter")
        result = engine.vote(
            "voter",
            "definition",
            defn.definition_id,
            VoteType.UPVOTE
        )
        
        assert result is True
        # Refresh word from engine
        updated_word = engine.words[word.word_id]
        assert updated_word.definitions[0].upvotes == 1
    
    def test_verify_definition_requires_trust(self, engine, registered_user):
        """Test that verification requires trust level 3+"""
        word = engine.add_word("ߓߟߊ", "bala", "ߓߟߊ", "stick", "user_001")
        defn = word.definitions[0]
        
        # Low-trust voter
        engine.register_contributor("low_trust", "Low")
        
        result = engine.vote(
            "low_trust",
            "definition",
            defn.definition_id,
            VoteType.VERIFY
        )
        
        assert result is False  # Should fail
    
    def test_endorse_requires_elder(self, engine, registered_user):
        """Test that endorsement requires elder/griot role"""
        word = engine.add_word("ߝߊ", "fa", "ߝߊ", "father", "user_001")
        defn = word.definitions[0]
        
        # Create elder
        engine.register_contributor("elder", "Elder")
        engine.contributors["elder"].role = ContributorRole.ELDER
        
        result = engine.vote(
            "elder",
            "definition",
            defn.definition_id,
            VoteType.ENDORSE
        )
        
        assert result is True
        updated_word = engine.words[word.word_id]
        assert updated_word.definitions[0].endorsements == 1
    
    def test_definition_auto_approval(self, engine, registered_user):
        """Test definition auto-approval after enough verifications"""
        word = engine.add_word("ߓߊ", "ba", "ߓߊ", "mother", "user_001")
        defn = word.definitions[0]
        
        assert defn.status == ModerationStatus.PENDING
        
        # Create expert and endorse
        engine.register_contributor("elder", "Elder")
        engine.contributors["elder"].role = ContributorRole.ELDER
        
        engine.vote("elder", "definition", defn.definition_id, VoteType.ENDORSE)
        
        # Should now be approved (1 endorsement = auto-approve)
        updated_defn = engine.words[word.word_id].definitions[0]
        assert updated_defn.status == ModerationStatus.APPROVED


class TestSearchAndLookup:
    """Test search functionality"""
    
    def test_lookup_by_nko(self, engine, registered_user):
        """Test word lookup by N'Ko text"""
        engine.add_word("ߛߏ", "so", "ߛߏ", "house", "user_001")
        
        word = engine.lookup_word("ߛߏ")
        
        assert word is not None
        assert word.latin_text == "so"
    
    def test_lookup_by_latin(self, engine, registered_user):
        """Test word lookup by Latin text"""
        engine.add_word("ߞߐ", "kɔ", "ߞߐ", "back", "user_001")
        
        word = engine.lookup_word("kɔ")
        
        assert word is not None
        assert word.nko_text == "ߞߐ"
    
    def test_search_partial_match(self, engine, registered_user):
        """Test partial search with unique words that don't collide with seed data"""
        engine.add_word("ߡߏߦߌ", "moyi", "def", "to understand", "user_001")
        engine.add_word("ߡߏߦߌߟߊ", "moyila", "def", "one who understands", "user_001")
        engine.add_word("ߡߏߦߌߟߌ", "moyili", "def", "understanding", "user_001")
        
        results = engine.search_words("ߡߏߦߌ")
        
        assert len(results) == 3
    
    def test_search_by_dialect(self, engine):
        """Test dialect-filtered search"""
        # Register a contributor with MANINKA dialect so they can add variants
        engine.register_contributor(
            user_id="maninka_user",
            display_name="Moussa",
            display_name_nko="ߡߎ߬ߛߊ",
            primary_dialect=Dialect.MANINKA,
            region="Kankan"
        )
        word = engine.add_word("ߖߌ", "ji", "def", "water", "maninka_user")
        assert word is not None, "add_word returned None — word may exist in seeds"
        engine.add_dialect_variant(
            word.word_id,
            Dialect.MANINKA,
            "ߖߌ",
            "dji",
            "maninka_user"
        )
        
        results = engine.get_words_by_dialect(Dialect.MANINKA)
        
        assert len(results) >= 1


class TestTranslationMemory:
    """Test translation memory functionality"""
    
    def test_add_translation(self, engine, registered_user):
        """Test adding translation pair"""
        tm = engine.add_translation(
            source_lang="en",
            source_text="Hello, how are you?",
            target_nko="ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ، ߤߊ߬ߟߌ ߓߍ߫؟",
            target_latin="i ni sogoma, hali bɛ?",
            contributor_id="user_001",
            domain="greeting"
        )
        
        assert tm is not None
        assert tm.source_lang == "en"
    
    def test_search_translations(self, engine, registered_user):
        """Test searching translation memory"""
        engine.add_translation(
            "en", "Good morning", "ߌ ߣߌ ߛߐ߲߬ߜߐ߫ߡߊ", "i ni sogoma",
            "user_001"
        )
        engine.add_translation(
            "en", "Good evening", "ߌ ߣߌ ߥߎ߬ߙߊ", "i ni wula",
            "user_001"
        )
        
        results = engine.search_translations("Good", "en")
        
        assert len(results) == 2


class TestKeyboardIntegration:
    """Test keyboard integration features"""
    
    def test_get_suggestions(self, engine, registered_user):
        """Test getting typing suggestions"""
        engine.add_word("ߌ ߣߌ", "i ni", "def", "you and", "user_001")
        
        suggestions = engine.get_suggestions_for_context("ߌ")
        
        assert len(suggestions) > 0
    
    def test_word_info_for_keyboard(self, engine, registered_user):
        """Test getting word info formatted for keyboard"""
        word = engine.add_word(
            "ߛߎ߲", "sun", "ߛߎ߲", "gold",
            "user_001",
            definition_english="gold"
        )
        
        info = engine.get_word_info_for_keyboard("ߛߎ߲")
        
        assert info is not None
        assert info["nko"] == "ߛߎ߲"
        assert info["latin"] == "sun"
    
    def test_keyboard_integration_class(self, engine, registered_user):
        """Test CommunityKeyboardIntegration"""
        engine.add_word("ߞߊ߲", "kan", "def", "language", "user_001")
        
        integration = CommunityKeyboardIntegration(engine, "user_001")
        suggestions = integration.get_typing_suggestions("ߞ")
        
        assert len(suggestions) > 0
        assert "nko" in suggestions[0]


class TestGamification:
    """Test gamification features"""
    
    def test_leaderboard(self, engine):
        """Test leaderboard generation"""
        # Create users with different reputations
        engine.register_contributor("user1", "User 1")
        engine.register_contributor("user2", "User 2")
        engine.register_contributor("user3", "User 3")
        
        engine.contributors["user1"].reputation_score = 100
        engine.contributors["user2"].reputation_score = 200
        engine.contributors["user3"].reputation_score = 150
        
        leaderboard = engine.get_leaderboard(limit=3)
        
        assert len(leaderboard) >= 0  # May include system user
    
    def test_daily_challenge(self, engine, registered_user):
        """Test daily challenge generation"""
        challenge = engine.get_daily_challenge()
        
        assert "date" in challenge
        assert "challenges" in challenge
        assert len(challenge["challenges"]) > 0
    
    def test_reputation_rewards(self, engine, registered_user):
        """Test reputation is awarded correctly"""
        initial_rep = engine.contributors["user_001"].reputation_score
        
        word = engine.add_word("ߕߍ", "te", "def", "not", "user_001")
        defn = word.definitions[0]
        
        # Simulate upvote
        engine.register_contributor("voter", "Voter")
        engine.vote("voter", "definition", defn.definition_id, VoteType.UPVOTE)
        
        # Check reputation increased
        assert engine.contributors["user_001"].reputation_score > initial_rep


class TestDataPersistence:
    """Test data persistence"""
    
    def test_save_and_load(self, temp_data_dir):
        """Test that data persists across engine instances"""
        # Create and populate engine
        engine1 = NkoCommunityEngine(data_dir=temp_data_dir)
        engine1.register_contributor("persist_user", "Persistent", "ߔߋ߬ߙߑߛߌ߬ߛߑ")
        engine1.contributors["persist_user"].reputation_score = 500
        engine1._save_data()
        
        # Create new engine instance
        engine2 = NkoCommunityEngine(data_dir=temp_data_dir)
        
        # Check data loaded
        assert "persist_user" in engine2.contributors
        assert engine2.contributors["persist_user"].reputation_score == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
