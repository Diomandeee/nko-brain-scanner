"""Tests for nko.cultural_tools — Cultural Tools Tab."""

import pytest

from nko.culture import NKoCulture
from nko.cultural_tools import (
    ProverbsBrowser,
    SigilsPlayer,
    CulturalCalendar,
    BlessingsBrowser,
    GreetingsProtocol,
    ClansBrowser,
    ConceptsBrowser,
)


@pytest.fixture
def culture():
    return NKoCulture()


# ── ProverbsBrowser ─────────────────────────────────────────────────


class TestProverbsBrowser:
    def test_random_returns_text(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.random()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_browse_first_page(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.browse(page=1, per_page=5)
        assert "Page 1/" in result
        assert "62 total" in result

    def test_browse_last_page(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.browse(page=13, per_page=5)
        assert "Page 13/" in result

    def test_browse_clamps_page(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.browse(page=999, per_page=5)
        assert "Page 13/" in result  # max page

    def test_search_finds_results(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.search("money")
        assert "Found 2 proverb" in result

    def test_search_no_results(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.search("xyznonexistent")
        assert "No proverbs found" in result

    def test_categories_lists_all(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.categories()
        assert "wisdom" in result
        assert "categories" in result

    def test_by_category(self, culture):
        browser = ProverbsBrowser(culture)
        result = browser.by_category("wisdom")
        assert "wisdom" in result
        assert len(result) > 50  # Non-trivial output


# ── SigilsPlayer ───────────────────────────────────────────────────


class TestSigilsPlayer:
    def test_list_sigils(self):
        player = SigilsPlayer()
        result = player.list_sigils()
        assert "stabilization" in result
        assert "echo" in result
        assert "ߛ" in result

    def test_info_by_name(self):
        player = SigilsPlayer()
        result = player.info("stabilization")
        assert "440" in result
        assert "Dispersion decreased" in result

    def test_info_by_char(self):
        player = SigilsPlayer()
        result = player.info("ߛ")
        assert "stabilization" in result

    def test_info_unknown(self):
        player = SigilsPlayer()
        result = player.info("xyz_unknown")
        assert "Unknown sigil" in result or "Error" in result

    def test_waveform_renders(self):
        player = SigilsPlayer()
        result = player.waveform("ߛ", width=30)
        assert "stabilization" in result
        assert "440" in result


# ── CulturalCalendar ────────────────────────────────────────────────


class TestCulturalCalendar:
    def test_list_events(self, culture):
        cal = CulturalCalendar(culture)
        result = cal.list_events()
        assert "7 events total" in result
        assert "N'Ko Don" in result
        assert "Aidu Fitiri" in result

    def test_upcoming(self, culture):
        cal = CulturalCalendar(culture)
        result = cal.upcoming()
        assert "Upcoming Events" in result

    def test_event_detail_by_id(self, culture):
        cal = CulturalCalendar(culture)
        result = cal.event_detail("cal-05")
        assert "N'Ko Don" in result
        assert "April 14" in result
        assert "Solomana Kanté" in result

    def test_event_detail_by_name(self, culture):
        cal = CulturalCalendar(culture)
        result = cal.event_detail("Mawloud")
        assert "Prophet" in result

    def test_event_not_found(self, culture):
        cal = CulturalCalendar(culture)
        result = cal.event_detail("nonexistent")
        assert "not found" in result


# ── BlessingsBrowser ────────────────────────────────────────────────


class TestBlessingsBrowser:
    def test_random(self, culture):
        browser = BlessingsBrowser(culture)
        result = browser.random()
        assert len(result) > 0

    def test_list_all(self, culture):
        browser = BlessingsBrowser(culture)
        result = browser.list_all()
        assert "29 total" in result
        assert "Blessing" in result

    def test_for_event_general(self, culture):
        browser = BlessingsBrowser(culture)
        result = browser.for_event("general")
        assert "general" in result

    def test_for_event_wedding(self, culture):
        browser = BlessingsBrowser(culture)
        result = browser.for_event("wedding")
        assert "wedding" in result


# ── GreetingsProtocol ───────────────────────────────────────────────


class TestGreetingsProtocol:
    def test_show_protocol(self, culture):
        proto = GreetingsProtocol(culture)
        result = proto.show_protocol()
        assert "OPENING" in result
        assert "RESPONSE" in result
        assert "WELFARE" in result
        assert "Protocol Flow" in result

    def test_by_phase_opening(self, culture):
        proto = GreetingsProtocol(culture)
        result = proto.by_phase("opening")
        assert "OPENING" in result

    def test_by_time_morning(self, culture):
        proto = GreetingsProtocol(culture)
        result = proto.by_time("morning")
        assert "Morning" in result

    def test_by_phase_not_found(self, culture):
        proto = GreetingsProtocol(culture)
        result = proto.by_phase("nonexistent")
        assert "No greetings found" in result


# ── ClansBrowser ────────────────────────────────────────────────────


class TestClansBrowser:
    def test_list_all(self, culture):
        browser = ClansBrowser(culture)
        result = browser.list_all()
        assert "9 families" in result
        assert "Keita" in result
        assert "Diomande" in result

    def test_info_by_name(self, culture):
        browser = ClansBrowser(culture)
        result = browser.info("Diomande")
        assert "Hippopotamus" in result
        assert "Noble Line: Yes" in result

    def test_info_not_found(self, culture):
        browser = ClansBrowser(culture)
        result = browser.info("Nonexistent")
        assert "not found" in result


# ── ConceptsBrowser ─────────────────────────────────────────────────


class TestConceptsBrowser:
    def test_list_all(self, culture):
        browser = ConceptsBrowser(culture)
        result = browser.list_all()
        assert "12 total" in result
        assert "Core Concepts" in result
        assert "Kinship Terms" in result

    def test_by_type_concept(self, culture):
        browser = ConceptsBrowser(culture)
        result = browser.by_type("concept")
        assert "sanaku" in result
        assert "jamu" in result

    def test_by_type_title(self, culture):
        browser = ConceptsBrowser(culture)
        result = browser.by_type("title")
        assert "mansa" in result

    def test_by_type_not_found(self, culture):
        browser = ConceptsBrowser(culture)
        result = browser.by_type("nonexistent")
        assert "No concepts found" in result
