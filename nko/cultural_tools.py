"""
nko.cultural_tools — Cultural Tools Tab
ߒߞߏ ߟߊ߬ߘߍ ߓߊ߯ߙߊ — Proverbs Browser, Sound Sigils Player, Cultural Calendar

Provides rich CLI-formatted output for browsing and interacting with
N'Ko cultural heritage data. Built on top of nko.culture.NKoCulture.

Usage::

    python -m nko culture proverbs [browse|random|search <q>|categories]
    python -m nko culture calendar [list|upcoming|event <id>]
    python -m nko culture sigils [list|info <sigil>|play <sigil>|waveform <sigil>]
    python -m nko culture blessings [random|list|for <event>]
    python -m nko culture greetings [protocol|phase <phase>|time <time>]
    python -m nko culture clans [list|info <name>]
    python -m nko culture concepts [list|type <type>]
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Proverbs Browser ────────────────────────────────────────────────


class ProverbsBrowser:
    """Browse, search, and filter the 62-entry proverbs collection."""

    def __init__(self, culture):
        self._culture = culture

    def random(self) -> str:
        """Display a random proverb with full formatting."""
        p = self._culture.random_proverb()
        return self._format_proverb(p)

    def browse(self, page: int = 1, per_page: int = 5) -> str:
        """Paginated proverb listing."""
        all_proverbs = self._culture.proverbs()
        total = len(all_proverbs)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        subset = all_proverbs[start:end]

        lines = [
            "",
            f"  Proverbs — Page {page}/{total_pages} ({total} total)",
            "  " + "─" * 60,
        ]
        for i, p in enumerate(subset, start=start + 1):
            lines.append(f"  [{i}] {self._format_proverb_compact(p)}")
            lines.append("")
        lines.append(f"  Page {page}/{total_pages} — use 'browse <page>' for more")
        return "\n".join(lines)

    def search(self, query: str) -> str:
        """Search proverbs and display results."""
        results = self._culture.search_proverbs(query)
        if not results:
            return f"  No proverbs found matching '{query}'"
        lines = [
            "",
            f"  Found {len(results)} proverb(s) matching '{query}':",
            "  " + "─" * 60,
        ]
        for p in results:
            lines.append(f"  {self._format_proverb_compact(p)}")
            lines.append("")
        return "\n".join(lines)

    def categories(self) -> str:
        """List all proverb categories with counts."""
        cat_counts: Dict[str, int] = {}
        for p in self._culture.proverbs():
            for cat in p.get("categories", []):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
        lines = [
            "",
            "  Proverb Categories:",
            "  " + "─" * 40,
        ]
        for cat, count in sorted_cats:
            lines.append(f"    {cat:<25} {count} proverbs")
        lines.append("")
        lines.append(f"  {len(sorted_cats)} categories across {sum(cat_counts.values())} tagged entries")
        return "\n".join(lines)

    def by_category(self, category: str) -> str:
        """List proverbs in a specific category."""
        results = [
            p for p in self._culture.proverbs()
            if category.lower() in [c.lower() for c in p.get("categories", [])]
        ]
        if not results:
            return f"  No proverbs found in category '{category}'"
        lines = [
            "",
            f"  Proverbs in '{category}' ({len(results)}):",
            "  " + "─" * 60,
        ]
        for p in results:
            lines.append(f"  {self._format_proverb_compact(p)}")
            lines.append("")
        return "\n".join(lines)

    def _format_proverb(self, p: Dict[str, Any]) -> str:
        """Full proverb display."""
        lines = []
        nko = p.get("text_nko", "")
        latin = p.get("text_latin", "")
        en = p.get("english_translation", "")
        meaning = p.get("meaning", "")
        lang = p.get("language", "")
        cats = p.get("categories", [])
        formality = p.get("formality", "")

        if nko:
            lines.append(f"    {nko}")
        if latin:
            lines.append(f"    {latin}")
        if en:
            lines.append(f"    \"{en}\"")
        if meaning:
            lines.append(f"    Meaning: {meaning}")
        meta_parts = []
        if lang:
            meta_parts.append(lang)
        if formality:
            meta_parts.append(formality)
        if cats:
            meta_parts.append(", ".join(cats))
        if meta_parts:
            lines.append(f"    [{' | '.join(meta_parts)}]")
        return "\n".join(lines)

    def _format_proverb_compact(self, p: Dict[str, Any]) -> str:
        """Single-line proverb with key info."""
        nko = p.get("text_nko", "")
        latin = p.get("text_latin", "")
        en = p.get("english_translation", "")
        display = nko or latin
        if en:
            display += f" — \"{en}\""
        return display


# ── Sound Sigils Player ─────────────────────────────────────────────


class SigilsPlayer:
    """Browse and play the 10 N'Ko sound sigils."""

    def __init__(self):
        self._engine = None
        self._load_error = None

    def _get_engine(self):
        """Lazy-load the sound sigils engine via nko.sigils bootstrap."""
        if self._engine is None and self._load_error is None:
            try:
                from nko.sigils import SoundSigils
                if SoundSigils is None:
                    self._load_error = "Sound sigils engine not available"
                else:
                    self._engine = SoundSigils()
            except Exception as e:
                self._load_error = str(e)
        return self._engine

    def list_sigils(self) -> str:
        """List all 10 sound sigils with descriptions."""
        engine = self._get_engine()
        if engine is None:
            return self._fallback_list()
        return engine.list_sigils()

    def info(self, key: str) -> str:
        """Show detailed info for a sigil."""
        engine = self._get_engine()
        if engine is None:
            return self._fallback_info(key)
        try:
            return engine.info(key)
        except ValueError as e:
            return f"  Error: {e}"

    def play(self, key: str) -> str:
        """Play a sigil through speakers."""
        engine = self._get_engine()
        if engine is None:
            return f"  Cannot play sigils: {self._load_error}"
        try:
            sigil = engine._require(key)
            engine.play(key)
            return f"  Playing {sigil.char} ({sigil.name}): {sigil.description}"
        except ValueError as e:
            return f"  Error: {e}"

    def play_sequence(self, keys: List[str]) -> str:
        """Play a sequence of sigils."""
        engine = self._get_engine()
        if engine is None:
            return f"  Cannot play sigils: {self._load_error}"
        try:
            engine.play_sequence(keys)
            return f"  Playing sequence: {' '.join(keys)}"
        except ValueError as e:
            return f"  Error: {e}"

    def waveform(self, key: str, width: int = 60) -> str:
        """Show ASCII waveform of a sigil."""
        engine = self._get_engine()
        if engine is None:
            return self._fallback_info(key)
        try:
            import sound_sigils.audio as _sa
            render_samples = _sa.render_samples
            sigil = engine._require(key)
            samples = render_samples(sigil)
            if not samples:
                return "  (empty)"

            bucket_size = max(1, len(samples) // width)
            buckets: list[float] = []
            for i in range(0, len(samples), bucket_size):
                chunk = samples[i : i + bucket_size]
                peak = max(abs(l) + abs(r) for l, r in chunk) / 2
                buckets.append(peak)

            if not buckets:
                return "  (empty)"

            max_peak = max(buckets) or 1.0
            blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

            lines = [f"  {sigil.char} {sigil.name} — {sigil.duration}s @ {sigil.base_freq}Hz"]
            lines.append("  " + "".join(
                blocks[min(int(b / max_peak * (len(blocks) - 1)), len(blocks) - 1)]
                for b in buckets[:width]
            ))
            return "\n".join(lines)
        except ValueError as e:
            return f"  Error: {e}"

    def export(self, key: str, output_path: str) -> str:
        """Export a sigil to WAV file."""
        engine = self._get_engine()
        if engine is None:
            return f"  Cannot export sigils: {self._load_error}"
        try:
            path = engine.export(key, output_path)
            return f"  Exported to {path}"
        except ValueError as e:
            return f"  Error: {e}"

    def _fallback_list(self) -> str:
        """Static sigil listing when engine not available."""
        sigils = [
            ("ߛ", "stabilization", "Dispersion decreased", 440.0, 1.5),
            ("ߜ", "dispersion", "Spread increased", 330.0, 1.5),
            ("ߕ", "transition", "Change point", 523.0, 0.8),
            ("ߙ", "return", "Re-entry to basin", 392.0, 1.2),
            ("ߡ", "dwell", "Sustained stay", 349.0, 2.0),
            ("ߚ", "oscillation", "Rapid alternation", 466.0, 1.0),
            ("ߞ", "recovery", "Return latency", 294.0, 1.8),
            ("ߣ", "novelty", "New basin", 587.0, 1.0),
            ("ߠ", "place_shift", "Location change", 415.0, 1.3),
            ("ߥ", "echo", "Pattern match", 370.0, 2.0),
        ]
        lines = [
            "",
            "  Sound Sigils — N'Ko Audio Signatures",
            "",
            f"  {'Sigil':<6} {'Name':<15} {'Description':<25} {'Freq':<8} {'Duration'}",
            "  " + "-" * 66,
        ]
        for char, name, desc, freq, dur in sigils:
            lines.append(f"  {char:<6} {name:<15} {desc:<25} {freq:<8.0f} {dur}s")
        lines.append("")
        if self._load_error:
            lines.append(f"  (Audio playback unavailable: {self._load_error})")
            lines.append("  Use 'play' or 'export' from the tools/sound-sigils directory.")
        return "\n".join(lines)

    def _fallback_info(self, key: str) -> str:
        """Static info when engine not available."""
        sigils = {
            "ߛ": ("stabilization", "Dispersion decreased", "Descending tone settling to steady hum", 440.0, 1.5),
            "ߜ": ("dispersion", "Spread increased", "Expanding stereo, rising harmonics", 330.0, 1.5),
            "ߕ": ("transition", "Change point", "Sharp frequency shift, brief silence", 523.0, 0.8),
            "ߙ": ("return", "Re-entry to basin", "Melodic resolution, home note return", 392.0, 1.2),
            "ߡ": ("dwell", "Sustained stay", "Long sustained tone, subtle warmth", 349.0, 2.0),
            "ߚ": ("oscillation", "Rapid alternation", "Tremolo, rapid frequency modulation", 466.0, 1.0),
            "ߞ": ("recovery", "Return latency", "Slow fade-in, gradual stabilization", 294.0, 1.8),
            "ߣ": ("novelty", "New basin", "Surprising interval, new timbre", 587.0, 1.0),
            "ߠ": ("place_shift", "Location change", "Spatial panning, Doppler effect", 415.0, 1.3),
            "ߥ": ("echo", "Pattern match", "Delayed repetition, reverb tail", 370.0, 2.0),
        }
        # Try by char first, then by name
        info = sigils.get(key)
        if info is None:
            normalized = key.lower().replace("-", "_")
            for char, data in sigils.items():
                if data[0] == normalized:
                    info = data
                    key = char
                    break
        if info is None:
            return f"  Unknown sigil: {key}"
        name, desc, sound, freq, dur = info
        return (
            f"  Character : {key}\n"
            f"  Name      : {name}\n"
            f"  Meaning   : {desc}\n"
            f"  Sound     : {sound}\n"
            f"  Frequency : {freq} Hz\n"
            f"  Duration  : {dur}s"
        )


# ── Cultural Calendar ───────────────────────────────────────────────


class CulturalCalendar:
    """Browse the N'Ko cultural and religious calendar."""

    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    def __init__(self, culture):
        self._culture = culture

    def list_events(self) -> str:
        """List all cultural calendar events."""
        events = self._culture.calendar()
        lines = [
            "",
            "  N'Ko Cultural Calendar",
            "  " + "═" * 60,
        ]
        # Group: fixed-date first, then lunar
        fixed = [e for e in events if e.get("date_fixed")]
        lunar = [e for e in events if not e.get("date_fixed")]

        if fixed:
            lines.append("")
            lines.append("  Fixed-Date Events:")
            lines.append("  " + "─" * 50)
            for e in sorted(fixed, key=lambda x: self._parse_date_key(x.get("date_fixed", ""))):
                lines.append(self._format_event_line(e))

        if lunar:
            lines.append("")
            lines.append("  Lunar Calendar Events (dates vary):")
            lines.append("  " + "─" * 50)
            for e in lunar:
                lines.append(self._format_event_line(e))

        lines.append("")
        lines.append(f"  {len(events)} events total")
        return "\n".join(lines)

    def upcoming(self) -> str:
        """Show events relative to the current month."""
        events = self._culture.calendar()
        now = datetime.now()
        current_month = now.month

        fixed = [e for e in events if e.get("date_fixed")]
        lunar = [e for e in events if not e.get("date_fixed")]

        # Sort fixed events by proximity to current date
        upcoming_fixed = []
        for e in fixed:
            month = self._parse_date_key(e.get("date_fixed", ""))
            if month >= current_month:
                upcoming_fixed.append((month - current_month, e))
            else:
                upcoming_fixed.append((month + 12 - current_month, e))
        upcoming_fixed.sort(key=lambda x: x[0])

        lines = [
            "",
            f"  Upcoming Events (from {now.strftime('%B %Y')}):",
            "  " + "═" * 60,
        ]

        if upcoming_fixed:
            lines.append("")
            for months_away, e in upcoming_fixed:
                if months_away == 0:
                    timing = "THIS MONTH"
                elif months_away == 1:
                    timing = "Next month"
                else:
                    timing = f"In {months_away} months"
                lines.append(f"  [{timing}] {self._format_event_line(e)}")
            lines.append("")

        if lunar:
            lines.append("  Lunar Calendar Events (check local announcements):")
            lines.append("  " + "─" * 50)
            for e in lunar:
                lines.append(self._format_event_line(e))
            lines.append("")

        return "\n".join(lines)

    def event_detail(self, event_id: str) -> str:
        """Show full detail for a specific event."""
        events = self._culture.calendar()
        event = None
        # Match by id or name
        for e in events:
            if e.get("id") == event_id or event_id.lower() in e.get("name_latin", "").lower():
                event = e
                break
        if event is None:
            return f"  Event not found: {event_id}"
        return self._format_event_full(event)

    def _format_event_line(self, e: Dict[str, Any]) -> str:
        """Compact one-line event display."""
        nko = e.get("name_nko", "")
        latin = e.get("name_latin", "")
        date = e.get("date_fixed") or "Lunar"
        desc = e.get("description", "")
        tags = []
        if e.get("is_islamic"):
            tags.append("Islamic")
        if e.get("is_regional"):
            tags.append("Regional")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"    {nko}  {latin:<20} {date:<15} {desc}{tag_str}"

    def _format_event_full(self, e: Dict[str, Any]) -> str:
        """Full event detail display."""
        lines = [
            "",
            f"  {e.get('name_nko', '')}",
            f"  {e.get('name_latin', '')}",
            "  " + "─" * 40,
            f"  Description: {e.get('description', '')}",
        ]
        date = e.get("date_fixed")
        if date:
            lines.append(f"  Date: {date}")
        note = e.get("date_note")
        if note:
            lines.append(f"  Note: {note}")
        tags = []
        if e.get("is_islamic"):
            tags.append("Islamic")
        if e.get("is_regional"):
            tags.append("Regional")
        if tags:
            lines.append(f"  Type: {', '.join(tags)}")
        themes = e.get("themes", [])
        if themes:
            lines.append(f"  Themes: {', '.join(themes)}")
        greetings = e.get("greetings", [])
        if greetings:
            lines.append("  Greetings:")
            for g in greetings:
                lines.append(f"    {g}")
        proverbs = e.get("associated_proverbs", [])
        if proverbs:
            lines.append("  Associated Proverbs:")
            for p in proverbs:
                lines.append(f"    {p}")
        lines.append("")
        return "\n".join(lines)

    def _parse_date_key(self, date_str: str) -> int:
        """Extract month number from a date string like 'April 14'."""
        if not date_str:
            return 13  # Sort lunar events last
        parts = date_str.lower().split()
        if parts:
            return self.MONTH_MAP.get(parts[0], 13)
        return 13


# ── Blessings Browser ───────────────────────────────────────────────


class BlessingsBrowser:
    """Browse blessings, condolences, and congratulations."""

    def __init__(self, culture):
        self._culture = culture

    def random(self) -> str:
        """Display a random blessing."""
        b = self._culture.random_blessing()
        return self._format_blessing(b)

    def list_all(self) -> str:
        """List all blessings grouped by category."""
        all_blessings = self._culture.blessings()
        by_category: Dict[str, List] = {}
        for b in all_blessings:
            cat = b.get("category", "other")
            by_category.setdefault(cat, []).append(b)

        lines = [
            "",
            f"  Blessings & Prayers ({len(all_blessings)} total)",
            "  " + "═" * 60,
        ]
        for cat, entries in sorted(by_category.items()):
            lines.append(f"\n  {cat.title()} ({len(entries)}):")
            lines.append("  " + "─" * 40)
            for b in entries:
                nko = b.get("text_nko", "")
                en = b.get("english_translation", "")
                lines.append(f"    {nko}  — {en}")
        lines.append("")
        return "\n".join(lines)

    def for_event(self, life_event: str) -> str:
        """Get blessings appropriate for a life event."""
        results = [
            b for b in self._culture.blessings()
            if b.get("life_event", "").lower() == life_event.lower()
            or life_event.lower() == "general"
        ]
        if not results:
            return f"  No blessings found for event '{life_event}'"
        lines = [
            "",
            f"  Blessings for '{life_event}' ({len(results)}):",
            "  " + "─" * 50,
        ]
        for b in results:
            lines.append(f"  {self._format_blessing(b)}")
            lines.append("")
        return "\n".join(lines)

    def _format_blessing(self, b: Dict[str, Any]) -> str:
        """Format a single blessing."""
        lines = []
        nko = b.get("text_nko", "")
        latin = b.get("text_latin", "")
        en = b.get("english_translation", "")
        cat = b.get("category", "")
        event = b.get("life_event", "")

        if nko:
            lines.append(f"    {nko}")
        if latin:
            lines.append(f"    {latin}")
        if en:
            lines.append(f"    \"{en}\"")
        meta = []
        if cat:
            meta.append(cat)
        if event:
            meta.append(event)
        if meta:
            lines.append(f"    [{' | '.join(meta)}]")
        return "\n".join(lines)


# ── Greetings Protocol ──────────────────────────────────────────────


class GreetingsProtocol:
    """Explore the multi-turn Manding greeting protocol."""

    def __init__(self, culture):
        self._culture = culture

    def show_protocol(self) -> str:
        """Show the full greeting protocol flow."""
        all_greetings = self._culture.greetings()
        phases = ["opening", "response", "welfare", "family", "work", "closing", "blessing"]
        by_phase: Dict[str, List] = {}
        for g in all_greetings:
            phase = g.get("phase", "other")
            by_phase.setdefault(phase, []).append(g)

        lines = [
            "",
            "  Manding Greeting Protocol",
            "  " + "═" * 60,
            "  Note: Manding greetings are multi-turn exchanges,",
            "  not simple hello/hi pairs.",
            "",
            "  Protocol Flow: OPENING > RESPONSE > WELFARE > FAMILY > WORK > CLOSING > BLESSING",
            "",
        ]
        for phase in phases:
            entries = by_phase.get(phase, [])
            if not entries:
                continue
            lines.append(f"  {phase.upper()} ({len(entries)}):")
            lines.append("  " + "─" * 40)
            for g in entries:
                nko = g.get("text_nko", "")
                en = g.get("english_translation", "")
                role = g.get("speaker_role", "")
                resp = g.get("expected_response", "")
                lines.append(f"    [{role}] {nko} — {en}")
                if resp:
                    lines.append(f"      Expected: {resp}")
            lines.append("")
        return "\n".join(lines)

    def by_phase(self, phase: str) -> str:
        """Show greetings for a specific phase."""
        results = self._culture.greetings_by_phase(phase)
        if not results:
            return f"  No greetings found for phase '{phase}'"
        lines = [
            "",
            f"  {phase.upper()} Phase Greetings ({len(results)}):",
            "  " + "─" * 50,
        ]
        for g in results:
            nko = g.get("text_nko", "")
            en = g.get("english_translation", "")
            role = g.get("speaker_role", "")
            lines.append(f"    [{role}] {nko} — {en}")
        lines.append("")
        return "\n".join(lines)

    def by_time(self, time_context: str) -> str:
        """Show greetings for a time of day."""
        results = [
            g for g in self._culture.greetings()
            if g.get("time_context", "").lower() == time_context.lower()
        ]
        if not results:
            return f"  No greetings found for time '{time_context}'"
        lines = [
            "",
            f"  {time_context.title()} Greetings ({len(results)}):",
            "  " + "─" * 50,
        ]
        for g in results:
            nko = g.get("text_nko", "")
            en = g.get("english_translation", "")
            phase = g.get("phase", "")
            lines.append(f"    [{phase}] {nko} — {en}")
        lines.append("")
        return "\n".join(lines)


# ── Clans Browser ───────────────────────────────────────────────────


class ClansBrowser:
    """Browse Manding clan/family name (jamu) database."""

    def __init__(self, culture):
        self._culture = culture

    def list_all(self) -> str:
        """List all clans."""
        clans = self._culture.clans()
        lines = [
            "",
            f"  Manding Clans — Jamu ({len(clans)} families)",
            "  " + "═" * 60,
            "",
            f"  {'N\'Ko':<10} {'Latin':<15} {'Praise Name':<25} {'Totem':<15} {'Region'}",
            "  " + "─" * 75,
        ]
        for c in clans:
            nko = c.get("name_nko") or ""
            latin = c.get("name_latin") or ""
            jamu = c.get("jamu_meaning") or ""
            totem = c.get("totem") or ""
            region = c.get("region") or ""
            lines.append(f"  {nko:<10} {latin:<15} {jamu:<25} {totem:<15} {region}")
        lines.append("")
        return "\n".join(lines)

    def info(self, name: str) -> str:
        """Show detailed clan info."""
        clans = self._culture.clans()
        clan = None
        for c in clans:
            if (name.lower() in c.get("name_latin", "").lower()
                    or name == c.get("name_nko", "")
                    or name == c.get("id", "")):
                clan = c
                break
        if clan is None:
            return f"  Clan not found: {name}"
        return self._format_clan(clan)

    def _format_clan(self, c: Dict[str, Any]) -> str:
        """Full clan detail display."""
        lines = [
            "",
            f"  {c.get('name_nko', '')}  {c.get('name_latin', '')}",
            "  " + "─" * 40,
            f"  Praise Name (Jamu): {c.get('jamu', '')} — {c.get('jamu_meaning', '')}",
            f"  Totem: {c.get('totem', '')}",
            f"  Region: {c.get('region', '')}",
            f"  Griot Line: {'Yes' if c.get('griot_line') else 'No'}",
            f"  Noble Line: {'Yes' if c.get('noble_line') else 'No'}",
        ]
        greetings = c.get("appropriate_greetings", [])
        if greetings:
            lines.append("  Appropriate Greetings:")
            for g in greetings:
                lines.append(f"    {g}")
        notes = c.get("notes", "")
        if notes:
            lines.append(f"  Notes: {notes}")
        lines.append("")
        return "\n".join(lines)


# ── Concepts Browser ────────────────────────────────────────────────


class ConceptsBrowser:
    """Browse Manding cultural concepts, titles, and kinship terms."""

    def __init__(self, culture):
        self._culture = culture

    def list_all(self) -> str:
        """List all concepts grouped by type."""
        concepts = self._culture.concepts()
        by_type: Dict[str, List] = {}
        for c in concepts:
            ctype = c.get("type", "other")
            by_type.setdefault(ctype, []).append(c)

        lines = [
            "",
            f"  Cultural Concepts ({len(concepts)} total)",
            "  " + "═" * 60,
        ]
        type_labels = {
            "concept": "Core Concepts",
            "title": "Titles & Honorifics",
            "kinship": "Kinship Terms",
            "life_events_reference": "Life Events",
        }
        for ctype, label in type_labels.items():
            entries = by_type.get(ctype, [])
            if not entries:
                continue
            lines.append(f"\n  {label} ({len(entries)}):")
            lines.append("  " + "─" * 40)
            for c in entries:
                nko = c.get("text_nko", "")
                latin = c.get("text_latin", "")
                en = c.get("english_translation", "")
                if nko:
                    lines.append(f"    {nko}  {latin} — {en}")
                else:
                    lines.append(f"    {latin or c.get('id', '')} — {en}")
        lines.append("")
        return "\n".join(lines)

    def by_type(self, concept_type: str) -> str:
        """List concepts of a specific type."""
        results = [
            c for c in self._culture.concepts()
            if c.get("type", "").lower() == concept_type.lower()
        ]
        if not results:
            return f"  No concepts found of type '{concept_type}'"
        lines = [
            "",
            f"  {concept_type.title()} Concepts ({len(results)}):",
            "  " + "─" * 50,
        ]
        for c in results:
            nko = c.get("text_nko", "")
            latin = c.get("text_latin", "")
            en = c.get("english_translation", "")
            usage = c.get("usage", "")
            lines.append(f"    {nko}  {latin}")
            lines.append(f"      {en}")
            if usage:
                lines.append(f"      Usage: {usage}")
            lines.append("")
        return "\n".join(lines)


# ── Main Cultural Tools Router ──────────────────────────────────────


def run_cultural_tools(args: List[str]) -> None:
    """Route 'culture' subcommands to the appropriate tool."""
    from nko.culture import NKoCulture

    if not args:
        _print_culture_help()
        return

    culture = NKoCulture()
    tool = args[0]
    sub_args = args[1:]

    if tool == "proverbs":
        browser = ProverbsBrowser(culture)
        if not sub_args or sub_args[0] == "random":
            print(browser.random())
        elif sub_args[0] == "browse":
            page = int(sub_args[1]) if len(sub_args) > 1 else 1
            print(browser.browse(page=page))
        elif sub_args[0] == "search" and len(sub_args) > 1:
            print(browser.search(" ".join(sub_args[1:])))
        elif sub_args[0] == "categories":
            print(browser.categories())
        elif sub_args[0] == "category" and len(sub_args) > 1:
            print(browser.by_category(sub_args[1]))
        else:
            print("  Usage: nko culture proverbs [random|browse <page>|search <query>|categories|category <name>]")

    elif tool == "calendar":
        cal = CulturalCalendar(culture)
        if not sub_args or sub_args[0] == "list":
            print(cal.list_events())
        elif sub_args[0] == "upcoming":
            print(cal.upcoming())
        elif sub_args[0] == "event" and len(sub_args) > 1:
            print(cal.event_detail(sub_args[1]))
        else:
            print("  Usage: nko culture calendar [list|upcoming|event <id>]")

    elif tool == "sigils":
        player = SigilsPlayer()
        if not sub_args or sub_args[0] == "list":
            print(player.list_sigils())
        elif sub_args[0] == "info" and len(sub_args) > 1:
            print(player.info(sub_args[1]))
        elif sub_args[0] == "play" and len(sub_args) > 1:
            print(player.play(sub_args[1]))
        elif sub_args[0] == "sequence" and len(sub_args) > 1:
            keys = sub_args[1].split()
            print(player.play_sequence(keys))
        elif sub_args[0] == "waveform" and len(sub_args) > 1:
            width = 60
            if len(sub_args) > 2:
                try:
                    width = int(sub_args[2])
                except ValueError:
                    pass
            print(player.waveform(sub_args[1], width=width))
        elif sub_args[0] == "export" and len(sub_args) > 2:
            print(player.export(sub_args[1], sub_args[2]))
        else:
            print("  Usage: nko culture sigils [list|info <sigil>|play <sigil>|waveform <sigil>|export <sigil> <path>]")

    elif tool == "blessings":
        browser = BlessingsBrowser(culture)
        if not sub_args or sub_args[0] == "random":
            print(browser.random())
        elif sub_args[0] == "list":
            print(browser.list_all())
        elif sub_args[0] == "for" and len(sub_args) > 1:
            print(browser.for_event(sub_args[1]))
        else:
            print("  Usage: nko culture blessings [random|list|for <event>]")

    elif tool == "greetings":
        proto = GreetingsProtocol(culture)
        if not sub_args or sub_args[0] == "protocol":
            print(proto.show_protocol())
        elif sub_args[0] == "phase" and len(sub_args) > 1:
            print(proto.by_phase(sub_args[1]))
        elif sub_args[0] == "time" and len(sub_args) > 1:
            print(proto.by_time(sub_args[1]))
        else:
            print("  Usage: nko culture greetings [protocol|phase <phase>|time <time>]")

    elif tool == "clans":
        browser = ClansBrowser(culture)
        if not sub_args or sub_args[0] == "list":
            print(browser.list_all())
        elif sub_args[0] == "info" and len(sub_args) > 1:
            print(browser.info(" ".join(sub_args[1:])))
        else:
            print("  Usage: nko culture clans [list|info <name>]")

    elif tool == "concepts":
        browser = ConceptsBrowser(culture)
        if not sub_args or sub_args[0] == "list":
            print(browser.list_all())
        elif sub_args[0] == "type" and len(sub_args) > 1:
            print(browser.by_type(sub_args[1]))
        else:
            print("  Usage: nko culture concepts [list|type <type>]")

    elif tool == "stats":
        st = culture.stats()
        print("\n  N'Ko Cultural Data Statistics:")
        print("  " + "─" * 40)
        total = 0
        for k, v in st.items():
            print(f"    {k:<15} {v:>4} entries")
            total += v
        print("  " + "─" * 40)
        print(f"    {'TOTAL':<15} {total:>4} entries")
        print()

    else:
        _print_culture_help()


def _print_culture_help() -> None:
    """Print cultural tools help."""
    print("""
  N'Ko Cultural Tools
  ═══════════════════

  Usage: nko culture <tool> [subcommand] [args]

  Tools:
    proverbs    Browse, search, and filter 62 Manding proverbs
                  random              Show a random proverb
                  browse [page]       Paginated browsing
                  search <query>      Search by text
                  categories          List all categories
                  category <name>     Filter by category

    calendar    Cultural and religious calendar
                  list                Show all events
                  upcoming            Events by proximity
                  event <id>          Full event detail

    sigils      N'Ko Sound Sigils — audio signatures
                  list                All 10 sigils
                  info <sigil>        Detailed sigil info
                  play <sigil>        Play through speakers
                  waveform <sigil>    ASCII waveform
                  export <sigil> <path>  Export to WAV

    blessings   Blessings, condolences, congratulations
                  random              Random blessing
                  list                All grouped by category
                  for <event>         For a life event

    greetings   Multi-turn Manding greeting protocol
                  protocol            Full protocol flow
                  phase <phase>       By phase (opening, welfare, etc.)
                  time <time>         By time of day

    clans       Manding clan/family database (jamu)
                  list                All clans
                  info <name>         Clan detail

    concepts    Cultural concepts, titles, kinship terms
                  list                All grouped by type
                  type <type>         Filter by type

    stats       Show cultural data statistics
""")
