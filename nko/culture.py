"""
nko.culture — Unified Cultural Data Access
ߒߞߏ ߟߊ߬ߘߍ — N'Ko Cultural Heritage Module

Provides programmatic access to the consolidated cultural data
created in NKO-1.4:
  • Proverbs (62 entries)
  • Blessings & Prayers (29 entries)
  • Greetings (23 entries)
  • Clan/Family Names
  • Cultural Calendar
  • Cultural Concepts

All data lives in data/cultural/*.json and is loaded lazily on first access.

Usage::

    from nko.culture import NKoCulture

    culture = NKoCulture()
    proverb = culture.random_proverb()
    print(proverb['text_nko'], '—', proverb['text_latin'])

    greetings = culture.greetings()
    clans = culture.clans()
    calendar = culture.calendar()
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_data_dir() -> Path:
    """Locate the data/cultural/ directory relative to this package."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "data" / "cultural",           # Inside installed package
        pkg_dir.parent / "data" / "cultural",     # Dev layout (project root)
        Path.cwd() / "data" / "cultural",         # Fallback: cwd
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Cannot locate data/cultural/ directory. "
        "Ensure you're running from the NKo project root or the package is installed correctly."
    )


class NKoCulture:
    """Unified access to N'Ko cultural data files."""

    _FILES = {
        "proverbs": "proverbs-unified.json",
        "blessings": "blessings-unified.json",
        "greetings": "greetings-unified.json",
        "clans": "clans-unified.json",
        "calendar": "cultural-calendar.json",
        "concepts": "cultural-concepts.json",
    }

    def __init__(self, data_dir: Optional[str | Path] = None):
        self._data_dir = Path(data_dir) if data_dir else _find_data_dir()
        self._cache: Dict[str, Any] = {}

    def _load(self, key: str) -> Dict[str, Any]:
        """Load and cache a cultural data file."""
        if key not in self._cache:
            fpath = self._data_dir / self._FILES[key]
            if not fpath.exists():
                raise FileNotFoundError(f"Cultural data file not found: {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                self._cache[key] = json.load(f)
        return self._cache[key]

    # ── Proverbs ─────────────────────────────────────────────

    def proverbs(self) -> List[Dict[str, Any]]:
        """Return all proverbs."""
        data = self._load("proverbs")
        return data.get("proverbs", data.get("entries", []))

    def random_proverb(self) -> Dict[str, Any]:
        """Return a random proverb."""
        entries = self.proverbs()
        if not entries:
            return {"text_nko": "", "text_latin": "No proverbs loaded"}
        return random.choice(entries)

    def search_proverbs(self, query: str) -> List[Dict[str, Any]]:
        """Search proverbs by text (case-insensitive, searches all text fields)."""
        q = query.lower()
        results = []
        for p in self.proverbs():
            searchable = " ".join(
                str(p.get(k, ""))
                for k in ("text_nko", "text_latin", "text_arabic",
                          "english_translation", "translation_en", "meaning",
                          "applied_context")
            ).lower()
            if q in searchable:
                results.append(p)
        return results

    # ── Blessings ────────────────────────────────────────────

    def blessings(self) -> List[Dict[str, Any]]:
        """Return all blessings and prayers."""
        data = self._load("blessings")
        return data.get("blessings", data.get("entries", []))

    def random_blessing(self) -> Dict[str, Any]:
        """Return a random blessing."""
        entries = self.blessings()
        if not entries:
            return {"text_nko": "", "text_latin": "No blessings loaded"}
        return random.choice(entries)

    # ── Greetings ────────────────────────────────────────────

    def greetings(self) -> List[Dict[str, Any]]:
        """Return all greetings."""
        data = self._load("greetings")
        return data.get("greetings", data.get("entries", []))

    def greetings_by_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Filter greetings by protocol phase (e.g. 'opening', 'welfare', 'closing')."""
        return [g for g in self.greetings() if g.get("phase", "").lower() == phase.lower()]

    # ── Clans ────────────────────────────────────────────────

    def clans(self) -> List[Dict[str, Any]]:
        """Return all clan/family name entries."""
        data = self._load("clans")
        return data.get("clans", data.get("entries", []))

    # ── Calendar ─────────────────────────────────────────────

    def calendar(self) -> List[Dict[str, Any]]:
        """Return cultural calendar entries."""
        data = self._load("calendar")
        return data.get("events", data.get("entries", data.get("months", [])))

    # ── Concepts ─────────────────────────────────────────────

    def concepts(self) -> List[Dict[str, Any]]:
        """Return cultural concepts."""
        data = self._load("concepts")
        return data.get("concepts", data.get("entries", []))

    # ── Metadata ─────────────────────────────────────────────

    def meta(self, dataset: str) -> Dict[str, Any]:
        """Return metadata for a given dataset."""
        data = self._load(dataset)
        return data.get("meta", {})

    def available_datasets(self) -> List[str]:
        """Return list of available dataset names."""
        return list(self._FILES.keys())

    def stats(self) -> Dict[str, int]:
        """Return record counts per dataset."""
        counts = {}
        for key in self._FILES:
            try:
                data = self._load(key)
                # Try common list keys
                for list_key in (key, "entries", "proverbs", "blessings", "greetings",
                                  "clans", "events", "months", "concepts"):
                    if list_key in data and isinstance(data[list_key], list):
                        counts[key] = len(data[list_key])
                        break
                else:
                    counts[key] = 0
            except FileNotFoundError:
                counts[key] = 0
        return counts

    def __repr__(self) -> str:
        return f"NKoCulture(data_dir={self._data_dir})"


# ── Module-level convenience functions ───────────────────────

_default: Optional[NKoCulture] = None


def _get_default() -> NKoCulture:
    global _default
    if _default is None:
        _default = NKoCulture()
    return _default


def proverbs() -> List[Dict[str, Any]]:
    """Return all proverbs."""
    return _get_default().proverbs()


def random_proverb() -> Dict[str, Any]:
    """Return a random proverb."""
    return _get_default().random_proverb()


def search_proverbs(query: str) -> List[Dict[str, Any]]:
    """Search proverbs."""
    return _get_default().search_proverbs(query)


def blessings() -> List[Dict[str, Any]]:
    """Return all blessings."""
    return _get_default().blessings()


def greetings() -> List[Dict[str, Any]]:
    """Return all greetings."""
    return _get_default().greetings()


def clans() -> List[Dict[str, Any]]:
    """Return all clans."""
    return _get_default().clans()


def calendar() -> List[Dict[str, Any]]:
    """Return cultural calendar."""
    return _get_default().calendar()


def concepts() -> List[Dict[str, Any]]:
    """Return cultural concepts."""
    return _get_default().concepts()


def stats() -> Dict[str, int]:
    """Return cultural dataset statistics."""
    return _get_default().stats()


__all__ = [
    "NKoCulture",
    "proverbs",
    "random_proverb",
    "search_proverbs",
    "blessings",
    "greetings",
    "clans",
    "calendar",
    "concepts",
    "stats",
]
