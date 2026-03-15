#!/usr/bin/env python3
"""
N'Ko Syllable Structure FSM — Admissibility Checker

N'Ko (Manding) has a regular CV/CVN syllable structure:
  - Consonant-Vowel (CV): ߓߊ (ba), ߞߏ (kɔ)
  - Consonant-Vowel-Nasal (CVN): ߞߊ߲ (kan)
  - Vowel-initial (V/VN): ߊ (a), ߊ߲ (an)

This FSM validates whether generated text maintains valid syllable
structure, used as a logits mask during MLX generation.

States:
  START  — beginning of syllable (expects consonant or vowel)
  ONSET  — consumed consonant (expects vowel nucleus)
  NUCLEUS — consumed vowel (expects consonant/tone/end)
  CODA   — consumed nasal coda or tone mark (expects next syllable)
"""

from __future__ import annotations

import sys
from enum import Enum, auto
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# Import N'Ko character classes from phonetics
_NKO_ROOT = Path.home() / "Desktop" / "NKo"
if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))

from nko.phonetics import (
    VOWEL_CHARS,
    CONSONANT_CHARS,
    TONE_MARK_CHARS,
    COMBINING_CHARS,
    DIGIT_CHARS,
    PUNCTUATION_CHARS,
    ALL_NKO_CHARS,
)

# N'Ko Unicode range
NKO_START = 0x07C0
NKO_END = 0x07FF

# Nasal consonants that can appear in coda position (CVN pattern)
NASAL_CHARS: frozenset = frozenset({"ߒ", "ߡ", "ߣ", "ߧ", "ߢ", "ߠ"})

# Nasalization combining marks
NASAL_MARKS: frozenset = frozenset({"߲", "߳"})


class FSMState(Enum):
    START = auto()    # Beginning of syllable
    ONSET = auto()    # After consonant (expecting vowel)
    NUCLEUS = auto()  # After vowel (expecting coda/new onset/end)
    CODA = auto()     # After nasal coda or tone (expecting new syllable)


class NKoSyllableFSM:
    """
    Finite state machine for N'Ko syllable structure validation.

    Tracks state across characters and determines whether appending
    a given token would maintain valid syllable structure.
    """

    def __init__(self):
        self.state: FSMState = FSMState.START
        self._partial: str = ""  # Tracks partial context for compound chars

    def reset(self):
        self.state = FSMState.START
        self._partial = ""

    def clone(self) -> "NKoSyllableFSM":
        fsm = NKoSyllableFSM()
        fsm.state = self.state
        fsm._partial = self._partial
        return fsm

    def _classify_char(self, ch: str) -> str:
        """Classify a single character for FSM transitions."""
        if ch in VOWEL_CHARS:
            return "vowel"
        if ch in CONSONANT_CHARS:
            return "consonant"
        if ch in TONE_MARK_CHARS or ch in COMBINING_CHARS:
            return "tone"
        if ch in NASAL_MARKS:
            return "nasal_mark"
        if ch in DIGIT_CHARS:
            return "digit"
        if ch in PUNCTUATION_CHARS:
            return "punct"
        if ch == " " or ch == "\n" or ch == "\t":
            return "space"
        return "other"

    def advance(self, ch: str) -> FSMState:
        """
        Advance FSM by one character. Returns new state.

        Transition table:
          START + consonant → ONSET
          START + vowel     → NUCLEUS
          ONSET + vowel     → NUCLEUS
          ONSET + consonant → ONSET  (compound onset, e.g., gb in ߜ is one char)
          NUCLEUS + tone    → NUCLEUS (tone attaches to vowel)
          NUCLEUS + nasal   → CODA    (nasal mark = nasalization)
          NUCLEUS + consonant → ONSET (new syllable)
          NUCLEUS + vowel   → NUCLEUS (vowel hiatus — allowed in Manding)
          CODA + consonant  → ONSET
          CODA + vowel      → NUCLEUS (vowel-initial syllable after coda)
        """
        cat = self._classify_char(ch)

        if cat in ("space", "punct", "digit", "other"):
            # Non-phonemic: reset to START
            self.state = FSMState.START
            return self.state

        if self.state == FSMState.START:
            if cat == "consonant":
                self.state = FSMState.ONSET
            elif cat == "vowel":
                self.state = FSMState.NUCLEUS
            elif cat == "tone" or cat == "nasal_mark":
                pass  # Stray tone at start — stay in START

        elif self.state == FSMState.ONSET:
            if cat == "vowel":
                self.state = FSMState.NUCLEUS
            elif cat == "consonant":
                # Compound onset (rare, but possible: gg, gb already single chars)
                self.state = FSMState.ONSET
            elif cat == "tone":
                # Tone on consonant (unusual but not invalid)
                pass

        elif self.state == FSMState.NUCLEUS:
            if cat == "tone" or cat == "nasal_mark":
                # Tone/nasalization attaches to vowel, stay in NUCLEUS
                pass
            elif cat == "consonant":
                # New syllable onset
                self.state = FSMState.ONSET
            elif cat == "vowel":
                # Vowel hiatus (new syllable, vowel-initial)
                self.state = FSMState.NUCLEUS

        elif self.state == FSMState.CODA:
            if cat == "consonant":
                self.state = FSMState.ONSET
            elif cat == "vowel":
                self.state = FSMState.NUCLEUS
            elif cat == "tone":
                pass  # Tone on coda — stay

        return self.state

    def advance_token(self, token_text: str) -> FSMState:
        """Advance FSM through all characters in a token."""
        for ch in token_text:
            self.advance(ch)
        return self.state

    def is_admissible(self, text: str) -> bool:
        """
        Check if a complete text has valid N'Ko syllable structure.

        Returns True if the text can be parsed as a sequence of valid
        CV/CVN/V/VN syllables. Non-N'Ko characters are ignored.
        """
        fsm = NKoSyllableFSM()
        for ch in text:
            if not (NKO_START <= ord(ch) <= NKO_END):
                # Non-N'Ko char resets (spaces, Latin, etc.)
                fsm.state = FSMState.START
                continue
            fsm.advance(ch)

        # Valid end states: START (after space), NUCLEUS (after vowel),
        # CODA (after nasal). ONSET alone is invalid (consonant without vowel)
        return fsm.state in (FSMState.START, FSMState.NUCLEUS, FSMState.CODA)

    def would_be_admissible(self, token_text: str) -> bool:
        """
        Check if appending this token would leave FSM in a valid state.

        Does NOT modify the current FSM state — works on a clone.
        """
        test = self.clone()
        for ch in token_text:
            cp = ord(ch)
            if NKO_START <= cp <= NKO_END:
                test.advance(ch)
            else:
                test.state = FSMState.START
        # ONSET is the only "mid-syllable" invalid end state
        return test.state != FSMState.ONSET

    def get_valid_token_mask(self, vocab_tokens: List[str]) -> List[bool]:
        """
        Precompute which tokens are admissible from the current state.

        Returns a boolean list aligned with vocab_tokens.
        """
        mask = []
        for token_text in vocab_tokens:
            # Non-N'Ko tokens always pass through
            has_nko = any(NKO_START <= ord(ch) <= NKO_END for ch in token_text)
            if not has_nko:
                mask.append(True)
            else:
                mask.append(self.would_be_admissible(token_text))
        return mask

    def syllable_count(self, text: str) -> int:
        """Count syllables in N'Ko text by counting vowel nuclei."""
        count = 0
        for ch in text:
            if ch in VOWEL_CHARS:
                count += 1
        return max(count, 0)

    def valid_syllable_ratio(self, text: str) -> float:
        """
        Measure syllable validity by detecting CC violations and stranded consonants.

        Counts transitions and violations:
        - CC violation: consonant immediately following consonant (ONSET → consonant)
        - Stranded consonant: text ending in ONSET state
        Returns 1.0 - (violations / total_transitions) for N'Ko characters.
        """
        nko_chars = [ch for ch in text if NKO_START <= ord(ch) <= NKO_END]
        if not nko_chars:
            return 1.0

        fsm = NKoSyllableFSM()
        total_transitions = 0
        violations = 0

        for ch in nko_chars:
            cat = fsm._classify_char(ch)
            if cat in ("tone", "nasal_mark"):
                # Tone marks don't count as structural transitions
                fsm.advance(ch)
                continue

            total_transitions += 1
            old_state = fsm.state

            # CC violation: consonant after consonant (no intervening vowel)
            if cat == "consonant" and old_state == FSMState.ONSET:
                violations += 1

            fsm.advance(ch)

        # Stranded consonant at end
        if fsm.state == FSMState.ONSET and total_transitions > 0:
            violations += 1

        if total_transitions == 0:
            return 1.0

        return 1.0 - (violations / total_transitions)
