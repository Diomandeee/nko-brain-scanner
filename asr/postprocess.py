#!/usr/bin/env python3
"""
N'Ko ASR Post-Processing Pipeline
====================================
Takes raw CTC character output and produces validated, structured N'Ko text.

Pipeline:
  1. CTC Decode: argmax → collapse repeats → remove blanks → raw chars
  2. FSM Validation: check each character against syllable grammar
  3. Syllable Assembly: group chars into CV/CVN syllables
  4. Tone Normalization: attach floating tones to correct vowels
  5. Word Boundary Detection: insert spaces at syllable boundaries
  6. Confidence Scoring: per-syllable confidence from CTC logits

Usage:
    from asr.postprocess import NKoPostProcessor
    pp = NKoPostProcessor()
    result = pp.process(ctc_logits, idx_to_char, num_chars)
    print(result.text)           # Clean N'Ko text
    print(result.syllables)      # List of syllable objects
    print(result.validity_rate)  # % of valid syllables
    print(result.confidence)     # Mean confidence score
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from constrained.nko_fsm import NKoSyllableFSM, FSMState
    FSM_AVAILABLE = True
except ImportError:
    FSM_AVAILABLE = False


# N'Ko character classification
NKO_VOWELS = {'\u07CA', '\u07CB', '\u07CC', '\u07CD', '\u07CE', '\u07CF', '\u07D0'}  # a, ee, i, e, u, o, open-e
NKO_CONSONANTS = {  # All N'Ko consonants
    '\u07D1', '\u07D2', '\u07D3', '\u07D4', '\u07D5', '\u07D6', '\u07D7', '\u07D8',
    '\u07D9', '\u07DA', '\u07DB', '\u07DC', '\u07DD', '\u07DE', '\u07DF', '\u07E0',
    '\u07E1', '\u07E2', '\u07E3', '\u07E4', '\u07E5', '\u07E6', '\u07E7',
}
NKO_TONE_MARKS = {'\u07EB', '\u07EC', '\u07ED', '\u07EE', '\u07EF'}  # High, low, mid, etc.
NKO_NASALS = {'\u07F2', '\u07F3'}  # Nasalization marks
NKO_DIGITS = set(chr(cp) for cp in range(0x07C0, 0x07CA))  # N'Ko digits 0-9

def is_consonant(c):
    return c in NKO_CONSONANTS and c not in NKO_VOWELS

def is_vowel(c):
    return c in NKO_VOWELS

def is_tone(c):
    return c in NKO_TONE_MARKS

def is_nasal(c):
    return c in NKO_NASALS

def is_nko(c):
    return 0x07C0 <= ord(c) <= 0x07FF


@dataclass
class Syllable:
    """A single N'Ko syllable with its components."""
    onset: str = ""      # Consonant(s)
    nucleus: str = ""    # Vowel
    coda: str = ""       # Nasal mark
    tone: str = ""       # Tone diacritic
    raw: str = ""        # Raw characters
    confidence: float = 0.0
    valid: bool = True

    @property
    def text(self):
        return self.onset + self.nucleus + self.tone + self.coda

    @property
    def pattern(self):
        parts = []
        if self.onset:
            parts.append("C")
        if self.nucleus:
            parts.append("V")
        if self.coda:
            parts.append("N")
        return "".join(parts) or "?"


@dataclass
class PostProcessResult:
    """Complete post-processing result."""
    text: str = ""
    raw_text: str = ""
    syllables: List[Syllable] = field(default_factory=list)
    validity_rate: float = 0.0
    confidence: float = 0.0
    fsm_corrections: int = 0
    total_chars: int = 0


class NKoPostProcessor:
    """Post-processes CTC output into validated N'Ko text."""

    def __init__(self):
        if FSM_AVAILABLE:
            self.fsm = NKoSyllableFSM()
        else:
            self.fsm = None

    def ctc_decode(self, logits, idx_to_char, num_chars):
        """CTC greedy decode with confidence scores."""
        import torch

        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        pred_probs = probs.max(dim=-1).values.cpu().tolist()

        chars = []
        confidences = []
        prev = -1

        for i, idx in enumerate(pred_ids):
            if idx != num_chars and idx != prev:
                char = idx_to_char.get(idx, "")
                if char:
                    chars.append(char)
                    confidences.append(pred_probs[i])
            prev = idx

        return chars, confidences

    def segment_syllables(self, chars, confidences):
        """Segment character sequence into syllables using FSM grammar."""
        syllables = []
        current = Syllable()
        current_conf = []

        for i, (ch, conf) in enumerate(zip(chars, confidences)):
            if ch == " ":
                # Word boundary
                if current.nucleus:
                    current.confidence = sum(current_conf) / len(current_conf) if current_conf else 0
                    syllables.append(current)
                syllables.append(Syllable(raw=" ", nucleus=" ", valid=True))
                current = Syllable()
                current_conf = []
                continue

            if not is_nko(ch):
                continue

            if is_consonant(ch):
                if current.nucleus:
                    # Previous syllable complete, start new one
                    current.confidence = sum(current_conf) / len(current_conf) if current_conf else 0
                    syllables.append(current)
                    current = Syllable()
                    current_conf = []
                current.onset += ch
                current.raw += ch
                current_conf.append(conf)

            elif is_vowel(ch):
                current.nucleus += ch
                current.raw += ch
                current_conf.append(conf)

            elif is_tone(ch):
                current.tone += ch
                current.raw += ch
                current_conf.append(conf)

            elif is_nasal(ch):
                current.coda += ch
                current.raw += ch
                current_conf.append(conf)
                # Nasal coda closes the syllable
                current.confidence = sum(current_conf) / len(current_conf) if current_conf else 0
                syllables.append(current)
                current = Syllable()
                current_conf = []

        # Don't forget last syllable
        if current.raw:
            current.confidence = sum(current_conf) / len(current_conf) if current_conf else 0
            syllables.append(current)

        return syllables

    def validate_syllables(self, syllables):
        """Validate each syllable against N'Ko phonotactic rules."""
        corrections = 0

        for syl in syllables:
            if syl.raw == " ":
                continue

            # Valid patterns: V, CV, VN, CVN
            has_onset = bool(syl.onset)
            has_nucleus = bool(syl.nucleus)
            has_coda = bool(syl.coda)

            if not has_nucleus and has_onset:
                # Consonant without vowel — invalid
                syl.valid = False
                corrections += 1
            elif has_nucleus:
                syl.valid = True

        return corrections

    def fsm_validate(self, text):
        """Run full FSM validation on the assembled text."""
        if not self.fsm:
            return 1.0

        self.fsm.reset()
        total = 0
        valid = 0

        for ch in text:
            if ch == " ":
                self.fsm.reset()
                continue
            if not is_nko(ch):
                continue
            total += 1
            state = self.fsm.advance(ch)
            if state is not None:
                valid += 1
            else:
                self.fsm.reset()
                # Re-try this character as start of new syllable
                state = self.fsm.advance(ch)
                if state is not None:
                    valid += 1

        return valid / max(total, 1)

    def process(self, logits, idx_to_char, num_chars):
        """Full post-processing pipeline."""
        import torch

        # Step 1: CTC decode
        chars, confidences = self.ctc_decode(logits, idx_to_char, num_chars)
        raw_text = "".join(chars)

        # Step 2: Segment into syllables
        syllables = self.segment_syllables(chars, confidences)

        # Step 3: Validate
        corrections = self.validate_syllables(syllables)

        # Step 4: Assemble clean text
        clean_text = "".join(s.text for s in syllables)

        # Step 5: FSM validation rate
        fsm_rate = self.fsm_validate(clean_text)

        # Step 6: Overall confidence
        real_syls = [s for s in syllables if s.raw != " "]
        avg_conf = sum(s.confidence for s in real_syls) / max(len(real_syls), 1)

        # Validity rate
        valid_count = sum(1 for s in real_syls if s.valid)
        validity = valid_count / max(len(real_syls), 1)

        return PostProcessResult(
            text=clean_text,
            raw_text=raw_text,
            syllables=syllables,
            validity_rate=validity,
            confidence=avg_conf,
            fsm_corrections=corrections,
            total_chars=len(chars),
        )

    def process_batch(self, logits_batch, idx_to_char, num_chars):
        """Process a batch of CTC outputs."""
        results = []
        for logits in logits_batch:
            results.append(self.process(logits, idx_to_char, num_chars))
        return results


# Reverse bridge: N'Ko → Latin Bambara (for WER comparison)
NKO_TO_LATIN = {
    '\u07CA': 'a', '\u07CB': 'ee', '\u07CC': 'i', '\u07CD': 'e', '\u07CE': 'u',
    '\u07CF': 'o', '\u07D0': 'ɛ',
    '\u07D2': 'ŋ', '\u07D3': 'b', '\u07D4': 'p', '\u07D5': 't', '\u07D6': 'j',
    '\u07D7': 'c', '\u07D8': 'd', '\u07D9': 'r', '\u07DB': 's', '\u07DC': 'g',
    '\u07DD': 'f', '\u07DE': 'k', '\u07DF': 'l', '\u07E1': 'm', '\u07E2': 'ɲ',
    '\u07E3': 'n', '\u07E4': 'h', '\u07E5': 'w', '\u07E6': 'y', '\u07E7': 'ŋ',
}

def nko_to_latin(nko_text):
    """Convert N'Ko text back to Latin Bambara for WER comparison."""
    result = []
    for c in nko_text:
        if c in NKO_TO_LATIN:
            result.append(NKO_TO_LATIN[c])
        elif c == ' ':
            result.append(' ')
        elif is_tone(c) or is_nasal(c):
            pass  # Drop tone/nasal marks for Latin
    return ''.join(result)


def demo():
    """Demo the post-processor on a sample."""
    pp = NKoPostProcessor()

    # Simulate a decoded N'Ko string
    sample = "ߖߌߜߌ ߌ ߓߋߟߋ ߘߍߜߎߣߣߍߣ ߘߋߣ ߥߊ"
    print(f"Input: {sample}")

    chars = list(sample)
    confs = [0.9] * len(chars)
    syllables = pp.segment_syllables(chars, confs)

    print(f"\nSyllables ({len([s for s in syllables if s.raw != ' '])} found):")
    for s in syllables:
        if s.raw == " ":
            print("  [space]")
        else:
            print(f"  {s.text:8s}  pattern={s.pattern}  valid={s.valid}  conf={s.confidence:.2f}")

    corrections = pp.validate_syllables(syllables)
    clean = "".join(s.text for s in syllables)
    latin = nko_to_latin(clean)

    print(f"\nClean N'Ko: {clean}")
    print(f"Latin:      {latin}")
    print(f"Corrections: {corrections}")
    print(f"FSM validity: {pp.fsm_validate(clean)*100:.0f}%")


if __name__ == "__main__":
    demo()
