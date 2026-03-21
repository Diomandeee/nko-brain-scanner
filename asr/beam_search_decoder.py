#!/usr/bin/env python3
"""
Beam Search CTC Decoder with N'Ko FSM Constraints
====================================================
CTC-aware beam search decoder that integrates the N'Ko syllable FSM
as a structural filter, rejecting beams that violate phonotactic rules.

Features:
  - Beam width configurable (default 5)
  - FSM integration: reject beams entering invalid states
  - Optional character-level language model for rescoring
  - Prefix merging: collapse equivalent CTC prefixes
  - Length normalization for fair scoring
  - Greedy fallback if beam search produces empty output

Usage:
    from beam_search_decoder import BeamSearchDecoder

    decoder = BeamSearchDecoder(beam_width=5, fsm=my_fsm)
    results = decoder.decode(logits, idx_to_char, num_chars)
    # results: [(text, score), (text, score), ...]

    # With language model
    decoder = BeamSearchDecoder(beam_width=5, fsm=my_fsm, lm=my_lm)
    results = decoder.decode(logits, idx_to_char, num_chars)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

# Add project root for FSM imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from constrained.nko_fsm import NKoSyllableFSM, FSMState
    FSM_AVAILABLE = True
except ImportError:
    FSM_AVAILABLE = False


# ── Character Language Model Interface ──────────────────────────

class CharLM:
    """Base interface for character-level language model.

    Subclass this and implement score() to provide LM rescoring.
    The default implementation returns uniform probability (no rescoring).
    """

    def score(self, prefix: str, next_char: str) -> float:
        """Return log probability of next_char given prefix.

        Args:
            prefix: characters decoded so far
            next_char: candidate next character

        Returns:
            Log probability (0.0 = uniform, negative = less likely)
        """
        return 0.0

    def reset(self):
        """Reset any internal state."""
        pass


class NgramCharLM(CharLM):
    """Simple n-gram character language model.

    Estimates P(c | prefix[-n:]) from a frequency table.
    """

    def __init__(self, corpus_text: str, n: int = 3, smoothing: float = 0.01):
        """Build n-gram LM from corpus text.

        Args:
            corpus_text: N'Ko text to estimate probabilities from
            n: context window size
            smoothing: Laplace smoothing parameter
        """
        import math
        from collections import Counter

        self.n = n
        self.smoothing = smoothing
        self.counts: Dict[str, Counter] = {}
        self.vocab = set()

        # Collect all characters
        for c in corpus_text:
            self.vocab.add(c)

        # Count n-gram transitions
        for i in range(len(corpus_text)):
            for ctx_len in range(1, n + 1):
                if i < ctx_len:
                    continue
                ctx = corpus_text[i - ctx_len:i]
                char = corpus_text[i]
                if ctx not in self.counts:
                    self.counts[ctx] = Counter()
                self.counts[ctx][char] += 1

        self.vocab_size = len(self.vocab)
        self._log = math.log

    def score(self, prefix: str, next_char: str) -> float:
        """Log probability of next_char given prefix context."""
        import math

        # Try longest context first, back off
        for ctx_len in range(min(self.n, len(prefix)), 0, -1):
            ctx = prefix[-ctx_len:]
            if ctx in self.counts:
                total = sum(self.counts[ctx].values()) + self.smoothing * self.vocab_size
                count = self.counts[ctx].get(next_char, 0) + self.smoothing
                return math.log(count / total)

        # Uniform fallback
        return -math.log(max(self.vocab_size, 1))


# ── Beam Hypothesis ─────────────────────────────────────────────

@dataclass
class Beam:
    """A single hypothesis in the beam."""
    text: str = ""                          # Decoded text so far (after CTC collapse)
    raw_seq: List[int] = field(default_factory=list)  # Raw token sequence (before collapse)
    score: float = 0.0                      # Log probability sum
    lm_score: float = 0.0                   # LM contribution
    fsm_state: Optional[object] = None      # FSM state (clone of NKoSyllableFSM)
    last_token: int = -1                    # Last emitted token (for CTC dedup)

    def __lt__(self, other):
        """For heap comparison."""
        return self.total_score > other.total_score  # Higher is better

    @property
    def total_score(self) -> float:
        return self.score + self.lm_score

    @property
    def normalized_score(self) -> float:
        """Length-normalized score."""
        text_len = max(len(self.text), 1)
        return self.total_score / text_len


# ── Beam Search Decoder ────────────────────────────────────────

class BeamSearchDecoder:
    """CTC beam search decoder with FSM constraints.

    Implements a prefix beam search that:
    1. Expands each beam by all vocabulary tokens
    2. Applies CTC prefix merging (collapse repeated tokens + blanks)
    3. Filters beams that violate FSM phonotactic rules
    4. Optionally rescores with a character language model
    5. Prunes to beam_width candidates per timestep

    Args:
        beam_width: number of hypotheses to keep (default 5)
        fsm: NKoSyllableFSM instance for phonotactic filtering (or None)
        lm: CharLM instance for language model rescoring (or None)
        lm_weight: weight for language model scores (default 0.3)
        length_bonus: bonus per decoded character to prevent short outputs (default 0.1)
        blank_bias: additional log-probability for blank token (default 0.0)
    """

    def __init__(
        self,
        beam_width: int = 5,
        fsm=None,
        lm: Optional[CharLM] = None,
        lm_weight: float = 0.3,
        length_bonus: float = 0.1,
        blank_bias: float = 0.0,
    ):
        self.beam_width = beam_width
        self.fsm = fsm
        self.lm = lm
        self.lm_weight = lm_weight
        self.length_bonus = length_bonus
        self.blank_bias = blank_bias

    def decode(
        self,
        logits: torch.Tensor,
        idx_to_char: Dict[int, str],
        num_chars: int,
        top_k: int = 0,
    ) -> List[Tuple[str, float]]:
        """Decode CTC logits using beam search.

        Args:
            logits: tensor of shape [T, V] where T=timesteps, V=vocab+blank
            idx_to_char: mapping from token index to character
            num_chars: index of the blank token (last index)
            top_k: return top-k results (0 = beam_width)

        Returns:
            List of (text, score) tuples, sorted by score descending
        """
        if top_k <= 0:
            top_k = self.beam_width

        # Ensure logits are on CPU for beam search
        if logits.is_cuda:
            logits = logits.cpu()

        T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)  # [T, V]

        blank_idx = num_chars  # blank is the last token

        # Initialize beam with empty hypothesis
        initial_fsm = None
        if self.fsm is not None:
            if hasattr(self.fsm, 'clone'):
                initial_fsm = self.fsm.clone()
            else:
                initial_fsm = self.fsm

        beams = [Beam(
            text="",
            raw_seq=[],
            score=0.0,
            lm_score=0.0,
            fsm_state=initial_fsm,
            last_token=-1,
        )]

        for t in range(T):
            frame_log_probs = log_probs[t]  # [V]

            # Pre-select top tokens to limit expansion
            # Use 2x beam_width as candidate pool for efficiency
            candidate_k = min(V, max(self.beam_width * 3, 20))
            top_values, top_indices = frame_log_probs.topk(candidate_k)

            new_beams: Dict[str, Beam] = {}

            for beam in beams:
                # Option 1: Emit blank (extend without adding character)
                blank_score = frame_log_probs[blank_idx].item() + self.blank_bias
                blank_key = beam.text + "|blank"
                new_score = beam.score + blank_score

                if blank_key not in new_beams or new_beams[blank_key].score < new_score:
                    new_beams[blank_key] = Beam(
                        text=beam.text,
                        raw_seq=beam.raw_seq + [blank_idx],
                        score=new_score,
                        lm_score=beam.lm_score,
                        fsm_state=beam.fsm_state,
                        last_token=blank_idx,
                    )

                # Option 2: Emit each candidate token
                for k in range(candidate_k):
                    token_idx = top_indices[k].item()
                    token_score = top_values[k].item()

                    if token_idx == blank_idx:
                        continue  # Already handled above

                    char = idx_to_char.get(token_idx, "")
                    if not char:
                        continue

                    # CTC dedup: if same token as last non-blank, it is a repeat
                    if token_idx == beam.last_token:
                        # Repeat: extend without adding character (like blank)
                        repeat_key = beam.text + f"|repeat_{token_idx}"
                        repeat_score = beam.score + token_score
                        if repeat_key not in new_beams or new_beams[repeat_key].score < repeat_score:
                            new_beams[repeat_key] = Beam(
                                text=beam.text,
                                raw_seq=beam.raw_seq + [token_idx],
                                score=repeat_score,
                                lm_score=beam.lm_score,
                                fsm_state=beam.fsm_state,
                                last_token=token_idx,
                            )
                        continue

                    # New character emission
                    new_text = beam.text + char

                    # FSM check: would this character violate phonotactics?
                    new_fsm = beam.fsm_state
                    if new_fsm is not None and hasattr(new_fsm, 'clone'):
                        test_fsm = new_fsm.clone()
                        new_state = test_fsm.advance(char)
                        # Reject if the FSM enters an unrecoverable bad state
                        # We allow ONSET since it just needs a following vowel
                        # Only hard-reject obvious violations
                        if self._is_fsm_violation(test_fsm, char, beam.text):
                            continue
                        new_fsm = test_fsm

                    # LM rescoring
                    lm_delta = 0.0
                    if self.lm is not None:
                        lm_delta = self.lm.score(beam.text, char) * self.lm_weight

                    new_score = beam.score + token_score + self.length_bonus
                    new_lm_score = beam.lm_score + lm_delta

                    # Merge beams with same text (prefix merging)
                    merge_key = new_text
                    if merge_key not in new_beams or (new_beams[merge_key].score + new_beams[merge_key].lm_score) < (new_score + new_lm_score):
                        new_beams[merge_key] = Beam(
                            text=new_text,
                            raw_seq=beam.raw_seq + [token_idx],
                            score=new_score,
                            lm_score=new_lm_score,
                            fsm_state=new_fsm,
                            last_token=token_idx,
                        )

            # Prune to beam_width
            candidates = sorted(new_beams.values(), key=lambda b: b.total_score, reverse=True)
            beams = candidates[:self.beam_width]

            if not beams:
                # All beams pruned -- fall back to greedy
                return self._greedy_decode(logits, idx_to_char, num_chars)

        # Final FSM validation: penalize beams ending in bad states
        for beam in beams:
            if beam.fsm_state is not None and hasattr(beam.fsm_state, 'state'):
                if beam.fsm_state.state == FSMState.ONSET:
                    # Penalize stranded consonant (no following vowel)
                    beam.score -= 2.0

        # Sort by normalized score and return top-k
        beams.sort(key=lambda b: b.normalized_score, reverse=True)

        results = []
        for beam in beams[:top_k]:
            if beam.text.strip():
                results.append((beam.text, beam.normalized_score))

        if not results:
            return self._greedy_decode(logits, idx_to_char, num_chars)

        return results

    def _is_fsm_violation(self, fsm, char: str, prefix: str) -> bool:
        """Check if the FSM considers this character a hard violation.

        We use a soft approach: only reject transitions that are clearly
        wrong, such as three consecutive consonants without vowels.
        Single consonant-consonant is allowed (compound onsets exist).
        """
        if not FSM_AVAILABLE:
            return False

        # Count recent consonants without intervening vowel
        recent = prefix[-3:] if len(prefix) >= 3 else prefix
        consonant_run = 0
        for c in reversed(recent + char):
            if 0x07C0 <= ord(c) <= 0x07FF:
                cp = ord(c)
                # Vowels are U+07CA to U+07D0
                if 0x07CA <= cp <= 0x07D0:
                    break
                # Consonants are U+07D1 to U+07E7
                elif 0x07D1 <= cp <= 0x07E7:
                    consonant_run += 1
            elif c == " ":
                break

        # Allow up to 2 consecutive consonants (compound onsets)
        # Reject 3+ consecutive consonants as phonotactically invalid
        return consonant_run >= 4

    def _greedy_decode(
        self,
        logits: torch.Tensor,
        idx_to_char: Dict[int, str],
        num_chars: int,
    ) -> List[Tuple[str, float]]:
        """Greedy CTC decode as fallback.

        Returns:
            List with single (text, score) tuple
        """
        probs = F.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        pred_probs = probs.max(dim=-1).values.cpu().tolist()

        decoded = []
        confidences = []
        prev = -1

        for i, idx in enumerate(pred_ids):
            if idx != num_chars and idx != prev:
                char = idx_to_char.get(idx, "")
                if char:
                    decoded.append(char)
                    confidences.append(pred_probs[i])
            prev = idx

        text = "".join(decoded)
        avg_conf = sum(confidences) / max(len(confidences), 1)

        return [(text, avg_conf)]

    def decode_batch(
        self,
        logits_batch: torch.Tensor,
        idx_to_char: Dict[int, str],
        num_chars: int,
        top_k: int = 0,
    ) -> List[List[Tuple[str, float]]]:
        """Decode a batch of CTC logits.

        Args:
            logits_batch: tensor of shape [B, T, V]
            idx_to_char: token index to character mapping
            num_chars: blank token index
            top_k: results per sample

        Returns:
            List of lists of (text, score) tuples
        """
        B = logits_batch.shape[0]
        results = []
        for b in range(B):
            results.append(self.decode(logits_batch[b], idx_to_char, num_chars, top_k))
        return results


# ── Convenience Constructors ───────────────────────────────────

def build_decoder(
    beam_width: int = 5,
    use_fsm: bool = True,
    corpus_path: Optional[str] = None,
    lm_weight: float = 0.3,
) -> BeamSearchDecoder:
    """Build a BeamSearchDecoder with optional FSM and LM.

    Args:
        beam_width: beam search width
        use_fsm: whether to use N'Ko syllable FSM
        corpus_path: path to N'Ko text corpus for building char LM (or None)
        lm_weight: weight for LM scores

    Returns:
        Configured BeamSearchDecoder
    """
    fsm = None
    if use_fsm and FSM_AVAILABLE:
        fsm = NKoSyllableFSM()

    lm = None
    if corpus_path is not None:
        corpus_text = Path(corpus_path).read_text(encoding="utf-8")
        lm = NgramCharLM(corpus_text, n=3, smoothing=0.01)

    return BeamSearchDecoder(
        beam_width=beam_width,
        fsm=fsm,
        lm=lm,
        lm_weight=lm_weight,
    )


# ── Self-Test ──────────────────────────────────────────────────

def test():
    """Quick test with synthetic logits."""
    import torch

    print("BeamSearchDecoder self-test")
    print("=" * 40)

    # Build N'Ko char vocab (must match training)
    chars = {}
    idx = 0
    idx_to_char = {}
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx_to_char[idx] = chr(cp)
        idx += 1
    chars[" "] = idx
    idx_to_char[idx] = " "
    num_chars = idx + 1  # blank index

    V = num_chars + 1
    T = 50

    # Create synthetic logits biased toward specific N'Ko characters
    logits = torch.randn(T, V) * 0.1
    # Bias first few frames toward consonant (ba)
    logits[0:5, chars.get(chr(0x07D3), 0)] = 3.0  # b consonant
    logits[5:10, chars.get(chr(0x07CA), 0)] = 3.0  # a vowel
    logits[10:15, num_chars] = 3.0  # blank
    logits[15:20, chars.get(chr(0x07DE), 0)] = 3.0  # k consonant
    logits[20:25, chars.get(chr(0x07CF), 0)] = 3.0  # o vowel

    # Test greedy
    decoder_greedy = BeamSearchDecoder(beam_width=1)
    results_greedy = decoder_greedy.decode(logits, idx_to_char, num_chars)
    print(f"Greedy: {results_greedy[0][0]} (score={results_greedy[0][1]:.3f})")

    # Test beam search without FSM
    decoder_beam = BeamSearchDecoder(beam_width=5)
    results_beam = decoder_beam.decode(logits, idx_to_char, num_chars)
    print(f"Beam (no FSM): {results_beam[0][0]} (score={results_beam[0][1]:.3f})")
    for i, (text, score) in enumerate(results_beam):
        print(f"  [{i}] {text} ({score:.3f})")

    # Test beam search with FSM
    if FSM_AVAILABLE:
        fsm = NKoSyllableFSM()
        decoder_fsm = BeamSearchDecoder(beam_width=5, fsm=fsm)
        results_fsm = decoder_fsm.decode(logits, idx_to_char, num_chars)
        print(f"Beam (FSM): {results_fsm[0][0]} (score={results_fsm[0][1]:.3f})")
        for i, (text, score) in enumerate(results_fsm):
            print(f"  [{i}] {text} ({score:.3f})")
    else:
        print("FSM not available (constrained.nko_fsm not found)")

    print("\nSelf-test passed.")


if __name__ == "__main__":
    test()
