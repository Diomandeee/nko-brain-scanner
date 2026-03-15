#!/usr/bin/env python3
"""
Admissibility-Constrained Logits Processor for MLX Generation

Hooks into mlx_lm's generate_step() as a logits_processor callable.
Masks tokens that would violate N'Ko syllable structure (CV/CVN FSM).

Usage with mlx_lm:
    from constrained.logits_processor import NKoAdmissibilityProcessor

    processor = NKoAdmissibilityProcessor(tokenizer)
    # Use with mlx_lm.utils.generate_step:
    for (token, logits) in generate_step(prompt, model, ...):
        logits = processor(tokens_so_far, logits)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

from constrained.nko_fsm import NKoSyllableFSM, FSMState, NKO_START, NKO_END


class NKoAdmissibilityProcessor:
    """
    MLX logits processor that constrains generation to valid N'Ko syllables.

    Only constrains tokens containing N'Ko characters (U+07C0-U+07FF).
    English/punctuation/special tokens pass through unconstrained.

    The processor precomputes per-state validity masks over the full
    vocabulary for efficiency (4 states x vocab_size boolean arrays).
    """

    def __init__(self, tokenizer, penalty: float = float("inf"), model_vocab_size: int = 0):
        """
        Args:
            tokenizer: HuggingFace-compatible tokenizer with decode().
            penalty: Logit penalty for inadmissible tokens (default: -inf).
            model_vocab_size: Actual model vocab size (if > tokenizer.vocab_size,
                              extended tokens are allowed by default).
        """
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.fsm = NKoSyllableFSM()
        self.tokenizer_vocab_size = tokenizer.vocab_size
        self.model_vocab_size = max(model_vocab_size, tokenizer.vocab_size)

        # Decode all tokens once
        self._token_texts: List[str] = []
        self._has_nko: List[bool] = []

        for tid in range(self.tokenizer_vocab_size):
            try:
                text = tokenizer.decode([tid])
            except Exception:
                text = ""
            self._token_texts.append(text)
            self._has_nko.append(
                any(NKO_START <= ord(ch) <= NKO_END for ch in text)
            )

        # Precompute masks per FSM state
        self._state_masks: dict[FSMState, mx.array] = {}
        self._precompute_masks()

    def _precompute_masks(self):
        """Build boolean mask for each FSM state, padded to model vocab size."""
        for state in FSMState:
            mask = []
            for tid in range(self.tokenizer_vocab_size):
                if not self._has_nko[tid]:
                    mask.append(True)
                    continue
                test_fsm = NKoSyllableFSM()
                test_fsm.state = state
                admissible = test_fsm.would_be_admissible(self._token_texts[tid])
                mask.append(admissible)

            # Pad for extended vocab tokens (N'Ko BPE tokens are always allowed)
            extra = self.model_vocab_size - self.tokenizer_vocab_size
            if extra > 0:
                mask.extend([True] * extra)

            self._state_masks[state] = mx.array(mask)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply admissibility constraint to logits.

        Args:
            tokens: Previously generated token IDs.
            logits: Raw logits for next token, shape [vocab_size] or [1, vocab_size].

        Returns:
            Modified logits with inadmissible N'Ko tokens masked to -inf.
        """
        current_mask = self._state_masks.get(self.fsm.state)
        if current_mask is None:
            return logits

        penalty_mask = mx.where(current_mask, 0.0, -self.penalty)

        # Handle both 1D and 2D logits (generate_step may pass either)
        if logits.ndim == 2:
            penalty_mask = penalty_mask[None, :]  # [1, vocab_size]

        # Trim or pad mask if logits size differs
        if penalty_mask.shape[-1] != logits.shape[-1]:
            v = logits.shape[-1]
            if penalty_mask.shape[-1] > v:
                penalty_mask = penalty_mask[..., :v]
            else:
                pad_size = v - penalty_mask.shape[-1]
                pad = mx.zeros((pad_size,) if logits.ndim == 1 else (1, pad_size))
                penalty_mask = mx.concatenate([penalty_mask, pad], axis=-1)

        return logits + penalty_mask

    def update_state(self, token_id: int):
        """
        Update FSM state after a token is selected.
        Call this after each generation step.
        """
        if token_id < len(self._token_texts):
            text = self._token_texts[token_id]
            self.fsm.advance_token(text)

    def reset(self):
        """Reset FSM to initial state for a new generation."""
        self.fsm.reset()


class RepetitionPenaltyProcessor:
    """Penalizes recently generated tokens to reduce mode collapse."""

    def __init__(self, penalty: float = 1.3, window: int = 32):
        self.penalty = penalty
        self.window = window
        self._recent: List[int] = []

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if not self._recent:
            return logits
        recent = self._recent[-self.window:]
        # Build penalty array
        penalties = mx.zeros(logits.shape)
        for tid in set(recent):
            if tid < logits.shape[-1]:
                if logits.ndim == 2:
                    val = logits[0, tid].item()
                else:
                    val = logits[tid].item()
                adj = -val * (1 - 1.0 / self.penalty) if val > 0 else -val * (self.penalty - 1)
                if logits.ndim == 2:
                    penalties = penalties.at[0, tid].add(adj)
                else:
                    penalties = penalties.at[tid].add(adj)
        return logits + penalties

    def update(self, token_id: int):
        self._recent.append(token_id)


def constrained_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    """
    Generate text with N'Ko admissibility constraints.

    Uses mlx_lm's generate_step with logits_processors parameter
    to mask tokens violating N'Ko syllable structure during sampling.
    Includes repetition penalty to reduce mode collapse.
    """
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    # Detect model vocab size from model config
    model_vocab = getattr(model, 'vocab_size', 0)
    if not model_vocab:
        try:
            model_vocab = model.model.embed_tokens.weight.shape[0]
        except Exception:
            model_vocab = 0
    fsm_processor = NKoAdmissibilityProcessor(tokenizer, model_vocab_size=model_vocab)
    rep_processor = RepetitionPenaltyProcessor(penalty=repetition_penalty)
    tokens = tokenizer.encode(prompt)
    prompt_array = mx.array(tokens)  # 1D — generate_step adds batch dim internally
    sampler = make_sampler(temp=temperature, top_p=top_p)

    generated = []
    for result in generate_step(
        prompt_array,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=[fsm_processor, rep_processor],
    ):
        token = result[0]
        token_id = token if isinstance(token, int) else token.item()
        fsm_processor.update_state(token_id)
        rep_processor.update(token_id)
        generated.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)
