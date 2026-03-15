#!/usr/bin/env python3
"""
Graph Kernel Semantic Scorer for N'Ko Generation

Queries the Graph Kernel for N'Ko word-level knowledge triples
and applies soft logit boosts to tokens forming known valid words.

This is an optional extension to the FSM-based admissibility
constraint — it adds semantic awareness on top of phonotactic rules.

GK API: GET /api/knowledge?subject=<word>&predicate=is_valid
Falls back gracefully if GK is offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import URLError

import mlx.core as mx


# GK endpoint (tunneled from cloud-vm to Mac1)
GK_BASE_URL = "http://localhost:8001"
GK_TIMEOUT = 2  # seconds — fail fast if GK is down

# N'Ko Unicode range
NKO_START = 0x07C0
NKO_END = 0x07FF


def _is_nko_word(text: str) -> bool:
    """Check if text is primarily N'Ko characters."""
    nko_count = sum(1 for ch in text if NKO_START <= ord(ch) <= NKO_END)
    return nko_count > len(text) * 0.5


class GKScorer:
    """
    Queries Graph Kernel for valid N'Ko words and collocations.

    Applies soft logit boosts (+boost_value) to tokens that would
    complete or form known valid N'Ko words from the knowledge graph.
    """

    def __init__(self, boost_value: float = 2.0, cache_size: int = 1000):
        self.boost_value = boost_value
        self.available = self._check_availability()
        self._word_cache: Dict[str, bool] = {}
        self._collocation_cache: Dict[str, List[str]] = {}
        self._cache_size = cache_size

        if self.available:
            self._preload_nko_words()

    def _check_availability(self) -> bool:
        """Check if GK is reachable."""
        try:
            req = Request(f"{GK_BASE_URL}/health", method="GET")
            resp = urlopen(req, timeout=GK_TIMEOUT)
            return resp.status == 200
        except Exception:
            return False

    def _gk_query(self, endpoint: str, params: dict) -> Optional[dict]:
        """Query GK API with params."""
        if not self.available:
            return None
        try:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{GK_BASE_URL}{endpoint}?{query}"
            req = Request(url, method="GET")
            resp = urlopen(req, timeout=GK_TIMEOUT)
            return json.loads(resp.read().decode())
        except Exception:
            return None

    def _preload_nko_words(self):
        """Load known N'Ko words from GK at startup."""
        result = self._gk_query("/api/knowledge", {
            "subject": "nko-suite",
            "predicate": "has_word",
            "limit": "500",
        })
        if result and isinstance(result, dict):
            triples = result.get("triples", [])
            for triple in triples:
                obj = triple.get("object", "")
                if _is_nko_word(obj):
                    self._word_cache[obj] = True

        # Also try to get common collocations
        result = self._gk_query("/api/knowledge", {
            "subject": "nko-suite",
            "predicate": "common_bigram",
            "limit": "200",
        })
        if result and isinstance(result, dict):
            triples = result.get("triples", [])
            for triple in triples:
                obj = triple.get("object", "")
                if " " in obj:
                    parts = obj.split()
                    if len(parts) == 2:
                        self._collocation_cache.setdefault(parts[0], []).append(parts[1])

    def is_valid_word(self, word: str) -> bool:
        """Check if a word is known valid N'Ko."""
        if word in self._word_cache:
            return self._word_cache[word]

        # Query GK for this specific word
        result = self._gk_query("/api/knowledge", {
            "subject": word,
            "predicate": "is_valid",
        })
        valid = False
        if result and isinstance(result, dict):
            triples = result.get("triples", [])
            valid = len(triples) > 0

        # Cache (with LRU-like eviction)
        if len(self._word_cache) >= self._cache_size:
            oldest = next(iter(self._word_cache))
            del self._word_cache[oldest]
        self._word_cache[word] = valid
        return valid

    def get_collocations(self, word: str) -> List[str]:
        """Get known words that commonly follow this word."""
        if word in self._collocation_cache:
            return self._collocation_cache[word]

        result = self._gk_query("/api/knowledge", {
            "subject": word,
            "predicate": "followed_by",
            "limit": "20",
        })
        collocations = []
        if result and isinstance(result, dict):
            triples = result.get("triples", [])
            for triple in triples:
                obj = triple.get("object", "")
                if _is_nko_word(obj):
                    collocations.append(obj)

        self._collocation_cache[word] = collocations
        return collocations

    def score_tokens(
        self,
        context_text: str,
        token_texts: List[str],
    ) -> mx.array:
        """
        Compute logit boosts for each token in vocabulary.

        Boosts tokens that:
        1. Complete a known valid N'Ko word
        2. Start a word that commonly follows the last word

        Args:
            context_text: Generated text so far.
            token_texts: List of decoded token strings (aligned with vocab).

        Returns:
            mx.array of logit adjustments (0.0 for no boost, +boost_value for boosted).
        """
        if not self.available:
            return mx.zeros(len(token_texts))

        boosts = [0.0] * len(token_texts)

        # Extract the last N'Ko word being formed
        words = context_text.split()
        last_word = ""
        partial_word = ""
        for ch in reversed(context_text):
            if ch == " " or ch == "\n":
                break
            partial_word = ch + partial_word

        if words:
            last_word = words[-1] if not partial_word else ""

        # Strategy 1: Boost tokens completing a valid word
        if partial_word and _is_nko_word(partial_word):
            for i, tok in enumerate(token_texts):
                candidate = partial_word + tok
                if self.is_valid_word(candidate):
                    boosts[i] = self.boost_value

        # Strategy 2: Boost tokens starting a collocated word
        if last_word and _is_nko_word(last_word):
            collocations = self.get_collocations(last_word)
            colloc_set = set(collocations)
            for i, tok in enumerate(token_texts):
                tok_clean = tok.strip()
                if tok_clean in colloc_set:
                    boosts[i] = self.boost_value * 0.5  # Softer boost for collocations
                elif any(c.startswith(tok_clean) for c in colloc_set if tok_clean):
                    boosts[i] = self.boost_value * 0.25

        return mx.array(boosts)


class GKScoringProcessor:
    """
    Combined logits processor using FSM constraint + GK semantic scoring.

    Composes with NKoAdmissibilityProcessor — apply FSM first, then GK boost.
    """

    def __init__(self, tokenizer, boost_value: float = 2.0):
        self.tokenizer = tokenizer
        self.scorer = GKScorer(boost_value=boost_value)
        self._generated_text = ""

        # Pre-decode vocabulary
        self._token_texts: List[str] = []
        for tid in range(tokenizer.vocab_size):
            try:
                self._token_texts.append(tokenizer.decode([tid]))
            except Exception:
                self._token_texts.append("")

    @property
    def available(self) -> bool:
        return self.scorer.available

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Apply GK semantic boosts to logits."""
        if not self.scorer.available:
            return logits

        boosts = self.scorer.score_tokens(self._generated_text, self._token_texts)
        return logits + boosts

    def update_state(self, token_id: int):
        """Update generated text context."""
        if token_id < len(self._token_texts):
            self._generated_text += self._token_texts[token_id]

    def reset(self):
        self._generated_text = ""
