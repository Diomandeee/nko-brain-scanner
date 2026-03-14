"""
Distribution-based partial credit scoring for evaluation probes.

Matches the scoring methodology from the RYS paper (Ng):
- Math probes: partial credit based on distance from correct answer
- Semantic probes: keyword matching with weighted scoring

Usage::

    from probes.scoring import score_math, score_semantic, score_probes
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Optional


def score_math(predicted: str, correct: int, tolerance: float = 0.1) -> float:
    """
    Score a math answer with distribution-based partial credit.

    Full credit (1.0) for exact match. Partial credit decays
    exponentially with relative distance from correct answer.

    Parameters
    ----------
    predicted : str
        Model's predicted answer (extracted number).
    correct : int
        The correct numerical answer.
    tolerance : float
        Relative tolerance for full credit (default 10%).

    Returns
    -------
    float
        Score between 0.0 and 1.0.
    """
    # Extract number from model output
    numbers = re.findall(r'-?\d+\.?\d*', predicted)
    if not numbers:
        return 0.0

    # Take the last number (models often show work then answer)
    try:
        pred_num = float(numbers[-1])
    except ValueError:
        return 0.0

    if correct == 0:
        return 1.0 if abs(pred_num) < 0.01 else 0.0

    # Relative error
    rel_error = abs(pred_num - correct) / abs(correct)

    # Full credit within tolerance
    if rel_error <= tolerance:
        return 1.0

    # Exponential decay for partial credit
    # Score = exp(-5 * relative_error) — drops to ~0.007 at 100% error
    score = math.exp(-5.0 * rel_error)
    return max(0.0, min(1.0, score))


def score_semantic(predicted: str, keywords: List[str], threshold: int = 2) -> float:
    """
    Score a semantic answer based on keyword presence.

    Parameters
    ----------
    predicted : str
        Model's response text.
    keywords : List[str]
        Expected keywords/phrases.
    threshold : int
        Minimum keywords for full credit.

    Returns
    -------
    float
        Score between 0.0 and 1.0.
    """
    predicted_lower = predicted.lower()
    matched = sum(1 for kw in keywords if kw.lower() in predicted_lower)

    if matched >= threshold:
        return 1.0
    elif matched > 0:
        return matched / threshold
    return 0.0


def load_probes(probe_type: str) -> List[dict]:
    """Load probe questions from JSON files."""
    path = Path(__file__).parent / f"{probe_type}_probes.json"
    with open(path) as f:
        data = json.load(f)
    return data["probes"]


def score_math_probes(responses: Dict[str, str]) -> Dict[str, float]:
    """
    Score all math probe responses.

    Parameters
    ----------
    responses : Dict[str, str]
        Mapping of probe_id -> model's text response.

    Returns
    -------
    Dict[str, float]
        Mapping of probe_id -> score (0.0 to 1.0).
    """
    probes = load_probes("math")
    scores = {}
    for probe in probes:
        pid = probe["id"]
        if pid in responses:
            scores[pid] = score_math(responses[pid], probe["answer"])
        else:
            scores[pid] = 0.0
    return scores


def score_semantic_probes(responses: Dict[str, str]) -> Dict[str, float]:
    """
    Score all semantic probe responses.

    Parameters
    ----------
    responses : Dict[str, str]
        Mapping of probe_id -> model's text response.

    Returns
    -------
    Dict[str, float]
        Mapping of probe_id -> score (0.0 to 1.0).
    """
    probes = load_probes("semantic")
    scores = {}
    for probe in probes:
        pid = probe["id"]
        if pid in responses:
            scores[pid] = score_semantic(
                responses[pid],
                probe["answer_keywords"],
            )
        else:
            scores[pid] = 0.0
    return scores


def aggregate_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Compute aggregate statistics for a set of scores."""
    values = list(scores.values())
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "perfect": sum(1 for v in values if v >= 0.99),
        "zero": sum(1 for v in values if v < 0.01),
    }
