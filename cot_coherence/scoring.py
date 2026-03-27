"""Scoring engine — converts incoherence flags into coherence scores."""

from __future__ import annotations

from .config import CoherenceConfig
from .models import IncoherenceFlag, IncoherenceType, Severity

SEVERITY_PENALTY: dict[Severity, float] = {
    Severity.LOW: 0.05,
    Severity.MEDIUM: 0.12,
    Severity.HIGH: 0.22,
    Severity.CRITICAL: 0.35,
}


def compute_pattern_scores(
    flags: list[IncoherenceFlag],
    config: CoherenceConfig,
) -> dict[IncoherenceType, float]:
    """Compute per-pattern coherence scores.

    Each pattern starts at 1.0. For each flag of that type,
    subtract SEVERITY_PENALTY[severity] * confidence. Clamp to [0, 1].
    """
    scores: dict[IncoherenceType, float] = {}

    for itype in IncoherenceType:
        if itype not in config.enabled_detectors:
            continue
        score = 1.0
        for flag in flags:
            if flag.type == itype:
                penalty = SEVERITY_PENALTY[flag.severity] * flag.confidence
                score -= penalty
        scores[itype] = max(0.0, min(1.0, score))

    return scores


def compute_overall_score(
    pattern_scores: dict[IncoherenceType, float],
    config: CoherenceConfig,
) -> float:
    """Compute weighted average of pattern scores."""
    if not pattern_scores:
        return 1.0

    total_weight = 0.0
    weighted_sum = 0.0

    for itype, score in pattern_scores.items():
        weight = config.weights.get(itype, 1.0)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 1.0

    return round(max(0.0, min(1.0, weighted_sum / total_weight)), 2)
