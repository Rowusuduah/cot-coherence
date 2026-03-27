"""Detector for circular return — steps that repeat earlier reasoning."""

from __future__ import annotations

import re

from ..models import IncoherenceFlag, IncoherenceType, ReasoningStep, Severity
from .base import BaseDetector

_WORD_RE = re.compile(r"\b[a-z][a-z]+\b")

STOP_WORDS = frozenset({
    "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "so", "than", "too", "very", "just", "but", "and",
    "or", "yet", "an", "that", "this", "it", "its", "we", "they", "them",
    "their", "what", "which", "who", "these", "those", "about", "also", "any",
    "like",
})


def _content_words(text: str) -> set[str]:
    words = set(_WORD_RE.findall(text.lower()))
    return words - STOP_WORDS


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class CircularReturnDetector(BaseDetector):
    """Detects when reasoning circles back to repeat earlier steps."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold

    def detect(
        self,
        steps: list[ReasoningStep],
        original_question: str = "",
    ) -> list[IncoherenceFlag]:
        flags: list[IncoherenceFlag] = []
        if len(steps) < 3:
            return flags

        # Compute fingerprints
        fingerprints = [_content_words(step.text) for step in steps]

        # Compare each step against non-adjacent prior steps
        flagged_pairs: set[tuple[int, int]] = set()

        for i in range(2, len(steps)):
            for j in range(0, i - 1):  # Skip adjacent (i-1)
                if (j, i) in flagged_pairs:
                    continue

                sim = _jaccard(fingerprints[j], fingerprints[i])
                if sim >= self.threshold:
                    flagged_pairs.add((j, i))

                    gap = i - j
                    severity = (
                        Severity.CRITICAL if sim >= 0.85
                        else Severity.HIGH if sim >= 0.7
                        else Severity.MEDIUM
                    )
                    confidence = round(sim, 2)

                    shared = sorted(fingerprints[j] & fingerprints[i])[:5]

                    flags.append(
                        IncoherenceFlag(
                            type=IncoherenceType.CIRCULAR_RETURN,
                            severity=severity,
                            confidence=confidence,
                            step_range=(j, i),
                            summary=(
                                f"Step {i} repeats the reasoning from step {j} "
                                f"({gap} steps apart, {sim:.0%} similarity)."
                            ),
                            evidence=(
                                f"Shared content words: {shared}. "
                                f"Jaccard similarity: {sim:.2f}."
                            ),
                            suggestion=(
                                "Avoid restating earlier reasoning. "
                                "Build forward or reference the earlier step instead."
                            ),
                        )
                    )

        return flags
