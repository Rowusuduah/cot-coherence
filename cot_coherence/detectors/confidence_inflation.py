"""Detector for confidence inflation — unjustified jumps from hedging to certainty."""

from __future__ import annotations

import re

from ..models import IncoherenceFlag, IncoherenceType, ReasoningStep, Severity
from .base import BaseDetector

HEDGE_WORDS = frozenset({
    "possibly", "might", "could", "perhaps", "may", "maybe", "likely",
    "probably", "uncertain", "unclear", "seems", "appears", "suggest",
    "suggests", "roughly", "approximately", "around", "arguably",
    "potentially", "presumably", "conceivably", "tentatively",
})

CERTAINTY_WORDS = frozenset({
    "certainly", "definitely", "clearly", "must", "always", "never",
    "obviously", "undoubtedly", "absolutely", "proven", "guarantees",
    "guaranteed", "without doubt", "unquestionably", "indisputably",
    "conclusively", "inevitably", "surely", "plainly", "evidently",
})

EVIDENCE_MARKERS = frozenset({
    "because", "since", "evidence", "data", "shows", "demonstrates",
    "according", "study", "research", "found", "results", "proves",
    "confirmed", "measured", "observed", "experiment", "analysis",
})

_WORD_RE = re.compile(r"\b[a-z]+\b")


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _hedge_ratio(words: set[str]) -> float:
    hedge_count = len(words & HEDGE_WORDS)
    certainty_count = len(words & CERTAINTY_WORDS)
    total = hedge_count + certainty_count
    if total == 0:
        return 0.5  # neutral
    return hedge_count / total


def _has_evidence(words: set[str]) -> bool:
    return len(words & EVIDENCE_MARKERS) >= 1


class ConfidenceInflationDetector(BaseDetector):
    """Detects unjustified jumps from hedging language to certainty."""

    def detect(
        self,
        steps: list[ReasoningStep],
        original_question: str = "",
    ) -> list[IncoherenceFlag]:
        flags: list[IncoherenceFlag] = []
        if len(steps) < 2:
            return flags

        prev_words = _word_set(steps[0].text)
        prev_ratio = _hedge_ratio(prev_words)

        for i in range(1, len(steps)):
            curr_words = _word_set(steps[i].text)
            curr_ratio = _hedge_ratio(curr_words)
            has_ev = _has_evidence(curr_words)

            # Detect: previous step was hedging (ratio > 0.5) and current is certain (ratio < 0.2)
            # without evidence markers to justify the shift
            if prev_ratio > 0.5 and curr_ratio < 0.2 and not has_ev:
                certainty_count = len(curr_words & CERTAINTY_WORDS)
                if certainty_count == 0:
                    prev_words = curr_words
                    prev_ratio = curr_ratio
                    continue

                hedge_examples = sorted(prev_words & HEDGE_WORDS)[:3]
                cert_examples = sorted(curr_words & CERTAINTY_WORDS)[:3]

                severity = Severity.HIGH if curr_ratio == 0.0 else Severity.MEDIUM
                confidence = min(1.0, 0.5 + (prev_ratio - curr_ratio) * 0.5)

                flags.append(
                    IncoherenceFlag(
                        type=IncoherenceType.CONFIDENCE_INFLATION,
                        severity=severity,
                        confidence=round(confidence, 2),
                        step_range=(i - 1, i),
                        summary=(
                            f"Confidence jumps from hedging to certainty between "
                            f"steps {i - 1} and {i} without supporting evidence."
                        ),
                        evidence=(
                            f"Step {i - 1} hedges: {hedge_examples}. "
                            f"Step {i} asserts: {cert_examples}. "
                            f"No evidence markers found in step {i}."
                        ),
                        suggestion=(
                            "Add evidence or reasoning to justify the increased confidence, "
                            "or maintain appropriate hedging."
                        ),
                    )
                )

            prev_words = curr_words
            prev_ratio = curr_ratio

        return flags
