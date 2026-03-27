"""Detector for premise abandonment — premises introduced but never followed up."""

from __future__ import annotations

import re

from ..models import IncoherenceFlag, IncoherenceType, ReasoningStep, Severity
from .base import BaseDetector

PREMISE_MARKERS = frozenset({
    "assume", "assuming", "given", "given that", "if", "since", "because",
    "suppose", "supposing", "let's say", "considering", "provided",
    "granted", "starting from", "based on",
})

_WORD_RE = re.compile(r"\b[a-z][a-z]+\b")

STOP_WORDS = frozenset({
    "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "dare", "ought", "used", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "yet", "an", "that", "this", "it", "its", "we", "they",
    "them", "their", "what", "which", "who", "whom", "these", "those",
    "am", "about", "also", "any", "like",
})


def _content_words(text: str) -> set[str]:
    words = set(_WORD_RE.findall(text.lower()))
    return words - STOP_WORDS


def _has_premise_marker(text: str) -> bool:
    lower = text.lower()
    for marker in PREMISE_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", lower):
            return True
    return False


def _extract_premise_entities(text: str) -> set[str]:
    """Extract content words from premise-containing sentence."""
    lower = text.lower()
    # Find the sentence containing a premise marker
    sentences = re.split(r"[.!?]+", lower)
    for sentence in sentences:
        for marker in PREMISE_MARKERS:
            if re.search(r"\b" + re.escape(marker) + r"\b", sentence):
                return _content_words(sentence)
    return _content_words(lower)


class PremiseAbandonmentDetector(BaseDetector):
    """Detects premises that are introduced but never referenced in subsequent steps."""

    def __init__(self, window: int = 3) -> None:
        self.window = window

    def detect(
        self,
        steps: list[ReasoningStep],
        original_question: str = "",
    ) -> list[IncoherenceFlag]:
        flags: list[IncoherenceFlag] = []
        if len(steps) < 2:
            return flags

        for i, step in enumerate(steps):
            if not _has_premise_marker(step.text):
                continue

            premise_entities = _extract_premise_entities(step.text)
            if len(premise_entities) < 2:
                continue

            # Check if subsequent steps reference premise entities
            end = min(i + self.window + 1, len(steps))
            subsequent_words: set[str] = set()
            for j in range(i + 1, end):
                subsequent_words |= _content_words(steps[j].text)

            overlap = premise_entities & subsequent_words
            overlap_ratio = len(overlap) / len(premise_entities) if premise_entities else 1.0

            if overlap_ratio < 0.15:
                # Almost no premise entities appear in subsequent steps
                last_step = min(i + self.window, len(steps) - 1)
                severity = Severity.HIGH if overlap_ratio == 0 else Severity.MEDIUM
                confidence = round(1.0 - overlap_ratio, 2)

                abandoned = sorted(premise_entities - subsequent_words)[:5]
                flags.append(
                    IncoherenceFlag(
                        type=IncoherenceType.PREMISE_ABANDONMENT,
                        severity=severity,
                        confidence=confidence,
                        step_range=(i, last_step),
                        summary=(
                            f"Premise introduced in step {i} is abandoned — "
                            f"key entities not referenced in steps {i + 1}-{last_step}."
                        ),
                        evidence=(
                            f"Premise entities abandoned: {abandoned}. "
                            f"Overlap ratio: {overlap_ratio:.0%} "
                            f"(checked steps {i + 1} through {last_step})."
                        ),
                        suggestion=(
                            "Either build on the premise in subsequent reasoning, "
                            "or remove it if it's not relevant to the conclusion."
                        ),
                    )
                )

        return flags
