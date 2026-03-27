"""Detector for conclusion drift — conclusions that shift topic mid-chain."""

from __future__ import annotations

import re

from ..models import IncoherenceFlag, IncoherenceType, ReasoningStep, Severity
from .base import BaseDetector

CONCLUSION_MARKERS = frozenset({
    "therefore", "thus", "so", "conclude", "concluding", "conclusion",
    "this means", "hence", "consequently", "as a result", "it follows",
    "in summary", "to summarize", "overall", "finally", "ultimately",
})

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
    "like", "therefore", "thus", "hence", "consequently", "means",
})


def _content_words(text: str) -> set[str]:
    words = set(_WORD_RE.findall(text.lower()))
    return words - STOP_WORDS


def _has_conclusion_marker(text: str) -> bool:
    lower = text.lower()
    for marker in CONCLUSION_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", lower):
            return True
    return False


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class ConclusionDriftDetector(BaseDetector):
    """Detects when conclusions shift topic between steps."""

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def detect(
        self,
        steps: list[ReasoningStep],
        original_question: str = "",
    ) -> list[IncoherenceFlag]:
        flags: list[IncoherenceFlag] = []
        if len(steps) < 2:
            return flags

        # Find steps with conclusion markers
        conclusion_steps: list[tuple[int, set[str]]] = []
        for i, step in enumerate(steps):
            if _has_conclusion_marker(step.text):
                conclusion_steps.append((i, _content_words(step.text)))

        if len(conclusion_steps) < 2:
            return flags

        # Compare adjacent conclusion topics
        for k in range(1, len(conclusion_steps)):
            prev_idx, prev_words = conclusion_steps[k - 1]
            curr_idx, curr_words = conclusion_steps[k]

            if not prev_words or not curr_words:
                continue

            similarity = _jaccard(prev_words, curr_words)

            if similarity < self.threshold:
                severity = Severity.CRITICAL if similarity == 0 else Severity.HIGH
                confidence = round(1.0 - similarity, 2)

                prev_topics = sorted(prev_words)[:5]
                curr_topics = sorted(curr_words)[:5]

                flags.append(
                    IncoherenceFlag(
                        type=IncoherenceType.CONCLUSION_DRIFT,
                        severity=severity,
                        confidence=confidence,
                        step_range=(prev_idx, curr_idx),
                        summary=(
                            f"Conclusion in step {curr_idx} drifts from the topic "
                            f"established in step {prev_idx} "
                            f"(similarity: {similarity:.0%})."
                        ),
                        evidence=(
                            f"Step {prev_idx} concludes about: {prev_topics}. "
                            f"Step {curr_idx} concludes about: {curr_topics}. "
                            f"Jaccard similarity: {similarity:.2f}."
                        ),
                        suggestion=(
                            "Ensure conclusions build on each other logically. "
                            "If the topic changed, explain the transition."
                        ),
                    )
                )

        return flags
