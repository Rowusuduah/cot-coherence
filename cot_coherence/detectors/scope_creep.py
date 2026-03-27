"""Detector for scope creep — reasoning that drifts away from the original question."""

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


class ScopeCreepDetector(BaseDetector):
    """Detects when reasoning steps drift away from the original question."""

    def __init__(self, overlap_threshold: float = 0.1, min_step: int = 3) -> None:
        self.overlap_threshold = overlap_threshold
        self.min_step = min_step

    def detect(
        self,
        steps: list[ReasoningStep],
        original_question: str = "",
    ) -> list[IncoherenceFlag]:
        flags: list[IncoherenceFlag] = []

        if not original_question or len(steps) < self.min_step + 1:
            return flags

        question_words = _content_words(original_question)
        if not question_words:
            return flags

        # Track consecutive off-topic steps
        off_topic_start: int | None = None
        consecutive_off = 0

        for i, step in enumerate(steps):
            if i < self.min_step:
                continue

            step_words = _content_words(step.text)
            if not step_words:
                continue

            overlap = len(question_words & step_words) / len(question_words)

            if overlap < self.overlap_threshold:
                if off_topic_start is None:
                    off_topic_start = i
                consecutive_off += 1
            else:
                # If we had 2+ consecutive off-topic steps, flag them
                if consecutive_off >= 2 and off_topic_start is not None:
                    severity = Severity.CRITICAL if consecutive_off >= 4 else (
                        Severity.HIGH if consecutive_off >= 3 else Severity.MEDIUM
                    )
                    confidence = round(min(1.0, 0.5 + consecutive_off * 0.15), 2)
                    end_step = off_topic_start + consecutive_off - 1

                    flags.append(
                        IncoherenceFlag(
                            type=IncoherenceType.SCOPE_CREEP,
                            severity=severity,
                            confidence=confidence,
                            step_range=(off_topic_start, end_step),
                            summary=(
                                f"Steps {off_topic_start}-{end_step} drift away from "
                                f"the original question ({consecutive_off} consecutive "
                                f"off-topic steps)."
                            ),
                            evidence=(
                                f"Question keywords: {sorted(question_words)[:5]}. "
                                f"These steps share <{self.overlap_threshold:.0%} "
                                f"overlap with the question."
                            ),
                            suggestion=(
                                "Bring the reasoning back to the original question, "
                                "or explain why this tangent is relevant."
                            ),
                        )
                    )
                off_topic_start = None
                consecutive_off = 0

        # Check trailing off-topic steps
        if consecutive_off >= 2 and off_topic_start is not None:
            severity = Severity.CRITICAL if consecutive_off >= 4 else (
                Severity.HIGH if consecutive_off >= 3 else Severity.MEDIUM
            )
            confidence = round(min(1.0, 0.5 + consecutive_off * 0.15), 2)
            end_step = off_topic_start + consecutive_off - 1

            flags.append(
                IncoherenceFlag(
                    type=IncoherenceType.SCOPE_CREEP,
                    severity=severity,
                    confidence=confidence,
                    step_range=(off_topic_start, end_step),
                    summary=(
                        f"Steps {off_topic_start}-{end_step} drift away from "
                        f"the original question ({consecutive_off} consecutive "
                        f"off-topic steps)."
                    ),
                    evidence=(
                        f"Question keywords: {sorted(question_words)[:5]}. "
                        f"These steps share <{self.overlap_threshold:.0%} "
                        f"overlap with the question."
                    ),
                    suggestion=(
                        "Bring the reasoning back to the original question, "
                        "or explain why this tangent is relevant."
                    ),
                )
            )

        return flags
