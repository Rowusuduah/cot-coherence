"""Base detector abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import IncoherenceFlag, ReasoningStep


class BaseDetector(ABC):
    """Abstract base class for all incoherence detectors."""

    @abstractmethod
    def detect(
        self,
        steps: list[ReasoningStep],
        original_question: str = "",
    ) -> list[IncoherenceFlag]:
        """Detect incoherence patterns in reasoning steps.

        Args:
            steps: Parsed reasoning steps.
            original_question: The original question being answered (optional).

        Returns:
            List of incoherence flags found.
        """
        ...
