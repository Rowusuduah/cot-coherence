"""Configuration for cot-coherence."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import IncoherenceType


class CoherenceConfig(BaseModel):
    """Configuration for coherence analysis."""

    use_llm: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    weights: dict[IncoherenceType, float] = Field(
        default_factory=lambda: {t: 1.0 for t in IncoherenceType}
    )
    enabled_detectors: set[IncoherenceType] = Field(
        default_factory=lambda: set(IncoherenceType)
    )
    analyze_horizon: bool = True
    premise_window: int = 3
    similarity_threshold: float = 0.35
    scope_overlap_threshold: float = 0.1
