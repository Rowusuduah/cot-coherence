"""Core data models for cot-coherence."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IncoherenceType(str, Enum):
    """Types of chain-of-thought incoherence."""

    PREMISE_ABANDONMENT = "premise_abandonment"
    CONCLUSION_DRIFT = "conclusion_drift"
    CONFIDENCE_INFLATION = "confidence_inflation"
    SCOPE_CREEP = "scope_creep"
    CIRCULAR_RETURN = "circular_return"


class Severity(str, Enum):
    """Severity levels for incoherence flags."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReasoningStep(BaseModel):
    """A single step in a chain-of-thought trace."""

    index: int
    text: str
    raw_text: str = ""
    metadata: dict = Field(default_factory=dict)


class IncoherenceFlag(BaseModel):
    """A detected incoherence in the reasoning chain."""

    type: IncoherenceType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    step_range: tuple[int, int]
    summary: str
    evidence: str
    suggestion: str = ""


class HorizonAnalysis(BaseModel):
    """Analysis of the reasoning horizon — where chain quality degrades."""

    chain_length: int
    estimated_horizon: int
    horizon_ratio: float = Field(ge=0.0, le=1.0)
    degradation_signals: list[str] = Field(default_factory=list)


class CoherenceReport(BaseModel):
    """Complete coherence analysis report."""

    steps: list[ReasoningStep]
    flags: list[IncoherenceFlag] = Field(default_factory=list)
    horizon: HorizonAnalysis | None = None
    overall_score: float = Field(ge=0.0, le=1.0)
    pattern_scores: dict[IncoherenceType, float] = Field(default_factory=dict)

    @property
    def is_coherent(self) -> bool:
        return self.overall_score >= 0.7

    @property
    def critical_flags(self) -> list[IncoherenceFlag]:
        return [f for f in self.flags if f.severity == Severity.CRITICAL]


class TraceInput(BaseModel):
    """Input specification for a reasoning trace."""

    text: str | None = None
    steps: list[str] | None = None
    original_question: str = ""
    trace_format: str = "auto"
