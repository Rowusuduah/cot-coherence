"""Incoherence detectors registry."""

from __future__ import annotations

from ..models import IncoherenceType
from .base import BaseDetector
from .circular_return import CircularReturnDetector
from .conclusion_drift import ConclusionDriftDetector
from .confidence_inflation import ConfidenceInflationDetector
from .premise_abandonment import PremiseAbandonmentDetector
from .scope_creep import ScopeCreepDetector

ALL_DETECTORS: dict[IncoherenceType, type[BaseDetector]] = {
    IncoherenceType.PREMISE_ABANDONMENT: PremiseAbandonmentDetector,
    IncoherenceType.CONCLUSION_DRIFT: ConclusionDriftDetector,
    IncoherenceType.CONFIDENCE_INFLATION: ConfidenceInflationDetector,
    IncoherenceType.SCOPE_CREEP: ScopeCreepDetector,
    IncoherenceType.CIRCULAR_RETURN: CircularReturnDetector,
}

__all__ = [
    "ALL_DETECTORS",
    "BaseDetector",
    "CircularReturnDetector",
    "ConclusionDriftDetector",
    "ConfidenceInflationDetector",
    "PremiseAbandonmentDetector",
    "ScopeCreepDetector",
]
