"""cot-coherence: Detect silent incoherence in AI chain-of-thought reasoning."""

__version__ = "0.1.0"

from .analyzer import analyze
from .config import CoherenceConfig
from .models import (
    CoherenceReport,
    HorizonAnalysis,
    IncoherenceFlag,
    IncoherenceType,
    ReasoningStep,
    Severity,
    TraceInput,
)
from .parser import parse_trace, register_parser

__all__ = [
    "CoherenceConfig",
    "CoherenceReport",
    "HorizonAnalysis",
    "IncoherenceFlag",
    "IncoherenceType",
    "ReasoningStep",
    "Severity",
    "TraceInput",
    "analyze",
    "parse_trace",
    "register_parser",
]
