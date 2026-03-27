"""Main analysis engine — orchestrates parsing, detection, scoring, and horizon."""

from __future__ import annotations

from .config import CoherenceConfig
from .detectors import ALL_DETECTORS
from .horizon import analyze_horizon
from .models import (
    CoherenceReport,
    IncoherenceFlag,
    ReasoningStep,
    TraceInput,
)
from .parser import parse_trace
from .scoring import compute_overall_score, compute_pattern_scores


def analyze(
    text: str | None = None,
    *,
    steps: list[str] | None = None,
    original_question: str = "",
    trace_format: str = "auto",
    config: CoherenceConfig | None = None,
) -> CoherenceReport:
    """Analyze a chain-of-thought trace for incoherence.

    Args:
        text: Raw trace text. Provide either text or steps, not both.
        steps: Pre-split reasoning steps.
        original_question: The original question being reasoned about.
        trace_format: Format hint — "auto", "numbered", "xml", "markdown", "newline".
        config: Analysis configuration. Uses defaults if not provided.

    Returns:
        CoherenceReport with scores, flags, and horizon analysis.
    """
    if config is None:
        config = CoherenceConfig()

    # Parse trace into steps
    trace_input = TraceInput(
        text=text,
        steps=steps,
        original_question=original_question,
        trace_format=trace_format,
    )
    parsed_steps: list[ReasoningStep] = parse_trace(trace_input)

    if not parsed_steps:
        return CoherenceReport(
            steps=[],
            flags=[],
            horizon=None,
            overall_score=1.0,
            pattern_scores={},
        )

    # Run all enabled detectors
    all_flags: list[IncoherenceFlag] = []
    for itype, detector_cls in ALL_DETECTORS.items():
        if itype not in config.enabled_detectors:
            continue
        detector = detector_cls()
        flags = detector.detect(parsed_steps, original_question)
        all_flags.extend(flags)

    # LLM-powered second pass (opt-in)
    if config.use_llm:
        try:
            from .llm import llm_analyze, merge_flags

            llm_flags = llm_analyze(parsed_steps, original_question, config)
            all_flags = merge_flags(all_flags, llm_flags)
        except (ImportError, RuntimeError, Exception):
            # If LLM fails for any reason, fall back to rule-based results
            pass

    # Sort flags by step range start, then severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_flags.sort(key=lambda f: (f.step_range[0], severity_order.get(f.severity.value, 4)))

    # Compute scores
    pattern_scores = compute_pattern_scores(all_flags, config)
    overall_score = compute_overall_score(pattern_scores, config)

    # Horizon analysis
    horizon = analyze_horizon(parsed_steps) if config.analyze_horizon else None

    return CoherenceReport(
        steps=parsed_steps,
        flags=all_flags,
        horizon=horizon,
        overall_score=overall_score,
        pattern_scores=pattern_scores,
    )
