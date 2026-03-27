"""Reasoning Horizon detection — finds where chain quality degrades."""

from __future__ import annotations

import re
import statistics

from .models import HorizonAnalysis, ReasoningStep

_WORD_RE = re.compile(r"\b[a-z]+\b")

HEDGE_WORDS = frozenset({
    "possibly", "might", "could", "perhaps", "may", "maybe", "likely",
    "probably", "uncertain", "unclear", "seems", "appears", "suggest",
    "roughly", "approximately", "around", "arguably", "potentially",
})


def _lexical_diversity(text: str) -> float:
    """Ratio of unique words to total words."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _avg_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(_WORD_RE.findall(s)) for s in sentences) / len(sentences)


def _hedge_frequency(text: str) -> float:
    """Proportion of words that are hedge words."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in HEDGE_WORDS) / len(words)


def _repetition_ratio(text: str, all_prior_text: str) -> float:
    """Proportion of content words that appeared in all prior steps."""
    if not all_prior_text:
        return 0.0
    words = set(_WORD_RE.findall(text.lower()))
    prior_words = set(_WORD_RE.findall(all_prior_text.lower()))
    if not words:
        return 0.0
    return len(words & prior_words) / len(words)


def _rolling_average(values: list[float], window: int = 3) -> list[float]:
    """Compute rolling average with given window size."""
    if len(values) < window:
        return values[:]
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(statistics.mean(values[start : i + 1]))
    return result


def analyze_horizon(steps: list[ReasoningStep]) -> HorizonAnalysis | None:
    """Detect the reasoning horizon — where chain quality starts to degrade.

    Returns None if the chain is too short for meaningful analysis.
    """
    if len(steps) < 4:
        return None

    # Compute per-step quality signals
    quality_signals: list[float] = []
    prior_text = ""

    for step in steps:
        diversity = _lexical_diversity(step.text)
        sent_len = min(_avg_sentence_length(step.text) / 30.0, 1.0)  # normalize
        hedge = 1.0 - _hedge_frequency(step.text)  # less hedging = higher quality
        repetition = 1.0 - _repetition_ratio(step.text, prior_text)

        # Composite quality (equal weights)
        composite = (diversity + sent_len + hedge + repetition) / 4.0
        quality_signals.append(composite)
        prior_text += " " + step.text

    # Smooth
    smoothed = _rolling_average(quality_signals, window=3)

    # Find inflection point: where smoothed drops below mean - 1 std dev
    if len(smoothed) < 3:
        return HorizonAnalysis(
            chain_length=len(steps),
            estimated_horizon=len(steps),
            horizon_ratio=1.0,
            degradation_signals=[],
        )

    mean_q = statistics.mean(smoothed)
    std_q = statistics.stdev(smoothed) if len(smoothed) > 1 else 0.0
    threshold = mean_q - std_q

    degradation_signals: list[str] = []
    estimated_horizon = len(steps)

    for i in range(len(smoothed)):
        if smoothed[i] < threshold:
            estimated_horizon = i
            # Determine which signals degraded
            if i < len(steps):
                div = _lexical_diversity(steps[i].text)
                if div < 0.5:
                    degradation_signals.append(f"step_{i}_low_lexical_diversity")
                rep = _repetition_ratio(steps[i].text, " ".join(s.text for s in steps[:i]))
                if rep > 0.7:
                    degradation_signals.append(f"step_{i}_high_repetition")
                hedge = _hedge_frequency(steps[i].text)
                if hedge > 0.1:
                    degradation_signals.append(f"step_{i}_increased_hedging")
            break

    horizon_ratio = estimated_horizon / len(steps) if len(steps) > 0 else 1.0

    return HorizonAnalysis(
        chain_length=len(steps),
        estimated_horizon=estimated_horizon,
        horizon_ratio=round(horizon_ratio, 2),
        degradation_signals=degradation_signals,
    )
