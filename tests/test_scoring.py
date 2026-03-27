"""Tests for scoring engine."""

from cot_coherence.config import CoherenceConfig
from cot_coherence.models import IncoherenceFlag, IncoherenceType, Severity
from cot_coherence.scoring import compute_overall_score, compute_pattern_scores


def _make_flag(
    itype: IncoherenceType = IncoherenceType.CONFIDENCE_INFLATION,
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.8,
) -> IncoherenceFlag:
    return IncoherenceFlag(
        type=itype,
        severity=severity,
        confidence=confidence,
        step_range=(0, 1),
        summary="test",
        evidence="test",
    )


class TestPatternScores:
    def test_perfect_score_no_flags(self):
        config = CoherenceConfig()
        scores = compute_pattern_scores([], config)
        for score in scores.values():
            assert score == 1.0

    def test_single_medium_flag_reduces_score(self):
        config = CoherenceConfig()
        flags = [_make_flag(severity=Severity.MEDIUM, confidence=1.0)]
        scores = compute_pattern_scores(flags, config)
        assert scores[IncoherenceType.CONFIDENCE_INFLATION] < 1.0
        assert scores[IncoherenceType.CONFIDENCE_INFLATION] == 1.0 - 0.12

    def test_critical_flag_large_penalty(self):
        config = CoherenceConfig()
        flags = [_make_flag(severity=Severity.CRITICAL, confidence=1.0)]
        scores = compute_pattern_scores(flags, config)
        assert scores[IncoherenceType.CONFIDENCE_INFLATION] == 1.0 - 0.35

    def test_multiple_flags_accumulate(self):
        config = CoherenceConfig()
        flags = [
            _make_flag(severity=Severity.MEDIUM, confidence=1.0),
            _make_flag(severity=Severity.MEDIUM, confidence=1.0),
        ]
        scores = compute_pattern_scores(flags, config)
        assert scores[IncoherenceType.CONFIDENCE_INFLATION] == 1.0 - 0.24

    def test_score_clamped_at_zero(self):
        config = CoherenceConfig()
        flags = [_make_flag(severity=Severity.CRITICAL, confidence=1.0)] * 5
        scores = compute_pattern_scores(flags, config)
        assert scores[IncoherenceType.CONFIDENCE_INFLATION] == 0.0

    def test_confidence_scales_penalty(self):
        config = CoherenceConfig()
        flags = [_make_flag(severity=Severity.HIGH, confidence=0.5)]
        scores = compute_pattern_scores(flags, config)
        # HIGH penalty = 0.22, confidence = 0.5, so penalty = 0.11
        assert scores[IncoherenceType.CONFIDENCE_INFLATION] == 1.0 - 0.11


class TestOverallScore:
    def test_perfect_overall(self):
        config = CoherenceConfig()
        pattern_scores = {t: 1.0 for t in IncoherenceType}
        assert compute_overall_score(pattern_scores, config) == 1.0

    def test_zero_overall(self):
        config = CoherenceConfig()
        pattern_scores = {t: 0.0 for t in IncoherenceType}
        assert compute_overall_score(pattern_scores, config) == 0.0

    def test_weighted_average(self):
        config = CoherenceConfig()
        pattern_scores = {t: 0.5 for t in IncoherenceType}
        assert compute_overall_score(pattern_scores, config) == 0.5

    def test_empty_scores(self):
        config = CoherenceConfig()
        assert compute_overall_score({}, config) == 1.0
