"""Tests for conclusion drift detector."""

from cot_coherence.detectors.conclusion_drift import ConclusionDriftDetector
from cot_coherence.models import IncoherenceType, ReasoningStep


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestConclusionDriftDetector:
    def setup_method(self):
        self.detector = ConclusionDriftDetector()

    def test_detects_topic_drift_between_conclusions(self):
        steps = _make_steps([
            "Therefore, database indexing improves query performance significantly.",
            "Thus, Kubernetes is the best platform for container orchestration.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert flags[0].type == IncoherenceType.CONCLUSION_DRIFT

    def test_no_flag_on_related_conclusions(self):
        steps = _make_steps([
            "Therefore, Python performance can be improved with NumPy vectorization.",
            "Thus, Python performance improvements come from NumPy vectorization techniques.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_no_flag_without_conclusion_markers(self):
        steps = _make_steps([
            "Database queries can be slow.",
            "Kubernetes manages containers.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_single_conclusion(self):
        steps = _make_steps([
            "Therefore, this approach works well.",
            "The system processes data efficiently.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_empty_steps(self):
        flags = self.detector.detect([])
        assert len(flags) == 0

    def test_fixture_trace(self, conclusion_drift_trace):
        from cot_coherence.models import TraceInput
        from cot_coherence.parser import parse_trace
        steps = parse_trace(TraceInput(text=conclusion_drift_trace))
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert any(f.type == IncoherenceType.CONCLUSION_DRIFT for f in flags)
