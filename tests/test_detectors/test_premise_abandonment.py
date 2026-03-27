"""Tests for premise abandonment detector."""

from cot_coherence.detectors.premise_abandonment import PremiseAbandonmentDetector
from cot_coherence.models import IncoherenceType, ReasoningStep


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestPremiseAbandonmentDetector:
    def setup_method(self):
        self.detector = PremiseAbandonmentDetector(window=3)

    def test_detects_abandoned_premise(self):
        steps = _make_steps([
            "Given that the database uses PostgreSQL with row-level security, we need to consider authentication.",
            "The frontend uses React with TypeScript components.",
            "The color scheme follows Material Design guidelines.",
            "Therefore the architecture is modern.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert flags[0].type == IncoherenceType.PREMISE_ABANDONMENT

    def test_no_flag_when_premise_followed(self):
        steps = _make_steps([
            "Given that the database uses PostgreSQL, we need to optimize queries.",
            "PostgreSQL supports advanced indexing strategies for query optimization.",
            "We should add indexes on the database columns most frequently queried.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_no_flag_without_premise_markers(self):
        steps = _make_steps([
            "The system processes data in real time.",
            "Weather patterns affect agricultural output.",
            "Ancient Rome had impressive architecture.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_empty_steps(self):
        flags = self.detector.detect([])
        assert len(flags) == 0

    def test_single_step(self):
        steps = _make_steps(["Given that X is true, we proceed."])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_fixture_trace(self, premise_abandonment_trace):
        from cot_coherence.parser import parse_trace
        from cot_coherence.models import TraceInput
        steps = parse_trace(TraceInput(text=premise_abandonment_trace))
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert any(f.type == IncoherenceType.PREMISE_ABANDONMENT for f in flags)
