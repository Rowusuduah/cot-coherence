"""Tests for confidence inflation detector."""

from cot_coherence.detectors.confidence_inflation import ConfidenceInflationDetector
from cot_coherence.models import IncoherenceType, ReasoningStep


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestConfidenceInflationDetector:
    def setup_method(self):
        self.detector = ConfidenceInflationDetector()

    def test_detects_hedge_to_certainty_jump(self):
        steps = _make_steps([
            "This might possibly work, though it is unclear and perhaps uncertain.",
            "This is definitely the best solution and certainly always works.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert flags[0].type == IncoherenceType.CONFIDENCE_INFLATION

    def test_no_flag_on_consistent_hedging(self):
        steps = _make_steps([
            "This might possibly work, perhaps with some effort.",
            "It could probably help, though likely not in all cases.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_no_flag_on_consistent_certainty(self):
        steps = _make_steps([
            "This is definitely the right approach, clearly proven.",
            "It is certainly effective and always delivers results.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_no_flag_when_evidence_provided(self):
        steps = _make_steps([
            "This might possibly improve performance, though it seems uncertain.",
            "According to the research data, this is definitely proven to work. The study demonstrates clear results.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_single_step_no_crash(self):
        steps = _make_steps(["Just one step here."])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_empty_steps(self):
        flags = self.detector.detect([])
        assert len(flags) == 0

    def test_fixture_trace(self, confidence_inflation_trace):
        from cot_coherence.models import TraceInput
        from cot_coherence.parser import parse_trace
        steps = parse_trace(TraceInput(text=confidence_inflation_trace))
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert any(f.type == IncoherenceType.CONFIDENCE_INFLATION for f in flags)
