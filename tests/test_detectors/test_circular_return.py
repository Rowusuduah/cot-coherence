"""Tests for circular return detector."""

from cot_coherence.detectors.circular_return import CircularReturnDetector
from cot_coherence.models import IncoherenceType, ReasoningStep


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestCircularReturnDetector:
    def setup_method(self):
        self.detector = CircularReturnDetector()

    def test_detects_repeated_reasoning(self):
        steps = _make_steps([
            "Machine learning models require large datasets for training. Data quality and quantity are important factors in model performance.",
            "Neural networks use backpropagation to adjust weights during training.",
            "Machine learning models need big datasets to train effectively. The quality and amount of data are key factors that determine how well the model performs.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert flags[0].type == IncoherenceType.CIRCULAR_RETURN

    def test_no_flag_on_distinct_steps(self):
        steps = _make_steps([
            "Python is a high-level programming language with dynamic typing.",
            "JavaScript runs in web browsers and enables interactive web pages.",
            "Rust provides memory safety guarantees without a garbage collector.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_no_flag_on_adjacent_similarity(self):
        # Adjacent steps can naturally be similar — only flag non-adjacent
        steps = _make_steps([
            "Python has great libraries for data science.",
            "Python data science libraries include NumPy and pandas.",
            "Rust is a systems programming language.",
        ])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_short_chain_no_crash(self):
        steps = _make_steps(["One step.", "Two steps."])
        flags = self.detector.detect(steps)
        assert len(flags) == 0

    def test_empty_steps(self):
        flags = self.detector.detect([])
        assert len(flags) == 0

    def test_fixture_trace(self, circular_return_trace):
        from cot_coherence.models import TraceInput
        from cot_coherence.parser import parse_trace
        steps = parse_trace(TraceInput(text=circular_return_trace))
        flags = self.detector.detect(steps)
        assert len(flags) >= 1
        assert any(f.type == IncoherenceType.CIRCULAR_RETURN for f in flags)
