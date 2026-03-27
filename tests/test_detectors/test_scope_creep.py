"""Tests for scope creep detector."""

from cot_coherence.detectors.scope_creep import ScopeCreepDetector
from cot_coherence.models import IncoherenceType, ReasoningStep


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestScopeCreepDetector:
    def setup_method(self):
        self.detector = ScopeCreepDetector()

    def test_detects_off_topic_drift(self):
        steps = _make_steps([
            "The user asks about Python memory management.",
            "Python uses reference counting for memory management.",
            "The garbage collector handles circular references in Python memory.",
            "Ancient Roman architecture featured many circular structures like the Colosseum.",
            "The Colosseum held approximately fifty thousand spectators for gladiatorial contests.",
            "Roman entertainment included chariot racing at the Circus Maximus venue.",
        ])
        flags = self.detector.detect(steps, "How does Python memory management work?")
        assert len(flags) >= 1
        assert flags[0].type == IncoherenceType.SCOPE_CREEP

    def test_no_flag_when_on_topic(self):
        steps = _make_steps([
            "The user asks about Python performance.",
            "Python uses an interpreter which adds overhead.",
            "Python performance can be improved with compiled extensions.",
            "NumPy provides fast Python array operations through C extensions.",
            "Cython compiles Python code to C for better performance.",
        ])
        flags = self.detector.detect(steps, "How can I improve Python performance?")
        assert len(flags) == 0

    def test_no_flag_without_question(self):
        steps = _make_steps([
            "Step one.", "Step two.", "Step three.", "Step four.", "Step five.",
        ])
        flags = self.detector.detect(steps, "")
        assert len(flags) == 0

    def test_no_flag_short_chain(self):
        steps = _make_steps(["Step one.", "Step two."])
        flags = self.detector.detect(steps, "What is Python?")
        assert len(flags) == 0

    def test_empty_steps(self):
        flags = self.detector.detect([], "question")
        assert len(flags) == 0

    def test_fixture_trace(self, scope_creep_trace):
        from cot_coherence.parser import parse_trace
        from cot_coherence.models import TraceInput
        steps = parse_trace(TraceInput(text=scope_creep_trace))
        flags = self.detector.detect(steps, "How does Python memory management work?")
        assert len(flags) >= 1
        assert any(f.type == IncoherenceType.SCOPE_CREEP for f in flags)
