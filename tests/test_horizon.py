"""Tests for reasoning horizon analysis."""

from cot_coherence.horizon import analyze_horizon
from cot_coherence.models import ReasoningStep


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestHorizonAnalysis:
    def test_returns_none_for_short_chain(self):
        steps = _make_steps(["One", "Two", "Three"])
        result = analyze_horizon(steps)
        assert result is None

    def test_returns_analysis_for_sufficient_chain(self):
        steps = _make_steps([
            "Python is a high-level programming language known for its readability and versatility.",
            "It supports multiple programming paradigms including procedural, object-oriented, and functional.",
            "Python's standard library provides modules for file I/O, networking, and data structures.",
            "The language uses dynamic typing and automatic memory management with garbage collection.",
            "Python's ecosystem includes frameworks like Django for web development and Flask for microservices.",
        ])
        result = analyze_horizon(steps)
        assert result is not None
        assert result.chain_length == 5
        assert 0.0 <= result.horizon_ratio <= 1.0

    def test_degrading_chain_detects_horizon(self):
        steps = _make_steps([
            "Machine learning algorithms learn patterns from training data to make predictions on new data.",
            "Supervised learning uses labeled examples where the algorithm maps inputs to known outputs.",
            "Neural networks with multiple layers can learn hierarchical feature representations automatically.",
            "Deep learning architectures process complex data through sequential transformation layers.",
            # Quality starts degrading — repetitive, short, less diverse
            "Learning learning data data patterns patterns training training.",
            "Data learning patterns data learning patterns data learning.",
            "Data data data learning learning learning patterns patterns.",
        ])
        result = analyze_horizon(steps)
        assert result is not None
        assert result.estimated_horizon < len(steps)
        assert result.horizon_ratio < 1.0

    def test_consistent_quality_full_horizon(self):
        steps = _make_steps([
            "Quantum computing leverages superposition and entanglement for parallel computation.",
            "Classical bits represent zero or one, while qubits can exist in superposition of both states.",
            "Quantum error correction codes protect information against decoherence and gate errors.",
            "Shor's algorithm demonstrates exponential speedup for integer factorization problems.",
            "Grover's algorithm provides quadratic speedup for unstructured database search tasks.",
        ])
        result = analyze_horizon(steps)
        assert result is not None
        # Good consistent quality — horizon should be at or near the end
        assert result.horizon_ratio >= 0.5

    def test_empty_steps(self):
        result = analyze_horizon([])
        assert result is None
