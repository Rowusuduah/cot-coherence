"""Tests for the main analyzer engine."""

import json
from unittest.mock import MagicMock, patch

import cot_coherence
from cot_coherence.config import CoherenceConfig
from cot_coherence.models import IncoherenceType


class TestAnalyze:
    def test_clean_trace_high_score(self, clean_trace):
        report = cot_coherence.analyze(clean_trace)
        assert report.overall_score >= 0.7
        assert report.is_coherent is True
        assert len(report.steps) == 5

    def test_multi_issue_trace_low_score(self, multi_issue_trace):
        report = cot_coherence.analyze(
            multi_issue_trace,
            original_question="What are Python performance optimization techniques?",
        )
        assert report.overall_score < 1.0
        assert len(report.flags) > 0

    def test_empty_text(self):
        report = cot_coherence.analyze("")
        assert report.overall_score == 1.0
        assert len(report.steps) == 0
        assert len(report.flags) == 0

    def test_none_text(self):
        report = cot_coherence.analyze(None)
        assert report.overall_score == 1.0

    def test_pre_split_steps(self):
        report = cot_coherence.analyze(
            steps=["Step one about Python.", "Step two about Python performance."]
        )
        assert len(report.steps) == 2

    def test_horizon_analysis_present(self, clean_trace):
        report = cot_coherence.analyze(clean_trace)
        assert report.horizon is not None
        assert report.horizon.chain_length == 5

    def test_horizon_disabled(self, clean_trace):
        config = CoherenceConfig(analyze_horizon=False)
        report = cot_coherence.analyze(clean_trace, config=config)
        assert report.horizon is None

    def test_disabled_detectors(self, multi_issue_trace):
        config = CoherenceConfig(
            enabled_detectors={IncoherenceType.CONFIDENCE_INFLATION}
        )
        report = cot_coherence.analyze(
            multi_issue_trace,
            original_question="What are Python performance optimization techniques?",
            config=config,
        )
        # Only confidence inflation flags should be present
        for flag in report.flags:
            assert flag.type == IncoherenceType.CONFIDENCE_INFLATION

    def test_version_exists(self):
        assert hasattr(cot_coherence, "__version__")
        assert cot_coherence.__version__ == "0.2.0"

    def test_report_properties(self):
        report = cot_coherence.analyze(
            "Step 1: Maybe this works.\nStep 2: Definitely always works.",
        )
        # Test critical_flags property
        assert isinstance(report.critical_flags, list)
        # Test is_coherent property
        assert isinstance(report.is_coherent, bool)

    def test_confidence_inflation_detected(self, confidence_inflation_trace):
        report = cot_coherence.analyze(confidence_inflation_trace)
        types = {f.type for f in report.flags}
        assert IncoherenceType.CONFIDENCE_INFLATION in types

    def test_premise_abandonment_detected(self, premise_abandonment_trace):
        report = cot_coherence.analyze(premise_abandonment_trace)
        types = {f.type for f in report.flags}
        assert IncoherenceType.PREMISE_ABANDONMENT in types

    def test_conclusion_drift_detected(self, conclusion_drift_trace):
        report = cot_coherence.analyze(conclusion_drift_trace)
        types = {f.type for f in report.flags}
        assert IncoherenceType.CONCLUSION_DRIFT in types

    def test_scope_creep_detected(self, scope_creep_trace):
        report = cot_coherence.analyze(
            scope_creep_trace,
            original_question="How does Python memory management work?",
        )
        types = {f.type for f in report.flags}
        assert IncoherenceType.SCOPE_CREEP in types

    def test_circular_return_detected(self, circular_return_trace):
        report = cot_coherence.analyze(circular_return_trace)
        types = {f.type for f in report.flags}
        assert IncoherenceType.CIRCULAR_RETURN in types

    def test_llm_mode_integrates(self, multi_issue_trace):
        """Test that LLM mode merges flags with rule-based results."""
        llm_response = json.dumps({
            "flags": [
                {
                    "type": "scope_creep",
                    "severity": "high",
                    "confidence": 0.95,
                    "step_range": [2, 5],
                    "summary": "LLM detected scope creep",
                    "evidence": "Steps 2-5 wander",
                    "suggestion": "Refocus",
                }
            ]
        })

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=llm_response)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        config = CoherenceConfig(use_llm=True)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                report = cot_coherence.analyze(
                    multi_issue_trace,
                    original_question="What are Python performance optimization techniques?",
                    config=config,
                )

        assert len(report.flags) > 0
        assert report.overall_score < 1.0

    def test_llm_failure_falls_back_to_rules(self, multi_issue_trace):
        """If LLM fails, rule-based results are still returned."""
        config = CoherenceConfig(use_llm=True)

        # No anthropic module and no API key — should fall back gracefully
        with patch.dict("sys.modules", {"anthropic": None}):
            report = cot_coherence.analyze(
                multi_issue_trace,
                original_question="What are Python performance optimization techniques?",
                config=config,
            )

        # Should still have rule-based flags
        assert len(report.flags) > 0
