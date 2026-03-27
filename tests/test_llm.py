"""Tests for LLM-powered detection module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from cot_coherence.llm import build_prompt, llm_analyze, merge_flags, parse_llm_response
from cot_coherence.models import (
    IncoherenceFlag,
    IncoherenceType,
    ReasoningStep,
    Severity,
)


def _make_steps(texts: list[str]) -> list[ReasoningStep]:
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


class TestBuildPrompt:
    def test_basic_prompt(self):
        steps = _make_steps(["First step.", "Second step."])
        system, user = build_prompt(steps)
        assert "coherence analyzer" in system.lower()
        assert "Step 0: First step." in user
        assert "Step 1: Second step." in user
        assert "PREMISE_ABANDONMENT" in user

    def test_prompt_with_question(self):
        steps = _make_steps(["First step."])
        system, user = build_prompt(steps, original_question="Is Python fast?")
        assert "Is Python fast?" in user
        assert "Original question:" in user

    def test_prompt_without_question(self):
        steps = _make_steps(["First step."])
        _, user = build_prompt(steps, original_question="")
        assert "Original question:" not in user

    def test_prompt_contains_all_patterns(self):
        steps = _make_steps(["Step."])
        _, user = build_prompt(steps)
        for pattern in [
            "PREMISE_ABANDONMENT",
            "CONCLUSION_DRIFT",
            "CONFIDENCE_INFLATION",
            "SCOPE_CREEP",
            "CIRCULAR_RETURN",
        ]:
            assert pattern in user


class TestParseLLMResponse:
    def test_valid_response(self):
        response = json.dumps({
            "flags": [
                {
                    "type": "premise_abandonment",
                    "severity": "high",
                    "confidence": 0.9,
                    "step_range": [0, 3],
                    "summary": "Premise dropped",
                    "evidence": "Step 0 says X, step 3 ignores it",
                    "suggestion": "Reference the premise",
                }
            ]
        })
        flags = parse_llm_response(response)
        assert len(flags) == 1
        assert flags[0].type == IncoherenceType.PREMISE_ABANDONMENT
        assert flags[0].severity == Severity.HIGH
        assert flags[0].confidence == 0.9
        assert flags[0].step_range == (0, 3)

    def test_empty_flags(self):
        response = json.dumps({"flags": []})
        flags = parse_llm_response(response)
        assert flags == []

    def test_multiple_flags(self):
        response = json.dumps({
            "flags": [
                {
                    "type": "scope_creep",
                    "severity": "medium",
                    "confidence": 0.7,
                    "step_range": [2, 4],
                    "summary": "Wandered off topic",
                    "evidence": "Steps 2-4 discuss unrelated topic",
                    "suggestion": "Stay focused",
                },
                {
                    "type": "confidence_inflation",
                    "severity": "high",
                    "confidence": 0.85,
                    "step_range": [3, 5],
                    "summary": "Unjustified certainty",
                    "evidence": "Hedging to certainty",
                    "suggestion": "Add evidence",
                },
            ]
        })
        flags = parse_llm_response(response)
        assert len(flags) == 2

    def test_markdown_code_fence(self):
        response = '```json\n{"flags": []}\n```'
        flags = parse_llm_response(response)
        assert flags == []

    def test_malformed_flag_skipped(self):
        response = json.dumps({
            "flags": [
                {"type": "invalid_type", "severity": "high", "confidence": 0.9,
                 "step_range": [0, 1], "summary": "Bad"},
                {"type": "scope_creep", "severity": "medium", "confidence": 0.7,
                 "step_range": [2, 4], "summary": "Good", "evidence": "E",
                 "suggestion": "S"},
            ]
        })
        flags = parse_llm_response(response)
        assert len(flags) == 1
        assert flags[0].type == IncoherenceType.SCOPE_CREEP

    def test_confidence_clamped(self):
        response = json.dumps({
            "flags": [
                {
                    "type": "scope_creep",
                    "severity": "low",
                    "confidence": 1.5,
                    "step_range": [0, 1],
                    "summary": "S",
                    "evidence": "E",
                    "suggestion": "S",
                }
            ]
        })
        flags = parse_llm_response(response)
        assert flags[0].confidence == 1.0

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_response("not json at all")


class TestMergeFlags:
    def _flag(self, itype: IncoherenceType, step_range: tuple[int, int],
              summary: str = "test") -> IncoherenceFlag:
        return IncoherenceFlag(
            type=itype,
            severity=Severity.HIGH,
            confidence=0.9,
            step_range=step_range,
            summary=summary,
            evidence="evidence",
            suggestion="suggestion",
        )

    def test_no_overlap(self):
        rule = [self._flag(IncoherenceType.SCOPE_CREEP, (0, 2))]
        llm = [self._flag(IncoherenceType.PREMISE_ABANDONMENT, (1, 4))]
        merged = merge_flags(rule, llm)
        assert len(merged) == 2

    def test_llm_overrides_on_overlap(self):
        rule = [self._flag(IncoherenceType.SCOPE_CREEP, (0, 2), summary="rule")]
        llm = [self._flag(IncoherenceType.SCOPE_CREEP, (0, 2), summary="llm")]
        merged = merge_flags(rule, llm)
        assert len(merged) == 1
        assert merged[0].summary == "llm"

    def test_empty_inputs(self):
        assert merge_flags([], []) == []

    def test_rule_only(self):
        rule = [self._flag(IncoherenceType.SCOPE_CREEP, (0, 2))]
        merged = merge_flags(rule, [])
        assert len(merged) == 1

    def test_llm_only(self):
        llm = [self._flag(IncoherenceType.SCOPE_CREEP, (0, 2))]
        merged = merge_flags([], llm)
        assert len(merged) == 1


class TestLLMAnalyze:
    def test_missing_anthropic_raises(self):
        steps = _make_steps(["Step 1."])
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                llm_analyze(steps)

    def test_missing_api_key_raises(self):
        steps = _make_steps(["Step 1."])
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                    llm_analyze(steps)

    def test_successful_call(self):
        steps = _make_steps(["First step.", "Second step."])
        response_json = json.dumps({
            "flags": [
                {
                    "type": "scope_creep",
                    "severity": "medium",
                    "confidence": 0.7,
                    "step_range": [0, 1],
                    "summary": "Drifted",
                    "evidence": "Evidence",
                    "suggestion": "Fix it",
                }
            ]
        })

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client_instance

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                flags = llm_analyze(steps, original_question="Test?")

        assert len(flags) == 1
        assert flags[0].type == IncoherenceType.SCOPE_CREEP
        mock_client_instance.messages.create.assert_called_once()
