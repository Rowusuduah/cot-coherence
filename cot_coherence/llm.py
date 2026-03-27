"""LLM-powered incoherence detection using Claude."""

from __future__ import annotations

import json
import os

from .config import CoherenceConfig
from .models import IncoherenceFlag, IncoherenceType, ReasoningStep, Severity

_SYSTEM_PROMPT = """\
You are an AI reasoning coherence analyzer. Your job is to detect incoherence \
patterns in chain-of-thought reasoning traces. Focus on structural coherence \
between steps, not factual accuracy."""

_USER_PROMPT_TEMPLATE = """\
Analyze the following chain-of-thought trace for incoherence patterns.

{question_section}
Trace:
{numbered_steps}

Check for these 5 patterns:
1. PREMISE_ABANDONMENT: A key assumption or premise is introduced then never \
referenced or built upon in subsequent steps.
2. CONCLUSION_DRIFT: Adjacent conclusions shift topic without connecting logic. \
The reasoning jumps between unrelated conclusions.
3. CONFIDENCE_INFLATION: Language shifts from hedging ("might", "possibly", \
"could") to certainty ("definitely", "clearly", "must") without new supporting \
evidence between the steps.
4. SCOPE_CREEP: The reasoning wanders away from the original question into \
unrelated territory.
5. CIRCULAR_RETURN: Non-adjacent steps repeat substantially the same content, \
indicating the reasoning is going in circles.

Return your analysis as JSON with this exact structure:
{{
  "flags": [
    {{
      "type": "PREMISE_ABANDONMENT",
      "severity": "low",
      "confidence": 0.85,
      "step_range": [0, 3],
      "summary": "One-line description of the issue",
      "evidence": "Specific text from the trace demonstrating the issue",
      "suggestion": "How to fix this incoherence"
    }}
  ]
}}

Rules:
- severity must be one of: low, medium, high, critical
- confidence must be between 0.0 and 1.0
- step_range uses 0-based indices matching the step numbers above
- If the trace is coherent, return {{"flags": []}}
- Return ONLY valid JSON, no other text"""


def build_prompt(
    steps: list[ReasoningStep],
    original_question: str = "",
) -> tuple[str, str]:
    """Build the system and user prompts for LLM analysis.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    numbered = "\n".join(f"Step {s.index}: {s.text}" for s in steps)

    question_section = ""
    if original_question:
        question_section = f"Original question: {original_question}\n\n"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question_section=question_section,
        numbered_steps=numbered,
    )

    return _SYSTEM_PROMPT, user_prompt


def parse_llm_response(response_text: str) -> list[IncoherenceFlag]:
    """Parse LLM JSON response into IncoherenceFlag objects.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        List of parsed IncoherenceFlag objects.
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)
    flags: list[IncoherenceFlag] = []

    for item in data.get("flags", []):
        try:
            itype = IncoherenceType(item["type"].lower())
            severity = Severity(item["severity"].lower())
            confidence = max(0.0, min(1.0, float(item["confidence"])))
            step_range = (int(item["step_range"][0]), int(item["step_range"][1]))

            flags.append(
                IncoherenceFlag(
                    type=itype,
                    severity=severity,
                    confidence=confidence,
                    step_range=step_range,
                    summary=str(item.get("summary", "")),
                    evidence=str(item.get("evidence", "")),
                    suggestion=str(item.get("suggestion", "")),
                )
            )
        except (KeyError, ValueError):
            # Skip malformed flags rather than crashing
            continue

    return flags


def merge_flags(
    rule_flags: list[IncoherenceFlag],
    llm_flags: list[IncoherenceFlag],
) -> list[IncoherenceFlag]:
    """Merge rule-based and LLM flags, deduplicating by type + step_range.

    When both sources flag the same type at the same step range,
    keep the LLM version (better summaries and evidence).
    """
    # Index rule-based flags by (type, step_range)
    rule_index: dict[tuple[IncoherenceType, tuple[int, int]], IncoherenceFlag] = {}
    for flag in rule_flags:
        key = (flag.type, flag.step_range)
        rule_index[key] = flag

    # LLM flags override rule-based when they overlap
    llm_index: dict[tuple[IncoherenceType, tuple[int, int]], IncoherenceFlag] = {}
    for flag in llm_flags:
        key = (flag.type, flag.step_range)
        llm_index[key] = flag

    # Start with all rule-based flags, override with LLM where overlap exists
    merged: dict[tuple[IncoherenceType, tuple[int, int]], IncoherenceFlag] = {}
    merged.update(rule_index)
    merged.update(llm_index)

    return list(merged.values())


def llm_analyze(
    steps: list[ReasoningStep],
    original_question: str = "",
    config: CoherenceConfig | None = None,
) -> list[IncoherenceFlag]:
    """Run LLM-powered incoherence detection.

    Requires the `anthropic` package and ANTHROPIC_API_KEY env var.

    Args:
        steps: Parsed reasoning steps.
        original_question: The original question being answered.
        config: Analysis configuration.

    Returns:
        List of incoherence flags detected by the LLM.

    Raises:
        ImportError: If anthropic package is not installed.
        RuntimeError: If ANTHROPIC_API_KEY is not set.
    """
    if config is None:
        config = CoherenceConfig()

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for LLM mode. "
            "Install it with: pip install cot-coherence[llm]"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY=your-key-here"
        )

    system_prompt, user_prompt = build_prompt(steps, original_question)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text
    return parse_llm_response(response_text)
