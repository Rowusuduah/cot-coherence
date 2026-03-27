"""Trace parser — converts raw reasoning traces into structured steps."""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import ReasoningStep, TraceInput

# Registry for custom parsers
_custom_parsers: dict[str, Callable[[str], list[ReasoningStep]]] = {}

# Patterns for format detection
_NUMBERED_RE = re.compile(r"^\s*(?:step\s+)?(\d+)\s*[\.\)\:\-]\s*(.+)", re.IGNORECASE)
_XML_STEP_RE = re.compile(
    r"<(?:step|step_\d+|thinking)>(.*?)</(?:step|step_\d+|thinking)>",
    re.DOTALL,
)
_MARKDOWN_STEP_RE = re.compile(r"^#{2,3}\s+.*(?:step|stage)\s*\d*", re.IGNORECASE | re.MULTILINE)


def register_parser(name: str, parser: Callable[[str], list[ReasoningStep]]) -> None:
    """Register a custom parser for a named format."""
    _custom_parsers[name] = parser


def detect_format(text: str) -> str:
    """Auto-detect the trace format."""
    lines = text.strip().splitlines()
    numbered_count = sum(1 for line in lines if _NUMBERED_RE.match(line))
    if numbered_count >= 2:
        return "numbered"

    if _XML_STEP_RE.search(text):
        return "xml"

    if _MARKDOWN_STEP_RE.search(text):
        return "markdown"

    return "newline"


def _parse_numbered(text: str) -> list[ReasoningStep]:
    """Parse numbered step format (e.g., 'Step 1: ...', '1. ...', '1) ...')."""
    steps: list[ReasoningStep] = []
    lines = text.strip().splitlines()

    current_index = -1
    current_lines: list[str] = []

    for line in lines:
        match = _NUMBERED_RE.match(line)
        if match:
            # Save previous step
            if current_index >= 0 and current_lines:
                step_text = " ".join(current_lines).strip()
                steps.append(
                    ReasoningStep(
                        index=len(steps),
                        text=step_text,
                        raw_text=step_text,
                    )
                )
            current_index = int(match.group(1))
            current_lines = [match.group(2).strip()]
        elif current_index >= 0 and line.strip():
            current_lines.append(line.strip())

    # Save last step
    if current_index >= 0 and current_lines:
        step_text = " ".join(current_lines).strip()
        steps.append(
            ReasoningStep(
                index=len(steps),
                text=step_text,
                raw_text=step_text,
            )
        )

    return steps


def _parse_xml(text: str) -> list[ReasoningStep]:
    """Parse XML-tagged step format."""
    matches = _XML_STEP_RE.findall(text)
    steps: list[ReasoningStep] = []
    for i, content in enumerate(matches):
        cleaned = content.strip()
        if cleaned:
            steps.append(
                ReasoningStep(
                    index=i,
                    text=cleaned,
                    raw_text=content,
                )
            )
    return steps


def _parse_markdown(text: str) -> list[ReasoningStep]:
    """Parse markdown heading step format."""
    sections = re.split(r"(?=^#{2,3}\s+)", text.strip(), flags=re.MULTILINE)
    steps: list[ReasoningStep] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Remove the heading line, keep the body
        lines = section.splitlines()
        heading = lines[0] if lines else ""
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        step_text = body if body else heading
        if step_text:
            steps.append(
                ReasoningStep(
                    index=len(steps),
                    text=step_text,
                    raw_text=section,
                )
            )
    return steps


def _parse_newline(text: str) -> list[ReasoningStep]:
    """Parse double-newline separated steps (fallback)."""
    blocks = re.split(r"\n\s*\n", text.strip())
    steps: list[ReasoningStep] = []
    for i, block in enumerate(blocks):
        cleaned = " ".join(block.strip().splitlines()).strip()
        if cleaned:
            steps.append(
                ReasoningStep(
                    index=i,
                    text=cleaned,
                    raw_text=block,
                )
            )
    return steps


_FORMAT_PARSERS = {
    "numbered": _parse_numbered,
    "xml": _parse_xml,
    "markdown": _parse_markdown,
    "newline": _parse_newline,
}


def parse_trace(trace_input: TraceInput) -> list[ReasoningStep]:
    """Parse a trace input into structured reasoning steps."""
    # If pre-split steps provided, use them directly
    if trace_input.steps is not None:
        return [
            ReasoningStep(index=i, text=s.strip(), raw_text=s)
            for i, s in enumerate(trace_input.steps)
            if s.strip()
        ]

    if trace_input.text is None:
        return []

    text = trace_input.text.strip()
    if not text:
        return []

    fmt = trace_input.trace_format
    if fmt == "auto":
        fmt = detect_format(text)

    # Check custom parsers first
    if fmt in _custom_parsers:
        return _custom_parsers[fmt](text)

    parser_fn = _FORMAT_PARSERS.get(fmt, _parse_newline)
    return parser_fn(text)
