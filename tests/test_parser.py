"""Tests for the trace parser."""

from cot_coherence.models import TraceInput
from cot_coherence.parser import detect_format, parse_trace, register_parser


class TestDetectFormat:
    def test_numbered_with_step_prefix(self):
        text = "Step 1: First\nStep 2: Second\nStep 3: Third"
        assert detect_format(text) == "numbered"

    def test_numbered_with_dot(self):
        text = "1. First thing\n2. Second thing\n3. Third thing"
        assert detect_format(text) == "numbered"

    def test_numbered_with_paren(self):
        text = "1) First\n2) Second\n3) Third"
        assert detect_format(text) == "numbered"

    def test_xml_format(self):
        text = "<step>First step</step><step>Second step</step>"
        assert detect_format(text) == "xml"

    def test_xml_thinking_tags(self):
        text = "<thinking>I need to think about this</thinking>"
        assert detect_format(text) == "xml"

    def test_markdown_format(self):
        text = "## Step 1\nFirst thing\n## Step 2\nSecond thing"
        assert detect_format(text) == "markdown"

    def test_newline_fallback(self):
        text = "First paragraph about something.\n\nSecond paragraph about another thing."
        assert detect_format(text) == "newline"


class TestParseTrace:
    def test_numbered_parsing(self):
        trace = TraceInput(text="Step 1: Hello world\nStep 2: Goodbye world")
        steps = parse_trace(trace)
        assert len(steps) == 2
        assert steps[0].text == "Hello world"
        assert steps[1].text == "Goodbye world"
        assert steps[0].index == 0
        assert steps[1].index == 1

    def test_numbered_multiline_step(self):
        trace = TraceInput(text="Step 1: First line\ncontinuation of first\nStep 2: Second")
        steps = parse_trace(trace)
        assert len(steps) == 2
        assert "continuation" in steps[0].text

    def test_xml_parsing(self):
        trace = TraceInput(text="<step>First step content</step><step>Second step</step>")
        steps = parse_trace(trace)
        assert len(steps) == 2
        assert steps[0].text == "First step content"

    def test_xml_thinking_tags(self):
        trace = TraceInput(text="<thinking>Deep thought here</thinking>")
        steps = parse_trace(trace)
        assert len(steps) == 1
        assert steps[0].text == "Deep thought here"

    def test_markdown_parsing(self):
        text = "## Step 1\nFirst content\n## Step 2\nSecond content"
        trace = TraceInput(text=text)
        steps = parse_trace(trace)
        assert len(steps) == 2

    def test_newline_parsing(self):
        text = "First block of reasoning.\n\nSecond block of reasoning."
        trace = TraceInput(text=text)
        steps = parse_trace(trace)
        assert len(steps) == 2

    def test_pre_split_steps(self):
        trace = TraceInput(steps=["First step", "Second step", "Third step"])
        steps = parse_trace(trace)
        assert len(steps) == 3
        assert steps[0].text == "First step"

    def test_empty_input(self):
        trace = TraceInput(text="")
        assert parse_trace(trace) == []

    def test_none_input(self):
        trace = TraceInput()
        assert parse_trace(trace) == []

    def test_explicit_format(self):
        text = "Step 1: This is numbered\nStep 2: But force newline"
        trace = TraceInput(text=text, trace_format="newline")
        steps = parse_trace(trace)
        # Forced newline format — no double newlines, so it's one block
        assert len(steps) == 1

    def test_clean_trace_fixture(self, clean_trace):
        trace = TraceInput(text=clean_trace)
        steps = parse_trace(trace)
        assert len(steps) == 5
        assert "Python" in steps[0].text

    def test_custom_parser_registration(self):
        from cot_coherence.models import ReasoningStep

        def pipe_parser(text):
            parts = text.split("|")
            return [
                ReasoningStep(index=i, text=p.strip(), raw_text=p)
                for i, p in enumerate(parts)
                if p.strip()
            ]

        register_parser("pipe", pipe_parser)
        trace = TraceInput(text="First | Second | Third", trace_format="pipe")
        steps = parse_trace(trace)
        assert len(steps) == 3
        assert steps[1].text == "Second"
