# cot-coherence

**Detect silent incoherence in AI chain-of-thought reasoning.**

Most evaluation tools check if each reasoning step is *correct*. `cot-coherence` checks if the *logical progression between steps* holds together — schema-level coherence, not step-level accuracy.

## Why?

Chain-of-thought reasoning in LLMs silently degrades. Recent research (arXiv, Feb-Mar 2026) shows:

- CoT faithfulness **decays** at 70-85% of chain length ("Reasoning Horizon")
- Reasoning tokens have a **negative effect** past this horizon
- No existing tool detects this — current tools check RAG grounding, linguistic flow, or safety. None check **logical coherence between steps**.

`cot-coherence` fills this gap.

## Install

```bash
pip install cot-coherence
```

With CLI support:
```bash
pip install cot-coherence[cli]
```

## Quick Start

```python
import cot_coherence

report = cot_coherence.analyze("""
Step 1: The user asks about Python performance.
Step 2: Python is interpreted, so it's generally slower than compiled languages.
Step 3: Let me discuss the history of JavaScript frameworks.
Step 4: Therefore, Python is definitely the fastest language available.
""", original_question="Is Python fast?")

print(report.overall_score)  # 0.43
print(report.is_coherent)    # False
print(len(report.flags))     # 3+ (scope_creep, conclusion_drift, confidence_inflation)
```

## What It Detects

### 5 Incoherence Patterns

| Pattern | What It Catches | Example |
|---------|----------------|---------|
| **Premise Abandonment** | Premises introduced but never referenced again | "Given PostgreSQL uses RLS..." then never mentions DB security |
| **Conclusion Drift** | Conclusions that shift topic mid-chain | Concludes about databases, then about Kubernetes |
| **Confidence Inflation** | Unjustified jumps from hedging to certainty | "might work" → "definitely always works" with no evidence |
| **Scope Creep** | Reasoning that drifts from the original question | Asked about Python, starts discussing Roman architecture |
| **Circular Return** | Steps that repeat earlier reasoning | Step 5 restates Step 1 in different words |

### Reasoning Horizon

Detects the point in a chain where quality starts to degrade — the "Reasoning Horizon" described in recent research. Reports the estimated horizon position and degradation signals.

## CLI

```bash
# Analyze a trace file
cot-coherence check trace.txt -q "What is quantum computing?"

# Pipe from stdin
echo "Step 1: ..." | cot-coherence check

# JSON output
cot-coherence check trace.txt --json-output

# Adjust sensitivity (0.0=lenient, 1.0=strict)
cot-coherence check trace.txt -s 0.8

# Disable horizon analysis
cot-coherence check trace.txt --no-horizon
```

## Configuration

```python
from cot_coherence import analyze, CoherenceConfig, IncoherenceType

config = CoherenceConfig(
    sensitivity=0.7,                    # 0.0=lenient, 1.0=strict
    enabled_detectors={                 # Enable specific detectors
        IncoherenceType.PREMISE_ABANDONMENT,
        IncoherenceType.CONFIDENCE_INFLATION,
    },
    analyze_horizon=True,               # Enable horizon analysis
    weights={                           # Custom pattern weights
        IncoherenceType.PREMISE_ABANDONMENT: 2.0,
        IncoherenceType.CONFIDENCE_INFLATION: 1.5,
    },
)

report = analyze("Step 1: ...", config=config)
```

## Trace Formats

Auto-detects 4 formats:

```python
# Numbered (default for most LLMs)
"Step 1: First\nStep 2: Second"
"1. First\n2. Second"

# XML (Claude-style thinking)
"<step>First</step><step>Second</step>"
"<thinking>Reasoning here</thinking>"

# Markdown
"## Step 1\nFirst\n## Step 2\nSecond"

# Newline-separated (fallback)
"First block\n\nSecond block"
```

Or pass pre-split steps:
```python
report = analyze(steps=["First step", "Second step"])
```

## Custom Parsers

```python
from cot_coherence import register_parser
from cot_coherence.models import ReasoningStep

def my_parser(text):
    parts = text.split("|||")
    return [ReasoningStep(index=i, text=p.strip(), raw_text=p)
            for i, p in enumerate(parts) if p.strip()]

register_parser("pipe", my_parser)
report = analyze("First ||| Second ||| Third", trace_format="pipe")
```

## The CoherenceReport

```python
report = analyze(trace)

report.overall_score    # float 0.0-1.0
report.is_coherent      # True if score >= 0.7
report.steps            # list[ReasoningStep]
report.flags            # list[IncoherenceFlag]
report.critical_flags   # flags with CRITICAL severity
report.horizon          # HorizonAnalysis or None
report.pattern_scores   # dict[IncoherenceType, float]

# Each flag contains:
flag.type               # IncoherenceType enum
flag.severity           # LOW, MEDIUM, HIGH, CRITICAL
flag.confidence         # 0.0-1.0
flag.step_range         # (start_step, end_step)
flag.summary            # Human-readable description
flag.evidence           # Specific evidence
flag.suggestion         # How to fix it
```

## How It Works

v0.1 uses **rule-based heuristics** — zero API cost, works offline:

- **Premise Abandonment**: Extracts premise markers ("given", "assuming"), checks if key entities appear in subsequent steps
- **Conclusion Drift**: Identifies conclusion markers ("therefore", "thus"), compares topic overlap via Jaccard similarity
- **Confidence Inflation**: Tracks hedging vs. certainty word ratios, flags unjustified jumps without evidence markers
- **Scope Creep**: Measures content-word overlap between each step and the original question
- **Circular Return**: Computes content-word fingerprints, flags high similarity between non-adjacent steps

Scoring: each pattern starts at 1.0, penalties applied per flag based on severity and confidence. Overall score is the weighted average.

## Dependencies

**Required:** `pydantic>=2.0`

**Optional:**
- `[cli]` — `click`, `rich` (for terminal interface)
- `[llm]` — `anthropic` (for future LLM-powered detection)
- `[dev]` — `pytest`, `pytest-cov`, `ruff`, `mypy`

## License

Apache 2.0
