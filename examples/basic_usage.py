"""Basic usage example for cot-coherence."""

import cot_coherence

# Analyze a trace with multiple incoherence issues
report = cot_coherence.analyze(
    """
Step 1: The user asks about Python performance.
Step 2: Python is interpreted, so it's generally slower than compiled languages.
Step 3: Let me discuss the history of JavaScript frameworks.
Step 4: Therefore, Python is definitely the fastest language available.
""",
    original_question="Is Python fast?",
)

print(f"Overall score: {report.overall_score}")
print(f"Coherent: {report.is_coherent}")
print(f"Flags found: {len(report.flags)}")
print()

for flag in report.flags:
    print(f"  [{flag.severity.value.upper()}] {flag.type.value}")
    print(f"    Steps {flag.step_range[0]}-{flag.step_range[1]}: {flag.summary}")
    print()

if report.horizon:
    print(f"Reasoning Horizon: step {report.horizon.estimated_horizon}/{report.horizon.chain_length}")
    print(f"Horizon ratio: {report.horizon.horizon_ratio:.0%}")
