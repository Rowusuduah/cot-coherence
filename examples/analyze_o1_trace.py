"""Example: analyzing an o1-style reasoning trace."""

import cot_coherence

# Simulated o1-style extended reasoning trace
trace = """
Step 1: The user wants to know if renewable energy can fully replace fossil fuels by 2050. Let me think through this systematically.

Step 2: Solar and wind energy costs have dropped 90% since 2010, making them cheaper than new fossil fuel plants in most regions. This is a strong foundation for the transition.

Step 3: However, intermittency remains a challenge. Solar produces nothing at night, and wind varies seasonally. Grid-scale storage solutions like batteries and pumped hydro are needed.

Step 4: Current battery technology using lithium-ion can store 4-8 hours of energy. For seasonal storage (weeks to months), we need alternatives like hydrogen or compressed air.

Step 5: Given that nuclear power provides stable baseload generation, it could complement renewables during the transition. France generates 70% of electricity from nuclear.

Step 6: The global shipping and aviation industries are particularly hard to electrify. These sectors may require synthetic fuels or hydrogen, which are currently expensive.

Step 7: Therefore, while renewable energy can likely provide 80-90% of electricity by 2050, complete replacement of all fossil fuels across all sectors faces significant technical and economic barriers.
"""

report = cot_coherence.analyze(
    trace,
    original_question="Can renewable energy fully replace fossil fuels by 2050?",
)

print(f"Score: {report.overall_score:.2f} ({'Coherent' if report.is_coherent else 'Incoherent'})")
print(f"Steps: {len(report.steps)}")
print(f"Issues: {len(report.flags)}")

if report.flags:
    for f in report.flags:
        print(f"  - {f.type.value}: {f.summary}")

if report.horizon:
    print(f"\nHorizon: {report.horizon.estimated_horizon}/{report.horizon.chain_length} "
          f"({report.horizon.horizon_ratio:.0%})")
