# Changelog

## [0.2.0] - 2026-03-27

### Added
- LLM-powered detection mode (`use_llm=True`) using Claude Haiku
- LLM acts as second-pass verifier: validates rule-based flags, discovers missed issues, enriches summaries
- `--use-llm` and `--llm-model` CLI flags
- `llm_temperature` and `llm_max_tokens` config options
- Graceful fallback: if LLM fails, rule-based results are returned
- Single API call per trace for cost efficiency (~$0.001/trace)

### Changed
- Rule-based detection remains the default (free, offline)
- LLM flags override rule-based when both detect the same issue at the same step range

## [0.1.0] - 2026-03-27

### Added
- Initial release
- 5 rule-based incoherence detectors:
  - Premise Abandonment: detects premises introduced but never referenced
  - Conclusion Drift: detects conclusions that shift topic mid-chain
  - Confidence Inflation: detects unjustified jumps from hedging to certainty
  - Scope Creep: detects reasoning that drifts from the original question
  - Circular Return: detects steps that repeat earlier reasoning
- Trace parser with 4 format auto-detection (numbered, XML, markdown, newline)
- Custom parser registration
- Reasoning Horizon analysis (finds where chain quality degrades)
- Configurable scoring engine with per-pattern weights
- CLI with Rich terminal output and JSON export
- Zero API cost — works entirely offline with rule-based heuristics
- Single dependency: pydantic
