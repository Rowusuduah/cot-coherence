"""Shared test fixtures for cot-coherence."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def clean_trace() -> str:
    return (FIXTURES_DIR / "clean_trace.txt").read_text(encoding="utf-8")


@pytest.fixture
def premise_abandonment_trace() -> str:
    return (FIXTURES_DIR / "premise_abandonment_trace.txt").read_text(encoding="utf-8")


@pytest.fixture
def conclusion_drift_trace() -> str:
    return (FIXTURES_DIR / "conclusion_drift_trace.txt").read_text(encoding="utf-8")


@pytest.fixture
def confidence_inflation_trace() -> str:
    return (FIXTURES_DIR / "confidence_inflation_trace.txt").read_text(encoding="utf-8")


@pytest.fixture
def scope_creep_trace() -> str:
    return (FIXTURES_DIR / "scope_creep_trace.txt").read_text(encoding="utf-8")


@pytest.fixture
def circular_return_trace() -> str:
    return (FIXTURES_DIR / "circular_return_trace.txt").read_text(encoding="utf-8")


@pytest.fixture
def multi_issue_trace() -> str:
    return (FIXTURES_DIR / "multi_issue_trace.txt").read_text(encoding="utf-8")
