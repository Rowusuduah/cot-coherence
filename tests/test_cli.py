"""Tests for CLI interface."""

import json
from pathlib import Path

from click.testing import CliRunner

from cot_coherence.cli import _cli

assert _cli is not None, "CLI dependencies not installed"


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_check_file(self, fixtures_dir):
        result = self.runner.invoke(_cli, ["check", str(fixtures_dir / "clean_trace.txt")])
        assert result.exit_code == 0
        assert "Coherence Score" in result.output

    def test_check_file_with_question(self, fixtures_dir):
        result = self.runner.invoke(
            _cli,
            ["check", str(fixtures_dir / "scope_creep_trace.txt"),
             "-q", "How does Python memory management work?"],
        )
        assert result.exit_code == 0

    def test_check_json_output(self, fixtures_dir):
        result = self.runner.invoke(
            _cli,
            ["check", str(fixtures_dir / "clean_trace.txt"), "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "overall_score" in data
        assert "flags" in data
        assert "steps" in data

    def test_check_no_horizon(self, fixtures_dir):
        result = self.runner.invoke(
            _cli,
            ["check", str(fixtures_dir / "clean_trace.txt"), "--no-horizon", "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["horizon"] is None

    def test_check_stdin(self):
        result = self.runner.invoke(
            _cli,
            ["check"],
            input="Step 1: First step.\nStep 2: Second step.",
        )
        assert result.exit_code == 0

    def test_check_multi_issue(self, fixtures_dir):
        result = self.runner.invoke(
            _cli,
            ["check", str(fixtures_dir / "multi_issue_trace.txt"),
             "-q", "What are Python performance optimization techniques?"],
        )
        assert result.exit_code == 0
        assert "Incoherence Flags" in result.output

    def test_version_command(self):
        result = self.runner.invoke(_cli, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_check_no_args_reads_empty_stdin(self):
        # In test runner, stdin is never a TTY, so empty stdin produces empty text
        result = self.runner.invoke(_cli, ["check"], input="")
        assert result.exit_code == 0
