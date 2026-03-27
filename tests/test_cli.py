"""Tests for CLI interface."""

import json

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
        assert "0.2.0" in result.output

    def test_check_no_args_reads_empty_stdin(self):
        # In test runner, stdin is never a TTY, so empty stdin produces empty text
        result = self.runner.invoke(_cli, ["check"], input="")
        assert result.exit_code == 0

    def test_use_llm_without_api_key(self, fixtures_dir):
        result = self.runner.invoke(
            _cli,
            ["check", str(fixtures_dir / "clean_trace.txt"), "--use-llm"],
            env={"ANTHROPIC_API_KEY": ""},
        )
        assert result.exit_code != 0
        assert "ANTHROPIC_API_KEY" in result.output or "ANTHROPIC_API_KEY" in (result.stderr or "")

    def test_use_llm_flag_accepted(self, fixtures_dir):
        # Just verify the flag is accepted (won't actually call LLM without key)
        result = self.runner.invoke(
            _cli,
            ["check", str(fixtures_dir / "clean_trace.txt"), "--use-llm"],
            env={"ANTHROPIC_API_KEY": ""},
        )
        # Should fail with API key error, not an unknown option error
        assert "no such option" not in (result.output or "").lower()

    def test_llm_model_option_accepted(self):
        result = self.runner.invoke(
            _cli,
            ["check", "--help"],
        )
        assert "--llm-model" in result.output
        assert "--use-llm" in result.output
