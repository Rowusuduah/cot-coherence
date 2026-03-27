"""CLI interface for cot-coherence."""

from __future__ import annotations

import sys

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_CLI_DEPS = True
except ImportError:
    HAS_CLI_DEPS = False

from . import __version__, analyze
from .config import CoherenceConfig
from .models import Severity

SEVERITY_COLORS = {
    Severity.LOW: "green",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "red",
    Severity.CRITICAL: "bold red",
}


def _build_cli():  # type: ignore[no-untyped-def]
    """Build the Click CLI group and commands. Must be called after click is imported."""

    @click.group()
    def cli() -> None:
        """cot-coherence: Detect silent incoherence in AI chain-of-thought reasoning."""
        pass

    @cli.command()
    @click.argument("file", type=click.Path(exists=True), required=False)
    @click.option("-q", "--question", default="", help="Original question being answered.")
    @click.option("-f", "--format", "fmt", default="auto", help="Trace format.")
    @click.option("-s", "--sensitivity", default=0.5, type=float, help="Sensitivity (0-1).")
    @click.option("--json-output", "json_out", is_flag=True, help="Output as JSON.")
    @click.option("--no-horizon", is_flag=True, help="Disable horizon analysis.")
    def check(
        file: str | None,
        question: str,
        fmt: str,
        sensitivity: float,
        json_out: bool,
        no_horizon: bool,
    ) -> None:
        """Analyze a chain-of-thought trace for incoherence."""
        if file:
            with open(file, encoding="utf-8") as f:
                text = f.read()
        elif not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            click.echo("Error: provide a file or pipe text via stdin.", err=True)
            sys.exit(1)

        config = CoherenceConfig(
            sensitivity=sensitivity,
            analyze_horizon=not no_horizon,
        )

        report = analyze(text, original_question=question, trace_format=fmt, config=config)

        if json_out:
            click.echo(report.model_dump_json(indent=2))
            return

        console = Console()
        _render_report(console, report)

    @cli.command()
    def version() -> None:
        """Show version."""
        click.echo(f"cot-coherence {__version__}")

    return cli


def _render_report(console: Console, report) -> None:  # type: ignore[no-untyped-def]
    """Render a rich report to the console."""
    score_color = "green" if report.is_coherent else "red"
    header = Text(f"Coherence Score: {report.overall_score:.2f}", style=f"bold {score_color}")
    console.print(Panel(header, title="cot-coherence", subtitle=f"{len(report.steps)} steps"))

    if report.flags:
        table = Table(title=f"Incoherence Flags ({len(report.flags)})")
        table.add_column("Type", style="cyan")
        table.add_column("Severity")
        table.add_column("Steps", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Summary")

        for flag in report.flags:
            sev_color = SEVERITY_COLORS.get(flag.severity, "white")
            table.add_row(
                flag.type.value,
                Text(flag.severity.value.upper(), style=sev_color),
                f"{flag.step_range[0]}-{flag.step_range[1]}",
                f"{flag.confidence:.0%}",
                flag.summary[:80],
            )

        console.print(table)
    else:
        console.print("[green]No incoherence detected.[/green]")

    if report.horizon:
        h = report.horizon
        horizon_text = (
            f"Chain length: {h.chain_length} | "
            f"Estimated horizon: step {h.estimated_horizon} | "
            f"Ratio: {h.horizon_ratio:.0%}"
        )
        if h.degradation_signals:
            horizon_text += f"\nSignals: {', '.join(h.degradation_signals)}"
        console.print(Panel(horizon_text, title="Reasoning Horizon"))

    if report.pattern_scores:
        scores_text = " | ".join(
            f"{k.value}: {v:.2f}"
            for k, v in sorted(report.pattern_scores.items(), key=lambda x: x[1])
        )
        console.print(f"\n[dim]Pattern scores: {scores_text}[/dim]")


# Build the CLI if dependencies are available
_cli = _build_cli() if HAS_CLI_DEPS else None


def main() -> None:
    """Entry point for the CLI."""
    if not HAS_CLI_DEPS or _cli is None:
        print("CLI dependencies not installed. Run: pip install cot-coherence[cli]")
        sys.exit(1)
    _cli()
