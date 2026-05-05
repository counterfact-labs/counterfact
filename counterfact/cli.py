"""
Command Line Interface for counterfact.

Usage:
    counterfact eval trace.json --domain rag
    counterfact discover logs.txt

Note: Full counterfactual diagnostics (diagnose) require access to the
actual pipeline and are only available via the Python API:

    report = compiled_pipeline.diagnose(input_state={...})

Requires: pip install counterfact[cli]
"""
import sys
import os
import json
import argparse
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
except ImportError:
    print("The 'rich' library is required for the CLI.")
    print("Install it with: pip install counterfact[cli]")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# TRACE FILE LOADING
# ═════════════════════════════════════════════════════════════════════════


def _load_trace(path: str, console: Console) -> dict:
    """
    Load a trace file. Supports two formats:
      1. Plain array: [{"node": ..., "output": ...}, ...]
      2. Full input: {"trace": [...], "query": "...", "output": "..."}
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {path}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in {path}: {e}[/red]")
        sys.exit(1)

    if isinstance(data, list):
        return {"trace": data}
    elif isinstance(data, dict) and "trace" in data:
        return data
    else:
        console.print("[red]Error: Trace file must be a JSON array or an object with a 'trace' key.[/red]")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# LLM PROVIDER SETUP
# ═════════════════════════════════════════════════════════════════════════


def _resolve_llm_fn(provider: Optional[str], console: Console):
    """
    Resolve the LLM function from environment variables.
    Used for Tier 2 eval checks (which require an LLM).
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

    if provider == "anthropic" or (provider is None and anthropic_key):
        if not anthropic_key:
            console.print("[red]Error: --provider anthropic requires ANTHROPIC_API_KEY[/red]")
            sys.exit(1)
        return _make_anthropic_caller(anthropic_key, console)

    elif provider == "google" or (provider is None and google_key):
        if not google_key:
            console.print("[red]Error: --provider google requires GOOGLE_API_KEY or GEMINI_API_KEY[/red]")
            sys.exit(1)
        return _make_google_caller(google_key, console)

    elif provider is not None:
        console.print(f"[red]Error: Unknown provider '{provider}'. Use 'anthropic' or 'google'.[/red]")
        sys.exit(1)

    return None


def _make_anthropic_caller(api_key: str, console: Console):
    """Create an LLM caller using Anthropic Claude."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        console.print("[red]Error: 'anthropic' package not installed.[/red]")
        console.print("Install it with: pip install counterfact[anthropic]")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("COUNTERFACT_MODEL", "claude-sonnet-4-20250514")
    console.print(f"  Provider: [bold cyan]Anthropic[/bold cyan] ({model})")

    def caller(prompt: str, temperature: float = 0.1) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        return getattr(block, "text", str(block))

    return caller


def _make_google_caller(api_key: str, console: Console):
    """Create an LLM caller using Google Gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError:
        console.print("[red]Error: 'google-genai' package not installed.[/red]")
        console.print("Install it with: pip install counterfact[google]")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model = os.environ.get("COUNTERFACT_MODEL", "gemini-2.5-flash")
    console.print(f"  Provider: [bold cyan]Google Gemini[/bold cyan] ({model})")

    def caller(prompt: str, temperature: float = 0.1) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2000,
            ),
        )
        return response.text or ""

    return caller


# ═════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="counterfact",
        description="Counterfactual diagnostics for multi-agent AI pipelines.",
        epilog=(
            "Note: Full counterfactual diagnostics require the actual pipeline "
            "and are only available via the Python API. See: "
            "https://github.com/counterfact-labs/counterfact"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── eval ──────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run ground-truth-free evaluation checks on a trace",
    )
    eval_parser.add_argument("trace_file", help="Path to trace JSON file")
    eval_parser.add_argument("--domain", type=str, default="rag", choices=["rag", "decision"], help="Classifier domain (default: rag)")
    eval_parser.add_argument("--provider", type=str, default=None, choices=["anthropic", "google"], help="LLM provider for Tier 2 checks (optional)")

    # ── discover ──────────────────────────────────────────────────────
    disc_parser = subparsers.add_parser(
        "discover",
        help="Discover pipeline topology from raw logs",
    )
    disc_parser.add_argument("log_file", help="Path to raw or JSON logs")

    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args)
    elif args.command == "discover":
        run_discover(args)
    else:
        parser.print_help()


# ═════════════════════════════════════════════════════════════════════════
# EVAL COMMAND
# ═════════════════════════════════════════════════════════════════════════


def run_eval(args):
    console = Console()
    console.print(Panel.fit(
        "[bold green]counterfact eval[/bold green] ✓",
        subtitle="Ground-truth-free evaluation checks",
        border_style="green",
    ))

    # Load trace
    trace_data = _load_trace(args.trace_file, console)
    trace = trace_data["trace"]
    output_text = trace_data.get("output", "")

    # Resolve LLM (optional — enables Tier 2 checks)
    llm_fn = _resolve_llm_fn(args.provider, console) if args.provider else None
    tiers = [1, 2] if llm_fn else [1]

    agents = list(dict.fromkeys(
        entry["node"] for entry in trace
        if entry.get("node") and entry.get("node") != "output"
    ))

    console.print(f"\n  Trace: [bold]{len(trace)}[/bold] events from [bold]{args.trace_file}[/bold]")
    console.print(f"  Agents: {', '.join(f'[cyan]{a}[/cyan]' for a in agents)}")
    console.print(f"  Tiers: {', '.join(str(t) for t in tiers)}")
    console.print()

    from counterfact.evals import run_eval_suite

    with console.status("[bold green]Running evaluation checks...", spinner="dots"):
        try:
            suite = run_eval_suite(
                trace=trace,
                final_output=output_text,
                llm_fn=llm_fn,
                tiers=tiers,
            )
        except Exception as e:
            console.print(f"[bold red]Eval failed:[/bold red] {e}")
            sys.exit(1)

    # Render results table
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Severity", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    for result in suite.results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        table.add_row(
            result.check_name,
            str(result.severity),
            status,
            result.message[:100] if result.message else str(result.details)[:100],
        )

    console.print(table)

    passed = sum(1 for r in suite.results if r.passed)
    total = len(suite.results)
    console.print(Panel(
        f"Passed: [bold]{passed}/{total}[/bold]  |  Score: [bold]{passed/total:.1%}[/bold]" if total > 0 else f"Passed: [bold]{passed}/{total}[/bold]  |  Score: [bold]N/A[/bold]",
        title="Summary",
        border_style="green" if passed == total else "yellow",
    ))


# ═════════════════════════════════════════════════════════════════════════
# DISCOVER COMMAND
# ═════════════════════════════════════════════════════════════════════════


def run_discover(args):
    console = Console()
    console.print(Panel.fit(
        "[bold magenta]counterfact discover[/bold magenta] 🕵️",
        subtitle="Pipeline topology discovery",
        border_style="magenta",
    ))

    from counterfact.discovery import discover_pipeline

    with console.status(f"[bold green]Parsing: {args.log_file}...", spinner="bouncingBar"):
        try:
            with open(args.log_file, "r") as f:
                logs = f.read()
            pipeline_def = discover_pipeline(logs)
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {args.log_file}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Discovery failed:[/bold red] {e}")
            sys.exit(1)

    # Render as a tree
    tree = Tree(f"[bold]{pipeline_def.get('name', 'Discovered Pipeline')}[/bold]")
    for agent, info in pipeline_def.get("agents", {}).items():
        node = tree.add(f"[cyan]{agent}[/cyan]")
        node.add(f"Inputs: {', '.join(info.get('inputs', []))}")
        node.add(f"Outputs: {', '.join(info.get('outputs', []))}")

    console.print(tree)


if __name__ == "__main__":
    main()
