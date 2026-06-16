"""
cf_diagnose.py — run counterfact's full counterfactual diagnosis on a live pipeline.

This is the runner the counterfact-debugger skill drives. It loads a user-supplied
pipeline FACTORY, wires an LLM caller for quality classifiers, runs the real ablation
diagnosis (Shapley attribution), and writes both machine- and human-readable reports.

Usage:
    python cf_diagnose.py \
        --factory myapp.pipeline:build \
        --inputs cases.json \
        --domain rag \
        --num-simulations 30 \
        --seed 42 \
        --out report.json

Contracts:
  --factory  "module:function" where function() takes no required args and returns a
             COMPILED counterfact graph (i.e. counterfact.StateGraph(...).compile()).
             A graph built with raw langgraph has no build recipe and cannot be diagnosed.
  --inputs   Path to a JSON file: either one input_state dict, or a list of them.
             Each is a dict passed to pipeline.invoke() / diagnose(input_state=...).
  --registry (optional) "module:function" returning a counterfact ClassifierRegistry.
             If omitted, the built-in classifiers for --domain are used.

Output:
  Writes <out> (JSON, full report.to_dict()) and <out-with-.md> (report.to_markdown()).
  For multiple inputs, writes one numbered pair per case plus an aggregate summary to stdout.
  Exit code is 0 on success, nonzero on setup/usage errors.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

# Make the sibling llm_fn.py importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_fn as _llm  # noqa: E402


def _die(msg: str, code: int = 2) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _resolve(ref: str, what: str):
    """Resolve a "module:attr" reference to the attribute object."""
    if ":" not in ref:
        _die(f"--{what} must be 'module:function', got {ref!r}")
    mod_name, attr = ref.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001
        _die(f"could not import module {mod_name!r} for --{what}: {e}")
    try:
        return getattr(mod, attr)
    except AttributeError:
        _die(f"module {mod_name!r} has no attribute {attr!r} (for --{what})")


def _load_inputs(path: str) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        _die(f"--inputs file not found: {path}")
    except json.JSONDecodeError as e:
        _die(f"--inputs is not valid JSON: {e}")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data
    _die("--inputs must be a JSON object or a list of objects")


def _build_graph(factory):
    """Call the factory and sanity-check that the result can be diagnosed."""
    try:
        graph = factory()
    except TypeError as e:
        _die(f"factory must be callable with no required args: {e}")
    except Exception as e:  # noqa: BLE001
        _die(f"factory raised while building the pipeline: {e}")
    if not hasattr(graph, "diagnose"):
        _die(
            "factory did not return a counterfact graph (no .diagnose). "
            "Build it with `from counterfact import StateGraph` and return `graph.compile()`."
        )
    # The recipe is required for ablation; surface the clear error early.
    if getattr(graph, "_recipe", "missing") is None:
        _die(
            "the compiled graph has no build recipe — it was built with raw langgraph. "
            "Swap to `from counterfact import StateGraph` to enable diagnosis."
        )
    return graph


def _reconcile_domain(registry, domain: str) -> str:
    """Avoid the silent-empty-classifier footgun.

    A custom registry organizes classifiers by domain. If --domain doesn't match any
    domain the registry actually has, ClassifierRegistry.get() returns an empty list,
    every output scores a flat 0.5, and attribution becomes meaningless noise — with
    no error. Here we detect that and either auto-correct (single-domain registry) or
    fail loudly with the available domains.
    """
    if registry is None:
        return domain
    domains = list(getattr(registry, "_classifiers", {}).keys())
    if not domains:
        _die("the provided --registry has no classifiers registered.")
    if registry.get(domain):
        return domain  # requested domain has classifiers — fine
    if len(domains) == 1:
        print(f"warning: --domain {domain!r} has no classifiers in this registry; "
              f"using its only domain {domains[0]!r} instead.", file=sys.stderr)
        return domains[0]
    _die(
        f"--domain {domain!r} matches no classifiers in this registry. "
        f"Available domains: {domains}. Re-run with --domain set to one of them."
    )


def _progress_printer(label: str, every: float = 5.0):
    """Throttled progress+ETA callback for graph.diagnose(progress_callback=...).
    Prints at most once per `every` seconds so long real-LLM diagnoses are watchable."""
    import time
    state = {"t0": time.monotonic(), "last": 0.0}

    def cb(current=0, total=0, status=""):
        now = time.monotonic()
        done = total and current >= total
        if now - state["last"] < every and not done:
            return
        state["last"] = now
        el = now - state["t0"]
        eta = (total - current) / (current / el) if current and el > 0 and total else 0
        pct = f"{100*current/total:.0f}%" if total else "?"
        print(f"  [{label}] {current}/{total or '?'} ({pct}) | elapsed {el:.0f}s"
              f"{f' | ETA {eta:.0f}s' if eta else ''}{f' | {status}' if status else ''}",
              file=sys.stderr, flush=True)

    return cb


def _short(value, n: int = 70) -> str:
    s = json.dumps(value, default=str) if not isinstance(value, str) else value
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _run_sensitivity(factory, inputs, args, registry, fn, degraders_map, mags) -> None:
    """Run graded-degradation sensitivity analysis per case and report classifications."""
    out_path = Path(args.out)
    multi = len(inputs) > 1
    print(f"counterfact sensitivity | magnitudes: {mags} | cases: {len(inputs)}", file=sys.stderr)
    summaries = []
    for i, state in enumerate(inputs):
        graph = _build_graph(factory)
        print(f"\n[case {i + 1}/{len(inputs)}] input: {_short(state)}", file=sys.stderr)
        report = graph.diagnose_sensitivity(
            state,
            degraders=degraders_map,
            magnitudes=mags,
            registry=registry,
            llm_fn=fn,
            domain=args.domain,
            seed=args.seed,
            progress_callback=_progress_printer(f"case {i + 1}/{len(inputs)}"),
        )
        case_json = out_path if not multi else out_path.with_name(f"{out_path.stem}_{i + 1}{out_path.suffix}")
        case_md = case_json.with_suffix(".md")
        case_json.write_text(json.dumps(report.to_dict(), indent=2))
        report.to_markdown(str(case_md))

        print(f"  baseline quality {report.baseline_quality:.3f} | per-node dose-response:", file=sys.stderr)
        for n in report.ranked():
            print(f"    {n.node:16} {n.module_type:9} {n.classification:13} "
                  f"sens {n.sensitivity:+.3f} (partial {n.partial_sensitivity:+.3f})", file=sys.stderr)
        drivers = [n.node for n in report.nodes if n.classification == "quality_driver"]
        harmful = [n.node for n in report.nodes if n.classification == "harmful"]
        structural = [n.node for n in report.nodes if n.classification == "structural"]
        if drivers:
            print(f"  -> quality drivers (improving these should help): {drivers}", file=sys.stderr)
        if harmful:
            print(f"  -> harmful (degrading/removing improves quality): {harmful}", file=sys.stderr)
        print(f"  -> wrote {case_json} and {case_md}", file=sys.stderr)
        top = report.most_sensitive()
        summaries.append({
            "case": i + 1,
            "baseline_quality": round(report.baseline_quality, 3),
            "most_sensitive": top.node if top else None,
            "quality_drivers": drivers,
            "harmful": harmful,
            "structural": structural,
            "report_json": str(case_json),
            "report_md": str(case_md),
        })
    if multi:
        agg_path = out_path.with_name(f"{out_path.stem}_summary.json")
        agg_path.write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote aggregate summary: {agg_path}", file=sys.stderr)
    print(json.dumps(summaries if multi else summaries[0], indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Run counterfact full counterfactual diagnosis.")
    p.add_argument("--factory", required=True, help="module:function -> compiled counterfact graph")
    p.add_argument("--inputs", required=True, help="JSON file: input_state dict or list of dicts")
    p.add_argument("--registry", default=None, help="module:function -> ClassifierRegistry (optional)")
    p.add_argument("--domain", default="rag", help="Built-in classifier domain (default: rag)")
    p.add_argument("--num-simulations", type=int, default=30, help="Monte Carlo simulations (default: 30)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--provider", default=None, choices=["anthropic", "google"], help="Force LLM provider")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM classifiers (structural only)")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run GRADED DEGRADATION instead of pure ablation. For each node, "
                        "progressively degrade its output (magnitude 1.0 = ablation) and "
                        "classify the dose-response: quality_driver / structural / harmful / "
                        "robust. Use this when a module (retriever, parser, context builder) "
                        "would just structurally collapse the pipeline if fully ablated.")
    p.add_argument("--degraders", default=None,
                   help="module:function -> {node_name: Degrader} overrides for --sensitivity "
                        "(optional; built-in degraders are auto-selected by module type otherwise)")
    p.add_argument("--magnitudes", default="0.25,0.5,1.0",
                   help="Comma-separated degradation magnitudes for --sensitivity (default 0.25,0.5,1.0)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the pipeline ONCE (baseline only), print the output and the "
                        "re-execution budget a full diagnosis would incur, then exit. Use this "
                        "to gauge side-effect/cost risk before committing to ablation re-runs.")
    p.add_argument("--out", default="report.json", help="Output JSON path (default: report.json)")
    args = p.parse_args()

    factory = _resolve(args.factory, "factory")
    inputs = _load_inputs(args.inputs)
    registry = _resolve(args.registry, "registry")() if args.registry else None
    args.domain = _reconcile_domain(registry, args.domain)

    # Resolve the LLM caller (or None for structural-only).
    if args.no_llm:
        fn = None
        provider_desc = "disabled (--no-llm)"
    else:
        try:
            fn = _llm.make_llm_fn(args.provider)
        except Exception as e:  # noqa: BLE001
            _die(str(e))
        provider_desc = _llm.describe() if fn else "none found — structural-only"
        if fn is None:
            print(
                "warning: no LLM key set (ANTHROPIC_API_KEY / GOOGLE_API_KEY); "
                "running structural classifiers only — attribution signal will be weak.",
                file=sys.stderr,
            )

    print(f"counterfact diagnose | provider: {provider_desc} | domain: {args.domain} "
          f"| sims: {args.num_simulations} | cases: {len(inputs)}", file=sys.stderr)

    # Friction #2 (destructive re-runs / cost): make the execution budget explicit.
    # Diagnosis re-executes the WHOLE pipeline once per simulation, per case.
    total_exec = args.num_simulations * len(inputs)
    print(f"NOTE: a full diagnosis re-executes your pipeline ~{total_exec} times "
          f"({args.num_simulations} sims x {len(inputs)} case(s)). If any node has real "
          f"side effects (writes, payments, emails, rate-limited APIs), point the factory "
          f"at a sandbox/mock first. Use --dry-run to preview one run.", file=sys.stderr)

    if args.dry_run:
        graph = _build_graph(factory)
        state = inputs[0]
        print(f"\n[dry-run] invoking pipeline once on: {_short(state)}", file=sys.stderr)
        result = graph.invoke(state)
        out_text = result.get("output", result) if isinstance(result, dict) else result
        print(json.dumps({
            "dry_run": True,
            "output": out_text,
            "full_diagnosis_would_execute_pipeline_times": total_exec,
        }, indent=2, default=str))
        return

    # ── Sensitivity mode: graded degradation instead of pure ablation ──
    if args.sensitivity:
        if fn is not None:
            # Built-in classifiers read a globally-injected LLM caller.
            from counterfact.classifiers import set_llm_caller
            set_llm_caller(fn)
        degraders_map = _resolve(args.degraders, "degraders")() if args.degraders else None
        try:
            mags = tuple(float(x) for x in args.magnitudes.split(","))
        except ValueError:
            _die(f"--magnitudes must be comma-separated floats, got {args.magnitudes!r}")
        _run_sensitivity(factory, inputs, args, registry, fn, degraders_map, mags)
        return

    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    multi = len(inputs) > 1

    summaries = []
    for i, state in enumerate(inputs):
        graph = _build_graph(factory)  # fresh build per case — no shared state
        print(f"\n[case {i + 1}/{len(inputs)}] input: {_short(state)}", file=sys.stderr)
        report = graph.diagnose(
            input_state=state,
            domain=args.domain,
            num_simulations=args.num_simulations,
            registry=registry,
            llm_fn=fn,
            seed=args.seed,
            progress_callback=_progress_printer(f"case {i + 1}/{len(inputs)}"),
        )

        case_json = out_path if not multi else out_path.with_name(f"{out_path.stem}_{i + 1}{out_path.suffix}")
        case_md = case_json.with_suffix(".md")
        case_json.write_text(json.dumps(report.to_dict(), indent=2))
        report.to_markdown(str(case_md))

        used = report.simulation_results_summary.get("classifiers_used", [])
        if not used:
            print("warning: NO classifiers ran — quality is a flat default and attribution "
                  "is meaningless. Check that --domain matches your registry, or that an LLM "
                  "key is set for the built-in classifiers.", file=sys.stderr)

        cls = report.classification
        top = max(report.shapley_values.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
        print(f"  -> {cls.failure_type} (conf {cls.confidence:.0%}) "
              f"| baseline quality {report.baseline_quality:.3f} "
              f"| top agent: {top[0]} ({top[1]:+.3f}) "
              f"| method: {report.attribution_method}", file=sys.stderr)

        # Auto-detect: a strongly POSITIVE top Shapley on a low-quality baseline
        # means removing that agent collapses the pipeline — i.e. ablation is the
        # blunt, structural signal. Nudge the agent toward graded degradation.
        if report.baseline_quality < 0.5 and top[1] > 0.3:
            print(f"  HINT: ablating '{top[0]}' mostly causes a structural collapse "
                  f"(large positive Shapley on a failing pipeline), which says it is "
                  f"NECESSARY but not whether its OUTPUT QUALITY is the problem. If it is a "
                  f"retriever/parser/context builder, re-run with --sensitivity to measure "
                  f"graded degradation and tell structural from quality-driving.", file=sys.stderr)
        print(f"  -> wrote {case_json} and {case_md}", file=sys.stderr)
        summaries.append({
            "case": i + 1,
            "failure_type": cls.failure_type,
            "confidence": round(cls.confidence, 3),
            "baseline_quality": round(report.baseline_quality, 3),
            "dominant_agent": cls.dominant_agent,
            "top_shapley": {"agent": top[0], "value": round(top[1], 4)},
            "attribution_method": report.attribution_method,
            "report_json": str(case_json),
            "report_md": str(case_md),
        })

    if multi:
        agg_path = out_path.with_name(f"{out_path.stem}_summary.json")
        agg_path.write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote aggregate summary: {agg_path}", file=sys.stderr)

    # Echo a compact summary to stdout for the agent to parse.
    print(json.dumps(summaries if multi else summaries[0], indent=2))


if __name__ == "__main__":
    main()
