"""
Perturbation engine for counterfactual analysis.

This module handles the "what if?" experiments that are at the heart
of the counterfact approach. It:

  1. Generates perturbations — decides which agents to ablate
  2. Applies perturbations — actually re-runs the pipeline with the
     agent's function replaced by a no-op
  3. Runs Monte Carlo — executes many ablation re-runs and measures
     quality changes via classifiers

The key principle: we never simulate or guess what a perturbed pipeline
would produce. We actually re-execute the pipeline with the node function
replaced, getting real outputs that we can then score.

Dependencies: types, classifiers (for quality scoring)
"""

from typing import Callable, Optional, TYPE_CHECKING

from counterfact.types import (
    Perturbation,
    SimulationResult,
)

if TYPE_CHECKING:
    from counterfact.graph import CounterfactualGraph


# ═════════════════════════════════════════════════════════════════════════
# PERTURBATION GENERATION
# Decide what to perturb.
# ═════════════════════════════════════════════════════════════════════════


def generate_perturbations_from_graph(
    graph: "CounterfactualGraph",
) -> list[Perturbation]:
    """
    Generate one ablation perturbation per agent node in the pipeline.

    Each perturbation represents removing one agent entirely (replacing
    its function with a no-op that passes state through unchanged).
    This maps directly to Leave-One-Out (LOO) attribution.

    Args:
        graph: The compiled CounterfactualGraph.

    Returns:
        List of Perturbation objects, one per agent.
    """
    perturbations = []
    for agent in graph.get_node_names():
        perturbations.append(Perturbation(
            agent=agent,
            strategy="ablate",
            description=f"Remove {agent} — replace with no-op and re-run pipeline",
            magnitude=1.0,
        ))
    return perturbations


def generate_perturbations(trace: list[dict]) -> list[Perturbation]:
    """
    Generate perturbations from a trace (for eval-only mode).

    When we only have a trace (no live pipeline), we can still
    generate perturbation definitions for reporting purposes,
    but cannot actually execute them.
    """
    perturbations = []
    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry["node"] != "output"
    ))
    for agent in agents:
        perturbations.append(Perturbation(
            agent=agent,
            strategy="ablate",
            description=f"Remove {agent} from the pipeline — measure its marginal contribution",
            magnitude=1.0,
        ))
    return perturbations


# ═════════════════════════════════════════════════════════════════════════
# REAL RE-EXECUTION ENGINE
# Actually re-run the pipeline with ablated nodes.
# ═════════════════════════════════════════════════════════════════════════


def _run_pipeline_safe(
    graph: "CounterfactualGraph",
    input_state: dict,
) -> tuple[dict, list[dict]]:
    """
    Run a pipeline and return (result, trace), catching errors gracefully.

    If the pipeline raises an exception (which can happen when an agent
    is ablated and downstream agents receive unexpected input), we
    capture the error and return a partial result.
    """
    try:
        result = graph.invoke(input_state)
        trace = graph.get_trace()
        return result, trace
    except Exception as e:
        # Pipeline crashed — this is valid data for attribution.
        # A crash when an agent is ablated means that agent was critical.
        trace = graph.get_trace()
        return {"_error": str(e), "_partial_trace": trace}, trace


def _extract_final_output(result: dict) -> str:
    """
    Extract a string representation of the pipeline's final output.

    Tries common output key names, falls back to string representation.
    """
    if "_error" in result:
        return f"[PIPELINE ERROR: {result['_error'][:200]}]"

    # Try common output keys
    for key in [
        "final_output", "output", "response", "answer", "result",
        "synthesis", "text", "content",
    ]:
        if key in result and isinstance(result[key], str):
            return result[key]

    # Fall back to the full state as a string (excluding internal keys)
    visible = {k: v for k, v in result.items() if not k.startswith("_")}
    if visible:
        # Return the last string value
        for v in reversed(list(visible.values())):
            if isinstance(v, str) and len(v) > 10:
                return v

    return str(result)[:500]


def run_coalition(
    graph: "CounterfactualGraph",
    coalition: frozenset,
    input_state: dict,
) -> tuple[str, list[dict]]:
    """
    Run the pipeline with only the coalition of agents active.

    All agents NOT in the coalition are ablated (replaced with no-ops).
    Returns (final_output_text, trace).

    This is used by Shapley value computation to evaluate arbitrary
    subsets of agents via real pipeline re-execution.

    Args:
        graph: The compiled CounterfactualGraph (with build recipe).
        coalition: Set of agent names to keep active.
        input_state: The input dict to invoke the pipeline with.

    Returns:
        (output_text, trace_dicts) from the perturbed pipeline.
    """
    all_agents = set(graph.get_node_names())
    agents_to_ablate = all_agents - coalition

    # Build the perturbed pipeline by ablating each excluded agent
    perturbed = graph
    for agent in agents_to_ablate:
        perturbed = perturbed.clone_with_ablation(agent)

    result, trace = _run_pipeline_safe(perturbed, input_state)
    output_text = _extract_final_output(result)
    return output_text, trace


# ═════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION ENGINE
# Run many real re-executions and measure quality.
# ═════════════════════════════════════════════════════════════════════════


def run_monte_carlo(
    graph: "CounterfactualGraph",
    input_state: dict,
    sources: str = "",
    domain: str = "rag",
    num_simulations: int = 30,
    progress_callback: Optional[Callable] = None,
    registry=None,
    llm_fn: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> list[SimulationResult]:
    """
    Run Monte Carlo simulations with real pipeline re-execution.

    This is the main simulation loop:
      1. Run baseline measurements (pipeline as-is, multiple times for stability)
      2. For each agent, ablate it (replace with no-op) and re-run the
         actual pipeline
      3. Score each real output using quality classifiers
      4. Return all results for attribution analysis

    Unlike LLM-simulated counterfactuals, every simulation result comes
    from actually executing the pipeline code.

    Args:
        graph: The compiled CounterfactualGraph (with build recipe)
        input_state: The input dict to invoke the pipeline with
        sources: Source documents (for classifier context)
        domain: Classifier domain ("rag", "decision", etc.)
        num_simulations: Total number of simulations to run
        progress_callback: Optional callback(current, total, status)
        registry: ClassifierRegistry instance (uses default if None)
        llm_fn: LLM function for quality classifiers
        seed: Random seed for reproducibility
    """
    from counterfact.classifiers import ClassifierRegistry, get_default_registry

    reg = registry or get_default_registry()
    results = []
    sim_id = 0

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    agents = graph.get_node_names()

    # ── Step 1: Baseline runs ────────────────────────────────────────
    # Run the pipeline as-is multiple times to get a stable baseline.
    baseline_runs = max(3, num_simulations // 10)

    # Extract query from input state for classifier context
    query = input_state.get("query", str(input_state)[:200])

    for i in range(baseline_runs):
        result, trace = _run_pipeline_safe(graph, input_state)
        output_text = _extract_final_output(result)

        clf_results = reg.run_all(query, output_text, sources, domain)
        quality = ClassifierRegistry.aggregate_quality(clf_results)

        # Convert trace to simulation format
        baseline_traces = []
        for entry in trace:
            if entry.get("node") == "output":
                continue
            baseline_traces.append({
                "agent": entry["node"],
                "status": entry.get("status", "pass"),
                "note": "Original baseline execution",
                "input": entry.get("input", {}),
                "output": entry.get("output", {}),
            })

        results.append(SimulationResult(
            simulation_id=sim_id,
            perturbation=None,
            quality_score=quality,
            classifier_results=clf_results,
            perturbed_output=output_text[:200],
            is_baseline=True,
            agent_traces=baseline_traces,
        ))
        sim_id += 1
        if progress_callback:
            progress_callback(sim_id, num_simulations, "baseline")

    # ── Step 2: Shapley Coalition Runs ───────────────────────────────
    # Generate all subsets evaluated during Shapley permutations
    import itertools
    N = len(agents)
    if N <= 4:
        perms = list(itertools.permutations(agents))
    else:
        n_perms = max(5, num_simulations // N)
        perms = [tuple(random.sample(agents, N)) for _ in range(n_perms)]

    coalitions_to_run = set()
    full_coalition = frozenset(agents)
    for perm in perms:
        coalition = frozenset()
        for agent in perm:
            coalition = coalition | {agent}
            if coalition != full_coalition and coalition:
                coalitions_to_run.add(coalition)

    remaining = num_simulations - baseline_runs
    sims_per_coalition = max(1, remaining // len(coalitions_to_run)) if coalitions_to_run else 0

    for coalition in coalitions_to_run:
        ablated_agents = sorted(list(set(agents) - coalition))
        agent_str = ", ".join(ablated_agents)
        
        pert = Perturbation(
            agent=agent_str,
            strategy="ablate",
            description=f"Remove {agent_str} — replace with no-op",
            magnitude=1.0,
        )

        for _ in range(sims_per_coalition):
            output_text, trace = run_coalition(graph, coalition, input_state)

            # Score the real output
            clf_results = reg.run_all(query, output_text, sources, domain)
            quality = ClassifierRegistry.aggregate_quality(clf_results)

            # Build simulation trace
            sim_traces = []
            for entry in trace:
                if entry.get("node") == "output":
                    continue
                is_ablated = entry["node"] in ablated_agents
                sim_traces.append({
                    "agent": entry["node"],
                    "status": "ablated" if is_ablated else entry.get("status", "pass"),
                    "note": (
                        "Agent replaced with no-op"
                        if is_ablated
                        else "Real execution with upstream ablation"
                    ),
                    "input": entry.get("input", {}),
                    "output": entry.get("output", {}),
                })

            results.append(SimulationResult(
                simulation_id=sim_id,
                perturbation=pert,
                quality_score=quality,
                classifier_results=clf_results,
                perturbed_output=output_text[:200],
                is_baseline=False,
                agent_traces=sim_traces,
            ))
            sim_id += 1
            if progress_callback:
                progress_callback(sim_id, num_simulations, f"ablating {pert.agent}")

    return results
