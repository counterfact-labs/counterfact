"""
Asynchronous execution engine for counterfactual diagnostics.

This module provides async equivalents of run_monte_carlo and run_full_diagnostic,
allowing multiple LLM simulations to run concurrently, significant speeding up
the diagnostic process.
"""

import asyncio
from typing import Callable, Optional

from counterfact.attribution import (
    classify_failure,
    compute_bootstrap_ci,
    compute_loo_attribution,
    compute_per_classifier_loo,
    compute_shapley_values,
    is_loo_inconclusive,
)
from counterfact.diagnostics import DiagnosticReport, _build_summary, _make_no_failure_classification
from counterfact.perturbation import generate_perturbations
from counterfact.recommendations import (
    detect_coverage_gaps,
    extract_empirical_fixes,
    rank_recommendations,
)
from counterfact.types import ConfidenceInterval, SimulationResult


async def run_monte_carlo_async(
    trace: list[dict],
    query: str,
    output_text: str = "",
    sources: str = "",
    domain: str = "rag",
    num_simulations: int = 20,
    progress_callback: Optional[Callable] = None,
    registry=None,
    llm_fn_async: Optional[Callable] = None,
    seed: Optional[int] = None,
    max_concurrent_sims: int = 5,
) -> list[SimulationResult]:
    """
    Async version of run_monte_carlo.

    Uses asyncio.Semaphore to limit concurrency and asyncio.gather to run
    simulations in parallel.
    """
    import numpy as np

    from counterfact.classifiers import ClassifierRegistry, get_default_registry

    reg = registry or get_default_registry()
    results = []
    perturbations = generate_perturbations(trace)

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    # ── Step 1: Baseline runs (Sync, few runs, fast)
    baseline_runs = max(3, num_simulations // 10)
    sim_id = 0

    for i in range(baseline_runs):
        clf_results = reg.run_all(query, output_text, sources, domain)
        quality = ClassifierRegistry.aggregate_quality(clf_results)
        results.append(SimulationResult(
            simulation_id=sim_id,
            perturbation=None,
            quality_score=quality,
            classifier_results=clf_results,
            perturbed_output=output_text[:200],
            is_baseline=True,
        ))
        sim_id += 1
        if progress_callback:
            progress_callback(sim_id, num_simulations, "baseline")

    # ── Step 2: Ablation simulations (Async) ─────────────────────────────
    # Mirror the sync run_monte_carlo: each ablation runs the N-1 coalition
    # sequentially. We wrap llm_fn_async as a sync function so run_coalition
    # (which calls _simulate_agent_step synchronously) can use it.
    # The async concurrency is at the coalition level (sem), not within it.
    all_agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry["node"] != "output"
    ))
    remaining = num_simulations - baseline_runs
    sims_per_pert = max(1, remaining // len(perturbations)) if perturbations else 0

    tasks = []
    sem = asyncio.Semaphore(max_concurrent_sims)

    # Shared agent_step_cache across all async ablation tasks.
    # Access is not truly concurrent (sem limits to max_concurrent_sims),
    # but since tasks are IO-bound and asyncio is single-threaded, this is safe.
    agent_step_cache: dict = {}

    def _sync_llm(prompt: str, temperature: float) -> str:
        """Synchronous wrapper around the async LLM for use in run_coalition."""
        if llm_fn_async is None:
            return ""
        # In an async context the loop is always running, so we cannot use
        # run_until_complete(). The async simulations are handled directly
        # in _run_sim via await instead.
        return ""

    async def _run_sim(pert, s_id):
        async with sem:
            coalition = frozenset(all_agents) - {pert.agent}
            if llm_fn_async:
                # Build the coalition output step-by-step.
                # We run each step as an awaited async call to avoid blocking.
                def _get_agent_output_from_trace(t, a):
                    for entry in t:
                        if entry.get("node") == a:
                            return entry.get("output", {})
                    return {}
                agents_ordered = list(dict.fromkeys(
                    entry["node"] for entry in trace if entry["node"] != "output"
                ))
                current_state = query
                final_output = ""
                for agent in agents_ordered:
                    if agent not in coalition:
                        current_state = ""
                    else:
                        cache_key = (agent, current_state)
                        if cache_key not in agent_step_cache:
                            original_output = _get_agent_output_from_trace(trace, agent)
                            # Build the prompt directly (same logic as _simulate_agent_step)
                            # then await the async LLM call.
                            # We need the prompt — re-use the function's logic by
                            # calling it with a sync shim and letting it build/return.
                            if not current_state:
                                prompt = (
                                    f'You are simulating agent "{agent}" in a multi-agent pipeline.\n\n'
                                    f"CONTEXT — original query: {query}\n"
                                    f"CONTEXT — sources: {sources[:400]}\n\n"
                                    f'In this counterfactual scenario, the agent that feeds input to "{agent}" was REMOVED.\n'
                                    f'"{agent}" therefore receives NO input (empty input).\n\n'
                                    f'For reference, "{agent}"\'s original output was:\n{original_output[:300]}\n\n'
                                    f'Simulate what "{agent}" produces when it receives no input.\n'
                                    f"Keep it under 150 words."
                                )
                            else:
                                prompt = (
                                    f'You are simulating agent "{agent}" in a multi-agent pipeline.\n\n'
                                    f"CONTEXT — original query: {query}\n"
                                    f"CONTEXT — sources: {sources[:400]}\n\n"
                                    f'In this counterfactual scenario, "{agent}" receives this input:\n'
                                    f"INPUT: {current_state[:500]}\n\n"
                                    f'For reference, "{agent}"\'s original output was:\n{original_output[:300]}\n\n'
                                    f'Given this specific new input, simulate what "{agent}" produces.\n'
                                    f"Keep it under 200 words."
                                )
                            result = await llm_fn_async(prompt, 0.4)
                            agent_step_cache[cache_key] = result if len(result) > 20 else ""
                        current_state = agent_step_cache[cache_key]
                        final_output = current_state
                perturbed = final_output
            else:
                perturbed = ""

            clf_results = reg.run_all(query, perturbed, sources, domain)
            quality = ClassifierRegistry.aggregate_quality(clf_results)

            if progress_callback:
                progress_callback(s_id, num_simulations, f"ablating {pert.agent}")

            return SimulationResult(
                simulation_id=s_id,
                perturbation=pert,
                quality_score=quality,
                classifier_results=clf_results,
                perturbed_output=perturbed[:200],
                is_baseline=False,
            )

    for pert in perturbations:
        for _ in range(sims_per_pert):
            tasks.append(_run_sim(pert, sim_id))
            sim_id += 1

    if tasks:
        pert_results = await asyncio.gather(*tasks)
        results.extend(pert_results)

    # Sort results by ID to maintain deterministic order
    results.sort(key=lambda r: r.simulation_id)
    return results


async def run_full_diagnostic_async(
    trace: list[dict],
    query: str,
    output_text: str = "",
    sources: str = "",
    domain: str = "rag",
    num_simulations: int = 30,
    quality_gate: float = 0.8,
    progress_callback: Optional[Callable] = None,
    registry=None,
    llm_fn_async: Optional[Callable] = None,
    run_evals: bool = True,
    seed: Optional[int] = None,
    max_concurrent_sims: int = 5,
) -> DiagnosticReport:
    """Async orchestrator. Steps 2-6 are mostly CPU-bound so they run synchronously after sims."""
    import numpy as np

    # Run async MC
    sim_results = await run_monte_carlo_async(
        trace, query, output_text, sources, domain,
        num_simulations, progress_callback, registry, llm_fn_async, seed, max_concurrent_sims
    )

    baseline_results = [r for r in sim_results if r.is_baseline]
    baseline_scores = [r.quality_score for r in baseline_results]
    baseline_quality = float(np.mean(baseline_scores)) if baseline_scores else 0.5

    agents = list(dict.fromkeys(
        entry["node"] for entry in trace if entry.get("node") != "output"
    ))
    zero_attribution = {agent: 0.0 for agent in agents}

    if baseline_quality >= quality_gate:
        classification = _make_no_failure_classification(baseline_quality, baseline_results)
        summary = _build_summary(sim_results, baseline_scores, baseline_quality, agents, "quality_gate", True, baseline_results)

        report = DiagnosticReport(
            query=query, domain=domain, baseline_quality=baseline_quality,
            shapley_values=zero_attribution, per_classifier_shapley={},
            classification=classification, recommendations=[], evaluations=[],
            num_simulations=num_simulations, simulation_results=sim_results,
            simulation_results_summary=summary, eval_suite=None,
            attribution_method="quality_gate", seed=seed,
        )
        report._trace = trace
        return report

    baseline_quality_ci = compute_bootstrap_ci(baseline_scores) if baseline_scores else None
    loo = compute_loo_attribution(sim_results, trace)
    attribution_method = "loo"
    cis: dict[str, ConfidenceInterval] = {}

    if is_loo_inconclusive(loo):
        # compute_shapley_values calls run_coalition → _simulate_agent_step,
        # which is synchronous. In the async path the event loop is already
        # running so we cannot use run_until_complete(). We pass llm_fn=None
        # which uses the LOO approximation from existing ablation results.
        # The method tag is "loo_approx" to make this explicit.
        attribution, cis, per_clf_from_shapley = compute_shapley_values(
            sim_results, trace,
            llm_fn=None,   # LOO approximation — see comment above
            query=query,
            output_text=output_text,
            sources=sources,
            domain=domain,
            registry=registry,
        )
        attribution_method = "loo_approx"  # NOT true Shapley in async path
        per_clf_shapley = compute_per_classifier_loo(sim_results, trace)
    else:
        attribution = loo
        per_clf_shapley = compute_per_classifier_loo(sim_results, trace)

    classification = classify_failure(attribution, sim_results, trace, per_clf_shapley)

    empirical_fixes = extract_empirical_fixes(sim_results, baseline_quality, trace)
    gap_fixes = []
    if classification.failure_type == "architectural_gap":
        gap_fixes = detect_coverage_gaps(per_clf_shapley, classification.failing_classifiers, sim_results, trace)

    all_candidates = empirical_fixes + gap_fixes
    recommendations = rank_recommendations(all_candidates)

    summary = _build_summary(
        sim_results, baseline_scores, baseline_quality,
        list(attribution.keys()), attribution_method, False, baseline_results,
    )

    report = DiagnosticReport(
        query=query, domain=domain, baseline_quality=baseline_quality,
        shapley_values=attribution, per_classifier_shapley=per_clf_shapley,
        classification=classification, recommendations=recommendations,
        evaluations=[], num_simulations=num_simulations,
        simulation_results=sim_results, simulation_results_summary=summary,
        eval_suite=None, attribution_method=attribution_method,
        shapley_cis=cis, baseline_quality_ci=baseline_quality_ci, seed=seed,
    )
    report._trace = trace
    return report
