"""Reproducible FinanceBench case study: diagnose -> fix -> re-diagnose.

Runs the honest, ADAPTIVE debugging loop the skill encodes:
  1. Diagnose the broken 8-agent pipeline across all queries (aggregate Shapley).
  2. Fix the single most-harmful still-broken agent (most-negative avg Shapley
     among the four editable agents).
  3. Re-diagnose, re-prioritize, repeat until no editable agent is harmful.
Then recompute the LLM-as-debugger baseline LIVE (no hardcoded numbers) and
compare its agent ranking to the causal Shapley ranking.

Deterministic-as-possible: seed=42, temperature=0.0, process-level call cache.
Writes reports/financebench_skill_casestudy.{json,md}.

Config (env overrides for cheaper runs):
    FB_SIMS=30      simulations per query per diagnosis
    FB_QUERIES=5    number of queries (1..5)
    FB_MAXFIX=4     max fix rounds

Run:
    export ANTHROPIC_API_KEY=...
    PYTHONPATH=examples python -m financebench_skill.run_casestudy
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import time

from financebench_skill import data, prompts
from financebench_skill.llm import SONNET, cache_stats, call
from financebench_skill.pipeline import build
from financebench_skill.progress import Progress
from financebench_skill.quality import build_registry

FIXABLE = ["table_extractor", "context_enricher", "fact_checker", "tone_editor"]
BROKEN0 = dict(prompts.INSTRUCTIONS)  # snapshot of the broken instructions at import
SIMS = int(os.environ.get("FB_SIMS", "30"))
NQ = int(os.environ.get("FB_QUERIES", "5"))
MAXFIX = int(os.environ.get("FB_MAXFIX", "4"))
SEED = 42


def _queries():
    return data.QUERIES[:NQ]


def _exact(query: str, output: str) -> bool:
    gt = data.GROUND_TRUTH.get(query, "")
    m = re.search(r"[\d,]+", gt)
    num = m.group() if m else ""
    return bool(num) and (num in output or num.replace(",", "") in output)


def _diagnose_one(q: dict, cb=None):
    """Diagnose one query; return (report, baseline_output, trace)."""
    g = build()
    init = data.make_input_state(q["query"])
    out = g.invoke(init)
    trace = g.get_trace()
    report = g.diagnose(input_state=init, domain="financebench",
                        num_simulations=SIMS, registry=build_registry(), seed=SEED,
                        progress_callback=cb)
    return q, report, out.get("analysis", ""), trace


def diagnose_all(label: str = "diagnose"):
    """Diagnose every query in parallel; return aggregate Shapley + quality + exacts + traces."""
    prog = Progress(total=len(_queries()) * SIMS, label=label)
    cb = lambda *a, **k: prog.tick()  # noqa: E731 — count each simulation
    results = []
    with ThreadPoolExecutor(max_workers=min(5, len(_queries()))) as pool:
        futs = [pool.submit(_diagnose_one, q, cb) for q in _queries()]
        for f in as_completed(futs):
            results.append(f.result())
    st = cache_stats()
    prog.done(status=f"cache {st['hits']} hits / {st['misses']} misses")
    # preserve query order
    order = {q["query"]: i for i, q in enumerate(_queries())}
    results.sort(key=lambda r: order[r[0]["query"]])

    agg: dict[str, list[float]] = {}
    per_clf: dict[str, dict[str, list[float]]] = {}
    quals, exacts, traces = [], 0, []
    for q, report, output, trace in results:
        for a, v in report.shapley_values.items():
            agg.setdefault(a, []).append(v)
        for clf, avs in (report.per_classifier_shapley or {}).items():
            for a, v in avs.items():
                per_clf.setdefault(clf, {}).setdefault(a, []).append(v)
        quals.append(report.baseline_quality)
        exacts += int(_exact(q["query"], output))
        traces.append({"short": q["short"], "query": q["query"],
                       "ground_truth": q["ground_truth"], "output": output, "trace": trace})
    avg_sv = {a: sum(v) / len(v) for a, v in agg.items()}
    avg_clf = {c: {a: sum(v) / len(v) for a, v in d.items()} for c, d in per_clf.items()}
    return {
        "avg_shapley": avg_sv,
        "per_classifier": avg_clf,
        "avg_quality": sum(quals) / len(quals),
        "exact": exacts,
        "n_queries": len(quals),
        "traces": traces,
    }


def worst_fixable(avg_sv: dict, already_fixed: set) -> str | None:
    """Most-negative still-broken editable agent (the next fix target)."""
    cands = [(a, avg_sv.get(a, 0.0)) for a in FIXABLE if a not in already_fixed]
    cands = [(a, v) for a, v in cands if v < 0]
    if not cands:
        return None
    return min(cands, key=lambda kv: kv[1])[0]


def llm_baseline(traces: list[dict]) -> dict:
    """Recompute the LLM-as-debugger baseline LIVE: hand Claude the traces and
    ask it to rank agents by how much each is degrading quality. No hardcoding."""
    blocks = []
    for t in traces:
        steps = "\n".join(f"    [{e['node']}] -> {str(e.get('output',''))[:300]}" for e in t["trace"])
        blocks.append(f"QUERY ({t['short']}): {t['query']}\nCORRECT: {t['ground_truth']}\n"
                      f"FINAL OUTPUT: {t['output'][:400]}\nAGENT TRACE:\n{steps}")
    agents = ", ".join(a for a, _ in [(n, None) for n in
              ["query_parser","doc_retriever","table_extractor","context_enricher",
               "synthesizer","fact_checker","tone_editor","output_formatter"]])
    prompt = f"""You are debugging an 8-agent financial QA pipeline. Agents: {agents}.
The outputs are wrong (rounded figures, fabricated peer comparisons). Below are
the full execution traces for several queries.

{chr(10).join(blocks)}

Rank the agents from MOST to LEAST responsible for the wrong outputs — i.e. which
agents should be fixed first. Respond with JSON only:
{{"ranking": ["agent1", "agent2", ...], "reasoning": "brief"}}"""
    try:
        r = call(prompt, model=SONNET, max_tokens=600)
        m = re.search(r"\{.*\}", r, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        return {"ranking": [], "reasoning": f"error: {e}"}
    return {"ranking": [], "reasoning": "parse error"}


def fmt_shapley(avg_sv: dict) -> str:
    lines = []
    for a, v in sorted(avg_sv.items(), key=lambda kv: kv[1]):
        bar = "█" * int(abs(v) * 20)
        lines.append(f"  {a:18s} {v:+.3f}  {bar}")
    return "\n".join(lines)


def main():
    try:
        call("Say OK.", max_tokens=5)
    except SystemExit:
        raise
    print(f"FinanceBench case study | sims={SIMS} queries={NQ} maxfix={MAXFIX}", file=sys.stderr)

    prompts.INSTRUCTIONS.update(BROKEN0)  # ensure we start from the broken state

    steps = []
    fixed: set = set()

    # Step 0: broken baseline
    t_step = time.monotonic()
    d0 = diagnose_all(label="step 0 (broken)")
    step_secs = time.monotonic() - t_step
    print(f"  step 0 took {step_secs:.0f}s; up to {MAXFIX} fix steps may follow "
          f"(~{step_secs * MAXFIX:.0f}s worst case, less with cache).", file=sys.stderr)
    llm = llm_baseline(d0["traces"])
    shapley_rank = [a for a, _ in sorted(d0["avg_shapley"].items(), key=lambda kv: kv[1])]
    steps.append({"step": 0, "fix": None, "avg_quality": d0["avg_quality"],
                  "exact": d0["exact"], "avg_shapley": d0["avg_shapley"],
                  "per_classifier": d0["per_classifier"]})
    print(f"  step 0: quality={d0['avg_quality']:.3f} exact={d0['exact']}/{d0['n_queries']}", file=sys.stderr)

    cur = d0
    for step in range(1, MAXFIX + 1):
        target = worst_fixable(cur["avg_shapley"], fixed)
        if target is None:
            print("  no remaining harmful editable agent — converged.", file=sys.stderr)
            break
        prompts.INSTRUCTIONS[target] = prompts.FIXED[target]
        fixed.add(target)
        cur = diagnose_all(label=f"step {step} (fixed {target})")
        steps.append({"step": step, "fix": target, "avg_quality": cur["avg_quality"],
                      "exact": cur["exact"], "avg_shapley": cur["avg_shapley"],
                      "per_classifier": cur["per_classifier"]})
        print(f"  step {step}: fixed {target} -> quality={cur['avg_quality']:.3f} "
              f"exact={cur['exact']}/{cur['n_queries']}", file=sys.stderr)

    # ── write reports ──
    rdir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    rdir = os.path.abspath(rdir)
    os.makedirs(rdir, exist_ok=True)
    result = {"config": {"sims": SIMS, "queries": NQ, "seed": SEED},
              "steps": steps,
              "llm_baseline": {"ranking": llm.get("ranking", []), "reasoning": llm.get("reasoning", "")},
              "shapley_ranking_step0": shapley_rank}
    with open(os.path.join(rdir, "financebench_skill_casestudy.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    md = ["# FinanceBench Case Study — Skill-Driven Reproduction",
          f"\n_Config: {NQ} queries, {SIMS} simulations/query, seed {SEED}, "
          f"Sonnet (synthesizer) + Haiku (other agents)._\n",
          "## Iterative fix arc\n",
          "| Step | Fix applied | Avg quality | Exact answers |",
          "|---|---|---|---|"]
    for s in steps:
        md.append(f"| {s['step']} | {s['fix'] or '— (broken baseline)'} | "
                  f"{s['avg_quality']:.3f} | {s['exact']}/{NQ} |")
    md.append("\n## Step 0 aggregate Shapley (most-harmful first)\n```")
    md.append(fmt_shapley(steps[0]["avg_shapley"]))
    md.append("```\n\n### Per-classifier worst agent (step 0)")
    for clf, avs in steps[0]["per_classifier"].items():
        worst = min(avs.items(), key=lambda kv: kv[1])
        md.append(f"- **{clf}**: {worst[0]} ({worst[1]:+.3f})")
    md.append("\n## Baseline: can an LLM diagnose this from traces?\n")
    md.append(f"**LLM ranking (most→least responsible):** {', '.join(llm.get('ranking', [])) or 'n/a'}\n")
    md.append(f"**Causal Shapley ranking (most-harmful→least, step 0):** {', '.join(shapley_rank)}\n")
    md.append(f"\n_LLM reasoning:_ {llm.get('reasoning','')}\n")
    with open(os.path.join(rdir, "financebench_skill_casestudy.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {rdir}/financebench_skill_casestudy.(json|md)", file=sys.stderr)
    print(json.dumps({"final_quality": steps[-1]["avg_quality"], "final_exact": steps[-1]["exact"],
                      "steps": len(steps) - 1, "llm_top": llm.get("ranking", [])[:3],
                      "shapley_top_harmful": shapley_rank[:3]}, indent=2))


if __name__ == "__main__":
    main()
