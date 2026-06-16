"""Why target selection matters: measure quality for fixing DIFFERENT single agents.

Cheap (no 30-sim ablation): for each candidate single-agent fix, run the pipeline
once per query and score it. Shows that fixing the diagnosis's top pick (tone_editor)
recovers the pipeline, while fixing a plausible-but-wrong agent (e.g. context_enricher,
or the LLM baseline's #2 pick) barely helps — i.e. WHICH agent you fix matters, and the
causal diagnosis picks the right one.

Run:
    PYTHONPATH=examples python -m financebench_casestudy.compare_fix_order
"""
from __future__ import annotations

import json
import os
import re
import sys

from financebench_casestudy import data, prompts
from financebench_casestudy.llm import cache_stats
from financebench_casestudy.pipeline import build
from financebench_casestudy.progress import Progress
from financebench_casestudy.quality import build_registry

BROKEN0 = dict(prompts.INSTRUCTIONS)


def _exact(query, output):
    num = re.search(r"[\d,]+", data.GROUND_TRUTH.get(query, "")).group()
    return (num in output) or (num.replace(",", "") in output)


def measure(fix_set: set, prog=None) -> dict:
    """Set INSTRUCTIONS per fix_set, run each query once, score quality + exact."""
    prompts.INSTRUCTIONS.update(BROKEN0)
    for a in fix_set:
        prompts.INSTRUCTIONS[a] = prompts.FIXED[a]
    reg = build_registry()
    quals, exact = [], 0
    for q in data.QUERIES:
        out = build().invoke(data.make_input_state(q["query"])).get("analysis", "")
        results = reg.run_all(q["query"], out, "", "financebench")
        quals.append(reg.aggregate_quality(results))
        exact += int(_exact(q["query"], out))
        if prog:
            prog.tick()
    return {"avg_quality": sum(quals) / len(quals), "exact": exact, "n": len(quals)}


def main():
    configs = [
        ("none (broken baseline)", set()),
        ("context_enricher only", {"context_enricher"}),
        ("table_extractor only", {"table_extractor"}),
        ("tone_editor only (diagnosis's top pick)", {"tone_editor"}),
    ]
    prog = Progress(total=len(configs) * len(data.QUERIES), label="compare-fix-order")
    rows = []
    for label, fs in configs:
        m = measure(fs, prog=prog)
        rows.append({"fix": label, **m})
        print(f"  {label:42s} quality={m['avg_quality']:.3f} exact={m['exact']}/{m['n']}", file=sys.stderr)
    st = cache_stats()
    prog.done(status=f"cache {st['hits']} hits / {st['misses']} misses")

    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "financebench_fix_order.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
