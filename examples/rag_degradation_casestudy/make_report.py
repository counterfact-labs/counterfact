"""Render a shareable HTML report for the RAG degradation case study.

    PYTHONPATH=examples python -m rag_degradation_casestudy.run_casestudy   # writes the JSON
    PYTHONPATH=examples python -m rag_degradation_casestudy.make_report     # writes the HTML

Single self-contained file with inline CSS; numbers are read from the JSON.
"""

from __future__ import annotations

import html
import json
import os

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
JSON_PATH = os.path.join(REPORTS, "rag_degradation_casestudy.json")
HTML_PATH = os.path.join(REPORTS, "rag_degradation_case_study.html")

_NODES = ["retriever", "reranker", "synthesizer"]


def _esc(text) -> str:
    return html.escape(str(text))


def _rows(abl: dict, auto: dict, strategies: dict) -> str:
    rows = []
    for n in _NODES:
        rows.append(
            f"<tr><td><code>{_esc(n)}</code></td>"
            f"<td class='num'>{abl.get(n, 0.0):+.3f}</td>"
            f"<td class='num'>{auto.get(n, 0.0):+.3f}</td>"
            f"<td>{_esc(strategies.get(n, ''))}</td></tr>"
        )
    return "".join(rows)


def render(report: dict) -> str:
    n = report["n_cases"]
    k = report["top_k"]
    abl = report["pure_ablation"]
    auto = report["auto_degrade"]
    strategies = report["removal_strategies"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Why diagnose degrades a retriever instead of ablating it</title>
<style>
  body {{ font: 15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c1e21;
         max-width: 880px; margin: 0 auto; padding: 48px 22px 80px; background:#fafafa; }}
  h1 {{ font-size: 26px; margin-bottom:6px; letter-spacing:-.01em; }}
  h2 {{ font-size: 20px; margin-top: 38px; border-bottom:2px solid #ececec; padding-bottom:7px; }}
  .sub {{ color:#666; margin-top:0; }}
  table {{ border-collapse: collapse; width:100%; margin:16px 0; background:#fff;
           border:1px solid #ececec; border-radius:8px; overflow:hidden; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid #f0f0f0; }}
  th {{ background:#f4f5f7; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#555; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; font-family: ui-monospace,Menlo,monospace; }}
  code,.mono {{ font-family: ui-monospace,Menlo,monospace; font-size:13px; }}
  code {{ background:#f1f1f4; padding:1px 5px; border-radius:4px; }}
  .flow {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:16px;
           font-family: ui-monospace,Menlo,monospace; font-size:14px; color:#333; }}
  .kpi {{ display:inline-block; background:#fff; border:1px solid #e3e3e3; border-radius:8px;
          padding:12px 16px; margin:6px 10px 6px 0; }}
  .kpi b {{ font-size:20px; display:block; }} .kpi span {{ color:#666; font-size:13px; }}
  .bad {{ color:#b3261e; }} .good {{ color:#2e7d4f; }}
  .footer {{ margin-top:46px; padding-top:18px; border-top:1px solid #ececec; color:#888; font-size:13px; }}
</style></head><body>

<h1>Why diagnose degrades a retriever instead of ablating it</h1>
<p class="sub">How counterfact removes a node during attribution, and why structural modules are
severely degraded rather than ablated.</p>

<h2>The system</h2>
<p>A three-stage retrieval pipeline. The synthesizer answers from the top {k} retrieved passages,
so it genuinely depends on retrieval: with no context there is nothing to answer from.</p>
<div class="flow">retriever &rarr; reranker &rarr; synthesizer&nbsp;&nbsp;(reads top {k})</div>

<h2>Ablating everything breaks the run</h2>
<p>To attribute a failure, counterfact removes each node from coalitions and measures the quality
change. If every removal is a plain ablation (a no-op), removing the retriever leaves the
synthesizer with no passages and the run structurally fails. Across {n} cases:</p>
<div>
  <span class="kpi"><b class="bad">{abl["structural_failures"]}/{abl["runs"]}</b><span>coalition runs failed (pure ablation)</span></span>
  <span class="kpi"><b class="good">{auto["structural_failures"]}/{auto["runs"]}</b><span>coalition runs failed (auto)</span></span>
</div>

<h2>Pure ablation vs auto-degradation</h2>
<table><tr><th>node</th><th>pure-ablation Shapley</th><th>auto Shapley</th><th>auto strategy</th></tr>
{_rows(abl["shapley"], auto["shapley"], strategies)}
</table>
<p>Under pure ablation the retriever and synthesizer dominate (removing either breaks or empties
the answer) and the reranker reads as dead weight (+0.00) because removing it is a harmless
pass-through. counterfact instead severely degrades the retriever and reranker: each still runs
and returns a non-empty doc list, but its content is replaced with low-relevance placeholders.
Every run stays live, and the reranker that ablation called irrelevant now shows the
contribution it actually makes.</p>

<h2>Takeaway</h2>
<p>Removing a structural module by no-op answers "is it load-bearing" and, when the rest of the
pipeline depends on it, just breaks the run. Severely degrading it (destroy the content, keep the
shape) measures how much its output quality is worth, with no structural failure. counterfact
makes that choice automatically, by inferred module type.</p>

<div class="footer">
Generated from <code>reports/rag_degradation_casestudy.json</code>. Reproduce offline with
<code>PYTHONPATH=examples python -m rag_degradation_casestudy.run_casestudy</code>.
</div>

</body></html>"""


def main():
    with open(JSON_PATH) as f:
        report = json.load(f)
    out = render(report)
    os.makedirs(REPORTS, exist_ok=True)
    with open(HTML_PATH, "w") as f:
        f.write(out)
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
