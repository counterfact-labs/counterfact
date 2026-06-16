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

_CLASS_NOTE = {
    "structural": "needed to run, but quality is insensitive to partial degradation",
    "quality_driver": "answer quality falls as this module's output degrades",
    "harmful": "degrading or removing this module improves quality",
    "robust": "little effect on quality even at full degradation",
}


def _esc(text) -> str:
    return html.escape(str(text))


def _ablation_rows(ablation: dict) -> str:
    rows = []
    for node, val in sorted(ablation.items(), key=lambda kv: kv[1], reverse=True):
        rows.append(f"<tr><td><code>{_esc(node)}</code></td><td class='num'>{val:+.3f}</td></tr>")
    return "".join(rows)


def _degradation_rows(degradation: dict, mags) -> str:
    rows = []
    order = {"quality_driver": 0, "harmful": 1, "structural": 2, "robust": 3}
    for node, d in sorted(degradation.items(), key=lambda kv: order.get(kv[1].get("classification"), 9)):
        cls = d.get("classification", "")
        curve = d.get("mean_curve", [])
        cells = "  ".join(f"{m:g}:{q:.2f}" for m, q in zip(mags, curve))
        cls_cls = "neg" if cls in ("quality_driver", "harmful") else ""
        rows.append(
            f"<tr><td><code>{_esc(node)}</code></td>"
            f"<td class='{cls_cls}'>{_esc(cls)}</td>"
            f"<td class='mono small'>{_esc(cells)}</td>"
            f"<td class='small'>{_esc(_CLASS_NOTE.get(cls, ''))}</td></tr>"
        )
    return "".join(rows)


def render(report: dict) -> str:
    n = report["n_cases"]
    k = report["top_k"]
    mags = report["magnitudes"]
    ablation = report["ablation_shapley"]
    degradation = report["degradation"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finding the quality lever in a RAG pipeline with counterfact</title>
<style>
  body {{ font: 15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c1e21;
         max-width: 880px; margin: 0 auto; padding: 48px 22px 80px; background:#fafafa; }}
  h1 {{ font-size: 27px; margin-bottom:6px; letter-spacing:-.01em; }}
  h2 {{ font-size: 20px; margin-top: 38px; border-bottom:2px solid #ececec; padding-bottom:7px; }}
  .sub {{ color:#666; margin-top:0; }}
  table {{ border-collapse: collapse; width:100%; margin:16px 0; background:#fff;
           border:1px solid #ececec; border-radius:8px; overflow:hidden; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid #f0f0f0; vertical-align:top; }}
  th {{ background:#f4f5f7; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#555; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; font-family: ui-monospace,Menlo,monospace; }}
  .neg {{ color:#b3261e; }}
  .small {{ font-size:13px; color:#444; }}
  code,.mono {{ font-family: ui-monospace,Menlo,monospace; font-size:13px; }}
  code {{ background:#f1f1f4; padding:1px 5px; border-radius:4px; }}
  .flow {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:16px;
           font-family: ui-monospace,Menlo,monospace; font-size:14px; color:#333; }}
  .footer {{ margin-top:46px; padding-top:18px; border-top:1px solid #ececec; color:#888; font-size:13px; }}
</style></head><body>

<h1>Finding the quality lever in a RAG pipeline</h1>
<p class="sub">A worked example of why counterfact measures graded degradation, not just
full ablation, when diagnosing retrieval pipelines.</p>

<h2>The system</h2>
<p>A standard three-stage retrieval pipeline. The synthesizer reads only the top {k} passages,
so where the relevant passage is ranked decides whether the answer is correct.</p>
<div class="flow">retriever &rarr; reranker &rarr; synthesizer&nbsp;&nbsp;(reads top {k})</div>
<p>The pipeline passes its eval on all {n} questions, so the question is which module's quality
is the real lever, and where the pipeline is fragile.</p>

<h2>Ablation alone is misleading</h2>
<p>Replacing each node with a no-op and re-running gives this attribution:</p>
<table><tr><th>node</th><th>ablation Shapley</th></tr>
{_ablation_rows(ablation)}
</table>
<p>Read literally, this says the retriever and synthesizer matter and the reranker is dead
weight. The reranker scores zero because removing it is a harmless pass-through: the retriever
already returns the relevant passage near the top on these cases, so skipping the reranker
changes nothing. A team acting on this would stop tuning the reranker.</p>

<h2>Graded degradation gives the right answer</h2>
<p>Instead of removing each module, degrade its output across magnitudes ({", ".join(str(m) for m in mags)};
1.0 is full ablation) and watch how answer quality responds:</p>
<table><tr><th>node</th><th>classification</th><th>quality vs magnitude</th><th></th></tr>
{_degradation_rows(degradation, mags)}
</table>
<p>The reranker, which ablation rated a flat zero, is a quality driver: decaying its ranking
pushes the relevant passage out of the top-{k} window and answers fail. The retriever, which
ablation rated important, is merely structural; as long as it returns the relevant passage
somewhere, its ranking quality does not change the answer. The reranker is what decides whether
that passage lands in the synthesizer's context window.</p>

<h2>Takeaway</h2>
<p>Ablation can call a module both necessary and irrelevant while missing where answer quality
is actually won or lost. Graded degradation re-runs the real pipeline under controlled quality
loss and separates structural dependencies from the modules whose quality drives the result.</p>

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
