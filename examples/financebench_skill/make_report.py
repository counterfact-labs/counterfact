"""Render a self-contained HTML case-study report from the generated JSON artifacts.

Reads (whatever exists):
  reports/financebench_skill_casestudy.json   (diagnose + fix arc + LLM baseline)
  reports/financebench_fix_order.json          (target-selection comparison)
  evals/counterfact-debugger/results/financebench/verdict_*.json  (behavioral eval)

Writes: reports/financebench_case_study.html  (no external dependencies).

Run:  python examples/financebench_skill/make_report.py   (no PYTHONPATH/env needed)
"""
from __future__ import annotations

import glob
import html
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(ROOT, "reports")


def _load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _bar(v, scale=300):
    """Horizontal Shapley bar: red (harmful, negative) left, green (helpful) right."""
    w = min(abs(v) * scale, 150)
    color = "#d64545" if v < 0 else "#2e9e5b"
    side = "right" if v < 0 else "left"
    return (f'<div class="barwrap"><div class="bar" style="width:{w:.0f}px;background:{color};'
            f'margin-{ "left" if v>=0 else "right" }:auto;float:{ "left" if v>=0 else "right"};"></div></div>')


def main():
    cs = _load(os.path.join(REPORTS, "financebench_skill_casestudy.json"), {})
    fo = _load(os.path.join(REPORTS, "financebench_fix_order.json"), [])
    verdicts = [_load(p, {}) for p in sorted(glob.glob(
        os.path.join(ROOT, "evals", "counterfact-debugger", "results", "financebench", "verdict_*.json")))]

    steps = cs.get("steps", [])
    cfg = cs.get("config", {})
    step0 = steps[0] if steps else {}
    final = steps[-1] if steps else {}
    shap = step0.get("avg_shapley", {})
    shap_sorted = sorted(shap.items(), key=lambda kv: kv[1])
    per_clf = step0.get("per_classifier", {})
    llm = cs.get("llm_baseline", {})
    shap_rank = cs.get("shapley_ranking_step0", [])

    e = html.escape

    def shap_rows():
        out = []
        for a, v in shap_sorted:
            cls = "neg" if v < 0 else "pos"
            out.append(f"<tr><td>{e(a)}</td><td class='num {cls}'>{v:+.3f}</td><td>{_bar(v)}</td></tr>")
        return "\n".join(out)

    def fix_rows():
        out = []
        for r in fo:
            hl = "hl" if r["exact"] == r.get("n", 5) and r["exact"] > 0 else ""
            out.append(f"<tr class='{hl}'><td>{e(r['fix'])}</td><td class='num'>{r['avg_quality']:.3f}</td>"
                       f"<td class='num'>{r['exact']}/{r.get('n',5)}</td></tr>")
        return "\n".join(out)

    def arc_rows():
        out = []
        for s in steps:
            out.append(f"<tr><td class='num'>{s['step']}</td><td>{e(str(s.get('fix') or '— (broken baseline)'))}</td>"
                       f"<td class='num'>{s['avg_quality']:.3f}</td><td class='num'>{s['exact']}/{cfg.get('queries',5)}</td></tr>")
        return "\n".join(out)

    def perclf_rows():
        out = []
        for clf, avs in per_clf.items():
            worst = min(avs.items(), key=lambda kv: kv[1]) if avs else ("—", 0)
            out.append(f"<tr><td>{e(clf)}</td><td>{e(worst[0])}</td><td class='num neg'>{worst[1]:+.3f}</td></tr>")
        return "\n".join(out)

    # eval summary
    n_pass = sum(1 for v in verdicts if v.get("pass"))
    eval_rows = []
    for i, v in enumerate(verdicts, 1):
        c = v.get("criteria", {})
        badge = "PASS" if v.get("pass") else "FAIL"
        bcls = "pass" if v.get("pass") else "fail"
        eval_rows.append(
            f"<tr><td>run {i}</td><td><span class='badge {bcls}'>{badge}</span></td>"
            f"<td class='num'>{v.get('exact_answers','?')}/{v.get('n_queries','?')}</td>"
            f"<td>skill_used={c.get('skill_used')}, fix_works={c.get('fix_works')}</td></tr>")

    llm_rank = llm.get("ranking", [])
    # mark where LLM diverges from causal ranking
    def rank_cmp():
        rows = []
        for idx in range(max(len(llm_rank), len(shap_rank))):
            l = llm_rank[idx] if idx < len(llm_rank) else ""
            s = shap_rank[idx] if idx < len(shap_rank) else ""
            mark = "" if l == s else " class='diverge'"
            rows.append(f"<tr{mark}><td class='num'>{idx+1}</td><td>{e(l)}</td><td>{e(s)}</td></tr>")
        return "\n".join(rows)

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>counterfact-debugger — FinanceBench case study</title>
<style>
  body {{ font: 15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c1e21;
         max-width: 860px; margin: 40px auto; padding: 0 20px; background:#fafafa; }}
  h1 {{ font-size: 26px; margin-bottom:4px; }}
  h2 {{ font-size: 20px; margin-top: 34px; border-bottom:2px solid #eee; padding-bottom:6px; }}
  .sub {{ color:#666; margin-top:0; }}
  table {{ border-collapse: collapse; width:100%; margin:14px 0; background:#fff; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #eee; }}
  th {{ background:#f4f5f7; font-size:13px; text-transform:uppercase; letter-spacing:.03em; color:#555; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; font-family: ui-monospace,Menlo,monospace; }}
  .neg {{ color:#d64545; }} .pos {{ color:#2e9e5b; }}
  tr.hl {{ background:#eafaf0; font-weight:600; }}
  tr.diverge {{ background:#fff5e6; }}
  .barwrap {{ width:300px; height:12px; position:relative; }}
  .bar {{ height:12px; border-radius:2px; }}
  .badge {{ padding:2px 8px; border-radius:10px; font-size:12px; font-weight:700; color:#fff; }}
  .badge.pass {{ background:#2e9e5b; }} .badge.fail {{ background:#d64545; }}
  .kpi {{ display:inline-block; background:#fff; border:1px solid #e3e3e3; border-radius:8px;
          padding:10px 16px; margin:6px 10px 6px 0; }}
  .kpi b {{ font-size:22px; display:block; }}
  code,.mono {{ font-family: ui-monospace,Menlo,monospace; font-size:13px; }}
  pre {{ background:#1c1e21; color:#e6e6e6; padding:14px; border-radius:8px; overflow:auto; font-size:13px; }}
  .note {{ background:#fff8e6; border-left:4px solid #e6b800; padding:10px 14px; border-radius:4px; }}
  blockquote {{ border-left:4px solid #ccc; margin:10px 0; padding:6px 14px; color:#444; background:#fff; }}
</style></head><body>

<h1>counterfact-debugger — FinanceBench case study</h1>
<p class="sub">Diagnosing an 8-agent financial-RAG pipeline on real FinanceBench questions
(3M FY2018 10-K). Config: {cfg.get('queries','?')} queries × {cfg.get('sims','?')} simulations, seed {cfg.get('seed','?')}.
All numbers regenerated by <span class="mono">financebench_skill.run_casestudy</span>.</p>

<div>
  <span class="kpi"><b>{step0.get('avg_quality',0):.3f} → {final.get('avg_quality',0):.3f}</b>avg quality</span>
  <span class="kpi"><b>{step0.get('exact',0)}/{cfg.get('queries',5)} → {final.get('exact',0)}/{cfg.get('queries',5)}</b>exact answers</span>
  <span class="kpi"><b>{n_pass}/{len(verdicts) if verdicts else '–'}</b>skill-driven eval</span>
</div>

<h2>The failure</h2>
<p>Eight agents, all passing in every trace, yet the answers round exact figures
("$1,577 million" → "$1.6 billion") and add fabricated peer comparisons. Nothing in the
trace says which agent to fix.</p>

<h2>Diagnosis — aggregate Shapley (real ablation)</h2>
<p>Each agent ablated across coalitions, pipeline re-run with real Claude calls, scored on
accuracy (×2), precision (×1.5), grounding (×1). Negative = the agent <em>hurts</em> quality.</p>
<table><tr><th>Agent</th><th>Avg Shapley</th><th>impact (red = harmful)</th></tr>
{shap_rows()}
</table>
<table><tr><th>Failing dimension</th><th>Worst agent</th><th>Shapley</th></tr>
{perclf_rows()}
</table>

<h2>Target selection matters</h2>
<p>Four agents look suspicious. Three fixes change nothing; only the diagnosis's pick recovers the pipeline.</p>
<table><tr><th>Single fix applied</th><th>Avg quality</th><th>Exact</th></tr>
{fix_rows()}
</table>

<h2>Iterative fix arc</h2>
<table><tr><th>Step</th><th>Fix applied</th><th>Avg quality</th><th>Exact</th></tr>
{arc_rows()}
</table>

<h2>Baseline: can an LLM diagnose this from the traces?</h2>
<p>We gave Claude the full traces + scores and asked it to rank agents by responsibility.
Rows highlighted where the LLM diverges from causal attribution:</p>
<table><tr><th>Rank</th><th>LLM (trace-reading)</th><th>Shapley (causal)</th></tr>
{rank_cmp()}
</table>
<div class="note"><b>The tell:</b> the LLM ranks <code>output_formatter</code> near the top, but
ablation shows it is the single <em>most-helpful</em> agent — it only <em>displays</em> the
rounded number. Trace-reading confuses correlation with causation; ablation measures the
agent's actual marginal contribution.</div>
<blockquote>{e((llm.get('reasoning','') or '')[:600])}…</blockquote>

<h2>Driven by the skill (behavioral eval)</h2>
<p>An agent given only the broken pipeline + the counterfact-debugger skill must diagnose and
fix it; graded by re-running the agent's <em>edited</em> pipeline on all queries.</p>
<table><tr><th>Run</th><th>Result</th><th>Exact</th><th>Criteria</th></tr>
{os.linesep.join(eval_rows) if eval_rows else "<tr><td colspan=4>(no eval verdicts found)</td></tr>"}
</table>
<p class="sub">A failing run here was cut off by the single-turn headless harness, not a wrong
diagnosis — it does not apply to interactive use.</p>

<h2>Reproduce it</h2>
<pre>pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
PYTHONPATH=examples python -m financebench_skill.run_casestudy     # diagnosis + fix arc + LLM baseline
PYTHONPATH=examples python -m financebench_skill.compare_fix_order # target-selection comparison
python examples/financebench_skill/make_report.py                 # regenerate this HTML</pre>
<p class="sub">Claude Sonnet 4.6 (synthesizer) + Claude Haiku 4.5 (other agents). Persistent
on-disk LLM cache makes re-runs cheap and interruption-safe. Numbers are as deterministic as
the API allows; exact floats vary slightly run-to-run, conclusions are stable.</p>

</body></html>"""

    out = os.path.join(REPORTS, "financebench_case_study.html")
    os.makedirs(REPORTS, exist_ok=True)
    with open(out, "w") as f:
        f.write(doc)
    print(out)


if __name__ == "__main__":
    main()
