"""Render a shareable HTML report for the agents-as-tools case study.

    PYTHONPATH=examples python -m agents_as_tools_skill.run_casestudy   # writes the JSON
    PYTHONPATH=examples python -m agents_as_tools_skill.make_report     # writes the HTML

Single self-contained file with inline CSS; numbers are read from the JSON.
"""

from __future__ import annotations

import html
import json
import os

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
JSON_PATH = os.path.join(REPORTS, "agents_as_tools_casestudy.json")
HTML_PATH = os.path.join(REPORTS, "agents_as_tools_case_study.html")


def _esc(text) -> str:
    return html.escape(str(text))


def _shapley_rows(shapley: dict) -> str:
    rows = []
    for node, val in sorted(shapley.items(), key=lambda kv: kv[1]):
        cls = "neg" if val < 0 else ("pos" if val > 0 else "")
        rows.append(f"<tr><td><code>{_esc(node)}</code></td><td class='num {cls}'>{val:+.3f}</td></tr>")
    return "".join(rows)


def render(report: dict) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Debugging an agents-as-tools system with counterfact</title>
<style>
  body {{ font: 15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c1e21;
         max-width: 880px; margin: 0 auto; padding: 48px 22px 80px; background:#fafafa; }}
  h1 {{ font-size: 27px; margin-bottom:6px; letter-spacing:-.01em; }}
  h2 {{ font-size: 20px; margin-top: 38px; border-bottom:2px solid #ececec; padding-bottom:7px; }}
  .sub {{ color:#666; margin-top:0; }}
  table {{ border-collapse: collapse; width:100%; margin:16px 0; background:#fff;
           border:1px solid #ececec; border-radius:8px; overflow:hidden; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid #f0f0f0; }}
  th {{ background:#f4f5f7; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#555; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; font-family: ui-monospace,Menlo,monospace; }}
  .neg {{ color:#b3261e; }} .pos {{ color:#2e7d4f; }}
  code,.mono {{ font-family: ui-monospace,Menlo,monospace; font-size:13px; }}
  code {{ background:#f1f1f4; padding:1px 5px; border-radius:4px; }}
  .flow {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:16px;
           font-family: ui-monospace,Menlo,monospace; font-size:14px; color:#333; }}
  blockquote {{ border-left:4px solid #d64545; margin:12px 0; padding:10px 16px; color:#444;
                background:#fff; border-radius:0 6px 6px 0; }}
  blockquote.good {{ border-left-color:#2e7d4f; }}
  .footer {{ margin-top:46px; padding-top:18px; border-top:1px solid #ececec; color:#888; font-size:13px; }}
</style></head><body>

<h1>Debugging an agents-as-tools system</h1>
<p class="sub">Attributing a wrong answer to the specific sub-agent tool that caused it, in an
OpenAI Agents SDK system where the top agent calls sub-agents as tools.</p>

<h2>The system</h2>
<p>An assistant answers a question by calling two sub-agent tools and a composer. counterfact
runs each tool-agent as a discrete, ablatable node rather than relying on the model's own
tool-calling loop.</p>
<div class="flow">assistant &rarr; fact_lookup (tool) &rarr; unit_converter (tool) &rarr; composer</div>

<h2>The symptom</h2>
<p>Asked "{_esc(report["query"])}", the system answers:</p>
<blockquote>{_esc(report["baseline_answer"])}</blockquote>
<p>The correct answer is {_esc(report["gold"])}. Every tool ran without error, so the trace
alone does not say which one is wrong.</p>

<h2>What counterfact found</h2>
<p>Ablating each tool-agent and re-running gives the following attribution (negative means the
tool's presence makes the answer worse):</p>
<table><tr><th>tool</th><th>Shapley</th></tr>
{_shapley_rows(report["shapley_values"])}
</table>
<p><code>fact_lookup</code> is positive: it supplies the correct figures, so removing it hurts.
The negative scores fall on <code>{_esc(report["culprit"])}</code> and the composer that
surfaces its output. The debugging loop fixes the most implicated editable tool,
<code>{_esc(report["culprit"])}</code> (its instruction used a wrong feet-to-meters factor),
and the answer becomes correct:</p>
<blockquote class="good">{_esc(report["fixed_answer"])}</blockquote>

<h2>Takeaway</h2>
<p>The same counterfact machinery handles all three OpenAI Agents SDK patterns: sequential,
orchestrator-with-handoffs, and agents-as-tools. A top agent calling sub-agents as tools is
just another wiring of ablatable nodes. For a tool that is retrieval-like, prefer graded
degradation over ablation.</p>

<div class="footer">
Generated from <code>reports/agents_as_tools_casestudy.json</code>. Reproduce offline with
<code>PYTHONPATH=examples python -m agents_as_tools_skill.run_casestudy</code>.
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
