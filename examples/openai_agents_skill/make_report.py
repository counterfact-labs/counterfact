"""Render a shareable, self-contained HTML report for the OpenAI Agents SDK
case study from the JSON produced by ``run_casestudy``.

    PYTHONPATH=examples python -m openai_agents_skill.run_casestudy   # writes the JSON
    PYTHONPATH=examples python -m openai_agents_skill.make_report     # writes the HTML

The HTML is a single file with inline CSS (no external assets), suitable for
sending to a customer. Numbers are read from the generated JSON; the before/after
customer replies are reproduced deterministically from the system itself.
"""

from __future__ import annotations

import html
import json
import os

from .scorer import refund_amount_scorer
from .system import FIXED, build_system

HERE = os.path.dirname(__file__)
REPORTS = os.path.normpath(os.path.join(HERE, "..", "..", "reports"))
JSON_PATH = os.path.join(REPORTS, "openai_agents_casestudy.json")
HTML_PATH = os.path.join(REPORTS, "openai_agents_case_study.html")

_SAMPLE = {
    "input": "Order #1042 was charged $250.00 but the customer cancelled within the window "
    "and is requesting a full refund.",
    "expected": "$250.00",
}


def _sample_replies():
    """Deterministically reproduce the before/after customer reply for one ticket."""
    state = {"input": _SAMPLE["input"], "expected": _SAMPLE["expected"]}
    before = build_system().invoke({**state}).get("final_output", "")
    after = build_system(FIXED).invoke({**state}).get("final_output", "")
    return before, after


def _esc(text) -> str:
    return html.escape(str(text))


def _shapley_table(shapley: dict) -> str:
    rows = sorted(shapley.items(), key=lambda kv: kv[1])
    span = max((abs(v) for v in shapley.values()), default=1.0) or 1.0
    out = ["<table><tr><th>Agent</th><th>Avg Shapley</th>"
           "<th>impact &nbsp;(red = hurts quality)</th></tr>"]
    for node, val in rows:
        width = int(round(abs(val) / span * 150))
        cls = "neg" if val < 0 else "pos"
        color = "#d64545" if val < 0 else "#2e9e5b"
        align = "right;float:right" if val < 0 else "left;float:left"
        culprit = " class='hl'" if val < 0 else ""
        bar = (f"<div class='barwrap'><div class='bar' style='width:{width}px;"
               f"background:{color};margin-{'left' if val < 0 else 'right'}:auto;{align};'></div></div>")
        out.append(f"<tr{culprit}><td><code>{_esc(node)}</code></td>"
                   f"<td class='num {cls}'>{val:+.3f}</td><td>{bar}</td></tr>")
    out.append("</table>")
    return "".join(out)


def render(report: dict, before: str, after: str) -> str:
    base = report["baseline_pass_rate"]
    final = report["final_pass_rate"]
    n = report["n_cases"]
    fixes = report["fixes_applied"]
    culprit = fixes[0] if fixes else "—"
    shapley = report["diagnosis_timeline"][0]["shapley"] if report["diagnosis_timeline"] else {}

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Debugging an OpenAI Agents SDK pipeline with counterfact</title>
<style>
  body {{ font: 15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c1e21;
         max-width: 880px; margin: 0 auto; padding: 48px 22px 80px; background:#fafafa; }}
  h1 {{ font-size: 28px; margin-bottom:6px; letter-spacing:-.01em; }}
  h2 {{ font-size: 20px; margin-top: 40px; border-bottom:2px solid #ececec; padding-bottom:7px; }}
  .sub {{ color:#666; margin-top:0; font-size:15px; }}
  .lede {{ font-size:16px; color:#333; }}
  table {{ border-collapse: collapse; width:100%; margin:16px 0; background:#fff;
           border:1px solid #ececec; border-radius:8px; overflow:hidden; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid #f0f0f0; }}
  th {{ background:#f4f5f7; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#555; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; font-family: ui-monospace,Menlo,monospace; }}
  .neg {{ color:#b3261e; }} .pos {{ color:#2e7d4f; }}
  tr.hl {{ background:#fbeceb; }}
  .barwrap {{ width:300px; height:12px; position:relative; }}
  .bar {{ height:12px; border-radius:2px; }}
  code,.mono {{ font-family: ui-monospace,Menlo,monospace; font-size:13px;
               background:#f1f1f4; padding:1px 5px; border-radius:4px; }}
  pre {{ background:#1c1e21; color:#e6e6e6; padding:16px; border-radius:8px; overflow:auto; font-size:13px; line-height:1.5; }}
  pre code {{ background:none; padding:0; color:inherit; }}
  .flow {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:16px;
           font-family: ui-monospace,Menlo,monospace; font-size:13px; color:#333; white-space:pre; overflow:auto; }}
  blockquote {{ border-left:4px solid #d64545; margin:12px 0; padding:10px 16px; color:#444;
                background:#fff; border-radius:0 6px 6px 0; }}
  blockquote.good {{ border-left-color:#2e9e5b; }}
  .note {{ background:#fff8e6; border-left:4px solid #e6b800; padding:12px 16px; border-radius:0 6px 6px 0; }}
  .footer {{ margin-top:48px; padding-top:18px; border-top:1px solid #ececec; color:#888; font-size:13px; }}
</style></head><body>

<h1>Debugging an OpenAI Agents SDK support pipeline with counterfact</h1>
<p class="sub">Finding which agent in a multi-agent system is responsible for a wrong answer,
by re-running the pipeline with each agent removed.</p>

<h2>The system</h2>
<p>This is an orchestrator-with-handoffs pipeline built with the OpenAI Agents SDK. A triage
agent routes each support ticket to a specialist, and a compliance editor writes the final
customer reply.</p>
<div class="flow">triage ──handoff──▶ {{ billing | technical | account }} ──▶ compliance_editor ──▶ reply</div>
<p>The evaluation set is {n} billing-refund tickets, scored with a Braintrust scorer. The rule
is the one a billing team would use: the exact refund amount has to appear in the reply.</p>

<h2>The symptom</h2>
<p>On the baseline pipeline, {int(round(base*n))} of {n} replies are correct. The rest are polite
acknowledgements that leave out the dollar figure the customer asked about:</p>
<blockquote>{_esc(before)}</blockquote>

<h2>The suspects</h2>
<p>Reading the transcript, several agents could be at fault. The usual first guess, and the one
an LLM asked to debug from the trace tends to make, is the <code>billing</code> agent: it owns
the amount, and the amount is missing from its eventual reply. That guess turns out to be wrong.</p>

<h2>What counterfact found</h2>
<p>counterfact removes one agent at a time, re-runs the real pipeline, and measures how the
Braintrust score changes, averaged over the failing tickets. A negative value means the agent's
presence lowers the score.</p>
{_shapley_table(shapley)}
<p>The <code>billing</code> agent has a non-negative score: removing it never improves the
answer, because it produces the correct figure in the first place. The negative value belongs to
<code>{_esc(culprit)}</code>. Its instruction to make the reply policy-compliant rewrites the
specialist's answer into a generic template and drops the exact amount along the way.</p>

<h2>The fix</h2>
<p>Correcting the <code>{_esc(culprit)}</code> instruction so it keeps the specialist's figures
brings the evaluation set to {int(round(final*n))} of {n}. The same ticket now answers
correctly:</p>
<blockquote class="good">{_esc(after)}</blockquote>

<h2>Why the trace alone is not enough</h2>
<p>The final transcript contains no sign that <code>{_esc(culprit)}</code> had the figure and
discarded it. By the time you read the reply, the amount is simply gone, and the natural reading
is that the billing agent never produced it. Removing the editor and re-running the pipeline is
what shows the figure was present all along. That is the part counterfact adds on top of an
ordinary observability trace.</p>

<div class="footer">
Generated from <code>reports/openai_agents_casestudy.json</code>. Reproduce offline with
<code>PYTHONPATH=examples python -m openai_agents_skill.run_casestudy</code>.
</div>

</body></html>"""


def main():
    with open(JSON_PATH) as f:
        report = json.load(f)
    before, after = _sample_replies()
    out = render(report, before, after)
    os.makedirs(REPORTS, exist_ok=True)
    with open(HTML_PATH, "w") as f:
        f.write(out)
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
