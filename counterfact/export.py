import json
from typing import TYPE_CHECKING, Optional
from counterfact.types import ConfidenceInterval

if TYPE_CHECKING:
    from counterfact.diagnostics import DiagnosticReport

def to_json(report: "DiagnosticReport", path: Optional[str] = None) -> str:
    """Export standard JSON representation of the diagnostic report."""
    data = report.to_dict()
    res = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(res)
    return res


def _format_ci(ci: Optional[ConfidenceInterval], percentage: bool = False) -> str:
    if not ci or ci.n_samples == 0:
        return "N/A"
    
    if percentage:
        return f"{ci.mean*100:+.1f}% [{ci.ci_low*100:+.1f}%, {ci.ci_high*100:+.1f}%]"
    return f"{ci.mean:+.3f} [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]"


def _ascii_bar(value: float, min_val: float, max_val: float, width: int = 20) -> str:
    if max_val == min_val:
        return "█" * width
    
    # Normalize to 0-1
    norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    norm = max(0.0, min(1.0, norm))
    
    filled = int(norm * width)
    return "█" * filled + "░" * (width - filled)


def to_markdown(report: "DiagnosticReport", path: Optional[str] = None) -> str:
    """Generate a human-readable markdown report."""
    lines = []
    lines.append("# Counterfact Diagnostic Report")
    lines.append(f"**Query**: `{report.query}`")
    lines.append(f"**Domain**: `{report.domain}`")
    
    bq = report.baseline_quality
    bq_ci_str = _format_ci(report.baseline_quality_ci) if getattr(report, "baseline_quality_ci", None) else f"{bq:.3f}"
    lines.append(f"**Baseline Quality**: {bq_ci_str}")
    lines.append(f"**Failure Type**: `{report.classification.failure_type}` (Confidence: {report.classification.confidence:.0%})")
    
    lines.append("\n## Evidence")
    for ev in report.classification.evidence:
        lines.append(f"- {ev}")
        
    lines.append("\n## Shapley Attribution Estimates")
    lines.append("Agent | Shapley Value | 95% CI | Impact")
    lines.append("---|---|---|---")
    
    # Determine min/max for ascii bars
    vals = list(report.shapley_values.values())
    if vals:
        min_v, max_v = min(vals), max(vals)
        for agent, val in report.shapley_values.items():
            ci = report.shapley_cis.get(agent)
            ci_str = _format_ci(ci)
            bar = _ascii_bar(val, min(0, min_v), max(0, max_v))
            lines.append(f"**{agent}** | {val:+.3f} | {ci_str} | `{bar}`")
            
    lines.append("\n## Recommended Fixes")
    if not report.recommendations:
        lines.append("No fixes recommended.")
    for i, rec in enumerate(report.recommendations, 1):
        target = rec.target_agent or "Pipeline"
        lines.append(f"### {i}. {rec.intervention_type.replace('_', ' ').title()} {target}")
        lines.append(f"**Description**: {rec.description}")
        lines.append(f"**Estimated Improvement**: {rec.estimated_failure_reduction*100:+.1f}%")
        lines.append(f"**Confidence**: `{rec.measurement_confidence}`")
        if rec.agent_spec:
            lines.append("\n**Agent Specification**:")
            lines.append(f"```json\n{{json.dumps(rec.agent_spec.to_dict(), indent=2)}}\n```")
            
    res = "\n".join(lines)
    if path:
        with open(path, "w") as f:
            f.write(res)
    return res


def to_html(report: "DiagnosticReport", path: Optional[str] = None) -> str:
    """Generate a minimal, self-contained HTML report."""
    import html
    
    md = to_markdown(report)
    escaped_md = html.escape(md)
    
    # Using a simple vanilla HTML template with minimal CSS
    html_out = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Diagnostic Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #fafafa;
    }}
    .container {{
        background: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    h1, h2, h3 {{ color: #111; }}
    pre {{
        background: #f4f4f4;
        padding: 15px;
        border-radius: 4px;
        overflow-x: auto;
        font-family: monospace;
    }}
    code {{
        background: #eee;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: monospace;
        font-size: 0.9em;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{ background-color: #f8f8f8; }}
</style>
</head>
<body>
<div class="container">
    <pre id="content">{escaped_md}</pre>
    <!-- Simple markdown parser for the browser to render the content nicely -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        document.getElementById('content').outerHTML = marked.parse(document.getElementById('content').textContent);
    </script>
</div>
</body>
</html>"""

    if path:
        with open(path, "w") as f:
            f.write(html_out)
            
    return html_out
