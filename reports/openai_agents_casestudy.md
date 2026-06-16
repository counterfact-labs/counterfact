# OpenAI Agents SDK + Braintrust — counterfact case study (generated)

- System: openai-agents-sdk orchestrator+handoffs (offline, deterministic)
- Scorer: `braintrust-style refund_amount_present`
- Cases: 5
- Baseline pass rate: **0%**
- Final pass rate: **100%**
- Fixes applied: `compliance_editor`

## Diagnosis timeline

**Round 1** — picked `compliance_editor` (Shapley -0.49)

| agent | Shapley |
|---|---|
| `compliance_editor` | -0.490 |
| `billing` | +0.000 |
| `technical` | +0.153 |
| `account` | +0.153 |
| `orchestrator` | +0.204 |
