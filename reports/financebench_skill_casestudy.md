# FinanceBench Case Study — Skill-Driven Reproduction

_Config: 5 queries, 30 simulations/query, seed 42, Sonnet (synthesizer) + Haiku (other agents)._

## Iterative fix arc

| Step | Fix applied | Avg quality | Exact answers |
|---|---|---|---|
| 0 | — (broken baseline) | 0.400 | 0/5 |
| 1 | tone_editor | 0.889 | 5/5 |

## Step 0 aggregate Shapley (most-harmful first)
```
  tone_editor        -0.146  ██
  table_extractor    -0.030  
  context_enricher   +0.002  
  query_parser       +0.012  
  synthesizer        +0.047  
  fact_checker       +0.129  ██
  doc_retriever      +0.138  ██
  output_formatter   +0.208  ████
```

### Per-classifier worst agent (step 0)
- **accuracy**: tone_editor (-0.106)
- **precision**: tone_editor (-0.244)
- **grounding**: synthesizer (-0.185)

## Baseline: can an LLM diagnose this from traces?

**LLM ranking (most→least responsible):** tone_editor, output_formatter, context_enricher, synthesizer, fact_checker, table_extractor, doc_retriever, query_parser

**Causal Shapley ranking (most-harmful→least, step 0):** tone_editor, table_extractor, context_enricher, query_parser, synthesizer, fact_checker, doc_retriever, output_formatter


_LLM reasoning:_ tone_editor is the primary culprit: it consistently rounds exact figures ($1,577→$1.6B, $1,488→$1.5B, $6,439→$6.4B, $4,870→$4.9B) and introduces fabricated qualitative narratives (peer comparisons, 'Dividend Aristocrat status', 'tariff disruptions', 'management confidence in valuation') that were never in the source data. output_formatter blindly propagates all of tone_editor's distortions without correction, making it the second most responsible. context_enricher introduces unsourced narrative framing and approximate figures ($6.4B, $3.2B) that prime downstream agents toward rounded outputs. synthesizer correctly preserves exact figures in most cases but occasionally inherits rounded approximations from context_enricher. fact_checker fails its core responsibility by passing through tone_editor-style rounding and fabricated context without flagging them as inaccurate — it verifies the exact figures but then allows downstream corruption. table_extractor, doc_retriever, and query_parser all perform correctly and are not responsible for the errors.
