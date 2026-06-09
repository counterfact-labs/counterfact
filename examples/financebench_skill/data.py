"""Canonical dataset for the FinanceBench case study.

ONE source document and ONE ground-truth map, used by every script and the
skill. Ground truth is keyed by the exact query string, which is what counterfact
passes to classifiers (perturbation.py: ``query = input_state.get("query", ...)``),
so classifiers can resolve the expected answer without any hidden global state.

Figures are 3M's FY2018 consolidated statement of cash flows (FinanceBench).
"""
from __future__ import annotations

# Single canonical source document (3M FY2018 cash flow statement).
CASH_FLOW = """
3M Company and Subsidiaries
Consolidated Statement of Cash Flows
Years ended December 31 (Millions)
                                                    2018        2017        2016
Cash Flows from Operating Activities
  Net income including noncontrolling interest     $5,363      $4,869      $5,058
  Depreciation and amortization                     1,488       1,544       1,474
  Net cash provided by operating activities         6,439       6,240       6,662

Cash Flows from Investing Activities
  Purchases of property, plant and equipment (PP&E) (1,577)     (1,373)     (1,420)
  Proceeds from sale of PP&E and other assets          262          49          58
  Acquisitions, net of cash acquired                    13      (2,023)        (16)

Cash Flows from Financing Activities
  Repurchases of common stock                       (4,870)    (2,068)     (3,753)
  Dividends paid                                    (3,193)    (2,803)     (2,678)
"""

# query -> (ground-truth answer, short label)
QUERIES = [
    {"query": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Use the cash flow statement.",
     "ground_truth": "$1,577 million", "short": "CapEx"},
    {"query": "What was 3M's depreciation and amortization expense in FY2018 (in USD millions)?",
     "ground_truth": "$1,488 million", "short": "D&A"},
    {"query": "What was 3M's net cash provided by operating activities in FY2018 (in USD millions)?",
     "ground_truth": "$6,439 million", "short": "OpCF"},
    {"query": "How much did 3M pay in dividends in FY2018 (in USD millions)?",
     "ground_truth": "$3,193 million", "short": "Dividends"},
    {"query": "What was 3M's total stock repurchase amount in FY2018 (in USD millions)?",
     "ground_truth": "$4,870 million", "short": "Buybacks"},
]

# query string -> ground-truth answer (stateless lookup for classifiers)
GROUND_TRUTH = {q["query"]: q["ground_truth"] for q in QUERIES}


def make_input_state(query: str) -> dict:
    """Build the initial pipeline state for a query."""
    return {
        "query": query,
        "source_document": CASH_FLOW,
        "parsed_query": "",
        "retrieved_sections": "",
        "extracted_data": "",
        "enriched_context": "",
        "analysis": "",
    }


def all_cases() -> list[dict]:
    """The five input states, one per FinanceBench query."""
    return [make_input_state(q["query"]) for q in QUERIES]
