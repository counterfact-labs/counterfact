"""Offline case study: why a retriever pipeline needs *graded degradation*, not
just ablation.

A three-stage RAG pipeline (retriever -> reranker -> synthesizer) answers document
questions. The synthesizer, like a real context-window-limited LLM, only reads the
top-k passages. The pipeline passes its eval, so the question is not "what is
broken" but "where is it fragile / which module's quality actually controls the
answer."

Pure ablation gives a misleading answer here:
  * Ablating the retriever collapses everything (no context) -> huge Shapley, but
    that only says "a retriever is necessary."
  * Ablating the reranker is a no-op pass-through (the retrieval order already puts
    the relevant passage in the top-k on these cases) -> ~0 Shapley, so ablation
    calls the reranker irrelevant.

Graded degradation tells the truth: the reranker is a **quality_driver** (decaying
its ranking buries the relevant passage out of the top-k and answers fail), while
the retriever is merely **structural**. That is the lesson teams over-tuning the
retriever miss.

Run it (deterministic, no API keys):

    PYTHONPATH=examples python -m rag_degradation_casestudy.run_casestudy
    PYTHONPATH=examples python -m rag_degradation_casestudy.make_report
"""
