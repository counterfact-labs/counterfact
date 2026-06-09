"""Skill-compatible FinanceBench case-study pipeline.

A clean, factory-driven rebuild of the 8-agent financial-RAG demo so the
counterfact-debugger Agent Skill can drive it directly:

    cf_diagnose.py --factory financebench_skill.pipeline:build \
                   --registry financebench_skill.quality:build_registry \
                   --inputs cases.json --domain financebench

Unlike the original research scripts, this package has NO thread-local global
state: ground truth is keyed by the query string and the source document is a
module constant, so build() is a true no-arg factory and the classifiers are
stateless. Editing the four buggy instructions in prompts.py is the fix surface.
"""
