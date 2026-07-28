"""Eval critic node — LLM-as-judge scoring of the draft script.

Scores the script against the shared rubric in `videoagent.evals.rubric` (the same
rubric the offline harness uses — CLAUDE.md 4d) using the judge-tier LLM, which must
be distinctly stronger than the drafter. The resulting score drives the conditional
retry edge in `graph.py`.

Populated in Phase 1c.
"""
