"""The scoring rubric — defined exactly once.

Criteria (hook strength, clarity, factual-risk flag, ...) and the structured score they
produce. Imported by both the `eval_critic` node and the offline `run_evals.py` harness;
duplicating it anywhere else is a bug (CLAUDE.md 4d).

Populated in Phase 1c.
"""
