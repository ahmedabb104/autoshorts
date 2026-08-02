"""Eval critic node — LLM-as-judge scoring of the draft script.

Phase 1a: a stub returning a fixed passing score. Phase 1c scores against the shared
rubric in `videoagent.evals.rubric` (the same rubric the offline harness uses —
CLAUDE.md 4d) using the judge-tier LLM, and its score drives the conditional retry edge.

The stub scores high on purpose: until Phase 1c adds that edge, a low score would have
nowhere to route and would just look like a silent pass.
"""

from __future__ import annotations

from typing import Any

from videoagent.graph.state import RunStatus, VideoState

#: A passing score on the 0-10 rubric scale (see EVAL_SCORE_THRESHOLD, default 7.0).
STUB_SCORE = 8.0


async def eval_critic_node(state: VideoState) -> dict[str, Any]:
    """Score the current script."""
    return {
        "eval_score": STUB_SCORE,
        "eval_notes": "stub score — the judge-tier LLM and the rubric arrive in Phase 1c.",
        "status": RunStatus.EVALUATED,
        "completed_nodes": ["eval_critic"],
    }
