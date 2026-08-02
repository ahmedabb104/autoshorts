"""Eval critic node — LLM-as-judge scoring of the draft script.

Runs on the **judge** tier, which must be a distinctly stronger model than the drafter
(CLAUDE.md section 5). The writer grading its own work is not an evaluation. This node
runs once per video, so if free judge models get throttled, pointing `LLM_JUDGE_MODEL` at
a cheap *paid* ID is a config change that keeps evals honest for a rounding error.

The node itself is deliberately thin: the criteria, the prompt, the parser, and the
pass/fail rule all live in `evals.rubric`, shared with the offline harness (CLAUDE.md 4d).
All this node does is call the rubric, write the verdict into state, and count the retry.

**Where the retry is counted.** This node increments `retry_count` when it rejects a
script; the conditional edge in `graph.py` only reads. A LangGraph router cannot write
state, and putting the increment in the scriptwriter would mean inferring "am I a retry?"
from the presence of a previous script — which is the same information, stored twice, and
free to disagree.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from videoagent.evals.rubric import score_script
from videoagent.graph.context import GraphContext
from videoagent.graph.state import CostEntry, RunStatus, VideoState
from videoagent.providers.llm import LLMError

__all__ = ["eval_critic_node"]


async def eval_critic_node(state: VideoState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    """Score the current script against the shared rubric."""
    if state.script is None:
        return {
            "status": RunStatus.FAILED,
            "error": "Eval critic reached without a script.",
            "completed_nodes": ["eval_critic"],
        }

    try:
        rubric_score, response = await score_script(
            runtime.context.llm,
            topic=state.topic or "",
            script=state.script,
        )
    except LLMError as error:
        # A judge outage is not the scriptwriter's fault, so this must NOT drive a retry —
        # rewriting a fine script because the grader was down would burn quota for nothing.
        # The run continues unscored and the human approval gate is what catches it.
        return {
            "status": RunStatus.FAILED,
            "error": f"Eval critic could not score this script: {error}",
            "completed_nodes": ["eval_critic"],
        }

    cost = [
        CostEntry(
            node="eval_critic",
            provider="openrouter",
            usd=response.usage.cost_usd,
            detail=response.model,
        )
    ]

    update: dict[str, Any] = {
        "eval_rubric": rubric_score,
        "eval_score": rubric_score.overall,
        "eval_notes": rubric_score.summary(),
        "status": RunStatus.EVALUATED,
        "costs": cost,
        "completed_nodes": ["eval_critic"],
    }

    if not rubric_score.passes(runtime.context.settings.eval_score_threshold):
        update["retry_count"] = state.retry_count + 1

    return update
