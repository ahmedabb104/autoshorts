"""Scriptwriter node — writes the hook, body, and CTA for the chosen topic.

Uses the draft tier: this is the high-volume call, and it is the one the retry loop
re-runs. When `eval_critic` scores below threshold (Phase 1c), the graph comes back here
with `retry_count` incremented, and the prompt then includes the rejected attempt and the
critic's notes — a retry that did not see why it failed would just resample the same
distribution and burn quota for nothing.

Output is requested as JSON rather than free text so the three parts stay separable. The
`response_format` parameter is deliberately *not* used: many OpenRouter `:free` models
reject it outright, and a hard 400 on a rotated model is exactly the time-bomb CLAUDE.md
4a warns about. Prompt instruction plus lenient parsing degrades instead of breaking.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from videoagent.config import LLMTier
from videoagent.graph.context import GraphContext
from videoagent.graph.state import CostEntry, RunStatus, Script, VideoState
from videoagent.providers.llm import LLMResponseError, extract_json_object

__all__ = ["SYSTEM_PROMPT", "scriptwriter_node"]

SYSTEM_PROMPT = """You write scripts for 30-45 second vertical short-form videos.

Return a JSON object with exactly these keys:
  "hook" — the first 1-2 sentences. It must earn the next three seconds. No throat-clearing,
           no "in this video", no greeting. Lead with the surprise.
  "body" — 60-100 words of narration that pays the hook off with something concrete.
  "cta"  — one short closing line.

Rules:
- Write narration to be spoken aloud, not read. Short sentences.
- Every claim must be one you are confident is true. If you are not sure, leave it out.
- No emoji, no hashtags, no stage directions, no speaker labels.

Reply with the JSON object and nothing else."""


def _retry_guidance(state: VideoState) -> str:
    """Tell the model what was wrong last time, so a retry is not just a resample."""
    previous = state.script
    if previous is None:
        return ""
    notes = state.eval_notes or "no specific feedback was recorded"
    score = "unknown" if state.eval_score is None else f"{state.eval_score:.1f}/10"
    return (
        f"\n\nYour previous attempt scored {score} and was rejected.\n"
        f"Reviewer notes: {notes}\n\n"
        f"Previous hook: {previous.hook}\n"
        f"Previous body: {previous.body}\n"
        f"Previous CTA: {previous.cta}\n\n"
        "Write a genuinely different script that fixes those problems. "
        "Do not lightly reword the previous attempt."
    )


async def scriptwriter_node(state: VideoState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    """Draft a script for the current topic, or a better one if this is a retry."""
    if not state.topic:
        return {
            "status": RunStatus.FAILED,
            "error": "Scriptwriter reached without a topic.",
            "completed_nodes": ["scriptwriter"],
        }

    request = f"Topic: {state.topic}{_retry_guidance(state)}"
    response = await runtime.context.llm.complete(
        [SystemMessage(SYSTEM_PROMPT), HumanMessage(request)],
        tier=LLMTier.DRAFT,
        temperature=0.8,
        max_tokens=600,
    )

    cost = [
        CostEntry(
            node="scriptwriter",
            provider="openrouter",
            usd=response.usage.cost_usd,
            detail=response.model,
        )
    ]

    try:
        script = _parse_script(response.text)
    except LLMResponseError as error:
        # Record the spend even though the output was unusable — a run that burned quota
        # on a malformed reply should show that in its cost, not hide it.
        return {
            "status": RunStatus.FAILED,
            "error": str(error),
            "costs": cost,
            "completed_nodes": ["scriptwriter"],
        }

    return {
        "script": script,
        "status": RunStatus.DRAFTED,
        "costs": cost,
        "completed_nodes": ["scriptwriter"],
    }


def _parse_script(text: str) -> Script:
    """Turn a model reply into a `Script`, or raise `LLMResponseError`."""
    payload = extract_json_object(text)
    missing = [key for key in ("hook", "body", "cta") if not str(payload.get(key, "")).strip()]
    if missing:
        raise LLMResponseError(f"Script JSON is missing {', '.join(missing)}: {payload!r}")
    return Script(
        hook=str(payload["hook"]).strip(),
        body=str(payload["body"]).strip(),
        cta=str(payload["cta"]).strip(),
    )
