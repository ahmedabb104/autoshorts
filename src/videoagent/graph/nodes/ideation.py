"""Ideation node — chooses the topic for this run.

Phase 1b: the draft-tier LLM sharpens one entry from a seed list into a specific angle.
Phase 3c replaces the seed list with retrieval over past top performers from
`videoagent.memory`, closing the loop from real performance data back into ideation —
the seed list is the placeholder for that retrieval, which is why the prompt is already
shaped as "here are candidates, pick and sharpen one" rather than "invent something".
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from videoagent.config import LLMTier
from videoagent.graph.context import GraphContext
from videoagent.graph.state import CostEntry, RunStatus, VideoState

__all__ = ["SEED_TOPICS", "SYSTEM_PROMPT", "ideation_node"]

#: Stand-in for the retrieval that lands in Phase 3c. Broad enough that the draft model
#: has room to find an angle, narrow enough to stay in a coherent niche.
SEED_TOPICS: tuple[str, ...] = (
    "counterintuitive facts about everyday physics",
    "how a familiar object is actually manufactured",
    "a historical decision that quietly shaped modern life",
    "a cognitive bias you are using right now",
    "the engineering trade-off hidden inside a common product",
    "an animal ability that outperforms human technology",
    "a number so large or small it breaks intuition",
    "why a piece of old technology refuses to die",
)

SYSTEM_PROMPT = """You pick topics for 30-45 second vertical short-form videos.

Given a list of broad themes, choose ONE and sharpen it into a single specific, concrete
topic that a narrator could cover completely in under 45 seconds.

Rules:
- Be specific. "Why bridges have gaps" beats "interesting facts about bridges".
- Pick something with one clear surprise in it, not a list of five things.
- No clickbait phrasing, no questions, no hashtags.

Reply with the topic and nothing else: one line, at most 15 words."""

#: A runaway model can return an essay; a topic is one line.
MAX_TOPIC_WORDS = 25


def _clean_topic(raw: str) -> str:
    """Reduce a model's reply to a single bare topic line."""
    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    unquoted = first_line.strip("\"'").removeprefix("Topic:").strip()
    words = unquoted.split()
    return " ".join(words[:MAX_TOPIC_WORDS])


async def ideation_node(state: VideoState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    """Choose this run's topic.

    A caller-supplied topic wins and skips the LLM entirely. That is not just an
    optimisation: the operator console needs to force a topic, and spending quota to
    re-derive one the human already chose would be wrong.
    """
    if state.topic:
        return {"status": RunStatus.IDEATED, "completed_nodes": ["ideation"]}

    candidates = "\n".join(f"- {topic}" for topic in SEED_TOPICS)
    response = await runtime.context.llm.complete(
        [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(f"Themes to choose from:\n{candidates}"),
        ],
        tier=LLMTier.DRAFT,
        temperature=1.0,  # Topic choice should vary between runs; scripts should not.
        max_tokens=64,
    )

    topic = _clean_topic(response.text)
    if not topic:
        return {
            "status": RunStatus.FAILED,
            "error": f"Ideation returned no usable topic: {response.text!r}",
            "completed_nodes": ["ideation"],
        }

    return {
        "topic": topic,
        "status": RunStatus.IDEATED,
        "costs": [
            CostEntry(
                node="ideation",
                provider="openrouter",
                usd=response.usage.cost_usd,
                detail=response.model,
            )
        ],
        "completed_nodes": ["ideation"],
    }
