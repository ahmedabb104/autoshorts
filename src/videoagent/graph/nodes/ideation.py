"""Ideation node — chooses the topic for this run.

Phase 1a: a stub. Phase 1b picks from a seed list using the draft-tier LLM; Phase 3c
retrieves past top performers from `videoagent.memory` and uses them as few-shot context,
closing the loop from real performance data back into topic selection.
"""

from __future__ import annotations

from typing import Any

from videoagent.graph.state import RunStatus, VideoState

#: Used only until Phase 1b wires the draft-tier LLM.
STUB_TOPIC = "stub topic (ideation is not implemented until Phase 1b)"


async def ideation_node(state: VideoState) -> dict[str, Any]:
    """Set `topic`, preserving one the caller supplied.

    Honouring a caller-provided topic is not just stub convenience: the operator console
    needs to be able to force a topic, so the real node keeps this behaviour.
    """
    return {
        "topic": state.topic or STUB_TOPIC,
        "status": RunStatus.IDEATED,
        "completed_nodes": ["ideation"],
    }
