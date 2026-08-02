"""Approval node — the human-in-the-loop gate.

Phase 1e calls LangGraph's `interrupt()` here so the graph genuinely pauses with its
state checkpointed and only advances when a human resumes it.

Phase 1a is a stub, and the important thing about this stub is what it does **not** do:
it does not approve. It moves the run to `AWAITING_APPROVAL` and leaves `approved` as
`None`. The publish node refuses to act on anything that is not explicitly `approved is
True`, so the absence of the interrupt cannot turn into an accidental auto-publish
(CLAUDE.md 4b). A caller can still pre-set `approved=True` in the input state — that is
an explicit human decision, which is exactly what the gate is for.

There is deliberately no "skip approval" path, now or later.
"""

from __future__ import annotations

from typing import Any

from videoagent.graph.state import RunStatus, VideoState


async def approval_node(state: VideoState) -> dict[str, Any]:
    """Mark the run as awaiting a human decision, without making one."""
    if state.approved is True:
        return {"status": RunStatus.APPROVED, "completed_nodes": ["approval"]}
    if state.approved is False:
        return {"status": RunStatus.REJECTED, "completed_nodes": ["approval"]}
    return {"status": RunStatus.AWAITING_APPROVAL, "completed_nodes": ["approval"]}
