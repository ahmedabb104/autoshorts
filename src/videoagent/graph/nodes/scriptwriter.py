"""Scriptwriter node — writes the hook, body, and CTA for the chosen topic.

Phase 1a: a stub. Phase 1b uses the draft-tier LLM (the high-volume call). This node is
re-entered when the eval critic scores below threshold, so the real prompt gets the
previous attempt and the critic's notes; the stub reflects `retry_count` in its output so
the retry loop is observable before the LLM exists.
"""

from __future__ import annotations

from typing import Any

from videoagent.graph.state import RunStatus, Script, VideoState


async def scriptwriter_node(state: VideoState) -> dict[str, Any]:
    """Produce a `Script`. Overwrites any previous attempt — retries replace, not append."""
    attempt = state.retry_count + 1
    script = Script(
        hook=f"stub hook for {state.topic!r} (attempt {attempt})",
        body="stub body — the draft-tier LLM writes this in Phase 1b.",
        cta="stub CTA — follow for more.",
    )
    return {
        "script": script,
        "status": RunStatus.DRAFTED,
        "completed_nodes": ["scriptwriter"],
    }
