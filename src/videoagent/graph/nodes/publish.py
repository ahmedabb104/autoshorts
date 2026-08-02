"""Publish node — hands the rendered video to the configured publish provider.

Phase 1e wires `PublishProvider` and the persisted set of published content hashes.
Phase 1a is a stub that publishes nothing anywhere.

Two guarantees are already enforced here, because they are the ones that are dangerous
to add late (CLAUDE.md 4b):

* **Approval is required.** Anything that is not explicitly `approved is True` is
  skipped, so a missing or broken approval gate fails closed rather than open.
* **Idempotency by content hash.** The hash is computed from the topic and script, so a
  retry or a resumed run recomputes the same value; a state that already carries a
  matching `content_hash` and a receipt is recognised as already published and is not
  published again.
"""

from __future__ import annotations

from typing import Any

from videoagent.graph.state import RunStatus, VideoState

#: Marks a receipt as having come from the Phase 1a no-op, not from a real platform.
STUB_RECEIPT_PREFIX = "stub://"


async def publish_node(state: VideoState) -> dict[str, Any]:
    """Publish the video, if and only if a human approved it and it is not a duplicate."""
    if state.approved is not True:
        # Fail closed. No approval, no publish — including when `approved` is None
        # because the Phase 1e interrupt has not been reached.
        return {"completed_nodes": ["publish"]}

    fingerprint = state.content_fingerprint()

    if state.content_hash == fingerprint and state.publish_receipt is not None:
        # Already published this exact content; a resume must not double-post.
        return {"status": RunStatus.PUBLISHED, "completed_nodes": ["publish"]}

    return {
        "content_hash": fingerprint,
        "publish_receipt": f"{STUB_RECEIPT_PREFIX}{fingerprint[:12]}",
        "status": RunStatus.PUBLISHED,
        "completed_nodes": ["publish"],
    }
