"""Render node — assembles clips, voiceover, and captions into a vertical short.

Phase 1a: a stub that produces no file. Phase 2 wraps ffmpeg and adds the QA gate
(duration within bounds, 9:16 aspect ratio, an audio track actually present). A failing
QA gate must not pass a broken video downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from videoagent.graph.state import RunStatus, VideoState


async def render_node(state: VideoState) -> dict[str, Any]:
    """Record a placeholder output path."""
    return {
        "video_path": Path(f"stub/{state.run_id}.mp4"),
        "status": RunStatus.RENDERED,
        "completed_nodes": ["render"],
    }
