"""Assets node — turns an approved script into a voiceover and visual clips.

Phase 1a: a stub that writes no files. Phase 2 calls the TTS and video providers for
real and appends their spend to the state cost ledger.

The stub records a zero-cost ledger entry rather than nothing at all, so the cost
accumulator is exercised end-to-end before any provider can charge us.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from videoagent.graph.state import CostEntry, RunStatus, VideoState


async def assets_node(state: VideoState) -> dict[str, Any]:
    """Record placeholder asset paths. Nothing is downloaded or synthesised yet."""
    return {
        "voiceover_path": Path("stub/voiceover.wav"),
        "clip_paths": [Path("stub/clip-01.mp4"), Path("stub/clip-02.mp4")],
        "status": RunStatus.ASSETS_READY,
        "costs": [
            CostEntry(node="assets", provider="stub", usd=0.0, detail="no provider wired yet")
        ],
        "completed_nodes": ["assets"],
    }
