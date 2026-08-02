"""Metrics node — ingests real performance data for published videos.

Phase 1a: a stub. Phase 3b pulls views/retention (starting with YouTube Analytics for our
own channel) and writes the results into `videoagent.memory`, so that ideation is shaped
by what actually performed. Simulated metrics sit behind a flag for platforms without
easy read access.
"""

from __future__ import annotations

from typing import Any

from videoagent.graph.state import VideoState


async def metrics_node(state: VideoState) -> dict[str, Any]:
    """No-op. There is nothing published to measure until Phase 3a."""
    return {"completed_nodes": ["metrics"]}
