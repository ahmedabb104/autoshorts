"""Runtime dependencies handed to nodes at invoke time.

This is how a node reaches an LLM without ever importing an SDK, constructing a client,
or naming a model (CLAUDE.md 4a). LangGraph's `context` is the right home for it: unlike
graph state it is *not* checkpointed, which is correct — a live HTTP client is not
something you want serialized into a SQLite row and rehydrated on resume.

A node receives it as `runtime.context` and asks for a capability tier; everything about
which model that is, what it costs, and how failures are retried stays behind the
provider interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from videoagent.config import Settings, get_settings
from videoagent.providers.llm import LLMProvider, open_llm_provider

__all__ = ["GraphContext", "open_graph_context"]


@dataclass(frozen=True)
class GraphContext:
    """Everything a node needs that is not part of the run's state."""

    llm: LLMProvider
    settings: Settings


@asynccontextmanager
async def open_graph_context(settings: Settings | None = None) -> AsyncIterator[GraphContext]:
    """Build the real, network-backed context and tear it down afterwards.

    Tests construct `GraphContext` directly with a fake provider instead of calling this.
    """
    settings = settings or get_settings()
    async with open_llm_provider(settings) as llm:
        yield GraphContext(llm=llm, settings=settings)
