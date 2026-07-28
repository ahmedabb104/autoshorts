"""Approval node — the human-in-the-loop gate.

Calls LangGraph's `interrupt()` so the graph pauses with its state checkpointed and
only advances to publish when a human resumes it. There is deliberately no
"skip approval" default path (CLAUDE.md 4b).

Populated in Phase 1e.
"""
