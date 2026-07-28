"""Publish node — hands the rendered video to the configured publish provider.

Idempotent by content hash: the hash is computed and persisted, and a re-run or resume
that reaches this node again must not publish a second time (CLAUDE.md 4b). The default
provider writes to disk; real posting is opt-in via config.

Populated in Phase 1e.
"""
