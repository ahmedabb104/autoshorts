"""`PublishProvider` protocol and its implementations.

`FileProvider` (the default) writes the video plus a metadata JSON locally and keeps the
persisted set of published content hashes that makes publishing idempotent. Real
posting — YouTube Data API directly, an aggregator for TikTok/Reels — is opt-in via
`PUBLISH_PROVIDER` and never the default (CLAUDE.md 4b).

`FileProvider` is populated in Phase 1e; the real providers in Phase 3a.
"""
