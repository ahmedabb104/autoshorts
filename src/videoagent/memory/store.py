"""Embedding + retrieval wrapper over the video/performance history.

Stores each published video with its script, topic, and real performance metrics, and
retrieves the top performers as few-shot context for ideation. Starts with the simplest
local option — no vector-DB server unless a task requires one.

Populated in Phase 3c.
"""
