"""Provider interfaces for everything external.

Each external dependency (LLM, TTS, video assets, publishing) is a Protocol with
swappable implementations, selected in `videoagent.config` from environment variables.
This seam is what makes the two-tier LLM strategy and the "swap a retired `:free` model
by changing env" story work (CLAUDE.md 4a).
"""
