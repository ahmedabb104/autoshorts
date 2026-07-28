"""Scriptwriter node — writes the hook, body, and CTA for the chosen topic.

Uses the draft-tier LLM (the high-volume call). Re-entered when the eval critic scores
a script below threshold, so its prompt has access to the previous attempt and the
critic's feedback.

Populated in Phase 1b.
"""
