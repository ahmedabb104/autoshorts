"""Ideation node — chooses the topic for this run.

Starts (Phase 1b) as a pick from a seed list using the draft-tier LLM. In Phase 3c it
retrieves past top-performing videos from `videoagent.memory` and uses them as few-shot
context, closing the loop from real performance data back into topic selection.
"""
