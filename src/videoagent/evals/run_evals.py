"""Offline eval harness — the regression suite for prompt changes.

Runs the judged step over the labeled dataset in `dataset/` and reports metrics
(agreement with the labels, precision on the factual-risk flag, mean score) as a summary
table, so a prompt edit that regresses quality is visible instead of silent. Run this
whenever a prompt changes.

Populated in Phase 1d.
"""
