"""Metrics node — ingests real performance data for published videos.

Pulls views/retention (starting with YouTube Analytics for our own channel) and writes
the results back into `videoagent.memory` so future ideation is informed by what
actually performed. Simulated metrics are available behind a flag for platforms
without easy read access.

Populated in Phase 3b.
"""
