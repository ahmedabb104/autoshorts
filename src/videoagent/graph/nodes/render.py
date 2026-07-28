"""Render node — assembles clips, voiceover, and captions into a vertical short.

Wraps ffmpeg, then applies a QA gate: duration within bounds, 9:16 aspect ratio, and
an audio track actually present. A failing QA gate must not silently pass a broken
video downstream.

Populated in Phase 2.
"""
