"""Assets node — turns an approved script into a voiceover and visual clips.

Calls the TTS provider and the video-asset provider, records their reported spend into
the state cost accumulator, and writes the resulting file paths back to state. Mocked
through Phase 1; wired to real providers in Phase 2.
"""
