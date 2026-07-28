"""FastAPI app — streams graph events to the operator console.

Exposes an endpoint to trigger a run, an SSE/WebSocket stream of graph events so the UI
can light up nodes live, and the approve/reject endpoint that resumes a run paused at
the human-approval interrupt.

Populated in Phase 3d.
"""
