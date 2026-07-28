"""`build_graph()` — assembles nodes, edges, and the checkpointer into a runnable graph.

Owns the graph topology: the linear happy path, the conditional edge that sends a
below-threshold script back to the scriptwriter (bounded by the retry counter in
state), and the human-approval interrupt before publish. The checkpointer is chosen
by config (SQLite locally, Postgres for a real deployment) and persists after each
node so a killed run resumes from the last completed node.

Populated in Phase 1a.
"""
