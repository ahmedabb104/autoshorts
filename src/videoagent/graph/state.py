"""The graph state schema — the single source of truth for everything the graph carries.

A Pydantic v2 model holding the topic, the script (hook/body/CTA), the eval score,
asset paths, the publish content hash, a status enum, the accumulated cost, and the
retry counter that bounds the scriptwriter loop. Every node takes this model in and
returns it out; no untyped dicts flow through the graph (CLAUDE.md 4c).

Populated in Phase 1a.
"""
