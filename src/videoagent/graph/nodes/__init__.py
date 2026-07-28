"""Graph nodes. Each is a pure state-in / state-out function over the model in `state.py`.

Nodes depend only on provider *interfaces* (`videoagent.providers`); importing a vendor
SDK or hardcoding a model ID inside a node is a bug (CLAUDE.md 4a).
"""
