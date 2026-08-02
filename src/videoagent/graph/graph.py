"""`build_graph()` — assembles nodes, edges, and the checkpointer into a runnable graph.

Owns the graph topology. Phase 1a wires the linear happy path over stub nodes; the two
branches that make this graph interesting land later and are noted at their insertion
points below:

* Phase 1c — a conditional edge after `eval_critic` routing a below-threshold script back
  to `scriptwriter`, bounded by `retry_count`.
* Phase 1e — `interrupt()` inside `approval`, so the graph pauses for a human.

Structure of this module:

* `build_graph(checkpointer)` is pure — it wires and compiles, and takes whatever
  checkpointer you hand it. Tests can pass an `InMemorySaver`; nothing here reads config.
* `open_checkpointer()` / `open_graph()` own the *lifecycle*, because a SQLite
  checkpointer is a live database connection that has to be opened, migrated, and closed.
  Keeping that out of `build_graph` is what lets Phase 3d's FastAPI app hold one
  connection open for the process lifetime while tests open a fresh one per test.

Nodes are `async` throughout. They do no I/O yet, but every one of them will (LLM calls
in 1b/1c, TTS and ffmpeg in 2, HTTP in 3a), and CLAUDE.md section 7 prefers async at the
I/O boundary — making them async now avoids rewriting the whole node layer later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from itertools import pairwise
from typing import Any, Final

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from videoagent.config import CheckpointerBackend, Settings, get_settings
from videoagent.graph.context import GraphContext
from videoagent.graph.nodes.approval import approval_node
from videoagent.graph.nodes.assets import assets_node
from videoagent.graph.nodes.eval_critic import eval_critic_node
from videoagent.graph.nodes.ideation import ideation_node
from videoagent.graph.nodes.metrics import metrics_node
from videoagent.graph.nodes.publish import publish_node
from videoagent.graph.nodes.render import render_node
from videoagent.graph.nodes.scriptwriter import scriptwriter_node
from videoagent.graph.state import CHECKPOINTED_TYPES, VideoState

__all__ = [
    "DEFAULT_DURABILITY",
    "NODE_NAMES",
    "RETRY_BRANCH_SOURCE",
    "RETRY_TARGET",
    "build_graph",
    "build_serde",
    "open_checkpointer",
    "open_graph",
    "route_after_eval",
]

#: Write the checkpoint before the next node starts, rather than concurrently with it.
#: LangGraph defaults to `"async"`, which is faster but means a hard kill can lose the
#: most recent node's checkpoint. CLAUDE.md 4c promises the graph resumes from the last
#: *completed* node, and only `"sync"` actually guarantees that.
DEFAULT_DURABILITY: Final = "sync"

#: The node whose outcome branches: pass and continue, or loop back and rewrite.
RETRY_BRANCH_SOURCE: Final = "eval_critic"
RETRY_TARGET: Final = "scriptwriter"

#: The linear path, in execution order. `eval_critic -> assets` is conditional.
NODE_NAMES: Final[tuple[str, ...]] = (
    "ideation",
    "scriptwriter",
    "eval_critic",
    "assets",
    "render",
    "approval",
    "publish",
    "metrics",
)


def _node_sequence() -> Sequence[tuple[str, Callable[[VideoState], Any]]]:
    """Pair each node name with its implementation.

    Resolved at call time rather than at import time so that a test can monkeypatch a
    node in this module's namespace (to simulate a crash, say) before building the graph.
    """
    return (
        ("ideation", ideation_node),
        ("scriptwriter", scriptwriter_node),
        ("eval_critic", eval_critic_node),
        ("assets", assets_node),
        ("render", render_node),
        ("approval", approval_node),
        ("publish", publish_node),
        ("metrics", metrics_node),
    )


def build_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledStateGraph:
    """Wire the nodes and edges and compile against `checkpointer`.

    Pure: no config is read and no connection is opened. Pass `None` only for topology
    inspection — without a checkpointer there is no persistence and no resumability.
    """
    # `context_schema` is how nodes reach the LLM provider without importing an SDK.
    # Context is passed per invocation and is never checkpointed — see `context.py`.
    builder: StateGraph = StateGraph(VideoState, context_schema=GraphContext)

    sequence = _node_sequence()
    for name, fn in sequence:
        builder.add_node(name, fn)

    builder.add_edge(START, sequence[0][0])
    for (previous, _), (following, _) in pairwise(sequence):
        if previous == RETRY_BRANCH_SOURCE:
            # Owned by the conditional edge below.
            continue
        builder.add_edge(previous, following)
    builder.add_edge(sequence[-1][0], END)

    continue_target = NODE_NAMES[NODE_NAMES.index(RETRY_BRANCH_SOURCE) + 1]
    builder.add_conditional_edges(
        RETRY_BRANCH_SOURCE,
        route_after_eval,
        # Naming both destinations keeps the retry loop visible in the rendered graph,
        # which the operator console draws.
        {RETRY_TARGET: RETRY_TARGET, continue_target: continue_target},
    )

    return builder.compile(checkpointer=checkpointer)


def route_after_eval(state: VideoState, runtime: Runtime[GraphContext]) -> str:
    """Decide whether to rewrite the script or move on.

    Three cases, in order:

    1. The judge passed it — continue.
    2. The judge rejected it and the retry budget is spent — continue anyway. The bound
       exists so a model that dislikes everything cannot spin forever; the run carries its
       low score and its notes onward, and the human approval gate is the real backstop.
       Failing the run outright here would throw away a script a human might still accept.
    3. Otherwise — back to the scriptwriter, which will see the rejected draft and the
       critic's notes in its prompt.

    An unscored script (the judge itself failed) takes case 2: retrying the *writer* would
    not fix the *grader*.
    """
    settings = runtime.context.settings
    continue_target = NODE_NAMES[NODE_NAMES.index(RETRY_BRANCH_SOURCE) + 1]

    if state.eval_rubric is None:
        return continue_target
    if state.eval_rubric.passes(settings.eval_score_threshold):
        return continue_target
    if state.retry_count > settings.max_script_retries:
        return continue_target
    return RETRY_TARGET


def build_serde() -> JsonPlusSerializer:
    """The checkpoint serializer, with this project's state types registered.

    LangGraph will only reconstruct types on its allowlist when reading a checkpoint
    back. Ours are not on it by default: today they deserialize with a warning, but a
    future release blocks them, which would break the resume guarantee (CLAUDE.md 4c) at
    upgrade time rather than here. Registering them is the difference between finding out
    now and finding out in production.

    The library's own `SAFE_MSGPACK_TYPES` (langchain message types, stdlib primitives)
    is always honoured on top of this list, so naming our types does not narrow anything.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINTED_TYPES))


@asynccontextmanager
async def open_checkpointer(
    settings: Settings | None = None,
) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """Open the configured checkpointer, creating its schema, and close it on exit.

    Which backend is used is a config decision (`CHECKPOINTER`), never a code one.
    """
    settings = settings or get_settings()

    if settings.checkpointer is CheckpointerBackend.POSTGRES:
        raise NotImplementedError(
            "The Postgres checkpointer is not wired yet (it belongs to the deployment "
            "story, Phase 3). Set CHECKPOINTER=sqlite for local runs."
        )

    path = settings.sqlite_checkpoint_path
    # `config.py` performs no filesystem side effects on purpose, so creating the
    # directory is this module's job.
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(path) as connection:
        checkpointer = AsyncSqliteSaver(connection, serde=build_serde())
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def open_graph(settings: Settings | None = None) -> AsyncIterator[CompiledStateGraph]:
    """Open a checkpointer and yield a graph compiled against it.

    The usual entry point::

        async with open_graph() as graph:
            state = await graph.ainvoke(
                {"topic": "..."},
                {"configurable": {"thread_id": run_id}},
                durability=DEFAULT_DURABILITY,
            )
    """
    async with open_checkpointer(settings) as checkpointer:
        yield build_graph(checkpointer)
