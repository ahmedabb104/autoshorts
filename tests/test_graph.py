"""Phase 1a tests: the graph runs on stubs, checkpoints every node, and resumes.

The resumability test is the load-bearing one. CLAUDE.md 4c promises that killing a run
mid-flight and restarting it continues from the last completed node rather than from the
start, and that promise is only worth anything if something actually proves it.
"""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Callable, Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from videoagent.config import CheckpointerBackend, Settings
from videoagent.graph import graph as graph_module
from videoagent.graph.context import GraphContext
from videoagent.graph.graph import (
    DEFAULT_DURABILITY,
    NODE_NAMES,
    build_graph,
    open_checkpointer,
    open_graph,
)
from videoagent.graph.nodes.publish import STUB_RECEIPT_PREFIX
from videoagent.graph.state import CostEntry, RunStatus, Script, VideoState

from .conftest import FakeLLM


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway checkpoint database.

    `_env_file=None` keeps a developer's real `.env` out of the test.
    """
    return Settings(
        _env_file=None,
        sqlite_checkpoint_path=tmp_path / "checkpoints" / "test.sqlite",
    )


@pytest.fixture
def thread() -> dict[str, Any]:
    """A LangGraph config naming the thread whose state is persisted."""
    return {"configurable": {"thread_id": "test-thread"}}


@pytest.fixture
def context(settings: Settings, fake_llm: FakeLLM) -> GraphContext:
    """Runtime dependencies for a run. No network, no API key."""
    return GraphContext(llm=fake_llm, settings=settings)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every node execution, in order, across as many graph builds as a test makes.

    Wraps each node in the `graph` module's namespace. `_node_sequence()` resolves those
    names at build time, so a graph built after this fixture runs picks up the wrappers.
    """
    recorded: list[str] = []
    _install_spies(monkeypatch, recorded)
    yield recorded


def _install_spies(
    monkeypatch: pytest.MonkeyPatch,
    recorded: list[str],
    crash_on: str | None = None,
    crash_times: int = 0,
) -> None:
    remaining = {"count": crash_times}

    def wrap(node_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        # LangGraph decides whether to pass `runtime` by inspecting the signature, so the
        # wrapper has to keep the wrapped node's arity rather than swallowing it in *args.
        wants_runtime = len(inspect.signature(original).parameters) > 1

        async def spy(state: VideoState, runtime: Any = None) -> dict[str, Any]:
            recorded.append(node_name)
            if node_name == crash_on and remaining["count"] > 0:
                remaining["count"] -= 1
                raise RuntimeError(f"simulated crash in {node_name}")
            return await (original(state, runtime) if wants_runtime else original(state))

        return spy

    for node_name in NODE_NAMES:
        attribute = f"{node_name}_node"
        original = getattr(graph_module, attribute)
        monkeypatch.setattr(graph_module, attribute, wrap(node_name, original))


def _checkpoint_row_count(database: Path) -> int:
    with sqlite3.connect(database) as connection:
        return connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]


# --------------------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------------------


def test_build_graph_is_pure_and_needs_no_checkpointer() -> None:
    """`build_graph` wires topology only — no config read, no connection opened."""
    compiled = build_graph()
    nodes = compiled.get_graph().nodes
    for node_name in NODE_NAMES:
        assert node_name in nodes


def test_graph_visits_every_node_in_order(settings: Settings, thread: dict[str, Any]) -> None:
    """The declared node order is the order the edges actually produce."""
    compiled = build_graph()
    edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
    for previous, following in pairwise(NODE_NAMES):
        assert (previous, following) in edges


# --------------------------------------------------------------------------------------
# End-to-end on stubs
# --------------------------------------------------------------------------------------


async def test_graph_runs_end_to_end_on_stubs(
    settings: Settings, thread: dict[str, Any], calls: list[str], context: GraphContext
) -> None:
    async with open_graph(settings) as graph:
        result = await graph.ainvoke(
            {"topic": "why the sky is blue"},
            thread,
            durability=DEFAULT_DURABILITY,
            context=context,
        )

    state = VideoState.model_validate(result)

    assert calls == list(NODE_NAMES)
    assert state.completed_nodes == list(NODE_NAMES)
    assert state.topic == "why the sky is blue"
    assert state.script is not None
    assert state.eval_score == 8.0
    assert state.video_path is not None
    assert state.error is None


async def test_caller_supplied_topic_survives_ideation(
    settings: Settings, thread: dict[str, Any], context: GraphContext
) -> None:
    """An operator forcing a topic must not have it overwritten."""
    async with open_graph(settings) as graph:
        result = await graph.ainvoke(
            {"topic": "forced"}, thread, durability=DEFAULT_DURABILITY, context=context
        )
    assert result["topic"] == "forced"


async def test_cost_ledger_accumulates(
    settings: Settings, thread: dict[str, Any], fake_llm: FakeLLM
) -> None:
    """Every charging node appends its own entry and the total is derived, not tracked.

    A per-node ledger rather than a running total is what makes the retry loop's real
    cost visible instead of averaged away (CLAUDE.md section 7).
    """
    fake_llm.cost_usd = 0.002
    context = GraphContext(llm=fake_llm, settings=settings)

    async with open_graph(settings) as graph:
        result = await graph.ainvoke({}, thread, durability=DEFAULT_DURABILITY, context=context)

    state = VideoState.model_validate(result)
    assert [entry.node for entry in state.costs] == ["ideation", "scriptwriter", "assets"]
    assert state.total_cost_usd == pytest.approx(0.004)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


async def test_a_checkpoint_is_written_after_each_node(
    settings: Settings, thread: dict[str, Any], context: GraphContext
) -> None:
    """Each node boundary gets its own checkpoint, and the rows genuinely hit disk.

    A checkpoint whose `next` is node N was committed *after* node N-1 finished and
    before N started — so one such checkpoint per node, plus the initial input one and a
    terminal one with nothing left to run, is exactly "a checkpoint after each node".
    """
    async with open_graph(settings) as graph:
        await graph.ainvoke({}, thread, durability=DEFAULT_DURABILITY, context=context)
        history = [snapshot async for snapshot in graph.aget_state_history(thread)]

    poised_before = [snapshot.next for snapshot in reversed(history)]
    assert poised_before == [
        ("__start__",),
        *[(node_name,) for node_name in NODE_NAMES],
        (),
    ]

    # Not just in memory — the rows are in the SQLite file after the connection closes.
    assert _checkpoint_row_count(settings.sqlite_checkpoint_path) == len(NODE_NAMES) + 2


async def test_every_state_type_round_trips_through_the_serializer(
    settings: Settings,
    thread: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    context: GraphContext,
) -> None:
    """Every custom type in the state is on the checkpoint allowlist.

    This guards a genuinely quiet failure. Because `build_serde` passes an explicit
    allowlist, a type that is missing from `CHECKPOINTED_TYPES` is *blocked* on read —
    LangGraph logs a warning and hands back something that is not the original object,
    so a resumed run would carry corrupted state rather than raise. Nothing else in the
    suite would notice.

    If this fails, a new field type reached `VideoState` without being added to
    `CHECKPOINTED_TYPES` in `state.py`.
    """
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        async with open_graph(settings) as graph:
            await graph.ainvoke(
                {"approved": True}, thread, durability=DEFAULT_DURABILITY, context=context
            )
        async with open_graph(settings) as reopened:
            snapshot = await reopened.aget_state(thread)

    complaints = [
        record.getMessage()
        for record in caplog.records
        if "allowed_msgpack_modules" in record.getMessage()
        or "unregistered type" in record.getMessage()
    ]
    assert complaints == []

    # And the reconstructed objects really are our types, not lookalike dicts.
    restored = VideoState.model_validate(snapshot.values)
    assert isinstance(restored.script, Script)
    assert isinstance(restored.status, RunStatus)
    assert all(isinstance(entry, CostEntry) for entry in restored.costs)


async def test_checkpointer_creates_its_parent_directory(settings: Settings) -> None:
    """`config.py` has no filesystem side effects, so the checkpointer must make the dir."""
    assert not settings.sqlite_checkpoint_path.parent.exists()
    async with open_checkpointer(settings):
        pass
    assert settings.sqlite_checkpoint_path.exists()


async def test_postgres_backend_fails_loudly(tmp_path: Path) -> None:
    """Unimplemented is better than silently falling back to SQLite."""
    postgres = Settings(
        _env_file=None,
        checkpointer=CheckpointerBackend.POSTGRES,
        sqlite_checkpoint_path=tmp_path / "unused.sqlite",
    )
    with pytest.raises(NotImplementedError, match="Postgres"):
        async with open_checkpointer(postgres):
            pass


# --------------------------------------------------------------------------------------
# Resumability — the core Phase 1a claim
# --------------------------------------------------------------------------------------


async def test_resume_after_crash_continues_from_the_last_completed_node(
    settings: Settings,
    thread: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    context: GraphContext,
) -> None:
    """Kill the run inside `render`, rebuild everything, resume: earlier nodes stay done.

    The graph object, the checkpointer, and the database connection are all torn down
    between the two halves — the only thing carried across is the thread ID and the
    SQLite file, which is exactly what surviving a process restart means.
    """
    recorded: list[str] = []
    _install_spies(monkeypatch, recorded, crash_on="render", crash_times=1)

    async with open_graph(settings) as graph:
        with pytest.raises(RuntimeError, match="simulated crash in render"):
            await graph.ainvoke(
                {"topic": "resumable"}, thread, durability=DEFAULT_DURABILITY, context=context
            )

    assert recorded == ["ideation", "scriptwriter", "eval_critic", "assets", "render"]

    # Everything is rebuilt from scratch, as it would be after a restart.
    async with open_graph(settings) as resumed_graph:
        snapshot = await resumed_graph.aget_state(thread)
        assert snapshot.next == ("render",), "should be poised to retry the node that died"
        assert snapshot.values["completed_nodes"] == [
            "ideation",
            "scriptwriter",
            "eval_critic",
            "assets",
        ]

        result = await resumed_graph.ainvoke(
            None, thread, durability=DEFAULT_DURABILITY, context=context
        )

    state = VideoState.model_validate(result)
    assert state.completed_nodes == list(NODE_NAMES)

    # The four nodes that completed before the crash ran exactly once across both halves.
    for node_name in ("ideation", "scriptwriter", "eval_critic", "assets"):
        assert recorded.count(node_name) == 1, f"{node_name} re-ran after the resume"
    # `render` is the one that died, so it is retried — that is correct, not a leak.
    assert recorded.count("render") == 2


async def test_state_survives_a_reopened_database(
    settings: Settings, thread: dict[str, Any], context: GraphContext
) -> None:
    """A completed run's state is readable from a freshly opened checkpointer."""
    async with open_graph(settings) as graph:
        await graph.ainvoke(
            {"topic": "persisted"}, thread, durability=DEFAULT_DURABILITY, context=context
        )

    async with open_graph(settings) as reopened:
        snapshot = await reopened.aget_state(thread)

    assert snapshot.values["topic"] == "persisted"
    assert snapshot.next == ()


# --------------------------------------------------------------------------------------
# The approval gate and publish idempotency (fully wired in Phase 1e; enforced already)
# --------------------------------------------------------------------------------------


async def test_publish_is_skipped_without_approval(
    settings: Settings, thread: dict[str, Any], context: GraphContext
) -> None:
    """No approval means no publish. The gate fails closed (CLAUDE.md 4b)."""
    async with open_graph(settings) as graph:
        result = await graph.ainvoke({}, thread, durability=DEFAULT_DURABILITY, context=context)

    state = VideoState.model_validate(result)
    assert state.approved is None
    assert state.publish_receipt is None
    assert state.content_hash is None
    assert state.status is not RunStatus.PUBLISHED


async def test_publish_proceeds_once_a_human_approved(
    settings: Settings, thread: dict[str, Any], context: GraphContext
) -> None:
    async with open_graph(settings) as graph:
        result = await graph.ainvoke(
            {"approved": True}, thread, durability=DEFAULT_DURABILITY, context=context
        )

    state = VideoState.model_validate(result)
    assert state.status is RunStatus.PUBLISHED
    assert state.publish_receipt is not None
    assert state.publish_receipt.startswith(STUB_RECEIPT_PREFIX)
    assert state.content_hash == state.content_fingerprint()


async def test_republishing_identical_content_does_not_issue_a_new_receipt(
    settings: Settings, thread: dict[str, Any], context: GraphContext
) -> None:
    """A resumed or re-run publish must recognise the content and not double-post."""
    async with open_graph(settings) as graph:
        first = await graph.ainvoke(
            {"approved": True}, thread, durability=DEFAULT_DURABILITY, context=context
        )

    second_thread = {"configurable": {"thread_id": "second-run"}}
    async with open_graph(settings) as graph:
        second = await graph.ainvoke(
            {
                "approved": True,
                "topic": first["topic"],
                "content_hash": first["content_hash"],
                "publish_receipt": first["publish_receipt"],
            },
            second_thread,
            durability=DEFAULT_DURABILITY,
            context=context,
        )

    assert second["publish_receipt"] == first["publish_receipt"]


def test_content_fingerprint_ignores_run_specific_noise() -> None:
    """Two attempts at identical content must hash the same, or dedupe is useless."""
    script = Script(hook="h", body="b", cta="c")
    first = VideoState(topic="t", script=script, video_path=Path("a.mp4"))
    second = VideoState(topic="t", script=script, video_path=Path("b.mp4"))

    assert first.run_id != second.run_id
    assert first.content_fingerprint() == second.content_fingerprint()

    different = VideoState(topic="t", script=Script(hook="different", body="b", cta="c"))
    assert different.content_fingerprint() != first.content_fingerprint()
