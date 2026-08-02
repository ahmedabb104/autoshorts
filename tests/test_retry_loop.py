"""Phase 1c tests for the eval critic and the bounded retry loop.

This is the branch that makes the graph more than a pipeline: a weak script goes back to
the scriptwriter with the critic's notes, a strong one proceeds, and the loop is bounded
so a judge that dislikes everything cannot spin forever burning the daily quota.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from videoagent.config import Settings
from videoagent.graph.context import GraphContext
from videoagent.graph.graph import (
    DEFAULT_DURABILITY,
    RETRY_BRANCH_SOURCE,
    RETRY_TARGET,
    build_graph,
    open_graph,
    route_after_eval,
)
from videoagent.graph.nodes.eval_critic import eval_critic_node
from videoagent.graph.state import RunStatus, Script, VideoState

from .conftest import FakeLLM, LLMCall, rubric_reply

SCRIPT = Script(hook="h", body="b", cta="c")


@dataclass
class FakeRuntime:
    context: GraphContext


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        sqlite_checkpoint_path=tmp_path / "cp.sqlite",
        eval_score_threshold=7.0,
        max_script_retries=2,
    )


@pytest.fixture
def thread() -> dict[str, Any]:
    return {"configurable": {"thread_id": "retry-thread"}}


def runtime_for(llm: FakeLLM, settings: Settings) -> FakeRuntime:
    return FakeRuntime(context=GraphContext(llm=llm, settings=settings))


def judging(reply: Callable[[LLMCall], str]) -> Callable[[LLMCall], str]:
    """Wrap a judge-reply function, leaving the other nodes on their defaults."""
    from .conftest import default_reply

    def route(call: LLMCall) -> str:
        return reply(call) if "grade scripts" in call.system else default_reply(call)

    return route


# --------------------------------------------------------------------------------------
# The critic node
# --------------------------------------------------------------------------------------


async def test_critic_records_the_full_rubric_and_the_scalar(settings: Settings) -> None:
    llm = FakeLLM(reply=judging(lambda call: rubric_reply(hook_strength=8, clarity=6, payoff=7)))
    update = await eval_critic_node(
        VideoState(topic="t", script=SCRIPT), runtime_for(llm, settings)
    )

    assert update["eval_score"] == pytest.approx(0.4 * 8 + 0.3 * 6 + 0.3 * 7)
    assert update["eval_rubric"].criteria[0].key == "hook_strength"
    assert "hook_strength: 8.0/10" in update["eval_notes"]
    assert update["status"] is RunStatus.EVALUATED


async def test_critic_does_not_count_a_retry_when_the_script_passes(settings: Settings) -> None:
    llm = FakeLLM(reply=judging(lambda call: rubric_reply()))
    update = await eval_critic_node(
        VideoState(topic="t", script=SCRIPT), runtime_for(llm, settings)
    )
    assert "retry_count" not in update


async def test_critic_counts_a_retry_when_the_script_fails(settings: Settings) -> None:
    """The critic owns the counter; the router only reads it."""
    llm = FakeLLM(reply=judging(lambda call: rubric_reply(hook_strength=2, clarity=3, payoff=2)))
    update = await eval_critic_node(
        VideoState(topic="t", script=SCRIPT, retry_count=1), runtime_for(llm, settings)
    )
    assert update["retry_count"] == 2


async def test_critic_counts_a_retry_for_a_factual_risk_flag(settings: Settings) -> None:
    llm = FakeLLM(reply=judging(lambda call: rubric_reply(factual_risk=True)))
    update = await eval_critic_node(
        VideoState(topic="t", script=SCRIPT), runtime_for(llm, settings)
    )
    assert update["retry_count"] == 1
    assert "FACTUAL RISK" in update["eval_notes"]


async def test_critic_records_what_the_judgment_cost(settings: Settings) -> None:
    llm = FakeLLM(reply=judging(lambda call: rubric_reply()), cost_usd=0.005)
    update = await eval_critic_node(
        VideoState(topic="t", script=SCRIPT), runtime_for(llm, settings)
    )
    (entry,) = update["costs"]
    assert entry.node == "eval_critic"
    assert entry.detail == "fake/judge-model"


async def test_critic_without_a_script_fails_instead_of_judging_nothing(
    settings: Settings,
) -> None:
    llm = FakeLLM()
    update = await eval_critic_node(VideoState(topic="t"), runtime_for(llm, settings))
    assert llm.calls == []
    assert update["status"] is RunStatus.FAILED


async def test_a_judge_outage_does_not_trigger_a_scriptwriter_retry(settings: Settings) -> None:
    """Rewriting a fine script because the grader was down would burn quota for nothing."""
    llm = FakeLLM(reply=judging(lambda call: "the judge is having a bad day"))
    update = await eval_critic_node(
        VideoState(topic="t", script=SCRIPT), runtime_for(llm, settings)
    )

    assert update["status"] is RunStatus.FAILED
    assert "could not score" in update["error"]
    assert "retry_count" not in update
    assert "eval_rubric" not in update


# --------------------------------------------------------------------------------------
# The router, in isolation
# --------------------------------------------------------------------------------------


def route(state: VideoState, settings: Settings) -> str:
    return route_after_eval(state, runtime_for(FakeLLM(), settings))


def scored(overall: float, *, factual_risk: bool = False) -> Any:
    from videoagent.evals.rubric import CRITERIA, CriterionScore, RubricScore

    return RubricScore(
        criteria=[CriterionScore(key=c.key, score=overall) for c in CRITERIA],
        factual_risk=factual_risk,
    )


def test_a_passing_score_continues(settings: Settings) -> None:
    state = VideoState(eval_rubric=scored(9.0), retry_count=0)
    assert route(state, settings) == "assets"


def test_a_failing_score_goes_back_to_the_scriptwriter(settings: Settings) -> None:
    state = VideoState(eval_rubric=scored(3.0), retry_count=1)
    assert route(state, settings) == RETRY_TARGET


def test_the_retry_budget_is_bounded(settings: Settings) -> None:
    """max_script_retries=2 means the third rejection stops looping."""
    assert route(VideoState(eval_rubric=scored(3.0), retry_count=1), settings) == RETRY_TARGET
    assert route(VideoState(eval_rubric=scored(3.0), retry_count=2), settings) == RETRY_TARGET
    assert route(VideoState(eval_rubric=scored(3.0), retry_count=3), settings) == "assets"


def test_an_unscored_script_continues_rather_than_looping(settings: Settings) -> None:
    assert route(VideoState(eval_rubric=None, retry_count=0), settings) == "assets"


def test_a_factual_risk_flag_routes_to_a_retry(settings: Settings) -> None:
    state = VideoState(eval_rubric=scored(10.0, factual_risk=True), retry_count=0)
    assert route(state, settings) == RETRY_TARGET


def test_zero_retries_configured_never_loops(tmp_path: Path) -> None:
    strict = Settings(
        _env_file=None, sqlite_checkpoint_path=tmp_path / "cp.sqlite", max_script_retries=0
    )
    assert route(VideoState(eval_rubric=scored(1.0), retry_count=1), strict) == "assets"


def test_the_retry_loop_is_visible_in_the_rendered_graph() -> None:
    """The operator console draws this graph; the loop has to show up in it."""
    edges = {(edge.source, edge.target) for edge in build_graph().get_graph().edges}
    assert (RETRY_BRANCH_SOURCE, RETRY_TARGET) in edges
    assert (RETRY_BRANCH_SOURCE, "assets") in edges


# --------------------------------------------------------------------------------------
# End to end through the real graph
# --------------------------------------------------------------------------------------


async def test_a_weak_script_is_rewritten_and_a_good_rewrite_proceeds(
    settings: Settings, thread: dict[str, Any]
) -> None:
    judgments = iter([rubric_reply(hook_strength=2, clarity=3, payoff=2), rubric_reply()])
    llm = FakeLLM(reply=judging(lambda call: next(judgments)))

    async with open_graph(settings) as graph:
        result = await graph.ainvoke(
            {"topic": "t"},
            thread,
            durability=DEFAULT_DURABILITY,
            context=GraphContext(llm=llm, settings=settings),
        )

    state = VideoState.model_validate(result)
    assert state.completed_nodes.count("scriptwriter") == 2
    assert state.completed_nodes.count("eval_critic") == 2
    assert state.retry_count == 1
    assert state.eval_score == pytest.approx(9.0)


async def test_a_strong_script_is_not_rewritten(settings: Settings, thread: dict[str, Any]) -> None:
    llm = FakeLLM(reply=judging(lambda call: rubric_reply()))

    async with open_graph(settings) as graph:
        result = await graph.ainvoke(
            {"topic": "t"},
            thread,
            durability=DEFAULT_DURABILITY,
            context=GraphContext(llm=llm, settings=settings),
        )

    state = VideoState.model_validate(result)
    assert state.completed_nodes.count("scriptwriter") == 1
    assert state.retry_count == 0


async def test_a_judge_that_hates_everything_still_terminates(
    settings: Settings, thread: dict[str, Any]
) -> None:
    """The bound is what stops an unbounded loop from eating the whole daily quota."""
    llm = FakeLLM(reply=judging(lambda call: rubric_reply(hook_strength=1, clarity=1, payoff=1)))

    async with open_graph(settings) as graph:
        result = await graph.ainvoke(
            {"topic": "t"},
            thread,
            durability=DEFAULT_DURABILITY,
            context=GraphContext(llm=llm, settings=settings),
        )

    state = VideoState.model_validate(result)
    # One initial attempt plus max_script_retries=2 rewrites.
    assert state.completed_nodes.count("scriptwriter") == 3
    assert state.retry_count == 3
    # It proceeds carrying its low score; the human approval gate is the real backstop.
    assert state.status is RunStatus.AWAITING_APPROVAL
    assert state.publish_receipt is None


async def test_the_rewrite_prompt_carries_the_critics_notes(
    settings: Settings, thread: dict[str, Any]
) -> None:
    """The whole point of the loop: attempt two must know why attempt one failed."""
    judgments = iter(
        [
            rubric_reply(hook_strength=2, clarity=3, payoff=2, reason="opens on a greeting"),
            rubric_reply(),
        ]
    )
    llm = FakeLLM(reply=judging(lambda call: next(judgments)))

    async with open_graph(settings) as graph:
        await graph.ainvoke(
            {"topic": "t"},
            thread,
            durability=DEFAULT_DURABILITY,
            context=GraphContext(llm=llm, settings=settings),
        )

    rewrite = llm.calls_to("write scripts")[1]
    assert "opens on a greeting" in rewrite.user
    assert "previous attempt" in rewrite.user.lower()


async def test_the_retry_loop_survives_a_resume(settings: Settings, thread: dict[str, Any]) -> None:
    """A run killed mid-retry must not restart the loop from zero and re-spend quota."""
    judgments = iter([rubric_reply(hook_strength=2, clarity=2, payoff=2), rubric_reply()])
    llm = FakeLLM(reply=judging(lambda call: next(judgments)))
    context = GraphContext(llm=llm, settings=settings)

    async with open_graph(settings) as graph:
        # Stop the moment the critic has rejected the first draft.
        await graph.ainvoke(
            {"topic": "t"},
            thread,
            durability=DEFAULT_DURABILITY,
            context=context,
            interrupt_after=[RETRY_BRANCH_SOURCE],
        )

    async with open_graph(settings) as resumed:
        snapshot = await resumed.aget_state(thread)
        assert snapshot.next == (RETRY_TARGET,), "should be poised to rewrite"
        assert snapshot.values["retry_count"] == 1

        result = await resumed.ainvoke(None, thread, durability=DEFAULT_DURABILITY, context=context)

    state = VideoState.model_validate(result)
    assert state.completed_nodes.count("scriptwriter") == 2
    assert state.retry_count == 1
