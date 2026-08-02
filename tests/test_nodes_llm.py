"""Phase 1b tests for the two LLM-backed nodes: ideation and scriptwriter.

These assert the two things that actually break in production — what we put *into* the
prompt, and whether we can parse what comes back out — against a fake provider, so no
quota is spent and no key is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from videoagent.config import LLMTier, Settings
from videoagent.graph.context import GraphContext
from videoagent.graph.nodes.ideation import SEED_TOPICS, ideation_node
from videoagent.graph.nodes.scriptwriter import scriptwriter_node
from videoagent.graph.state import RunStatus, Script, VideoState

from .conftest import FakeLLM, LLMCall


@dataclass
class FakeRuntime:
    """The slice of LangGraph's `Runtime` that a node actually touches."""

    context: GraphContext


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def runtime_for(llm: FakeLLM, settings: Settings) -> FakeRuntime:
    return FakeRuntime(context=GraphContext(llm=llm, settings=settings))


# --------------------------------------------------------------------------------------
# Ideation
# --------------------------------------------------------------------------------------


async def test_ideation_asks_the_draft_tier_and_offers_the_seed_list(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    update = await ideation_node(VideoState(), runtime_for(fake_llm, settings))

    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    # Ideation is high-volume, so it must never reach for the judge tier.
    assert call.tier is LLMTier.DRAFT
    # Every seed is offered; the model chooses, we do not pre-filter.
    for topic in SEED_TOPICS:
        assert topic in call.user
    assert update["topic"] == "why aeroplane windows have a tiny hole in them"
    assert update["status"] is RunStatus.IDEATED


async def test_ideation_skips_the_llm_when_a_topic_was_supplied(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    """Forcing a topic from the console must not spend quota re-deriving it."""
    state = VideoState(topic="operator's choice")
    update = await ideation_node(state, runtime_for(fake_llm, settings))

    assert fake_llm.calls == []
    assert "topic" not in update
    assert update["status"] is RunStatus.IDEATED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a clean topic", "a clean topic"),
        ('"a quoted topic"', "a quoted topic"),
        ("Topic: a labelled topic", "a labelled topic"),
        ("the first line\nand some rambling after it", "the first line"),
        ("  padded  \n", "padded"),
    ],
    ids=["clean", "quoted", "labelled", "multiline", "padded"],
)
async def test_ideation_cleans_up_how_models_actually_reply(
    settings: Settings, raw: str, expected: str
) -> None:
    llm = FakeLLM(reply=lambda call: raw)
    update = await ideation_node(VideoState(), runtime_for(llm, settings))
    assert update["topic"] == expected


async def test_ideation_truncates_a_runaway_reply(settings: Settings) -> None:
    llm = FakeLLM(reply=lambda call: " ".join(["word"] * 200))
    update = await ideation_node(VideoState(), runtime_for(llm, settings))
    assert len(update["topic"].split()) == 25


async def test_ideation_fails_loudly_on_an_empty_reply(settings: Settings) -> None:
    llm = FakeLLM(reply=lambda call: "   \n  ")
    update = await ideation_node(VideoState(), runtime_for(llm, settings))

    assert update["status"] is RunStatus.FAILED
    assert "no usable topic" in update["error"]
    assert "topic" not in update


async def test_ideation_records_what_it_spent(settings: Settings) -> None:
    llm = FakeLLM(cost_usd=0.0011)
    update = await ideation_node(VideoState(), runtime_for(llm, settings))

    (entry,) = update["costs"]
    assert entry.node == "ideation"
    assert entry.provider == "openrouter"
    assert entry.usd == pytest.approx(0.0011)
    # The model that actually served the call, for a truthful cost-per-video breakdown.
    assert entry.detail == "fake/draft-model"


# --------------------------------------------------------------------------------------
# Scriptwriter
# --------------------------------------------------------------------------------------


async def test_scriptwriter_produces_a_script_on_the_draft_tier(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    state = VideoState(topic="why aeroplane windows have a tiny hole")
    update = await scriptwriter_node(state, runtime_for(fake_llm, settings))

    call = fake_llm.calls[0]
    assert call.tier is LLMTier.DRAFT
    assert "why aeroplane windows have a tiny hole" in call.user

    script = update["script"]
    assert isinstance(script, Script)
    assert script.hook.startswith("That tiny hole")
    assert update["status"] is RunStatus.DRAFTED


async def test_scriptwriter_refuses_to_run_without_a_topic(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    update = await scriptwriter_node(VideoState(), runtime_for(fake_llm, settings))

    assert fake_llm.calls == []
    assert update["status"] is RunStatus.FAILED
    assert "without a topic" in update["error"]


async def test_a_first_attempt_prompt_carries_no_retry_baggage(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    await scriptwriter_node(VideoState(topic="t"), runtime_for(fake_llm, settings))
    assert "previous attempt" not in fake_llm.calls[0].user.lower()


async def test_a_retry_prompt_includes_the_rejected_script_and_the_critic_notes(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    """A retry that cannot see why it failed just resamples and burns quota."""
    state = VideoState(
        topic="t",
        retry_count=1,
        eval_score=4.5,
        eval_notes="the hook buries the surprise in the second sentence",
        script=Script(hook="weak hook", body="weak body", cta="weak cta"),
    )
    await scriptwriter_node(state, runtime_for(fake_llm, settings))

    prompt = fake_llm.calls[0].user
    assert "4.5/10" in prompt
    assert "the hook buries the surprise" in prompt
    assert "weak hook" in prompt
    assert "Do not lightly reword" in prompt


async def test_a_retry_without_recorded_notes_still_explains_itself(
    fake_llm: FakeLLM, settings: Settings
) -> None:
    state = VideoState(
        topic="t",
        retry_count=1,
        script=Script(hook="h", body="b", cta="c"),
    )
    await scriptwriter_node(state, runtime_for(fake_llm, settings))
    assert "no specific feedback was recorded" in fake_llm.calls[0].user


@pytest.mark.parametrize(
    "raw",
    [
        '{"hook": "H", "body": "B", "cta": "C"}',
        '```json\n{"hook": "H", "body": "B", "cta": "C"}\n```',
        'Here you go:\n{"hook": "H", "body": "B", "cta": "C"}',
    ],
    ids=["bare", "fenced", "prose-wrapped"],
)
async def test_scriptwriter_parses_the_replies_small_models_really_send(
    settings: Settings, raw: str
) -> None:
    llm = FakeLLM(reply=lambda call: raw)
    update = await scriptwriter_node(VideoState(topic="t"), runtime_for(llm, settings))

    script = update["script"]
    assert (script.hook, script.body, script.cta) == ("H", "B", "C")


@pytest.mark.parametrize(
    "raw",
    [
        "just prose, no json at all",
        '{"hook": "H", "body": "B"}',
        '{"hook": "", "body": "B", "cta": "C"}',
    ],
    ids=["no-json", "missing-key", "empty-value"],
)
async def test_an_unusable_reply_fails_the_run_instead_of_shipping_a_half_script(
    settings: Settings, raw: str
) -> None:
    llm = FakeLLM(reply=lambda call: raw)
    update = await scriptwriter_node(VideoState(topic="t"), runtime_for(llm, settings))

    assert update["status"] is RunStatus.FAILED
    assert "script" not in update


async def test_spend_is_recorded_even_when_the_reply_was_unusable(settings: Settings) -> None:
    """A run that burned quota on garbage should show that in its cost, not hide it."""
    llm = FakeLLM(reply=lambda call: "not json", cost_usd=0.003)
    update = await scriptwriter_node(VideoState(topic="t"), runtime_for(llm, settings))

    assert update["status"] is RunStatus.FAILED
    (entry,) = update["costs"]
    assert entry.usd == pytest.approx(0.003)


async def test_scriptwriter_strips_whitespace_from_parsed_fields(settings: Settings) -> None:
    payload: dict[str, Any] = {"hook": "  H  ", "body": "\nB\n", "cta": " C "}
    llm = FakeLLM(reply=lambda call: json.dumps(payload))
    update = await scriptwriter_node(VideoState(topic="t"), runtime_for(llm, settings))

    script = update["script"]
    assert (script.hook, script.body, script.cta) == ("H", "B", "C")


# --------------------------------------------------------------------------------------
# The interface boundary itself
# --------------------------------------------------------------------------------------


def test_no_node_imports_a_vendor_sdk() -> None:
    """CLAUDE.md 4a: a node that imports an SDK client is a bug.

    Cheap to check and easy to violate by reflex, which is exactly the kind of rule worth
    enforcing mechanically rather than in review.
    """
    import videoagent.graph.nodes as nodes_package

    banned = ("httpx", "openai", "elevenlabs", "openrouter", "anthropic")
    package_dir = Path(nodes_package.__path__[0])
    offenders: list[str] = []

    for source_path in sorted(package_dir.glob("*.py")):
        for line in source_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            tokens = stripped.replace(",", " ").replace(".", " ").split()
            if any(name in tokens for name in banned):
                offenders.append(f"{source_path.name}: {stripped}")

    assert offenders == []


def test_the_fake_provider_records_the_call_shape(fake_llm: FakeLLM) -> None:
    """Guards the fixture the rest of these tests lean on."""
    call = LLMCall(tier=LLMTier.DRAFT, messages=[], temperature=0.5, max_tokens=None)
    assert call.tier is LLMTier.DRAFT
    assert fake_llm.calls == []
