"""Shared test fixtures.

The important one is `FakeLLM`. Every test in this suite runs with no network and no
API key: the LLM provider is an interface (CLAUDE.md 4a) precisely so the graph can be
exercised end-to-end without spending free-tier quota, and a test that needed a key
would mean that seam had leaked.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import BaseMessage

from videoagent.config import LLMTier
from videoagent.evals.rubric import CRITERIA
from videoagent.providers.llm import LLMResponse, LLMUsage

__all__ = ["FakeLLM", "LLMCall", "rubric_reply"]


@dataclass
class LLMCall:
    """One recorded call, so tests can assert on prompt shape."""

    tier: LLMTier
    messages: list[BaseMessage]
    temperature: float
    max_tokens: int | None

    @property
    def system(self) -> str:
        """The system prompt text."""
        return str(self.messages[0].content)

    @property
    def user(self) -> str:
        """The user turn's text."""
        return str(self.messages[-1].content)


def rubric_reply(
    *,
    hook_strength: float = 9.0,
    clarity: float = 9.0,
    payoff: float = 9.0,
    factual_risk: bool = False,
    reason: str = "fake judgment",
) -> str:
    """A well-formed judge reply. Defaults to a comfortable pass.

    Built from `CRITERIA` rather than hardcoded keys so that adding a criterion to the
    rubric breaks these tests loudly instead of silently scoring three of four.
    """
    payload: dict[str, object] = {
        criterion.key: {
            "score": {"hook_strength": hook_strength, "clarity": clarity, "payoff": payoff}[
                criterion.key
            ],
            "reason": reason,
        }
        for criterion in CRITERIA
    }
    payload["factual_risk"] = factual_risk
    payload["factual_risk_reason"] = "unverifiable claim" if factual_risk else ""
    return json.dumps(payload)


def default_reply(call: LLMCall) -> str:
    """Answer based on which node is asking, so call *order* is not baked into tests.

    Routing on the system prompt rather than a scripted queue means a test that skips
    ideation (because a topic was supplied) does not desynchronise every later reply.
    """
    if "pick topics" in call.system:
        return "why aeroplane windows have a tiny hole in them"
    if "grade scripts" in call.system:
        return rubric_reply()
    if "write scripts" in call.system:
        return json.dumps(
            {
                "hook": "That tiny hole in your aeroplane window is keeping you alive.",
                "body": "Cabin windows are three panes deep. "
                "The little hole sits in the middle pane and bleeds pressure onto the "
                "outer one, so the outer pane carries the load alone. "
                "If it ever fails, the middle pane silently takes over.",
                "cta": "Follow for more things hiding in plain sight.",
            }
        )
    return "fake reply"


@dataclass
class FakeLLM:
    """An `LLMProvider` that records calls and answers from a pure function.

    Satisfies the Protocol structurally — there is no import from a provider
    implementation here, which is the point.
    """

    reply: Callable[[LLMCall], str] = default_reply
    cost_usd: float = 0.0
    calls: list[LLMCall] = field(default_factory=list)

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        tier: LLMTier,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        call = LLMCall(
            tier=tier,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.calls.append(call)
        return LLMResponse(
            text=self.reply(call),
            model=f"fake/{tier.value}-model",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, cost_usd=self.cost_usd),
        )

    def calls_to(self, fragment: str) -> list[LLMCall]:
        """Recorded calls whose system prompt contains `fragment`."""
        return [call for call in self.calls if fragment in call.system]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
