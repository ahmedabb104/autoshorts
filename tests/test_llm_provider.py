"""Phase 1b tests for the OpenRouter provider.

The backoff tests are the ones that matter. OpenRouter's free tier bills a *failed*
request against the daily quota, so a blind retry loop does not merely fail — it spends
the rest of the day's budget discovering that it is rate limited. These assert that we
back off, that we only retry what is worth retrying, and that a server-supplied
`Retry-After` beats our own guess.

No test here touches the network: every one drives the provider through an
`httpx.MockTransport`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from videoagent.config import LLMTier, Settings
from videoagent.providers.llm import (
    MAX_BACKOFF_SECONDS,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponseError,
    OpenRouterProvider,
    extract_json_object,
)


class RecordingSleep:
    """Stands in for `asyncio.sleep` so backoff is asserted, not waited on."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class NoJitter(random.Random):
    """Pins jitter to its upper bound so delays are exactly the nominal schedule."""

    def uniform(self, a: float, b: float) -> float:
        return b


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "openrouter_api_key": "test-key",
        "llm_draft_model": "vendor/draft-model:free",
        "llm_judge_model": "vendor/judge-model:free",
        "llm_max_retries": 3,
        "llm_backoff_base_seconds": 2.0,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


def make_provider(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    settings: Settings | None = None,
    sleep: RecordingSleep | None = None,
) -> tuple[OpenRouterProvider, RecordingSleep]:
    recorder = sleep or RecordingSleep()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://openrouter.test/api/v1",
    )
    provider = OpenRouterProvider(
        settings or make_settings(),
        client,
        sleep=recorder,
        rng=NoJitter(),
    )
    return provider, recorder


def completion(text: str, *, cost: float = 0.0, model: str = "vendor/draft-model:free") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "cost": cost},
    }


PROMPT = [SystemMessage("be terse"), HumanMessage("hello")]


# --------------------------------------------------------------------------------------
# The happy path and the request we actually send
# --------------------------------------------------------------------------------------


def test_openrouter_provider_satisfies_the_protocol() -> None:
    provider, _ = make_provider(lambda request: httpx.Response(200, json=completion("hi")))
    assert isinstance(provider, LLMProvider)


async def test_request_shape_carries_the_tier_model_and_auth() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=completion("hi"))

    provider, _ = make_provider(handler)
    await provider.complete(PROMPT, tier=LLMTier.JUDGE, temperature=0.1, max_tokens=99)

    body = seen["body"]
    assert seen["url"] == "https://openrouter.test/api/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    # The judge tier resolves to the judge model — the node never names a model itself.
    assert body["model"] == "vendor/judge-model:free"
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 99
    # Ask OpenRouter to report spend, so the cost ledger is measured and not estimated.
    assert body["usage"] == {"include": True}
    # `response_format` is deliberately absent — many `:free` models reject it outright.
    assert "response_format" not in body


async def test_usage_and_cost_are_read_back() -> None:
    provider, _ = make_provider(
        lambda request: httpx.Response(200, json=completion("hi", cost=0.00042))
    )
    response = await provider.complete(PROMPT, tier=LLMTier.DRAFT)

    assert response.text == "hi"
    assert response.usage.cost_usd == pytest.approx(0.00042)
    assert response.usage.prompt_tokens == 11
    assert response.attempts == 1


async def test_a_missing_model_id_fails_with_a_useful_message() -> None:
    """No hardcoded fallback model — the failure must name the variable to set."""
    provider, _ = make_provider(
        lambda request: httpx.Response(200, json=completion("hi")),
        settings=make_settings(llm_draft_model=None),
    )
    with pytest.raises(ValueError, match="LLM_DRAFT_MODEL"):
        await provider.complete(PROMPT, tier=LLMTier.DRAFT)


# --------------------------------------------------------------------------------------
# Backoff — the quota-protecting behaviour
# --------------------------------------------------------------------------------------


async def test_a_429_backs_off_exponentially_rather_than_retrying_blind() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] <= 3:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=completion("finally"))

    provider, sleep = make_provider(handler)
    response = await provider.complete(PROMPT, tier=LLMTier.DRAFT)

    assert response.text == "finally"
    assert response.attempts == 4
    # base 2.0, doubling: not a tight loop, and each wait is longer than the last.
    assert sleep.delays == [2.0, 4.0, 8.0]


async def test_retries_are_bounded_and_then_the_error_propagates() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(429, text="always limited")

    provider, sleep = make_provider(handler)
    with pytest.raises(LLMRateLimitError, match="giving up after 4 attempt"):
        await provider.complete(PROMPT, tier=LLMTier.DRAFT)

    # llm_max_retries=3 means 4 attempts total and 3 waits — never an unbounded loop.
    assert attempts["count"] == 4
    assert len(sleep.delays) == 3


async def test_retry_after_header_overrides_our_own_backoff() -> None:
    """The server knows when quota frees up; our exponential guess does not."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "17"}, text="wait")
        return httpx.Response(200, json=completion("ok"))

    provider, sleep = make_provider(handler)
    await provider.complete(PROMPT, tier=LLMTier.DRAFT)
    assert sleep.delays == [17.0]


async def test_an_unparseable_retry_after_falls_back_to_our_backoff() -> None:
    """A date-formatted Retry-After is legal HTTP and must not crash the retry path."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(
                429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, text="wait"
            )
        return httpx.Response(200, json=completion("ok"))

    provider, sleep = make_provider(handler)
    await provider.complete(PROMPT, tier=LLMTier.DRAFT)
    assert sleep.delays == [2.0]


async def test_backoff_is_capped() -> None:
    provider, _ = make_provider(lambda request: httpx.Response(200, json=completion("hi")))
    assert provider._backoff_delay(attempt=40, retry_after=None) == MAX_BACKOFF_SECONDS
    assert provider._backoff_delay(attempt=0, retry_after="99999") == MAX_BACKOFF_SECONDS


async def test_jitter_spreads_retries_within_bounds() -> None:
    """Concurrent callers must not re-collide in lockstep after a shared 429."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion("hi"))),
        base_url="https://openrouter.test/api/v1",
    )
    provider = OpenRouterProvider(make_settings(), client, rng=random.Random(0))

    delays = {provider._backoff_delay(attempt=2, retry_after=None) for _ in range(50)}
    assert len(delays) > 1, "identical delays would defeat the point of jitter"
    assert all(4.0 <= delay <= 8.0 for delay in delays)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
async def test_client_errors_are_not_retried(status: int) -> None:
    """Retrying a bad key or a malformed request only burns quota."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(status, text="nope")

    provider, sleep = make_provider(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete(PROMPT, tier=LLMTier.DRAFT)

    assert attempts["count"] == 1
    assert sleep.delays == []


async def test_transport_errors_are_retried() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=completion("recovered"))

    provider, sleep = make_provider(handler)
    response = await provider.complete(PROMPT, tier=LLMTier.DRAFT)

    assert response.text == "recovered"
    assert sleep.delays == [2.0]


async def test_a_persistent_transport_failure_reports_the_cause() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection reset")

    provider, _ = make_provider(handler)
    with pytest.raises(LLMError, match="transport failure"):
        await provider.complete(PROMPT, tier=LLMTier.DRAFT)


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------


async def test_a_malformed_payload_raises_rather_than_returning_nonsense() -> None:
    provider, _ = make_provider(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMResponseError):
        await provider.complete(PROMPT, tier=LLMTier.DRAFT)


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here is the JSON:\n{"a": 1}\nHope that helps.',
        '  \n {"a": 1}  ',
    ],
    ids=["bare", "fenced-json", "fenced-plain", "surrounded-by-prose", "whitespace"],
)
def test_extract_json_object_survives_how_small_models_actually_reply(raw: str) -> None:
    """Re-prompting to fix a stray ``` fence would cost quota for nothing."""
    assert extract_json_object(raw) == {"a": 1}


def test_extract_json_object_handles_braces_inside_strings() -> None:
    assert extract_json_object('{"a": "not } the end", "b": 2}') == {"a": "not } the end", "b": 2}


def test_extract_json_object_handles_nesting() -> None:
    assert extract_json_object('prose {"a": {"b": 1}} more') == {"a": {"b": 1}}


@pytest.mark.parametrize(
    "raw",
    ["no json here", "", "[1, 2, 3]", '{"unterminated": '],
    ids=["prose", "empty", "array", "truncated"],
)
def test_extract_json_object_rejects_what_it_cannot_use(raw: str) -> None:
    with pytest.raises(LLMResponseError):
        extract_json_object(raw)
