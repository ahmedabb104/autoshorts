"""`LLMProvider` protocol and its implementations.

The shipping implementation targets OpenRouter's OpenAI-compatible endpoint and exposes
two capability tiers behind one interface: a cheap draft tier for the high-volume
scriptwriter/ideation calls and a stronger judge tier for the eval critic. Model IDs come
from `LLM_DRAFT_MODEL` / `LLM_JUDGE_MODEL` and are resolved per call — never hardcoded,
because `:free` IDs get retired without notice (CLAUDE.md 4a).

**Retry policy is the load-bearing part of this module.** The free tier allows 20
requests/minute and 1,000/day, and a *failed* request still burns daily quota. A blind
retry loop against a 429 therefore does not just fail — it actively destroys the day's
remaining budget. So:

* only 429, 5xx, and transport errors are retried at all; a 400 or 401 is a bug or a bad
  key and retrying it is pure waste,
* delays grow exponentially and are jittered, so a burst of concurrent calls does not
  re-collide in lockstep,
* a `Retry-After` header, when the server sends one, always wins over our own guess,
* retries are bounded by `LLM_MAX_RETRIES`, after which the error propagates.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import httpx
from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import BaseModel, Field

from videoagent.config import LLMTier, Settings, get_settings

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMResponseError",
    "LLMUsage",
    "OpenRouterProvider",
    "extract_json_object",
    "open_llm_provider",
]

#: Status codes worth trying again. Everything else is a client error we own.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Never wait longer than this between attempts, whatever the maths or the server says.
MAX_BACKOFF_SECONDS: float = 60.0


class LLMError(RuntimeError):
    """Any failure talking to the LLM."""


class LLMRateLimitError(LLMError):
    """Rate limited, and we exhausted our retry budget."""


class LLMResponseError(LLMError):
    """The call succeeded but the payload was not what we can use."""


class LLMUsage(BaseModel):
    """Token and cost accounting for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: OpenRouter reports this when `usage.include` is set. Free models report 0.0.
    cost_usd: float = Field(default=0.0, ge=0.0)


class LLMResponse(BaseModel):
    """One completion, plus what it cost and how hard it was to get."""

    text: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    #: Total attempts including the successful one. >1 means we were throttled.
    attempts: int = 1


@runtime_checkable
class LLMProvider(Protocol):
    """What a node is allowed to know about an LLM.

    Deliberately narrow: a node picks a *tier*, not a model, and never sees a client, a
    key, or a retry policy.
    """

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        tier: LLMTier,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a chat completion on the given capability tier."""
        ...


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model's reply.

    Small models wrap JSON in prose or ``` fences even when told not to, and that is not
    worth a retry (which costs quota) when the object is sitting right there. Strict
    parsing is tried first; the brace-matching scan is the fallback.

    Raises `LLMResponseError` if there is no usable object.
    """
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.MULTILINE)
    candidate = _first_balanced_object(fenced)
    if candidate is not None:
        try:
            recovered = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise LLMResponseError(f"Model returned malformed JSON: {text!r}") from error
        if isinstance(recovered, dict):
            return recovered

    raise LLMResponseError(f"Model returned no JSON object: {text!r}")


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced `{...}` span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


class OpenRouterProvider:
    """`LLMProvider` backed by OpenRouter's OpenAI-compatible chat completions endpoint.

    Takes its HTTP client rather than making one, so tests can hand it a transport with
    no network and the FastAPI layer can share one pool. `open_llm_provider()` is the
    convenience wrapper that builds both.
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        # Injected so a backoff test asserts on delays instead of waiting for them.
        self._sleep = sleep
        self._rng = rng or random.Random()

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        tier: LLMTier,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a chat completion, retrying only what is worth retrying."""
        model = self._settings.require_model(tier)
        payload: dict[str, Any] = {
            "model": model,
            "messages": convert_to_openai_messages(list(messages)),
            "temperature": temperature,
            # Ask OpenRouter to report spend so the cost ledger reflects reality
            # rather than an estimate.
            "usage": {"include": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Authorization": f"Bearer {self._settings.require_openrouter_api_key()}"}
        budget = self._settings.llm_max_retries
        last_error: Exception | None = None

        for attempt in range(budget + 1):
            try:
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=headers
                )
            except httpx.TransportError as error:
                last_error = error
                if attempt == budget:
                    break
                await self._sleep(self._backoff_delay(attempt, None))
                continue

            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()
                return self._parse(response.json(), model=model, attempts=attempt + 1)

            last_error = self._error_for(response)
            if attempt == budget:
                break
            await self._sleep(self._backoff_delay(attempt, response.headers.get("Retry-After")))

        raise self._exhausted(last_error, budget)

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        """Seconds to wait before attempt `attempt + 1`.

        A server-supplied `Retry-After` is authoritative — it is the only number that
        reflects when quota actually frees up. Otherwise: exponential growth with jitter
        in [0.5, 1.0] of the nominal delay, so concurrent callers spread out instead of
        retrying in lockstep and colliding again.
        """
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), MAX_BACKOFF_SECONDS)
            except ValueError:
                # A date-formatted Retry-After is legal HTTP; fall through to our own
                # backoff rather than guessing at a clock skew.
                pass

        nominal = self._settings.llm_backoff_base_seconds * (2**attempt)
        return min(nominal * self._rng.uniform(0.5, 1.0), MAX_BACKOFF_SECONDS)

    @staticmethod
    def _error_for(response: httpx.Response) -> LLMError:
        detail = response.text[:200]
        if response.status_code == 429:
            return LLMRateLimitError(f"OpenRouter rate limited (429): {detail}")
        return LLMError(f"OpenRouter returned {response.status_code}: {detail}")

    @staticmethod
    def _exhausted(last_error: Exception | None, budget: int) -> LLMError:
        summary = f"giving up after {budget + 1} attempt(s)"
        if isinstance(last_error, LLMRateLimitError):
            return LLMRateLimitError(f"{last_error} — {summary}")
        if isinstance(last_error, LLMError):
            return LLMError(f"{last_error} — {summary}")
        return LLMError(f"OpenRouter transport failure ({last_error}) — {summary}")

    @staticmethod
    def _parse(body: dict[str, Any], *, model: str, attempts: int) -> LLMResponse:
        try:
            choices = body["choices"]
            text = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMResponseError(f"Unexpected completion payload: {body!r}") from error

        if not isinstance(text, str):
            raise LLMResponseError(f"Completion content was not text: {text!r}")

        usage_body = body.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(usage_body.get("prompt_tokens") or 0),
            completion_tokens=int(usage_body.get("completion_tokens") or 0),
            cost_usd=float(usage_body.get("cost") or 0.0),
        )
        return LLMResponse(
            text=text,
            model=body.get("model") or model,
            usage=usage,
            attempts=attempts,
        )


@asynccontextmanager
async def open_llm_provider(settings: Settings | None = None) -> AsyncIterator[LLMProvider]:
    """Open an `OpenRouterProvider` with its own HTTP client, and close it on exit."""
    settings = settings or get_settings()
    async with httpx.AsyncClient(
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_request_timeout_seconds,
    ) as client:
        yield OpenRouterProvider(settings, client)
