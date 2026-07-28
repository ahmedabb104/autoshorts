"""`LLMProvider` protocol and its implementations.

The shipping implementation targets OpenRouter's OpenAI-compatible endpoint
(`https://openrouter.ai/api/v1`) and exposes two capability tiers behind one interface:
a cheap draft tier for the high-volume scriptwriter/ideation calls and a stronger judge
tier for the eval critic. Model IDs come from `LLM_DRAFT_MODEL` / `LLM_JUDGE_MODEL` and
are never hardcoded — `:free` IDs get retired without notice.

Free-tier rate limits are load-bearing, and a failed call still burns daily quota, so
the provider retries a 429 with exponential backoff rather than blind retry.

Populated in Phase 1b.
"""
