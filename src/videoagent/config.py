"""Application settings.

Reads configuration from the environment (and `.env`) and is the single place that
decides *which* provider implementation is used for each external dependency —
LLM tier, TTS, video assets, publishing. Nodes never select a provider themselves.

Design rules enforced here (CLAUDE.md Section 4):

* **No model ID is ever hardcoded.** `LLM_DRAFT_MODEL` / `LLM_JUDGE_MODEL` have *no*
  default: OpenRouter `:free` IDs are retired without notice, so a baked-in literal is a
  time-bomb that would silently keep "working" against a dead or newly-paid model. They
  are `None` until set, and `require_model()` raises a clear error at the moment a
  provider actually needs one. Importing this module and constructing `Settings()` in an
  empty environment must always succeed, so config errors surface at use, not at import.
* **Secrets are optional at construction.** Every API key defaults to `None` and is typed
  `SecretStr` so it never leaks into a repr, log line, or traceback.
* **`PUBLISH_PROVIDER` defaults to `file`.** A default that posts for real is forbidden
  (CLAUDE.md 4b); real posting is strictly opt-in.
* **Checkpointer swapping is config-only.** `CHECKPOINTER` selects the backend; both the
  SQLite path and the Postgres URL live here so no code change is needed to switch.

This module holds settings and selectors *only*. Provider construction (the factory that
turns `TTSProvider.ELEVENLABS` into an actual client) lives with the providers, so that
`config.py` never imports an SDK.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "CheckpointerBackend",
    "LLMTier",
    "LogLevel",
    "PublishProvider",
    "Settings",
    "TTSProvider",
    "VideoProvider",
    "get_settings",
]


class LLMTier(StrEnum):
    """Which capability tier an LLM call runs on.

    Two tiers exist and both are always in play: `draft` (small/fast, used by the
    high-volume scriptwriter and ideation nodes) and `judge` (distinctly stronger, used
    once per video by the eval critic).

    ``LLM_TIER`` is therefore *not* a "pick the one model to use" switch — that would
    contradict CLAUDE.md Section 5, where the whole point is that the writer and the
    grader are different models. It is a **global tier override**:

    * ``auto`` (default) — every node uses its own natural tier. This is the correct
      production behaviour and the only mode the eval story is valid under.
    * ``draft`` — force *all* LLM calls, including the judge, onto the draft model.
      Useful when the daily free quota is nearly exhausted or for cheap smoke runs.
      Eval scores produced in this mode are not trustworthy.
    * ``judge`` — force all calls onto the judge model. Useful for a one-off
      "how good could this get" comparison, and for isolating whether a bad script is
      the drafter's fault or the prompt's.

    Nodes ask for the tier they want and call `Settings.resolve_tier()`, which applies
    the override. Because the default is `auto`, the override is inert unless explicitly
    set — it can never silently degrade the eval loop.
    """

    AUTO = "auto"
    DRAFT = "draft"
    JUDGE = "judge"


class TTSProvider(StrEnum):
    """Text-to-speech backend. Default is the zero-cost local one."""

    PIPER = "piper"
    ELEVENLABS = "elevenlabs"


class VideoProvider(StrEnum):
    """Visual-asset backend. Generative video is the opt-in alternate."""

    STOCK = "stock"
    GENERATIVE = "generative"


class PublishProvider(StrEnum):
    """Publishing backend.

    `FILE` writes the video plus a metadata JSON to disk and is the mandatory default
    (CLAUDE.md 4b) — the pipeline must never post to a real platform unless an operator
    explicitly opted in.
    """

    FILE = "file"
    YOUTUBE = "youtube"
    AGGREGATOR = "aggregator"


class CheckpointerBackend(StrEnum):
    """LangGraph checkpointer backend. SQLite for local dev, Postgres for deployment."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"


class LogLevel(StrEnum):
    """Standard library logging levels, validated up front."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Every environment-driven knob in the project.

    Constructing this with a completely empty environment must succeed; anything that is
    genuinely required (an API key, a model ID) is `None` here and is demanded by the
    component that needs it, via `require_model()` or an equivalent provider-side check.

    Tests should construct `Settings(_env_file=None, ...)` so a developer's real `.env`
    cannot leak into assertions.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (OpenRouter, two tiers behind one interface) --------------------------
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key. Optional at construction; required before any "
        "real LLM call.",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible base URL for OpenRouter.",
    )
    llm_draft_model: str | None = Field(
        default=None,
        description="Model ID for the draft tier (scriptwriter, ideation). No default "
        "on purpose — `:free` IDs get retired without notice.",
    )
    llm_judge_model: str | None = Field(
        default=None,
        description="Model ID for the judge tier (eval critic). Must be distinctly "
        "stronger than the drafter. No default on purpose.",
    )
    llm_tier: LLMTier = Field(
        default=LLMTier.AUTO,
        description="Global tier override; see LLMTier. `auto` keeps each node on its "
        "own tier and is the only mode under which eval scores are meaningful.",
    )
    llm_request_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Per-request timeout for LLM calls.",
    )
    llm_max_retries: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Max retry attempts on a retryable LLM error (e.g. HTTP 429). "
        "Retries are exponentially backed off — failed calls still burn daily quota.",
    )
    llm_backoff_base_seconds: float = Field(
        default=2.0,
        gt=0,
        description="Base delay for the exponential backoff between LLM retries.",
    )

    # --- Provider selection ---------------------------------------------------------
    tts_provider: TTSProvider = Field(
        default=TTSProvider.PIPER,
        description="Text-to-speech backend.",
    )
    video_provider: VideoProvider = Field(
        default=VideoProvider.STOCK,
        description="Visual-asset backend.",
    )
    publish_provider: PublishProvider = Field(
        default=PublishProvider.FILE,
        description="Publishing backend. MUST default to `file` (CLAUDE.md 4b).",
    )

    # --- TTS credentials / options --------------------------------------------------
    elevenlabs_api_key: SecretStr | None = Field(
        default=None,
        description="ElevenLabs API key; required only when TTS_PROVIDER=elevenlabs.",
    )
    elevenlabs_voice_id: str | None = Field(
        default=None,
        description="ElevenLabs voice ID to narrate with.",
    )
    piper_voice_model: str | None = Field(
        default=None,
        description="Path or name of the local Piper voice model (.onnx).",
    )

    # --- Stock / generative video credentials ---------------------------------------
    pexels_api_key: SecretStr | None = Field(
        default=None,
        description="Pexels API key for stock clip search.",
    )
    pixabay_api_key: SecretStr | None = Field(
        default=None,
        description="Pixabay API key for stock clip search.",
    )
    generative_video_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the generative-video backend "
        "(VIDEO_PROVIDER=generative). Off by default.",
    )

    # --- Publishing credentials -----------------------------------------------------
    youtube_client_id: str | None = Field(
        default=None,
        description="YouTube Data API OAuth client ID; only for PUBLISH_PROVIDER=youtube.",
    )
    youtube_client_secret: SecretStr | None = Field(
        default=None,
        description="YouTube Data API OAuth client secret.",
    )
    youtube_refresh_token: SecretStr | None = Field(
        default=None,
        description="Long-lived OAuth refresh token for the target channel.",
    )
    aggregator_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the TikTok/Reels posting aggregator "
        "(PUBLISH_PROVIDER=aggregator).",
    )
    aggregator_base_url: str | None = Field(
        default=None,
        description="Base URL of the posting aggregator's API.",
    )

    # --- Checkpointer ---------------------------------------------------------------
    checkpointer: CheckpointerBackend = Field(
        default=CheckpointerBackend.SQLITE,
        description="Which LangGraph checkpointer backend to use.",
    )
    sqlite_checkpoint_path: Path = Field(
        default=Path("data/checkpoints.sqlite"),
        description="SQLite file backing the checkpointer when CHECKPOINTER=sqlite.",
    )
    postgres_url: str | None = Field(
        default=None,
        description="Postgres DSN backing the checkpointer when CHECKPOINTER=postgres.",
    )

    # --- Graph tuning ---------------------------------------------------------------
    eval_score_threshold: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="Minimum rubric score (0-10) a script must reach; below this the "
        "conditional edge loops back to the scriptwriter.",
    )
    max_script_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Bounded scriptwriter retries after a failed eval (CLAUDE.md: ~2).",
    )

    # --- Output / observability -----------------------------------------------------
    output_dir: Path = Field(
        default=Path("out"),
        description="Directory the FileProvider writes rendered videos + metadata into.",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Root log level for structured logging.",
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing. Optional; off by default.",
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key; only needed when LANGSMITH_TRACING=true.",
    )
    langsmith_project: str | None = Field(
        default=None,
        description="LangSmith project name to file traces under.",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint.",
    )

    # --- Validators -----------------------------------------------------------------
    @field_validator(
        "llm_tier",
        "tts_provider",
        "video_provider",
        "publish_provider",
        "checkpointer",
        mode="before",
    )
    @classmethod
    def _normalise_selector(cls, value: Any) -> Any:
        """Accept `ElevenLabs`, ` elevenlabs ` etc. for selector enums.

        Genuinely unknown values still fail validation — this only forgives casing and
        stray whitespace, which are the common `.env` typos.
        """
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator(
        "openrouter_base_url",
        "llm_draft_model",
        "llm_judge_model",
        "aggregator_base_url",
        "langsmith_endpoint",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """Treat an empty/whitespace `.env` value as unset rather than as `""`.

        `LLM_DRAFT_MODEL=` in a `.env` file is a placeholder the operator never filled
        in; letting `""` through would defer the failure to a confusing 404 from the
        provider instead of our own clear error.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # --- Use-time accessors ---------------------------------------------------------
    def resolve_tier(self, requested: LLMTier) -> LLMTier:
        """Apply the global `LLM_TIER` override to a node's requested tier.

        Returns `requested` unchanged under the default `auto`. Never returns `AUTO`.
        """
        if requested is LLMTier.AUTO:
            raise ValueError("A node must request a concrete tier (draft or judge), not 'auto'.")
        if self.llm_tier is LLMTier.AUTO:
            return requested
        return self.llm_tier

    def require_model(self, tier: LLMTier) -> str:
        """Return the model ID for `tier`, raising if it was never configured.

        Called by the LLM provider at request time — deliberately not at import or
        construction time, so the app boots (and the test suite runs) without secrets.
        """
        resolved = self.resolve_tier(tier)
        env_var = "LLM_DRAFT_MODEL" if resolved is LLMTier.DRAFT else "LLM_JUDGE_MODEL"
        model = self.llm_draft_model if resolved is LLMTier.DRAFT else self.llm_judge_model
        if not model:
            raise ValueError(
                f"{env_var} is not set, so the {resolved.value} tier has no model. "
                "Model IDs are never hardcoded (OpenRouter ':free' IDs are retired "
                f"without notice) — set {env_var} in your environment or .env file."
            )
        return model

    def require_openrouter_api_key(self) -> str:
        """Return the OpenRouter key, raising a clear error if it is unset."""
        if self.openrouter_api_key is None:
            raise ValueError(
                "OPENROUTER_API_KEY is not set; no LLM call can be made. "
                "Set it in your environment or .env file."
            )
        return self.openrouter_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton.

    Cached so the whole app agrees on one configuration. Tests that need a variant
    should construct `Settings(_env_file=None, ...)` directly, or call
    `get_settings.cache_clear()` after patching the environment.
    """
    return Settings()
