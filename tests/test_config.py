"""Tests for `videoagent.config`.

Every construction passes `_env_file=None` so a developer's real `.env` can never leak
into an assertion, and `monkeypatch` is used for the environment so tests stay hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from videoagent.config import (
    CheckpointerBackend,
    LLMTier,
    LogLevel,
    PublishProvider,
    Settings,
    TTSProvider,
    VideoProvider,
    get_settings,
)

# Env vars a stray shell/CI environment might set; cleared before each test.
_MANAGED_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LLM_DRAFT_MODEL",
    "LLM_JUDGE_MODEL",
    "LLM_TIER",
    "LLM_REQUEST_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_BACKOFF_BASE_SECONDS",
    "TTS_PROVIDER",
    "VIDEO_PROVIDER",
    "PUBLISH_PROVIDER",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "PIPER_VOICE_MODEL",
    "PEXELS_API_KEY",
    "PIXABAY_API_KEY",
    "GENERATIVE_VIDEO_API_KEY",
    "YOUTUBE_CLIENT_ID",
    "YOUTUBE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN",
    "AGGREGATOR_API_KEY",
    "AGGREGATOR_BASE_URL",
    "CHECKPOINTER",
    "SQLITE_CHECKPOINT_PATH",
    "POSTGRES_URL",
    "EVAL_SCORE_THRESHOLD",
    "MAX_SCRIPT_RETRIES",
    "OUTPUT_DIR",
    "LOG_LEVEL",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every managed var so tests see a genuinely empty environment."""
    for name in _MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def make_settings(**overrides: object) -> Settings:
    """Construct `Settings` ignoring any on-disk `.env`."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --- defaults ----------------------------------------------------------------------


def test_defaults_load_with_clean_env() -> None:
    settings = make_settings()

    assert settings.openrouter_api_key is None
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_tier is LLMTier.AUTO
    assert settings.tts_provider is TTSProvider.PIPER
    assert settings.video_provider is VideoProvider.STOCK
    assert settings.checkpointer is CheckpointerBackend.SQLITE
    assert settings.sqlite_checkpoint_path == Path("data/checkpoints.sqlite")
    assert settings.postgres_url is None
    assert settings.eval_score_threshold == 7.0
    assert settings.max_script_retries == 2
    assert settings.output_dir == Path("out")
    assert settings.log_level is LogLevel.INFO
    assert settings.langsmith_tracing is False


def test_publish_provider_defaults_to_file() -> None:
    """CLAUDE.md 4b: the default must never post to a real platform."""
    assert make_settings().publish_provider is PublishProvider.FILE


def test_model_ids_have_no_hardcoded_default() -> None:
    """`:free` IDs rotate out, so a literal default would be a silent time-bomb."""
    settings = make_settings()
    assert settings.llm_draft_model is None
    assert settings.llm_judge_model is None


def test_all_secrets_optional_so_import_and_construction_never_need_them() -> None:
    settings = make_settings()
    assert settings.elevenlabs_api_key is None
    assert settings.pexels_api_key is None
    assert settings.pixabay_api_key is None
    assert settings.generative_video_api_key is None
    assert settings.youtube_client_secret is None
    assert settings.aggregator_api_key is None
    assert settings.langsmith_api_key is None


# --- env overrides -------------------------------------------------------------------


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_DRAFT_MODEL", "vendor/small-model:free")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "vendor/big-model:free")
    monkeypatch.setenv("LLM_TIER", "judge")
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("VIDEO_PROVIDER", "generative")
    monkeypatch.setenv("PUBLISH_PROVIDER", "youtube")
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://user:pw@localhost:5432/videoagent")
    monkeypatch.setenv("EVAL_SCORE_THRESHOLD", "8.5")
    monkeypatch.setenv("MAX_SCRIPT_RETRIES", "1")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/videoagent-out")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    settings = make_settings()

    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-test"
    assert settings.llm_draft_model == "vendor/small-model:free"
    assert settings.llm_judge_model == "vendor/big-model:free"
    assert settings.llm_tier is LLMTier.JUDGE
    assert settings.tts_provider is TTSProvider.ELEVENLABS
    assert settings.video_provider is VideoProvider.GENERATIVE
    assert settings.publish_provider is PublishProvider.YOUTUBE
    assert settings.checkpointer is CheckpointerBackend.POSTGRES
    assert settings.postgres_url == "postgresql://user:pw@localhost:5432/videoagent"
    assert settings.eval_score_threshold == 8.5
    assert settings.max_script_retries == 1
    assert settings.output_dir == Path("/tmp/videoagent-out")
    assert settings.log_level is LogLevel.DEBUG
    assert settings.langsmith_tracing is True


def test_env_var_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("publish_provider", "aggregator")
    assert make_settings().publish_provider is PublishProvider.AGGREGATOR


def test_selector_values_tolerate_casing_and_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TTS_PROVIDER", " ElevenLabs ")
    assert make_settings().tts_provider is TTSProvider.ELEVENLABS


def test_blank_model_id_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DRAFT_MODEL", "   ")
    assert make_settings().llm_draft_model is None


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_VAR", "whatever")
    assert make_settings().publish_provider is PublishProvider.FILE


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-supersecret")
    assert "sk-or-supersecret" not in repr(make_settings())


# --- validation errors ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_var", "bad_value"),
    [
        ("PUBLISH_PROVIDER", "twitter"),
        ("TTS_PROVIDER", "coqui"),
        ("VIDEO_PROVIDER", "sora"),
        ("CHECKPOINTER", "mysql"),
        ("LLM_TIER", "cheap"),
        ("LOG_LEVEL", "verbose"),
    ],
)
def test_invalid_enum_value_raises(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize(
    ("env_var", "bad_value"),
    [
        ("EVAL_SCORE_THRESHOLD", "11"),
        ("EVAL_SCORE_THRESHOLD", "-1"),
        ("MAX_SCRIPT_RETRIES", "-1"),
        ("MAX_SCRIPT_RETRIES", "99"),
        ("LLM_MAX_RETRIES", "-1"),
        ("LLM_REQUEST_TIMEOUT_SECONDS", "0"),
    ],
)
def test_out_of_range_numeric_raises(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValidationError):
        make_settings()


# --- use-time accessors --------------------------------------------------------------


def test_resolve_tier_is_identity_under_auto() -> None:
    settings = make_settings()
    assert settings.resolve_tier(LLMTier.DRAFT) is LLMTier.DRAFT
    assert settings.resolve_tier(LLMTier.JUDGE) is LLMTier.JUDGE


def test_resolve_tier_applies_global_override() -> None:
    settings = make_settings(llm_tier=LLMTier.DRAFT)
    assert settings.resolve_tier(LLMTier.JUDGE) is LLMTier.DRAFT


def test_resolve_tier_rejects_auto_from_a_node() -> None:
    with pytest.raises(ValueError, match="concrete tier"):
        make_settings().resolve_tier(LLMTier.AUTO)


def test_require_model_returns_configured_id() -> None:
    settings = make_settings(llm_draft_model="vendor/small:free", llm_judge_model="vendor/big:free")
    assert settings.require_model(LLMTier.DRAFT) == "vendor/small:free"
    assert settings.require_model(LLMTier.JUDGE) == "vendor/big:free"


def test_require_model_raises_at_use_time_not_construction_time() -> None:
    settings = make_settings()  # constructing without model IDs must be fine
    with pytest.raises(ValueError, match="LLM_JUDGE_MODEL is not set"):
        settings.require_model(LLMTier.JUDGE)


def test_require_openrouter_api_key_raises_when_unset() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        make_settings().require_openrouter_api_key()


# --- cached accessor ------------------------------------------------------------------


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # chdir somewhere with no `.env` so the real one cannot influence this test.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        first = get_settings()
        assert first is get_settings()
        assert first.publish_provider is PublishProvider.FILE
    finally:
        get_settings.cache_clear()
