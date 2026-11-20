"""Tests for settings loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from winnow.config import Settings, load_settings


def test_settings_from_env(settings_env):
    s = load_settings()
    assert s.immich_url == "http://immich.test:2283"
    assert s.immich_base == "http://immich.test:2283"
    assert s.triage_model == "claude-haiku-4-5"
    assert s.rank_model == "claude-sonnet-5"
    assert s.finals_model == "claude-opus-5"
    assert s.image_edge == 768
    assert s.db_path == Path(settings_env / "test.db")


def test_immich_base_strips_trailing_slash(settings_env, monkeypatch):
    monkeypatch.setenv("IMMICH_URL", "http://immich.test:2283/")
    assert load_settings().immich_base == "http://immich.test:2283"


def test_overrides_win(settings_env):
    s = Settings(triage_model="claude-opus-5", image_edge=512)  # type: ignore[call-arg]
    assert s.triage_model == "claude-opus-5"
    assert s.image_edge == 512


def test_env_overrides(settings_env, monkeypatch):
    monkeypatch.setenv("BWS_SET_SIZE", "6")
    monkeypatch.setenv("BURST_GAP_SECONDS", "5.5")
    s = load_settings()
    assert s.bws_set_size == 6
    assert s.burst_gap_seconds == 5.5


def test_env_example_documents_every_knob():
    """Every setting must be discoverable in .env.example, or it may as well
    not be configurable. Secrets are named there too, just unfilled."""
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    documented = {
        line.lstrip("# ").split("=", 1)[0].strip()
        for line in example.splitlines()
        if "=" in line
    }
    missing = sorted(
        name.upper() for name in Settings.model_fields if name.upper() not in documented
    )
    assert missing == [], f"undocumented settings: {missing}"


def test_compose_file_documents_every_knob():
    """The reference docker-compose.yml must mention every setting by its
    environment-variable name, active or commented, so compose users can
    discover every knob without leaving the file."""
    compose = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    missing = sorted(
        name.upper() for name in Settings.model_fields if name.upper() not in compose
    )
    assert missing == [], f"settings not documented in docker-compose.yml: {missing}"


def test_openai_base_url_works_without_anthropic_api_key(settings_env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    s = load_settings()
    assert s.use_openai is True
    assert s.openai_base_url == "http://localhost:11434/v1"
    assert s.anthropic_api_key is None


def test_missing_both_anthropic_key_and_openai_url_loads_but_fails_llm_check(
    settings_env, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    s = load_settings()
    assert s.anthropic_api_key is None
    assert s.openai_base_url is None
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        s.require_llm_credentials()


def test_effort_thinking_tokens_settings_configure_effort_budgets(settings_env, monkeypatch):
    monkeypatch.setenv("LOW_THINKING_TOKENS", "500")
    monkeypatch.setenv("MEDIUM_THINKING_TOKENS", "1500")
    monkeypatch.setenv("HIGH_THINKING_TOKENS", "3000")
    s = load_settings(openai_base_url="http://localhost:11434/v1")
    assert (
        s.thinking_tokens_for("triage"),
        s.thinking_tokens_for("rank"),
        s.thinking_tokens_for("finals"),
    ) == (500, 1500, 3000)


def test_stage_thinking_effort_and_token_defaults(settings_env):
    s = load_settings(openai_base_url="http://localhost:11434/v1")
    assert (s.triage_effort, s.rank_effort, s.finals_effort) == ("low", "medium", "high")
    assert (
        s.thinking_tokens_for("triage"),
        s.thinking_tokens_for("rank"),
        s.thinking_tokens_for("finals"),
    ) == (1024, 2048, 4096)
    assert (
        s.api_max_tokens_for("triage"),
        s.api_max_tokens_for("rank"),
        s.api_max_tokens_for("finals"),
    ) == (2048, 3072, 5120)


def test_explicit_thinking_tokens_override_effort_and_zero_disables_thinking(settings_env):
    s = load_settings(
        openai_base_url="http://localhost:11434/v1",
        triage_thinking_tokens=0,
        rank_thinking_tokens=512,
        rank_max_tokens=2048,
    )
    assert s.thinking_tokens_for("triage") == 0
    assert s.effort_for("triage") == "none"
    assert s.api_max_tokens_for("triage") == 1024
    assert s.stage_request_kwargs("triage") == {
        "max_tokens": 1024,
        "effort": "none",
        "thinking_tokens": 0,
    }
    assert s.thinking_tokens_for("rank") == 512
    assert s.effort_for("rank") == "medium"
    assert s.api_max_tokens_for("rank") == 2560
    assert s.stage_request_kwargs("rank") == {
        "max_tokens": 2560,
        "effort": "medium",
        "thinking_tokens": 512,
    }
