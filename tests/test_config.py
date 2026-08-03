"""Tests for settings loading."""

from __future__ import annotations

from pathlib import Path

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
