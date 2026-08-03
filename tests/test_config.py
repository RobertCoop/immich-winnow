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
