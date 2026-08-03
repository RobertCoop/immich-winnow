"""Shared fixtures for the Winnow test suite. No network, no secrets."""

from __future__ import annotations

import io

import pytest
from PIL import Image


def make_jpeg(
    width: int = 320, height: int = 240, color: tuple[int, int, int] = (120, 40, 200)
) -> bytes:
    """Return raw JPEG bytes of a solid-color image."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture()
def jpeg_bytes() -> bytes:
    return make_jpeg()


@pytest.fixture()
def settings_env(monkeypatch, tmp_path):
    """Minimal valid environment for Settings, isolated from any real .env."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.test:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "test-immich-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)  # avoid picking up the project .env
    return tmp_path
