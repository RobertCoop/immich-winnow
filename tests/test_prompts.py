"""Tests for Jinja2 system prompt templates, context wiring, and CLI support."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import make_jpeg
from winnow import cli
from winnow.cli import app
from winnow.config import load_settings
from winnow.images import prepare_image
from winnow.judge import (
    BURST_SYSTEM,
    BWS_SYSTEM,
    ITEM_ERRORS,
    PAIR_SYSTEM,
    TRIAGE_SYSTEM,
    Judge,
    JudgeError,
)
from winnow.ledger import Ledger
from winnow.pipeline.finals import run_finals
from winnow.pipeline.rank import run_rank
from winnow.pipeline.triage import run_triage_direct, submit_triage_batch
from winnow.prompts import (
    DEFAULT_TEMPLATE_FILES,
    PROMPT_TASKS,
    PromptError,
    PromptLibrary,
    build_prompt_context,
    safe_settings_dict,
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return ANSI.sub("", text)


def _msg(payload: dict[str, Any], model: str = "claude-test") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model=model,
    )


class _RecordingMessages:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []
        self.submitted: dict[str, list[dict[str, Any]]] = {}
        self.batches = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._handler(kwargs)

    def create_batch(self, requests: list[dict[str, Any]]) -> SimpleNamespace:
        batch_id = f"msgbatch_{len(self.submitted) + 1}"
        self.submitted[batch_id] = list(requests)
        return SimpleNamespace(id=batch_id)


def _make_batch_client() -> Any:
    batches = SimpleNamespace(submitted={})

    def create(requests: list[dict[str, Any]]) -> SimpleNamespace:
        batch_id = f"msgbatch_{len(batches.submitted) + 1}"
        batches.submitted[batch_id] = list(requests)
        return SimpleNamespace(id=batch_id)

    batches.create = create
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _seed_photo(
    ledger: Ledger,
    cache_dir: Path,
    asset_id: str,
    *,
    filename: str | None = None,
    camera: str = "NIKON CORPORATION|NIKON Z 9",
    taken_at: str = "2024-06-01T10:00:00+00:00",
    burst_id: str | None = None,
    width: int = 6000,
    height: int = 4000,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    thumb = cache_dir / f"{asset_id}.jpg"
    thumb.write_bytes(prepare_image(make_jpeg(width=200, height=150)))
    ledger.upsert_assets(
        [
            {
                "id": asset_id,
                "filename": filename or f"{asset_id}.jpg",
                "taken_at": taken_at,
                "camera": camera,
                "width": width,
                "height": height,
                "dhash": 0,
                "thumb_path": str(thumb),
                "immich_rating": None,
                "immich_description": None,
            }
        ]
    )
    if burst_id is not None:
        ledger.assign_burst(burst_id, [asset_id])


# ----------------------------------------------------------------------
# packaged defaults & overrides
# ----------------------------------------------------------------------


def test_default_templates_preserve_legacy_prompts_and_include_json_only() -> None:
    lib = PromptLibrary.default()
    expected = {
        "triage": TRIAGE_SYSTEM,
        "burst": BURST_SYSTEM,
        "bws": BWS_SYSTEM,
        "pair": PAIR_SYSTEM,
    }
    for task, text in expected.items():
        rendered = lib.render(task, model="claude-test", image_count=2)
        assert rendered == text
        assert text.endswith(
            "Reply with JSON matching the schema. JSON only — no prose, no markdown "
            "fences, no commentary before or after.\n"
        )
        assert lib.source(task) is None


def test_partial_override_only_changes_specified_task(settings_env, monkeypatch, tmp_path) -> None:
    custom = tmp_path / "triage.j2"
    custom.write_text("Custom triage prompt for {{ model }}.\n", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(custom))

    settings = load_settings()
    lib = PromptLibrary.from_settings(settings)

    assert lib.source("triage") == custom.resolve()
    assert lib.source("burst") is None
    assert lib.source("bws") is None
    assert lib.source("pair") is None
    assert lib.render("triage", model="m-1", image_count=1) == "Custom triage prompt for m-1.\n"
    assert lib.render("burst", model="m-1", image_count=2) == BURST_SYSTEM
    assert lib.render("bws", model="m-1", image_count=2) == BWS_SYSTEM
    assert lib.render("pair", model="m-1", image_count=2) == PAIR_SYSTEM


def test_empty_prompt_path_env_normalizes_to_none(settings_env, monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", "")
    monkeypatch.setenv("BURST_PROMPT_PATH", "   ")
    settings = load_settings()
    assert settings.triage_prompt_path is None
    assert settings.burst_prompt_path is None
    assert PromptLibrary.from_settings(settings).source("triage") is None


# ----------------------------------------------------------------------
# context variables, inheritance, and includes
# ----------------------------------------------------------------------


def test_templates_condition_on_counts_settings_and_photo_metadata(
    settings_env, monkeypatch, tmp_path
) -> None:
    triage_tpl = tmp_path / "triage.j2"
    triage_tpl.write_text(
        "task={{ task }} model={{ model }} count={{ image_count }} "
        "batch={{ batch_size }} total={{ total_photos }} lib={{ library_size }} "
        "min={{ settings.candidate_score_min }} "
        "cam={{ photo.camera if photo else 'none' }} "
        "size={{ photo.width }}x{{ photo.height }}\n",
        encoding="utf-8",
    )
    pair_tpl = tmp_path / "pair.j2"
    pair_tpl.write_text(
        "A={{ photo_a.filename if photo_a else 'none' }} "
        "B={{ photo_b.filename if photo_b else 'none' }} "
        "count={{ photos | length }}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(triage_tpl))
    monkeypatch.setenv("PAIR_PROMPT_PATH", str(pair_tpl))

    settings = load_settings(candidate_score_min=6)
    lib = PromptLibrary.from_settings(settings)

    triage_out = lib.render(
        "triage",
        model="gemma4",
        image_count=1,
        batch_size=12,
        total_photos=15,
        library_size=90,
        photos=[{"filename": "a.jpg", "camera": "Nikon|Z9", "width": 8256, "height": 5504}],
    )
    assert (
        triage_out
        == "task=triage model=gemma4 count=1 batch=12 total=15 lib=90 "
        "min=6 cam=Nikon|Z9 size=8256x5504\n"
    )

    pair_out = lib.render(
        "pair",
        model="opus",
        image_count=2,
        photos=[{"filename": "first.jpg"}, {"filename": "second.jpg"}],
    )
    assert pair_out == "A=first.jpg B=second.jpg count=2\n"


def test_include_sibling_and_extends_default_precedence(
    settings_env, monkeypatch, tmp_path
) -> None:
    (tmp_path / "shared.j2").write_text("SHARED RUBRIC LINE", encoding="utf-8")
    # Even if a user directory happens to contain a "default/" subfolder,
    # "default/..." must resolve to Winnow's packaged defaults first.
    local_default = tmp_path / "default"
    local_default.mkdir()
    (local_default / "triage.j2").write_text("SHADOWED LOCAL DEFAULT", encoding="utf-8")

    triage_tpl = tmp_path / "triage.j2"
    triage_tpl.write_text(
        '{% extends "default/triage.j2" %}\n'
        "{% block extra %}\n"
        '{% include "shared.j2" %}\n'
        "{% endblock %}\n",
        encoding="utf-8",
    )
    burst_tpl = tmp_path / "burst.j2"
    burst_tpl.write_text(
        'Custom burst with {% include "shared.j2" %}.\n\n{% include "default/_json_only.j2" %}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(triage_tpl))
    monkeypatch.setenv("BURST_PROMPT_PATH", str(burst_tpl))

    settings = load_settings()
    lib = PromptLibrary.from_settings(settings)

    triage_out = lib.render("triage", model="m", image_count=1)
    assert "You are the culling eye" in triage_out
    assert "SHARED RUBRIC LINE" in triage_out
    assert "SHADOWED LOCAL DEFAULT" not in triage_out
    assert triage_out.endswith(
        "Reply with JSON matching the schema. JSON only — no prose, no markdown "
        "fences, no commentary before or after.\n"
    )

    burst_out = lib.render("burst", model="m", image_count=3)
    assert "Custom burst with SHARED RUBRIC LINE." in burst_out
    assert "Reply with JSON matching the schema." in burst_out


# ----------------------------------------------------------------------
# security & fail-fast validation
# ----------------------------------------------------------------------


def test_secrets_excluded_from_settings_context_and_sandbox_blocks_dunder(
    settings_env, monkeypatch, tmp_path
) -> None:
    settings = load_settings()
    safe = safe_settings_dict(settings)
    for secret in ("immich_api_key", "anthropic_api_key", "openai_api_key", "immich_url"):
        assert secret not in safe
    ctx = build_prompt_context("triage", model="m", image_count=1, settings=settings)
    assert "immich_api_key" not in ctx["settings"]
    assert "anthropic_api_key" not in ctx["settings"]

    bad_secret_tpl = tmp_path / "secret.j2"
    bad_secret_tpl.write_text("{{ settings.anthropic_api_key }}", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(bad_secret_tpl))
    with pytest.raises(PromptError, match="triage"):
        PromptLibrary.from_settings(load_settings())

    bad_sandbox_tpl = tmp_path / "sandbox.j2"
    bad_sandbox_tpl.write_text("{{ ''.__class__.__mro__ }}", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(bad_sandbox_tpl))
    with pytest.raises(PromptError, match="triage"):
        PromptLibrary.from_settings(load_settings())


def test_invalid_prompt_templates_fail_fast_with_prompt_error(
    settings_env, monkeypatch, tmp_path
) -> None:
    assert not issubclass(PromptError, (JudgeError, OSError, ValueError))
    assert PromptError not in ITEM_ERRORS

    # missing file
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(tmp_path / "missing.j2"))
    with pytest.raises(PromptError, match="not found"):
        PromptLibrary.from_settings(load_settings())

    # directory path
    subdir = tmp_path / "dir"
    subdir.mkdir()
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(subdir))
    with pytest.raises(PromptError, match="not a file"):
        PromptLibrary.from_settings(load_settings())

    # syntax error
    syntax_tpl = tmp_path / "syntax.j2"
    syntax_tpl.write_text("{% if batch_size > 1 %}oops", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(syntax_tpl))
    with pytest.raises(PromptError, match="failed to load triage prompt"):
        PromptLibrary.from_settings(load_settings())

    # StrictUndefined error
    undef_tpl = tmp_path / "undef.j2"
    undef_tpl.write_text("{{ nonexistent_variable }}", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(undef_tpl))
    with pytest.raises(PromptError, match="invalid triage prompt"):
        PromptLibrary.from_settings(load_settings())


# ----------------------------------------------------------------------
# pipeline integration
# ----------------------------------------------------------------------


def test_run_triage_direct_and_batch_pass_counts_and_metadata(
    settings_env, monkeypatch, tmp_path
) -> None:
    burst_tpl = tmp_path / "burst.j2"
    burst_tpl.write_text(
        "BURST count={{ image_count }} batch={{ batch_size }} total={{ total_photos }} "
        "lib={{ library_size }} names={{ photos | map(attribute='filename') | join(',') }}",
        encoding="utf-8",
    )
    triage_tpl = tmp_path / "triage.j2"
    triage_tpl.write_text(
        "TRIAGE count={{ image_count }} batch={{ batch_size }} total={{ total_photos }} "
        "lib={{ library_size }} cam={{ photo.camera if photo else '' }} "
        "file={{ photo.filename if photo else '' }}",
        encoding="utf-8",
    )
    monkeypatch.setenv("BURST_PROMPT_PATH", str(burst_tpl))
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(triage_tpl))

    settings = load_settings()
    prompts = PromptLibrary.from_settings(settings)
    ledger = Ledger(settings.db_path)
    cache_dir = settings.cache_dir

    _seed_photo(ledger, cache_dir, "b1", filename="burst1.jpg", camera="Nikon|Z9")
    _seed_photo(ledger, cache_dir, "b2", filename="burst2.jpg", camera="Nikon|Z9")
    ledger.assign_burst("burst_grp", ["b1", "b2"])
    _seed_photo(ledger, cache_dir, "s1", filename="single1.jpg", camera="Canon|R5")

    def handler(kwargs: dict[str, Any]) -> SimpleNamespace:
        if "BURST" in kwargs["system"]:
            return _msg({"best_index": 1, "reject_indices": [2], "note": "frame 1 sharper"})
        return _msg(
            {
                "category": "photo",
                "verdict": "candidate",
                "technical_score": 9,
                "reasons": ["sharp"],
                "confidence": "high",
            }
        )

    rec = _RecordingMessages(handler)
    judge = Judge(SimpleNamespace(messages=rec), prompts=prompts)
    stats = run_triage_direct(settings, ledger, judge)

    assert stats.bursts_judged == 1
    assert stats.singles_judged == 2  # b1 (burst winner) + s1
    assert rec.calls[0]["system"] == (
        "BURST count=2 batch=1 total=2 lib=3 names=burst1.jpg,burst2.jpg"
    )
    single_systems = {c["system"] for c in rec.calls[1:]}
    assert single_systems == {
        "TRIAGE count=1 batch=2 total=2 lib=3 cam=Nikon|Z9 file=burst1.jpg",
        "TRIAGE count=1 batch=2 total=2 lib=3 cam=Canon|R5 file=single1.jpg",
    }

    # Verify batch path on fresh ledger too
    batch_ledger = Ledger(tmp_path / "batch.db")
    _seed_photo(batch_ledger, cache_dir, "b1", filename="burst1.jpg", camera="Nikon|Z9")
    _seed_photo(batch_ledger, cache_dir, "b2", filename="burst2.jpg", camera="Nikon|Z9")
    batch_ledger.assign_burst("burst_grp", ["b1", "b2"])
    _seed_photo(batch_ledger, cache_dir, "s1", filename="single1.jpg", camera="Canon|R5")
    batch_client = _make_batch_client()
    batch_judge = Judge(batch_client, prompts=prompts)
    batch_id = submit_triage_batch(settings, batch_ledger, batch_judge)
    assert batch_id is not None
    reqs = batch_client.messages.batches.submitted[batch_id]
    assert reqs[0]["params"]["system"] == (
        "BURST count=2 batch=1 total=2 lib=3 names=burst1.jpg,burst2.jpg"
    )
    assert reqs[1]["params"]["system"] == (
        "TRIAGE count=1 batch=1 total=1 lib=3 cam=Canon|R5 file=single1.jpg"
    )


def test_runtime_prompt_error_propagates_out_of_pipeline(
    settings_env, monkeypatch, tmp_path
) -> None:
    # A template that passes sample_context (camera is a str) but fails at
    # runtime when an asset in the ledger has camera=None.
    bad_runtime_tpl = tmp_path / "triage.j2"
    bad_runtime_tpl.write_text("{{ photo.camera.upper() }}", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(bad_runtime_tpl))

    settings = load_settings()
    prompts = PromptLibrary.from_settings(settings)
    ledger = Ledger(settings.db_path)
    _seed_photo(ledger, settings.cache_dir, "a1", camera="")
    ledger.upsert_assets(
        [
            {
                "id": "a1",
                "filename": "a1.jpg",
                "taken_at": "2024-06-01T10:00:00+00:00",
                "camera": None,
                "width": 100,
                "height": 100,
                "dhash": 0,
                "thumb_path": str(settings.cache_dir / "a1.jpg"),
                "immich_rating": None,
                "immich_description": None,
            }
        ]
    )
    judge = Judge(SimpleNamespace(messages=_RecordingMessages(lambda _kw: None)), prompts=prompts)
    with pytest.raises(PromptError):
        run_triage_direct(settings, ledger, judge)


def test_run_rank_and_run_finals_wire_ordered_and_swapped_photo_context(
    settings_env, monkeypatch, tmp_path
) -> None:
    bws_tpl = tmp_path / "bws.j2"
    bws_tpl.write_text(
        "BWS count={{ image_count }} batch={{ batch_size }} total={{ total_photos }} "
        "lib={{ library_size }} files={{ photos | map(attribute='filename') | join(',') }}",
        encoding="utf-8",
    )
    pair_tpl = tmp_path / "pair.j2"
    pair_tpl.write_text(
        "PAIR count={{ image_count }} batch={{ batch_size }} total={{ total_photos }} "
        "lib={{ library_size }} A={{ photo_a.filename }} B={{ photo_b.filename }}",
        encoding="utf-8",
    )
    monkeypatch.setenv("BWS_PROMPT_PATH", str(bws_tpl))
    monkeypatch.setenv("PAIR_PROMPT_PATH", str(pair_tpl))

    settings = load_settings(bws_set_size=3, bws_appearances=1, finals_rounds=1)
    prompts = PromptLibrary.from_settings(settings)
    ledger = Ledger(settings.db_path)
    for idx, aid_ in enumerate(("p1", "p2", "p3")):
        _seed_photo(
            ledger,
            settings.cache_dir,
            aid_,
            filename=f"photo_{aid_}.jpg",
            taken_at=f"2024-06-01T10:0{idx}:00+00:00",
        )
        ledger.record_triage(
            aid_,
            SimpleNamespace(
                category="photo",
                verdict="candidate",
                technical_score=9,
                reasons=["great"],
                confidence="high",
                caption=None,
                keywords=[],
            ),
            "claude-test",
        )

    bws_rec = _RecordingMessages(
        lambda _kw: _msg({"best_index": 1, "worst_index": 3, "note": "clear order"})
    )
    bws_judge = Judge(SimpleNamespace(messages=bws_rec), prompts=prompts)
    rank_stats = run_rank(settings, ledger, bws_judge)
    assert rank_stats.sets_judged == 1
    assert bws_rec.calls[0]["system"].startswith("BWS count=3 batch=1 total=3 lib=3 files=")
    shown_files = bws_rec.calls[0]["system"].split("files=")[1].split(",")
    assert set(shown_files) == {"photo_p1.jpg", "photo_p2.jpg", "photo_p3.jpg"}

    pair_rec = _RecordingMessages(lambda _kw: _msg({"winner": "A", "note": "stronger"}))
    pair_judge = Judge(SimpleNamespace(messages=pair_rec), prompts=prompts)
    finals_stats = run_finals(settings, ledger, pair_judge, five_count=1)
    assert finals_stats.pairs_judged == 1
    assert len(pair_rec.calls) == 2
    sys_fwd = pair_rec.calls[0]["system"]
    sys_rev = pair_rec.calls[1]["system"]
    assert "PAIR count=2 batch=1 total=3 lib=3 " in sys_fwd
    m_fwd = re.search(r"A=(\S+) B=(\S+)$", sys_fwd)
    m_rev = re.search(r"A=(\S+) B=(\S+)$", sys_rev)
    assert m_fwd is not None and m_rev is not None
    assert (m_rev.group(1), m_rev.group(2)) == (m_fwd.group(2), m_fwd.group(1))


# ----------------------------------------------------------------------
# export & CLI
# ----------------------------------------------------------------------


def test_export_defaults_and_force_guard(tmp_path) -> None:
    export_dir = tmp_path / "exported"
    written = PromptLibrary.export_defaults(export_dir)
    assert sorted(p.name for p in written) == sorted(DEFAULT_TEMPLATE_FILES.values())

    with pytest.raises(PromptError, match="--force"):
        PromptLibrary.export_defaults(export_dir)

    (export_dir / "triage.j2").write_text("modified", encoding="utf-8")
    PromptLibrary.export_defaults(export_dir, force=True)
    assert "You are the culling eye" in (export_dir / "triage.j2").read_text(encoding="utf-8")


def test_cli_prompts_and_check_integration(
    monkeypatch, settings_env, tmp_path
) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli, "console", cli.Console(width=200))
    monkeypatch.setattr(cli, "err_console", cli.Console(width=200, stderr=True))

    # --export works even with missing credentials
    monkeypatch.delenv("IMMICH_URL", raising=False)
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    export_dir = tmp_path / "bare_export"
    res = runner.invoke(app, ["prompts", "--export", str(export_dir)])
    assert res.exit_code == 0, _plain(res.output)
    for name in DEFAULT_TEMPLATE_FILES.values():
        assert (export_dir / name).is_file()

    # Re-export without --force fails with exit code 2
    res_dup = runner.invoke(app, ["prompts", "--export", str(export_dir)])
    assert res_dup.exit_code == 2

    # Restore valid env for inspect & render
    monkeypatch.setenv("IMMICH_URL", "http://immich.test:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    custom = tmp_path / "custom_triage.j2"
    custom.write_text("Custom rendered triage count={{ image_count }}\n", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(custom))

    res_table = runner.invoke(app, ["prompts"])
    assert res_table.exit_code == 0, _plain(res_table.output)
    out = _plain(res_table.stdout)
    for task in PROMPT_TASKS:
        assert task in out
    assert "custom_triage.j2" in out

    res_render = runner.invoke(app, ["prompts", "--render", "triage"])
    assert res_render.exit_code == 0, _plain(res_render.output)
    assert "Custom rendered triage count=1" in _plain(res_render.stdout)

    res_all = runner.invoke(app, ["prompts", "--render", "all"])
    assert res_all.exit_code == 0, _plain(res_all.output)
    all_out = _plain(res_all.stdout)
    for task in PROMPT_TASKS:
        assert f"Prompt: {task}" in all_out

    res_bad_arg = runner.invoke(app, ["prompts", "--render", "unknown"])
    assert res_bad_arg.exit_code != 0

    # `winnow check` checks prompts too
    monkeypatch.setattr(
        cli,
        "Anthropic",
        lambda **_kw: SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="m")]))
        ),
    )
    with respx.mock(assert_all_called=False) as router:
        router.get("http://immich.test:2283/api/server/about").mock(
            return_value=httpx.Response(200, json={"version": "v3.1.0"})
        )
        router.get("http://immich.test:2283/api/users/me/preferences").mock(
            return_value=httpx.Response(200, json={"ratings": {"enabled": True}})
        )
        ok_check = runner.invoke(app, ["check"])
        assert ok_check.exit_code == 0, _plain(ok_check.output)
        assert "Prompts loaded (custom: triage)" in _plain(ok_check.stdout)

        monkeypatch.setenv("TRIAGE_PROMPT_PATH", str(tmp_path / "does_not_exist.j2"))
        bad_check = runner.invoke(app, ["check"])
        assert bad_check.exit_code == 1
        assert "FAIL Prompts:" in _plain(bad_check.stderr)

        bad_triage = runner.invoke(app, ["triage"])
        assert bad_triage.exit_code == 2
        assert "Prompt error." in _plain(bad_triage.stderr)
