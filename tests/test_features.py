"""Tests for sticky five-stars, the best-photos album, and write-back toggles."""

from __future__ import annotations

import json
from itertools import combinations

import pytest
import respx
from httpx import Response

from winnow.config import load_settings
from winnow.immich import ImmichClient
from winnow.ledger import Ledger
from winnow.pipeline import writeback
from winnow.pipeline.finals import run_finals

BASE = "http://immich.test:2283"

A, B, C, D, E = "aaa1", "bbb2", "ccc3", "ddd4", "eee5"


def _asset_row(asset_id: str, rating: int | None = None) -> dict:
    return {
        "id": asset_id,
        "filename": f"{asset_id}.jpg",
        "taken_at": "2024-06-01T10:00:00+00:00",
        "camera": "Apple|iPhone",
        "width": 100,
        "height": 100,
        "dhash": None,
        "thumb_path": None,
        "immich_rating": rating,
    }


def _seed_ranked_ledger(tmp_path, *, external_five: bool = False) -> Ledger:
    """A ledger where fresh pairs rank A > B > C > D (and E last, if present).

    Every possible finals pair is pre-recorded, so ``run_finals`` plays no
    rounds and calls no judge — the star logic runs on existing evidence.
    """
    ledger = Ledger(tmp_path / "features.db")
    ids = [A, B, C, D] + ([E] if external_five else [])
    ledger.upsert_assets(
        [_asset_row(i, rating=5 if (external_five and i == E) else None) for i in ids]
    )
    order = ids  # strongest first
    for x, y in combinations(order, 2):
        # earlier in `order` beats later, twice for weight
        ledger.record_pair(x, y, x, "rank", "test-model")
        ledger.record_pair(y, x, x, "finals", "test-model")
    ledger.upsert_scores(
        {aid: (float(len(order) - idx), idx + 1) for idx, aid in enumerate(order)}
    )
    return ledger


class _NeverCalledJudge:
    """Every finals pairing is already played; judging would be a bug."""

    def __getattr__(self, name):  # pragma: no cover - only fires on regression
        raise AssertionError("judge should not be consulted")


@pytest.fixture()
def settings(settings_env):
    return load_settings()


def test_prior_five_star_is_sticky_by_default(settings, tmp_path):
    ledger = _seed_ranked_ledger(tmp_path)
    # A previous run crowned D; the fresh ranking puts D dead last.
    ledger.set_stars(D, 5)
    ledger.set_decision(D, "five_star", {"stars": 5, "rank": 1})

    stats = run_finals(settings, ledger, _NeverCalledJudge(), five_count=1)

    stars = {r["asset_id"]: r.get("stars") for r in ledger.score_rows()}
    assert stars[A] == 5  # the fresh winner is still crowned
    assert stars[D] == 5  # ...and D keeps its old crown
    assert stats.protected == 1
    buckets = {r["asset_id"]: r["bucket"] for r in ledger.decisions()}
    assert buckets[D] == "five_star"


def test_allow_demotions_lets_a_five_star_fall(settings, tmp_path):
    ledger = _seed_ranked_ledger(tmp_path)
    ledger.set_stars(D, 5)
    ledger.set_decision(D, "five_star", {"stars": 5, "rank": 1})

    stats = run_finals(settings, ledger, _NeverCalledJudge(), five_count=1, allow_demotions=True)

    stars = {r["asset_id"]: r.get("stars") for r in ledger.score_rows()}
    assert stars[A] == 5
    assert stars.get(D) in (None, 4)  # no longer pinned at five
    assert stats.protected == 0


def test_externally_rated_five_never_gets_a_lower_rating(settings, tmp_path):
    ledger = _seed_ranked_ledger(tmp_path, external_five=True)
    # E is rated five in Immich itself but ranks last here. With a big four
    # band E would otherwise be written down to four stars.
    run_finals(settings, ledger, _NeverCalledJudge(), five_count=1, four_frac=1.0)

    buckets = {r["asset_id"]: r["bucket"] for r in ledger.decisions()}
    assert E not in buckets  # left untouched, not demoted to four_star
    assert buckets[B] == "four_star"  # the band itself still exists


def _seed_starred_ledger(tmp_path) -> Ledger:
    ledger = Ledger(tmp_path / "album.db")
    ledger.upsert_assets([_asset_row(i) for i in (A, B, C)])
    ledger.set_decision(A, "five_star", {"stars": 5, "rank": 1})
    ledger.set_decision(B, "four_star", {"stars": 4, "rank": 2})
    ledger.set_decision(C, "reject", {})
    return ledger


def test_album_action_collects_five_stars_only_by_default(tmp_path):
    ledger = _seed_starred_ledger(tmp_path)
    actions = writeback.plan(ledger, album="Five-Stars", album_min_stars=5)
    album = [a for a in actions if a.bucket == "album"]
    assert len(album) == 1
    assert album[0].group == "stars"
    assert album[0].api_ops == [{"op": "album", "name": "Five-Stars", "asset_ids": [A]}]


def test_album_min_stars_four_includes_both_bands(tmp_path):
    ledger = _seed_starred_ledger(tmp_path)
    actions = writeback.plan(ledger, album="Best", album_min_stars=4)
    (album,) = [a for a in actions if a.bucket == "album"]
    assert album.api_ops[0]["asset_ids"] == sorted([A, B])


def test_no_album_action_when_disabled(tmp_path):
    ledger = _seed_starred_ledger(tmp_path)
    assert all(a.bucket != "album" for a in writeback.plan(ledger, album=None))


def test_album_includes_already_applied_stars(tmp_path):
    ledger = _seed_starred_ledger(tmp_path)
    ledger.mark_applied([A])
    (album,) = [a for a in writeback.plan(ledger, album="X") if a.bucket == "album"]
    assert album.api_ops[0]["asset_ids"] == [A]


def test_favorite_toggle_off_drops_is_favorite(tmp_path):
    ledger = _seed_starred_ledger(tmp_path)
    actions = writeback.plan(ledger, favorite_five=False)
    (crown,) = [a for a in actions if a.bucket == "five_star"]
    update = crown.api_ops[0]
    assert update["rating"] == 5
    assert "is_favorite" not in update
    on = next(a for a in writeback.plan(ledger) if a.bucket == "five_star")
    assert on.api_ops[0]["is_favorite"] is True


@respx.mock
def test_apply_creates_album_and_adds_members(settings, tmp_path):
    ledger = _seed_starred_ledger(tmp_path)
    respx.put(f"{BASE}/api/tags").mock(
        return_value=Response(
            200, json=[{"id": "t1", "name": "winnow/best", "value": "winnow/best"}]
        )
    )
    respx.put(f"{BASE}/api/tags/t1/assets").mock(return_value=Response(200, json=[]))
    update_route = respx.put(url__regex=rf"{BASE}/api/assets/.*").mock(
        return_value=Response(200, json={})
    )
    list_route = respx.get(f"{BASE}/api/albums").mock(return_value=Response(200, json=[]))
    create_route = respx.post(f"{BASE}/api/albums").mock(
        return_value=Response(201, json={"id": "alb1", "albumName": "Five-Stars"})
    )
    add_route = respx.put(f"{BASE}/api/albums/alb1/assets").mock(
        return_value=Response(200, json=[{"id": A, "success": True}])
    )

    with ImmichClient(BASE, "k") as immich:
        stats = writeback.apply(
            settings, ledger, immich, {"stars"}, dry_run=False, album="Five-Stars"
        )

    assert list_route.called and create_route.called and add_route.called
    body = json.loads(add_route.calls[0].request.content)
    assert body == {"ids": [A]}
    assert stats.album_assets == 1
    assert update_route.called  # the crown itself was still written


@respx.mock
def test_upsert_album_reuses_existing_album():
    respx.get(f"{BASE}/api/albums").mock(
        return_value=Response(
            200, json=[{"id": "old", "albumName": "Five-Stars"}, {"id": "x", "albumName": "Trips"}]
        )
    )
    with ImmichClient(BASE, "k") as immich:
        assert immich.upsert_album("Five-Stars") == "old"


def test_settings_defaults_for_writeback(settings):
    assert settings.best_album == "Five-Stars"
    assert settings.best_album_min_stars == 5
    assert settings.five_star_favorite is True


# ----------------------------------------------------------------------
# anchor cap (scoring limit)
# ----------------------------------------------------------------------

from pathlib import Path  # noqa: E402

import typer  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from winnow import cli as cli_mod  # noqa: E402
from winnow.pipeline.finals import FinalsStats  # noqa: E402
from winnow.pipeline.rank import RankStats, cap_candidates  # noqa: E402
from winnow.pipeline.scan import ScanStats  # noqa: E402
from winnow.pipeline.triage import TriageStats  # noqa: E402


def test_cap_keeps_every_newcomer_and_spreads_anchors():
    fresh = [f"new{i}" for i in range(5)]
    scores = {f"vet{i}": float(100 - i) for i in range(10)}  # vet0 ranks first
    pool = fresh + sorted(scores)  # deliberately unsorted by score
    capped = cap_candidates(pool, scores, limit=3)
    assert [i for i in capped if i.startswith("new")] == fresh
    anchors = [i for i in capped if i.startswith("vet")]
    # top, middle and bottom of the existing ranking pin the scales together
    assert anchors[0] == "vet0"
    assert anchors[-1] == "vet9"
    assert len(anchors) == 3


def test_cap_none_or_zero_reprocesses_every_veteran():
    scores = {"a": 2.0, "b": 1.0}
    assert set(cap_candidates(["x", "a", "b"], scores, None)) == {"x", "a", "b"}
    assert set(cap_candidates(["x", "a", "b"], scores, 0)) == {"x", "a", "b"}


def test_cap_limit_one_keeps_the_strongest_anchor():
    scores = {"a": 1.0, "b": 9.0}
    assert cap_candidates(["n", "a", "b"], scores, 1) == ["n", "b"]


# ----------------------------------------------------------------------
# watch mode
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90s", 90), ("30m", 1800), ("6h", 21600), ("7d", 604800), ("1w", 604800), ("45", 45)],
)
def test_parse_every(text, seconds):
    assert cli_mod.parse_every(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "-4h", "0"])
def test_parse_every_rejects_nonsense(text):
    with pytest.raises(typer.BadParameter):
        cli_mod.parse_every(text)


def _stub_watch_stages(monkeypatch, calls, rank_kwargs, scan_args):
    def scan(settings, ledger, immich, after, before, on_progress=None):
        calls.append("scan")
        scan_args.update(after=after, before=before)
        return ScanStats()

    def triage(settings, ledger, judge, limit, on_progress=None):
        calls.append("triage")
        return TriageStats()

    def rank(settings, ledger, judge, *, limit=None, on_progress=None):
        calls.append("rank")
        rank_kwargs["limit"] = limit
        return RankStats()

    def finals(settings, ledger, judge, on_progress=None, **kwargs):
        calls.append("finals")
        return FinalsStats()

    def apply_stub(settings, ledger, immich, buckets=None, dry_run=True, **kwargs):
        calls.append("apply")
        return writeback.ApplyStats(dry_run=dry_run)

    monkeypatch.setattr(cli_mod, "run_scan", scan)
    monkeypatch.setattr(cli_mod, "run_triage_direct", triage)
    monkeypatch.setattr(cli_mod, "run_rank", rank)
    monkeypatch.setattr(cli_mod, "run_finals", finals)
    monkeypatch.setattr(cli_mod.writeback, "apply", apply_stub)
    monkeypatch.setattr(
        cli_mod, "write_html_report", lambda ledger, cache, out: Path("winnow-report.html")
    )


def test_watch_once_chains_scan_judge_rank_finals_apply_report(settings_env, monkeypatch):
    calls: list[str] = []
    rank_kwargs: dict = {}
    scan_args: dict = {}
    _stub_watch_stages(monkeypatch, calls, rank_kwargs, scan_args)

    result = CliRunner().invoke(
        cli_mod.app, ["watch", "--once", "--scoring-limit", "7"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert calls == ["scan", "triage", "rank", "finals", "apply"]
    assert rank_kwargs["limit"] == 7
    # default full sweep: every new photo is picked up, even backdated imports
    assert scan_args["after"] == "1970-01-01"


def test_watch_once_no_apply_and_window(settings_env, monkeypatch):
    calls: list[str] = []
    rank_kwargs: dict = {}
    scan_args: dict = {}
    _stub_watch_stages(monkeypatch, calls, rank_kwargs, scan_args)

    result = CliRunner().invoke(
        cli_mod.app,
        ["watch", "--once", "--no-apply", "--window-days", "14"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "apply" not in calls
    assert scan_args["after"] != "1970-01-01"


def test_watch_scoring_limit_defaults_to_setting(settings_env, monkeypatch):
    monkeypatch.setenv("SCORING_LIMIT", "11")
    calls: list[str] = []
    rank_kwargs: dict = {}
    scan_args: dict = {}
    _stub_watch_stages(monkeypatch, calls, rank_kwargs, scan_args)

    CliRunner().invoke(cli_mod.app, ["watch", "--once"], catch_exceptions=False)
    assert rank_kwargs["limit"] == 11


def test_watch_options_come_from_environment(settings_env, monkeypatch):
    """Every CLI flag doubles as a WINNOW_* env var for compose files."""
    monkeypatch.setenv("WINNOW_ONCE", "1")
    monkeypatch.setenv("WINNOW_WINDOW_DAYS", "14")
    monkeypatch.setenv("WINNOW_APPLY", "false")
    calls: list[str] = []
    rank_kwargs: dict = {}
    scan_args: dict = {}
    _stub_watch_stages(monkeypatch, calls, rank_kwargs, scan_args)

    result = CliRunner().invoke(cli_mod.app, ["watch"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "apply" not in calls  # WINNOW_APPLY=false respected
    assert scan_args["after"] != "1970-01-01"  # WINNOW_WINDOW_DAYS respected
