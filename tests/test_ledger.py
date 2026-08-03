"""Tests for the SQLite ledger: round-trips, idempotency, filtering, counts."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from winnow.ledger import Ledger
from winnow.schemas import TriageVerdict

# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def ledger(tmp_path: Path):
    """A fresh ledger on a throwaway database file."""
    led = Ledger(tmp_path / "winnow.db")
    yield led
    led.close()


def asset(asset_id: str, **over):
    """Build an asset row dict with sane defaults."""
    row = {
        "id": asset_id,
        "filename": f"{asset_id}.jpg",
        "taken_at": f"2024-06-01T10:00:{int(asset_id[-1]) if asset_id[-1].isdigit() else 0:02d}",
        "camera": "Canon|R6",
        "width": 6000,
        "height": 4000,
        "burst_id": None,
        "dhash": 0xDEADBEEF,
        "thumb_path": f"/cache/{asset_id}.jpg",
        "immich_rating": 0,
    }
    row.update(over)
    return row


def verdict(**over) -> TriageVerdict:
    """Build a TriageVerdict with sane defaults."""
    data = {
        "category": "photo",
        "verdict": "neutral",
        "technical_score": 5,
        "reasons": ["sharp", "well exposed"],
        "confidence": "medium",
    }
    data.update(over)
    return TriageVerdict(**data)


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_creates_file_and_tables(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "winnow.db"
    with Ledger(path) as led:
        assert path.exists()
        names = {
            r["name"]
            for r in led.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {
        "assets",
        "triage",
        "bursts",
        "bws_sets",
        "pairs",
        "scores",
        "decisions",
        "batches",
        "batch_items",
    } <= names


def test_wal_mode_enabled(ledger: Ledger):
    mode = ledger.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_context_manager_closes(tmp_path: Path):
    led = Ledger(tmp_path / "w.db")
    with led:
        led.upsert_assets([asset("a1")])
    with pytest.raises(sqlite3.ProgrammingError):
        led.connection.execute("SELECT 1")


def test_reopening_keeps_data(tmp_path: Path):
    path = tmp_path / "w.db"
    with Ledger(path) as led:
        led.upsert_assets([asset("a1")])
    with Ledger(path) as led:
        assert [r["id"] for r in led.get_assets()] == ["a1"]


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------


def test_upsert_assets_round_trip(ledger: Ledger):
    assert ledger.upsert_assets([asset("a1"), asset("a2")]) == 2
    rows = ledger.get_assets()
    assert [r["id"] for r in rows] == ["a1", "a2"]
    first = rows[0]
    assert first["filename"] == "a1.jpg"
    assert first["camera"] == "Canon|R6"
    assert first["width"] == 6000 and first["height"] == 4000
    assert first["dhash"] == 0xDEADBEEF  # stored as TEXT, returned as int
    assert first["thumb_path"] == "/cache/a1.jpg"
    assert first["scanned_at"]  # auto-stamped


def test_upsert_assets_is_idempotent_and_updates(ledger: Ledger):
    ledger.upsert_assets([asset("a1", filename="old.jpg", immich_rating=0)])
    ledger.upsert_assets([asset("a1", filename="new.jpg", immich_rating=4)])
    rows = ledger.get_assets()
    assert len(rows) == 1
    assert rows[0]["filename"] == "new.jpg"
    assert rows[0]["immich_rating"] == 4


def test_upsert_assets_partial_update_preserves_other_columns(ledger: Ledger):
    ledger.upsert_assets([asset("a1")])
    ledger.assign_burst("b1", ["a1"])
    ledger.upsert_assets([{"id": "a1", "immich_rating": 5}])
    row = ledger.get_assets(["a1"])[0]
    assert row["burst_id"] == "b1"  # not clobbered
    assert row["filename"] == "a1.jpg"
    assert row["immich_rating"] == 5


def test_upsert_assets_coerces_datetime_and_path(ledger: Ledger):
    when = datetime(2024, 6, 1, 12, 30, tzinfo=UTC)
    ledger.upsert_assets([asset("a1", taken_at=when, thumb_path=Path("/cache/a1.jpg"), dhash=None)])
    row = ledger.get_assets(["a1"])[0]
    assert row["taken_at"] == when.isoformat()
    assert row["thumb_path"] == "/cache/a1.jpg"
    assert row["dhash"] is None


def test_upsert_assets_requires_id(ledger: Ledger):
    with pytest.raises(ValueError, match="id"):
        ledger.upsert_assets([{"filename": "x.jpg"}])


def test_get_assets_subset_and_missing_ids(ledger: Ledger):
    ledger.upsert_assets([asset("a1"), asset("a2"), asset("a3")])
    rows = ledger.get_assets(["a3", "a1", "nope"])
    assert [r["id"] for r in rows] == ["a1", "a3"]
    assert ledger.get_assets([]) == []


def test_assign_burst_and_burst_groups(ledger: Ledger):
    ledger.upsert_assets([asset("a1"), asset("a2"), asset("a3"), asset("a4")])
    ledger.assign_burst("b1", ["a2", "a1"])
    ledger.assign_burst("b2", ["a3"])
    assert ledger.burst_groups() == {"b1": ["a1", "a2"], "b2": ["a3"]}
    assert ledger.get_assets(["a4"])[0]["burst_id"] is None


def test_unjudged_asset_ids_excludes_bursts_and_judged(ledger: Ledger):
    ledger.upsert_assets([asset("a1"), asset("a2"), asset("a3")])
    ledger.assign_burst("b1", ["a3"])
    ledger.record_triage("a1", verdict(), "claude-haiku-4-5", "{}")
    assert ledger.unjudged_asset_ids(exclude_bursts=True) == ["a2"]
    assert ledger.unjudged_asset_ids(exclude_bursts=False) == ["a2", "a3"]


def test_unjudged_burst_ids(ledger: Ledger):
    ledger.upsert_assets([asset("a1"), asset("a2"), asset("a3")])
    ledger.assign_burst("b1", ["a1", "a2"])
    ledger.assign_burst("b2", ["a3"])
    assert ledger.unjudged_burst_ids() == ["b1", "b2"]
    ledger.record_burst("b1", "a1", ["a2"], "a1 is sharpest", "claude-haiku-4-5")
    assert ledger.unjudged_burst_ids() == ["b2"]


def test_clear_bursts_unassigns_every_member(ledger: Ledger):
    ledger.upsert_assets([asset("a1"), asset("a2"), asset("a3")])
    ledger.assign_burst("b1", ["a1", "a2"])
    assert ledger.clear_bursts() == 2
    assert ledger.burst_groups() == {}
    assert ledger.clear_bursts() == 0  # nothing left to clear
    # the assets themselves survive, and are visible to triage again
    assert ledger.unjudged_asset_ids(exclude_bursts=True) == ["a1", "a2", "a3"]


def test_mark_stacked_is_tracked_per_burst(ledger: Ledger):
    ledger.record_burst("b1", "a1", ["a2"], "note", "m")
    ledger.record_burst("b2", "a3", ["a4"], "note", "m")
    assert all(row["applied_at"] is None for row in ledger.burst_rows())

    assert ledger.mark_stacked(["b1", "missing"]) == 1
    rows = {row["burst_id"]: row for row in ledger.burst_rows()}
    assert rows["b1"]["applied_at"]
    assert rows["b2"]["applied_at"] is None


def test_re_recording_the_same_burst_keeps_it_stacked(ledger: Ledger):
    """Ingesting the same verdict twice must not produce a second stack."""
    ledger.record_burst("b1", "a1", ["a2"], "note", "m")
    ledger.mark_stacked(["b1"])
    ledger.record_burst("b1", "a1", ["a2"], "note again", "m")
    assert ledger.burst_rows()[0]["applied_at"]


def test_a_changed_burst_verdict_needs_restacking(ledger: Ledger):
    """A different winner means the existing stack is wrong, so the burst is
    queued for write-back again."""
    ledger.record_burst("b1", "a1", ["a2"], "note", "m")
    ledger.mark_stacked(["b1"])
    ledger.record_burst("b1", "a2", ["a1"], "changed my mind", "m")
    assert ledger.burst_rows()[0]["applied_at"] is None


def test_migration_adds_applied_at_to_an_old_database(tmp_path: Path):
    """A v0.1 ledger predates bursts.applied_at; opening it must not fail."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE bursts (burst_id TEXT PRIMARY KEY, winner_id TEXT, reject_ids TEXT,"
        " note TEXT, model TEXT, judged_at TEXT)"
    )
    conn.execute("INSERT INTO bursts VALUES ('b1', 'a1', '[\"a2\"]', 'n', 'm', 'then')")
    conn.commit()
    conn.close()

    with Ledger(path) as led:
        row = led.burst_rows()[0]
        assert row["burst_id"] == "b1"
        assert row["applied_at"] is None
        led.mark_stacked(["b1"])
        assert led.burst_rows()[0]["applied_at"]


# --------------------------------------------------------------------------
# triage / bursts
# --------------------------------------------------------------------------


def test_record_triage_round_trip(ledger: Ledger):
    ledger.upsert_assets([asset("a1")])
    v = verdict(category="photo", verdict="candidate", technical_score=9, confidence="high")
    ledger.record_triage("a1", v, "claude-haiku-4-5", '{"verdict": "candidate"}')
    rows = ledger.triage_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["asset_id"] == "a1"
    assert row["category"] == "photo"
    assert row["verdict"] == "candidate"
    assert row["technical_score"] == 9
    assert row["reasons"] == ["sharp", "well exposed"]
    assert row["confidence"] == "high"
    assert row["model"] == "claude-haiku-4-5"
    assert row["raw"] == '{"verdict": "candidate"}'
    assert row["judged_at"]


def test_record_triage_stores_reasons_as_json_text(ledger: Ledger):
    ledger.record_triage("a1", verdict(reasons=["a", "b"]), "m", None)
    stored = ledger.connection.execute("SELECT reasons FROM triage WHERE asset_id='a1'").fetchone()
    assert json.loads(stored["reasons"]) == ["a", "b"]


def test_record_triage_is_idempotent(ledger: Ledger):
    ledger.record_triage("a1", verdict(technical_score=3), "m1", None)
    ledger.record_triage("a1", verdict(technical_score=8, verdict="candidate"), "m2", None)
    rows = ledger.triage_rows()
    assert len(rows) == 1
    assert rows[0]["technical_score"] == 8
    assert rows[0]["verdict"] == "candidate"
    assert rows[0]["model"] == "m2"


def test_record_burst_round_trip_and_idempotent(ledger: Ledger):
    ledger.record_burst("b1", "a1", ["a2", "a3"], "a1 sharpest", "claude-haiku-4-5")
    rows = ledger.burst_rows()
    assert len(rows) == 1
    assert rows[0]["winner_id"] == "a1"
    assert rows[0]["reject_ids"] == ["a2", "a3"]
    assert rows[0]["note"] == "a1 sharpest"
    ledger.record_burst("b1", "a2", ["a1", "a3"], "revised", "m2")
    rows = ledger.burst_rows()
    assert len(rows) == 1
    assert rows[0]["winner_id"] == "a2"
    assert rows[0]["reject_ids"] == ["a1", "a3"]


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


def _seed_candidates(led: Ledger) -> None:
    led.upsert_assets([asset(f"a{i}") for i in range(1, 8)])
    led.record_triage("a1", verdict(verdict="candidate", technical_score=4), "m", None)
    led.record_triage("a2", verdict(technical_score=9), "m", None)
    led.record_triage("a3", verdict(technical_score=7), "m", None)  # below threshold
    led.record_triage("a4", verdict(category="screenshot", technical_score=10), "m", None)
    led.record_triage("a5", verdict(verdict="candidate", technical_score=9), "m", None)
    led.record_triage("a6", verdict(verdict="candidate", technical_score=9), "m", None)
    led.record_triage("a7", verdict(technical_score=8), "m", None)
    led.set_decision("a5", "burst_loser", {"burst_id": "b1"})
    led.set_decision("a6", "reject", None)
    led.set_decision("a7", "middle", None)


def test_candidates_filters(ledger: Ledger):
    _seed_candidates(ledger)
    ids = [r["asset_id"] for r in ledger.candidates(min_score=8)]
    assert ids == ["a2", "a7", "a1"]  # score desc, then asset_id


def test_candidates_excludes_nonphoto_burst_losers_and_rejects(ledger: Ledger):
    _seed_candidates(ledger)
    ids = {r["asset_id"] for r in ledger.candidates(min_score=8)}
    assert "a4" not in ids  # screenshot category
    assert "a5" not in ids  # burst loser
    assert "a6" not in ids  # rejected
    assert "a3" not in ids  # neutral, below threshold


def test_candidates_threshold_moves(ledger: Ledger):
    _seed_candidates(ledger)
    assert {r["asset_id"] for r in ledger.candidates(min_score=7)} == {"a1", "a2", "a3", "a7"}
    assert {r["asset_id"] for r in ledger.candidates(min_score=11)} == {"a1"}


def test_candidates_decodes_reasons(ledger: Ledger):
    ledger.record_triage("a1", verdict(verdict="candidate", reasons=["x"]), "m", None)
    assert ledger.candidates(min_score=10)[0]["reasons"] == ["x"]


def test_candidates_empty_ledger(ledger: Ledger):
    assert ledger.candidates(min_score=8) == []


# --------------------------------------------------------------------------
# bws / pairs / scores
# --------------------------------------------------------------------------


def test_record_bws_round_trip(ledger: Ledger):
    row_id = ledger.record_bws(1, ["a1", "a2", "a3"], "a1", "a3", "claude-sonnet-5")
    second = ledger.record_bws(2, ["a4", "a5"], "a4", "a5", "claude-sonnet-5")
    assert second == row_id + 1
    rows = ledger.bws_rows()
    assert [r["id"] for r in rows] == [row_id, second]
    assert rows[0]["round"] == 1
    assert rows[0]["member_ids"] == ["a1", "a2", "a3"]
    assert rows[0]["best_id"] == "a1"
    assert rows[0]["worst_id"] == "a3"
    assert rows[0]["model"] == "claude-sonnet-5"
    assert rows[0]["judged_at"]


def test_record_pair_round_trip(ledger: Ledger):
    ledger.record_pair("a1", "a2", "a1", "finals", "claude-opus-5")
    ledger.record_pair("a1", "a2", "tie", "finals", "claude-opus-5")
    rows = ledger.pair_rows()
    assert len(rows) == 2
    assert rows[0]["a_id"] == "a1" and rows[0]["b_id"] == "a2"
    assert rows[0]["winner"] == "a1"
    assert rows[1]["winner"] == "tie"
    assert rows[0]["stage"] == "finals"
    assert rows[0]["model"] == "claude-opus-5"


def test_upsert_scores_round_trip_and_idempotent(ledger: Ledger):
    ledger.upsert_scores({"a1": (1.5, 1), "a2": (0.5, 2)})
    rows = ledger.score_rows()
    assert [r["asset_id"] for r in rows] == ["a1", "a2"]
    assert rows[0]["bt_score"] == pytest.approx(1.5)
    assert rows[0]["rank"] == 1
    assert rows[0]["stars"] is None

    ledger.upsert_scores({"a1": (0.2, 2), "a2": (2.0, 1)})
    rows = ledger.score_rows()
    assert len(rows) == 2
    assert [r["asset_id"] for r in rows] == ["a2", "a1"]
    assert rows[0]["rank"] == 1


def test_set_stars_creates_and_survives_rescore(ledger: Ledger):
    ledger.set_stars("a1", 5)
    row = ledger.score_rows()[0]
    assert row["asset_id"] == "a1" and row["stars"] == 5 and row["bt_score"] is None

    ledger.upsert_scores({"a1": (1.0, 1)})
    row = ledger.score_rows()[0]
    assert row["stars"] == 5  # preserved by upsert_scores
    ledger.set_stars("a1", 4)
    row = ledger.score_rows()[0]
    assert row["stars"] == 4
    assert row["bt_score"] == pytest.approx(1.0)  # preserved by set_stars


def test_score_rows_limit(ledger: Ledger):
    ledger.upsert_scores({"a1": (1.0, 2), "a2": (3.0, 1), "a3": (2.0, 3)})
    assert [r["asset_id"] for r in ledger.score_rows(limit=2)] == ["a2", "a3"]


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------


def test_set_decision_round_trip(ledger: Ledger):
    ledger.set_decision("a1", "reject", {"why": "blurry", "score": 2})
    rows = ledger.decisions()
    assert len(rows) == 1
    assert rows[0]["asset_id"] == "a1"
    assert rows[0]["bucket"] == "reject"
    assert rows[0]["detail"] == {"why": "blurry", "score": 2}
    assert rows[0]["decided_at"]
    assert rows[0]["applied_at"] is None


def test_set_decision_detail_variants(ledger: Ledger):
    ledger.set_decision("a1", "middle", None)
    ledger.set_decision("a2", "middle", "plain string")
    ledger.set_decision("a3", "middle", ["a", "b"])
    details = {r["asset_id"]: r["detail"] for r in ledger.decisions()}
    assert details == {"a1": None, "a2": "plain string", "a3": ["a", "b"]}


def test_set_decision_idempotent_same_bucket_keeps_applied(ledger: Ledger):
    ledger.set_decision("a1", "reject", None)
    ledger.mark_applied(["a1"])
    ledger.set_decision("a1", "reject", {"note": "re-run"})
    rows = ledger.decisions()
    assert len(rows) == 1
    assert rows[0]["applied_at"] is not None
    assert rows[0]["detail"] == {"note": "re-run"}


def test_set_decision_bucket_change_clears_applied(ledger: Ledger):
    ledger.set_decision("a1", "reject", None)
    ledger.mark_applied(["a1"])
    ledger.set_decision("a1", "five_star", None)
    rows = ledger.decisions()
    assert len(rows) == 1
    assert rows[0]["bucket"] == "five_star"
    assert rows[0]["applied_at"] is None


def test_decisions_filtering(ledger: Ledger):
    ledger.set_decision("a1", "reject", None)
    ledger.set_decision("a2", "reject", None)
    ledger.set_decision("a3", "nonphoto", None)
    ledger.set_decision("a4", "five_star", None)
    ledger.mark_applied(["a1", "a4"])

    assert [r["asset_id"] for r in ledger.decisions(bucket="reject")] == ["a1", "a2"]
    assert [r["asset_id"] for r in ledger.decisions(unapplied_only=True)] == ["a2", "a3"]
    assert [r["asset_id"] for r in ledger.decisions(bucket="reject", unapplied_only=True)] == ["a2"]
    assert [r["asset_id"] for r in ledger.decisions(bucket="nope")] == []
    assert len(ledger.decisions()) == 4


def test_clear_decisions_removes_rows(ledger: Ledger):
    ledger.set_decision("a1", "burst_loser", {"burst_id": "b1"})
    ledger.set_decision("a2", "reject", None)
    assert ledger.clear_decisions(["a1", "never-seen"]) == 1
    assert [row["asset_id"] for row in ledger.decisions()] == ["a2"]


def test_set_stars_accepts_none_to_clear(ledger: Ledger):
    ledger.set_stars("a1", 5)
    assert ledger.score_rows()[0]["stars"] == 5
    ledger.set_stars("a1", None)
    assert ledger.score_rows()[0]["stars"] is None


def test_mark_applied_counts_only_existing(ledger: Ledger):
    ledger.set_decision("a1", "reject", None)
    assert ledger.mark_applied(["a1", "ghost"]) == 1
    assert ledger.decisions()[0]["applied_at"] is not None
    assert ledger.mark_applied([]) == 0


# --------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------


def test_add_batch_and_items_round_trip(ledger: Ledger):
    items = {
        "triage_a1": {"model": "claude-haiku-4-5", "max_tokens": 1024},
        "triage_a2": {"model": "claude-haiku-4-5", "max_tokens": 1024},
        "burst_b1": {"model": "claude-haiku-4-5", "max_tokens": 1024},
    }
    ledger.add_batch("batch_123", "triage", items)

    rows = ledger.batch_items_for("batch_123")
    assert [r["custom_id"] for r in rows] == ["burst_b1", "triage_a1", "triage_a2"]
    assert rows[0]["kind"] == "burst"  # derived from custom-id prefix
    assert rows[1]["kind"] == "triage"
    assert rows[1]["payload"] == {"model": "claude-haiku-4-5", "max_tokens": 1024}
    assert rows[1]["result"] is None and rows[1]["error"] is None

    open_rows = ledger.open_batches()
    assert len(open_rows) == 1
    assert open_rows[0]["batch_id"] == "batch_123"
    assert open_rows[0]["kind"] == "triage"
    assert open_rows[0]["status"] == "submitted"
    assert open_rows[0]["ingested_at"] is None


def test_add_batch_unknown_prefix_falls_back_to_batch_kind(ledger: Ledger):
    ledger.add_batch("b1", "rank", {"weird-id": {"x": 1}})
    assert ledger.batch_items_for("b1")[0]["kind"] == "rank"


def test_add_batch_is_idempotent(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {"v": 1}})
    ledger.add_batch("b1", "triage", {"triage_a1": {"v": 2}})
    rows = ledger.batch_items_for("b1")
    assert len(rows) == 1
    assert rows[0]["payload"] == {"v": 2}
    assert len(ledger.open_batches()) == 1


def test_requeuing_an_item_clears_its_old_outcome(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {"v": 1}})
    ledger.record_batch_result("triage_a1", error="errored")
    ledger.add_batch("b2", "triage", {"triage_a1": {"v": 2}})
    row = ledger.batch_items_for("b2")[0]
    assert row["batch_id"] == "b2"
    assert row["error"] is None and row["result"] is None


def test_inflight_custom_ids_tracks_uningested_batches(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {}, "burst_b9": {}})
    ledger.add_batch("b2", "triage", {"triage_a2": {}})
    assert ledger.inflight_custom_ids() == {"triage_a1", "burst_b9", "triage_a2"}

    ledger.set_batch_status("b1", "ingested")
    assert ledger.inflight_custom_ids() == {"triage_a2"}


def test_set_batch_status_and_open_batches(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {}})
    ledger.add_batch("b2", "triage", {"triage_a2": {}})

    ledger.set_batch_status("b1", "ended")
    assert {r["batch_id"] for r in ledger.open_batches()} == {"b1", "b2"}
    assert ledger.open_batches()[0]["status"] == "ended"

    ledger.set_batch_status("b1", "ingested")
    open_rows = ledger.open_batches()
    assert [r["batch_id"] for r in open_rows] == ["b2"]
    row = ledger.connection.execute("SELECT * FROM batches WHERE batch_id='b1'").fetchone()
    assert row["ingested_at"] is not None


def test_record_batch_result_success_and_error(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {}, "triage_a2": {}})
    ledger.record_batch_result("triage_a1", result_json={"verdict": "candidate"})
    ledger.record_batch_result("triage_a2", error="errored")

    rows = {r["custom_id"]: r for r in ledger.batch_items_for("b1")}
    assert rows["triage_a1"]["result"] == {"verdict": "candidate"}
    assert rows["triage_a1"]["error"] is None
    assert rows["triage_a2"]["result"] is None
    assert rows["triage_a2"]["error"] == "errored"


def test_record_batch_result_accepts_json_string(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {}})
    ledger.record_batch_result("triage_a1", result_json='{"verdict": "reject"}')
    assert ledger.batch_items_for("b1")[0]["result"] == {"verdict": "reject"}


def test_record_batch_result_unknown_custom_id_is_noop(ledger: Ledger):
    ledger.record_batch_result("ghost", result_json={"a": 1})
    assert ledger.batch_items_for("b1") == []


def test_batch_items_for_unknown_batch(ledger: Ledger):
    assert ledger.batch_items_for("nope") == []


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def test_summary_empty(ledger: Ledger):
    s = ledger.summary()
    for table in ("assets", "triage", "bursts", "bws_sets", "pairs", "scores", "decisions"):
        assert s[table] == 0
    assert s["buckets"] == {}
    assert s["verdicts"] == {}
    assert s["categories"] == {}
    assert s["untriaged"] == 0
    assert s["open_batches"] == 0


def test_summary_counts(ledger: Ledger):
    ledger.upsert_assets([asset(f"a{i}") for i in range(1, 6)])
    ledger.assign_burst("b1", ["a4", "a5"])
    ledger.record_triage("a1", verdict(verdict="candidate", technical_score=9), "m", None)
    ledger.record_triage("a2", verdict(verdict="reject", technical_score=2), "m", None)
    ledger.record_triage("a3", verdict(category="screenshot"), "m", None)
    ledger.record_burst("b1", "a4", ["a5"], "note", "m")
    ledger.record_bws(1, ["a1", "a2"], "a1", "a2", "m")
    ledger.record_pair("a1", "a2", "a1", "finals", "m")
    ledger.upsert_scores({"a1": (1.0, 1)})
    ledger.set_decision("a2", "reject", None)
    ledger.set_decision("a3", "nonphoto", None)
    ledger.set_decision("a5", "burst_loser", None)
    ledger.mark_applied(["a2"])
    ledger.add_batch("b1", "triage", {"triage_a1": {}, "triage_a2": {}})

    s = ledger.summary()
    assert s["assets"] == 5
    assert s["triage"] == 3
    assert s["bursts"] == 1
    assert s["bws_sets"] == 1
    assert s["pairs"] == 1
    assert s["scores"] == 1
    assert s["decisions"] == 3
    assert s["batches"] == 1
    assert s["batch_items"] == 2
    assert s["buckets"] == {"burst_loser": 1, "nonphoto": 1, "reject": 1}
    assert s["verdicts"] == {"candidate": 1, "neutral": 1, "reject": 1}
    assert s["categories"] == {"photo": 2, "screenshot": 1}
    assert s["burst_groups"] == 1
    assert s["untriaged"] == 2
    assert s["applied"] == 1
    assert s["unapplied"] == 2
    assert s["open_batches"] == 1


def test_summary_after_reingest(ledger: Ledger):
    ledger.add_batch("b1", "triage", {"triage_a1": {}})
    ledger.set_batch_status("b1", "ingested")
    assert ledger.summary()["open_batches"] == 0


# --------------------------------------------------------------------------
# scale / chunking
# --------------------------------------------------------------------------


def test_large_id_lists_are_chunked(ledger: Ledger):
    ids = [f"a{i:04d}" for i in range(1200)]
    ledger.upsert_assets([asset(i) for i in ids])
    assert len(ledger.get_assets(ids)) == 1200
    ledger.assign_burst("b1", ids)
    assert len(ledger.burst_groups()["b1"]) == 1200
    for asset_id in ids:
        ledger.set_decision(asset_id, "middle", None)
    assert ledger.mark_applied(ids) == 1200
