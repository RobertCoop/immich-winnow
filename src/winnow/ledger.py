"""Durable SQLite state for a Winnow run.

Every stage (scan, triage, rank, finals, write-back) reads and writes through
:class:`Ledger`, which makes the whole pipeline resumable: re-running a stage
skips work that is already recorded.

The database is plain stdlib :mod:`sqlite3` in WAL mode. Structured values
(lists, dicts) live in TEXT columns as JSON and are encoded/decoded at the
boundary, so every method takes and returns plain Python dicts and lists.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from .schemas import TriageVerdict

__all__ = ["SCHEMA", "Ledger"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    filename TEXT,
    taken_at TEXT,
    camera TEXT,
    immich_description TEXT,
    width INTEGER,
    height INTEGER,
    burst_id TEXT,
    dhash TEXT,
    thumb_path TEXT,
    immich_rating INTEGER,
    scanned_at TEXT
);

CREATE TABLE IF NOT EXISTS triage (
    asset_id TEXT PRIMARY KEY REFERENCES assets(id),
    category TEXT,
    verdict TEXT,
    technical_score INTEGER,
    reasons TEXT,
    caption TEXT,
    keywords TEXT,
    enriched_at TEXT,
    confidence TEXT,
    model TEXT,
    raw TEXT,
    judged_at TEXT
);

CREATE TABLE IF NOT EXISTS bursts (
    burst_id TEXT PRIMARY KEY,
    winner_id TEXT,
    reject_ids TEXT,
    note TEXT,
    model TEXT,
    judged_at TEXT,
    applied_at TEXT
);

CREATE TABLE IF NOT EXISTS bws_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round INTEGER,
    member_ids TEXT,
    best_id TEXT,
    worst_id TEXT,
    model TEXT,
    judged_at TEXT
);

CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a_id TEXT,
    b_id TEXT,
    winner TEXT,
    stage TEXT,
    model TEXT,
    judged_at TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    asset_id TEXT PRIMARY KEY,
    bt_score REAL,
    rank INTEGER,
    stars INTEGER,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    asset_id TEXT PRIMARY KEY,
    bucket TEXT,
    detail TEXT,
    decided_at TEXT,
    applied_at TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    kind TEXT,
    status TEXT,
    submitted_at TEXT,
    ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_items (
    custom_id TEXT PRIMARY KEY,
    batch_id TEXT,
    kind TEXT,
    payload TEXT,
    result TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_assets_burst ON assets(burst_id);
CREATE INDEX IF NOT EXISTS idx_decisions_bucket ON decisions(bucket);
CREATE INDEX IF NOT EXISTS idx_batch_items_batch ON batch_items(batch_id);
CREATE INDEX IF NOT EXISTS idx_pairs_stage ON pairs(stage);
"""

#: Columns of the ``assets`` table that :meth:`Ledger.upsert_assets` accepts.
ASSET_COLUMNS: tuple[str, ...] = (
    "id",
    "filename",
    "taken_at",
    "camera",
    "width",
    "height",
    "burst_id",
    "dhash",
    "thumb_path",
    "immich_rating",
    "immich_description",
    "scanned_at",
)

#: Decision buckets excluded from the stage-2 candidate pool.
_CANDIDATE_EXCLUDED_BUCKETS: tuple[str, ...] = ("reject", "burst_loser")

_BATCH_KINDS: frozenset[str] = frozenset({"triage", "burst", "bws", "pair"})

_SQL_VAR_CHUNK = 500

#: Columns added after v0.1 databases were first written, as
#: ``table -> {column: type}``. Applied on open so old ledgers keep working.
_MIGRATIONS: dict[str, dict[str, str]] = {
    "bursts": {"applied_at": "TEXT"},
    "assets": {"immich_description": "TEXT"},
    "triage": {"caption": "TEXT", "keywords": "TEXT", "enriched_at": "TEXT"},
}


def _now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dumps(value: Any) -> str | None:
    """JSON-encode a value for a TEXT column (``None`` stays ``None``)."""
    if value is None:
        return None
    return json.dumps(value)


def _loads(text: Any) -> Any:
    """Decode a JSON TEXT column, falling back to the raw string."""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _dumps_result(value: Any) -> str | None:
    """Encode a batch result: strings are stored verbatim, anything else as JSON."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _coerce_asset_value(column: str, value: Any) -> Any:
    """Normalise an asset field into something sqlite3 can bind."""
    if value is None:
        return None
    if column == "dhash":
        return str(value)
    if column in {"width", "height", "immich_rating"}:
        return int(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _asset_row(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an ``assets`` row into a plain dict (``dhash`` back to ``int``)."""
    out = dict(row)
    raw = out.get("dhash")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            out["dhash"] = int(raw)
    return out


def _chunked(items: Sequence[str], size: int = _SQL_VAR_CHUNK) -> Iterator[Sequence[str]]:
    """Yield slices small enough to bind as SQL parameters."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _item_kind(custom_id: str, default: str) -> str:
    """Derive a batch item's kind from its ``<kind>_<id>`` custom id."""
    prefix = custom_id.split("_", 1)[0]
    return prefix if prefix in _BATCH_KINDS else default


class Ledger:
    """Read/write access to the Winnow SQLite database.

    Usable as a context manager::

        with Ledger(settings.db_path) as ledger:
            ledger.upsert_assets(rows)

    Foreign keys are intentionally left unenforced so partial state (a triage
    row for an asset that has not been scanned yet) never raises; the
    ``REFERENCES`` clauses document intent only.
    """

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the ledger database at ``path``."""
        self.path = Path(path)
        if str(path) != ":memory:" and self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        for table, columns in _MIGRATIONS.items():
            existing = {
                str(row["name"])
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, sql_type in columns.items():
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    # --- lifecycle -----------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying sqlite3 connection (read-only convenience)."""
        return self._conn

    def close(self) -> None:
        """Commit and close the database connection."""
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- assets --------------------------------------------------------

    def upsert_assets(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Insert or update asset rows, keyed on ``id``.

        Only the columns present in each mapping are written, so a re-scan
        never clobbers fields it does not know about (a later
        :meth:`assign_burst` survives, for example). ``scanned_at`` defaults
        to now. Returns the number of rows processed.
        """
        count = 0
        for row in rows:
            if "id" not in row:
                raise ValueError("asset rows require an 'id' key")
            data = {col: _coerce_asset_value(col, row[col]) for col in ASSET_COLUMNS if col in row}
            data.setdefault("scanned_at", _now())
            columns = list(data)
            placeholders = ", ".join("?" for _ in columns)
            updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "id")
            sql = (
                f"INSERT INTO assets ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}"
            )
            self._conn.execute(sql, [data[c] for c in columns])
            count += 1
        self._conn.commit()
        return count

    def assign_burst(self, burst_id: str, asset_ids: Sequence[str]) -> None:
        """Tag every asset in ``asset_ids`` as a member of ``burst_id``."""
        for chunk in _chunked(list(asset_ids)):
            marks = ", ".join("?" for _ in chunk)
            self._conn.execute(
                f"UPDATE assets SET burst_id = ? WHERE id IN ({marks})",
                [burst_id, *chunk],
            )
        self._conn.commit()

    def clear_bursts(self) -> int:
        """Un-assign every asset's burst id; returns how many rows changed.

        Burst grouping is re-derived from scratch on every scan, so the old
        stamps must be cleared first — otherwise an asset that drops out of a
        group keeps a stale ``burst_id`` and becomes invisible to the triage
        queues (which skip grouped assets).
        """
        cur = self._conn.execute("UPDATE assets SET burst_id = NULL WHERE burst_id IS NOT NULL")
        self._conn.commit()
        return cur.rowcount

    def get_assets(self, ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return asset rows (all of them, or just ``ids``), oldest first."""
        if ids is None:
            cur = self._conn.execute("SELECT * FROM assets ORDER BY taken_at, id")
            return [_asset_row(r) for r in cur.fetchall()]
        out: list[dict[str, Any]] = []
        for chunk in _chunked(list(ids)):
            marks = ", ".join("?" for _ in chunk)
            cur = self._conn.execute(
                f"SELECT * FROM assets WHERE id IN ({marks}) ORDER BY taken_at, id",
                list(chunk),
            )
            out.extend(_asset_row(r) for r in cur.fetchall())
        return out

    def unjudged_asset_ids(self, exclude_bursts: bool = True) -> list[str]:
        """Asset ids with no triage row yet.

        With ``exclude_bursts`` (the default) assets that belong to a burst
        group are skipped, since those are resolved by the burst judge first.
        """
        sql = "SELECT id FROM assets WHERE id NOT IN (SELECT asset_id FROM triage)"
        if exclude_bursts:
            sql += " AND burst_id IS NULL"
        sql += " ORDER BY taken_at, id"
        return [r["id"] for r in self._conn.execute(sql).fetchall()]

    def burst_groups(self) -> dict[str, list[str]]:
        """Map every burst id to its member asset ids, ordered by capture time."""
        cur = self._conn.execute(
            "SELECT burst_id, id FROM assets WHERE burst_id IS NOT NULL "
            "ORDER BY burst_id, taken_at, id"
        )
        groups: dict[str, list[str]] = {}
        for row in cur.fetchall():
            groups.setdefault(row["burst_id"], []).append(row["id"])
        return groups

    def unjudged_burst_ids(self) -> list[str]:
        """Burst ids that have members but no recorded burst verdict."""
        cur = self._conn.execute(
            "SELECT DISTINCT burst_id FROM assets WHERE burst_id IS NOT NULL "
            "AND burst_id NOT IN (SELECT burst_id FROM bursts) ORDER BY burst_id"
        )
        return [r["burst_id"] for r in cur.fetchall()]

    # --- stage 1: triage ------------------------------------------------

    def record_triage(
        self,
        asset_id: str,
        verdict: TriageVerdict,
        model: str,
        raw: str | None = None,
    ) -> None:
        """Store a stage-1 verdict, replacing any previous verdict for the asset."""
        self._conn.execute(
            """
            INSERT INTO triage (asset_id, category, verdict, technical_score, reasons,
                                caption, keywords, enriched_at,
                                confidence, model, raw, judged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                category = excluded.category,
                verdict = excluded.verdict,
                technical_score = excluded.technical_score,
                reasons = excluded.reasons,
                caption = excluded.caption,
                keywords = excluded.keywords,
                enriched_at = NULL,
                confidence = excluded.confidence,
                model = excluded.model,
                raw = excluded.raw,
                judged_at = excluded.judged_at
            """,
            (
                asset_id,
                verdict.category,
                verdict.verdict,
                int(verdict.technical_score),
                _dumps(list(verdict.reasons)),
                verdict.caption or None,
                _dumps(list(verdict.keywords)) if verdict.keywords else None,
                verdict.confidence,
                model,
                raw if raw is None or isinstance(raw, str) else json.dumps(raw),
                _now(),
            ),
        )
        self._conn.commit()

    def record_burst(
        self,
        burst_id: str,
        winner_id: str,
        reject_ids: Sequence[str],
        note: str,
        model: str,
    ) -> None:
        """Store the winner / losers of a burst group.

        Re-recording the *same* verdict keeps any ``applied_at``, so ingesting
        a batch twice cannot produce a second Immich stack; a verdict that
        actually changed clears it, because the old stack no longer matches.
        """
        self._conn.execute(
            """
            INSERT INTO bursts (burst_id, winner_id, reject_ids, note, model, judged_at,
                                applied_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(burst_id) DO UPDATE SET
                winner_id = excluded.winner_id,
                reject_ids = excluded.reject_ids,
                note = excluded.note,
                model = excluded.model,
                judged_at = excluded.judged_at,
                applied_at = CASE
                    WHEN bursts.winner_id = excluded.winner_id
                     AND bursts.reject_ids = excluded.reject_ids THEN bursts.applied_at
                    ELSE NULL
                END
            """,
            (burst_id, winner_id, _dumps(list(reject_ids)), note, model, _now()),
        )
        self._conn.commit()

    def triage_rows(self) -> list[dict[str, Any]]:
        """Every stage-1 verdict, ``reasons`` decoded back into a list."""
        cur = self._conn.execute("SELECT * FROM triage ORDER BY asset_id")
        rows = []
        for row in cur.fetchall():
            data = dict(row)
            data["reasons"] = _loads(data.get("reasons")) or []
            data["keywords"] = _loads(data.get("keywords")) or []
            rows.append(data)
        return rows

    def unenriched_rows(self) -> list[dict[str, Any]]:
        """Triage rows with a caption or keywords not yet written to Immich.

        Joined with the asset's server-side description as captured at scan
        time, so write-back can honor the never-overwrite rule.
        """
        cur = self._conn.execute(
            """
            SELECT t.asset_id, t.caption, t.keywords, a.immich_description
            FROM triage t JOIN assets a ON a.id = t.asset_id
            WHERE t.enriched_at IS NULL
              AND (t.caption IS NOT NULL OR t.keywords IS NOT NULL)
            ORDER BY t.asset_id
            """
        )
        rows = []
        for row in cur.fetchall():
            data = dict(row)
            data["keywords"] = _loads(data.get("keywords")) or []
            rows.append(data)
        return rows

    def mark_enriched(self, asset_ids: Sequence[str]) -> int:
        """Stamp captions/keywords as written to Immich for these assets."""
        ids = [str(asset_id) for asset_id in asset_ids]
        if not ids:
            return 0
        stamp = _now()
        cur = self._conn.executemany(
            "UPDATE triage SET enriched_at = ? WHERE asset_id = ?",
            [(stamp, asset_id) for asset_id in ids],
        )
        self._conn.commit()
        return cur.rowcount if cur.rowcount is not None else len(ids)

    def burst_rows(self) -> list[dict[str, Any]]:
        """Every burst verdict, ``reject_ids`` decoded back into a list."""
        cur = self._conn.execute("SELECT * FROM bursts ORDER BY burst_id")
        rows = []
        for row in cur.fetchall():
            data = dict(row)
            data["reject_ids"] = _loads(data.get("reject_ids")) or []
            rows.append(data)
        return rows

    def mark_stacked(self, burst_ids: Sequence[str]) -> int:
        """Stamp ``applied_at`` on bursts whose Immich stack now exists.

        Stacking is a group action with no asset of its own, so it needs its
        own applied-state: without this a failed stack would be forgotten (the
        losers' decisions having been marked applied) and a successful one
        would be recreated whenever a sibling call failed.
        """
        now = _now()
        changed = 0
        for chunk in _chunked(list(burst_ids)):
            marks = ", ".join("?" for _ in chunk)
            cur = self._conn.execute(
                f"UPDATE bursts SET applied_at = ? WHERE burst_id IN ({marks})",
                [now, *chunk],
            )
            changed += cur.rowcount
        self._conn.commit()
        return changed

    def candidates(self, min_score: int) -> list[dict[str, Any]]:
        """Photos that earned a place in the stage-2 pool.

        A row qualifies when its category is ``photo``, it is neither a burst
        loser nor a reject, and it was either called a ``candidate`` outright
        or scored at least ``min_score`` technically. Best scores first.
        """
        marks = ", ".join("?" for _ in _CANDIDATE_EXCLUDED_BUCKETS)
        cur = self._conn.execute(
            f"""
            SELECT t.* FROM triage t
            LEFT JOIN decisions d ON d.asset_id = t.asset_id
            WHERE t.category = 'photo'
              AND (t.verdict = 'candidate' OR t.technical_score >= ?)
              AND (d.bucket IS NULL OR d.bucket NOT IN ({marks}))
            ORDER BY t.technical_score DESC, t.asset_id
            """,
            [int(min_score), *_CANDIDATE_EXCLUDED_BUCKETS],
        )
        rows = []
        for row in cur.fetchall():
            data = dict(row)
            data["reasons"] = _loads(data.get("reasons")) or []
            rows.append(data)
        return rows

    # --- stage 2: best-worst scaling -----------------------------------

    def record_bws(
        self,
        round: int,
        member_ids: Sequence[str],
        best_id: str,
        worst_id: str,
        model: str,
    ) -> int:
        """Store one best-worst-scaling judgment; returns the new set's row id."""
        cur = self._conn.execute(
            """
            INSERT INTO bws_sets (round, member_ids, best_id, worst_id, model, judged_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(round), _dumps(list(member_ids)), best_id, worst_id, model, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def bws_rows(self) -> list[dict[str, Any]]:
        """Every best-worst set, ``member_ids`` decoded back into a list."""
        cur = self._conn.execute("SELECT * FROM bws_sets ORDER BY id")
        rows = []
        for row in cur.fetchall():
            data = dict(row)
            data["member_ids"] = _loads(data.get("member_ids")) or []
            rows.append(data)
        return rows

    # --- stage 3: pairs and scores --------------------------------------

    def record_pair(self, a: str, b: str, winner: str, stage: str, model: str) -> int:
        """Store one head-to-head outcome (``winner`` is an asset id or ``'tie'``)."""
        cur = self._conn.execute(
            """
            INSERT INTO pairs (a_id, b_id, winner, stage, model, judged_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (a, b, winner, stage, model, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def pair_rows(self) -> list[dict[str, Any]]:
        """Every recorded pairwise outcome, oldest first."""
        return [dict(r) for r in self._conn.execute("SELECT * FROM pairs ORDER BY id").fetchall()]

    def upsert_scores(self, mapping: Mapping[str, tuple[float, int]]) -> None:
        """Write Bradley-Terry strengths and ranks; existing stars are preserved."""
        now = _now()
        rows = [
            (asset_id, float(score), int(rank), now) for asset_id, (score, rank) in mapping.items()
        ]
        self._conn.executemany(
            """
            INSERT INTO scores (asset_id, bt_score, rank, stars, updated_at)
            VALUES (?, ?, ?, NULL, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                bt_score = excluded.bt_score,
                rank = excluded.rank,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self._conn.commit()

    def set_stars(self, asset_id: str, stars: int | None) -> None:
        """Set the star band for an asset, creating the score row if needed.

        ``None`` clears the band — used when a re-run drops a photo out of the
        star bands it earned last time.
        """
        self._conn.execute(
            """
            INSERT INTO scores (asset_id, bt_score, rank, stars, updated_at)
            VALUES (?, NULL, NULL, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                stars = excluded.stars,
                updated_at = excluded.updated_at
            """,
            (asset_id, None if stars is None else int(stars), _now()),
        )
        self._conn.commit()

    def score_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Score rows, strongest first (``limit`` caps the number returned)."""
        sql = "SELECT * FROM scores ORDER BY bt_score DESC, asset_id"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # --- decisions ------------------------------------------------------

    def set_decision(self, asset_id: str, bucket: str, detail: Any = None) -> None:
        """Record the write-back bucket for an asset.

        ``detail`` may be any JSON-serialisable value. Re-deciding into the
        same bucket keeps an existing ``applied_at``; changing the bucket
        clears it so the new action is queued for write-back again.
        """
        self._conn.execute(
            """
            INSERT INTO decisions (asset_id, bucket, detail, decided_at, applied_at)
            VALUES (?, ?, ?, ?, NULL)
            ON CONFLICT(asset_id) DO UPDATE SET
                bucket = excluded.bucket,
                detail = excluded.detail,
                decided_at = excluded.decided_at,
                applied_at = CASE
                    WHEN decisions.bucket = excluded.bucket THEN decisions.applied_at
                    ELSE NULL
                END
            """,
            (asset_id, bucket, _dumps(detail), _now()),
        )
        self._conn.commit()

    def decisions(
        self, bucket: str | None = None, unapplied_only: bool = False
    ) -> list[dict[str, Any]]:
        """Decisions, optionally filtered to one ``bucket`` and/or not-yet-applied."""
        sql = "SELECT * FROM decisions"
        clauses: list[str] = []
        params: list[Any] = []
        if bucket is not None:
            clauses.append("bucket = ?")
            params.append(bucket)
        if unapplied_only:
            clauses.append("applied_at IS NULL")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY asset_id"
        rows = []
        for row in self._conn.execute(sql, params).fetchall():
            data = dict(row)
            data["detail"] = _loads(data.get("detail"))
            rows.append(data)
        return rows

    def select_rejudge_targets(
        self,
        *,
        after: str | None = None,
        before: str | None = None,
        bucket: str | None = None,
        missing_captions: bool = False,
    ) -> list[str]:
        """Asset ids whose stage-1 judgment matches the rejudge filters.

        Filters combine with AND. ``after``/``before`` bound the taken date
        (ISO prefixes compare correctly against stored timestamps),
        ``bucket`` matches the asset's current decision, and
        ``missing_captions`` selects verdicts recorded before captions existed.
        Only assets that HAVE a triage row are returned — anything unjudged is
        already pending.
        """
        sql = ["SELECT t.asset_id FROM triage t JOIN assets a ON a.id = t.asset_id"]
        clauses: list[str] = []
        params: list[Any] = []
        if bucket is not None:
            sql.append("JOIN decisions d ON d.asset_id = t.asset_id")
            clauses.append("d.bucket = ?")
            params.append(bucket)
        if after is not None:
            clauses.append("a.taken_at >= ?")
            params.append(after)
        if before is not None:
            clauses.append("a.taken_at < ?")
            params.append(before)
        if missing_captions:
            clauses.append("t.caption IS NULL")
        if clauses:
            sql.append("WHERE " + " AND ".join(clauses))
        sql.append("ORDER BY t.asset_id")
        cur = self._conn.execute(" ".join(sql), params)
        return [str(row["asset_id"]) for row in cur.fetchall()]

    def clear_triage(self, asset_ids: Sequence[str]) -> int:
        """Delete stage-1 verdicts so the assets are re-judged next triage.

        Also drops the burst verdict of any burst containing one of the
        assets, so its contest is replayed too. Decisions are left alone —
        the re-judgment overwrites them (an identical verdict keeps its
        applied stamp, a changed one clears it).
        """
        ids = [str(asset_id) for asset_id in asset_ids]
        removed = 0
        for chunk in _chunked(ids):
            marks = ", ".join("?" for _ in chunk)
            burst_rows = self._conn.execute(
                f"SELECT DISTINCT burst_id FROM assets "
                f"WHERE id IN ({marks}) AND burst_id IS NOT NULL",
                chunk,
            ).fetchall()
            burst_ids = [str(row["burst_id"]) for row in burst_rows]
            if burst_ids:
                burst_marks = ", ".join("?" for _ in burst_ids)
                self._conn.execute(
                    f"DELETE FROM bursts WHERE burst_id IN ({burst_marks})", burst_ids
                )
            cur = self._conn.execute(f"DELETE FROM triage WHERE asset_id IN ({marks})", chunk)
            removed += cur.rowcount
        self._conn.commit()
        return removed

    def clear_rank(self) -> int:
        """Forget stage 2: every best-worst set, rank pair, and fitted score.

        Stars and star decisions survive — they belong to the finals, and the
        sticky rules there decide their fate on the next run.
        """
        removed = self._conn.execute("DELETE FROM bws_sets").rowcount
        removed += self._conn.execute("DELETE FROM pairs WHERE stage = 'rank'").rowcount
        self._conn.execute("UPDATE scores SET bt_score = NULL, rank = NULL")
        self._conn.commit()
        return removed

    def clear_finals(self) -> int:
        """Forget stage 3's head-to-head outcomes (star decisions survive)."""
        cur = self._conn.execute("DELETE FROM pairs WHERE stage = 'finals'")
        self._conn.commit()
        return cur.rowcount

    def clear_decisions(self, asset_ids: Sequence[str]) -> int:
        """Delete decision rows outright; returns how many rows were removed.

        Used when a judgment is superseded — a burst also-ran promoted to
        winner, or a photo that has dropped out of the star bands — so the
        stale bucket never reaches write-back.
        """
        removed = 0
        for chunk in _chunked(list(asset_ids)):
            marks = ", ".join("?" for _ in chunk)
            cur = self._conn.execute(f"DELETE FROM decisions WHERE asset_id IN ({marks})", chunk)
            removed += cur.rowcount
        self._conn.commit()
        return removed

    def mark_applied(self, asset_ids: Sequence[str]) -> int:
        """Stamp ``applied_at`` on decisions; returns how many rows changed."""
        now = _now()
        changed = 0
        for chunk in _chunked(list(asset_ids)):
            marks = ", ".join("?" for _ in chunk)
            cur = self._conn.execute(
                f"UPDATE decisions SET applied_at = ? WHERE asset_id IN ({marks})",
                [now, *chunk],
            )
            changed += cur.rowcount
        self._conn.commit()
        return changed

    # --- batches --------------------------------------------------------

    def add_batch(self, batch_id: str, kind: str, items: Mapping[str, Mapping[str, Any]]) -> None:
        """Record a submitted Batch API job and its per-request payloads.

        ``items`` maps ``custom_id`` -> request payload. Each item's kind is
        taken from the ``<kind>_<id>`` custom-id prefix when recognisable,
        otherwise from the batch ``kind``.
        """
        self._conn.execute(
            """
            INSERT INTO batches (batch_id, kind, status, submitted_at, ingested_at)
            VALUES (?, ?, 'submitted', ?, NULL)
            ON CONFLICT(batch_id) DO UPDATE SET
                kind = excluded.kind,
                status = excluded.status,
                submitted_at = excluded.submitted_at
            """,
            (batch_id, kind, _now()),
        )
        self._conn.executemany(
            """
            INSERT INTO batch_items (custom_id, batch_id, kind, payload, result, error)
            VALUES (?, ?, ?, ?, NULL, NULL)
            ON CONFLICT(custom_id) DO UPDATE SET
                batch_id = excluded.batch_id,
                kind = excluded.kind,
                payload = excluded.payload,
                result = NULL,
                error = NULL
            """,
            [
                (custom_id, batch_id, _item_kind(custom_id, kind), _dumps(payload))
                for custom_id, payload in items.items()
            ],
        )
        self._conn.commit()

    def inflight_custom_ids(self) -> set[str]:
        """Custom ids belonging to a batch that has not been ingested yet.

        Queueing something twice would bill it twice *and* steal the item row
        from the earlier batch, so every pending-work query filters on this.
        """
        cur = self._conn.execute(
            "SELECT custom_id FROM batch_items WHERE batch_id IN "
            "(SELECT batch_id FROM batches WHERE ingested_at IS NULL)"
        )
        return {str(row["custom_id"]) for row in cur.fetchall()}

    def open_batches(self) -> list[dict[str, Any]]:
        """Batches that have not been ingested yet, oldest submission first."""
        cur = self._conn.execute(
            "SELECT * FROM batches WHERE ingested_at IS NULL ORDER BY submitted_at, batch_id"
        )
        return [dict(r) for r in cur.fetchall()]

    def set_batch_status(self, batch_id: str, status: str) -> None:
        """Update a batch's status; ``'ingested'`` also stamps ``ingested_at``."""
        if status == "ingested":
            self._conn.execute(
                "UPDATE batches SET status = ?, ingested_at = ? WHERE batch_id = ?",
                (status, _now(), batch_id),
            )
        else:
            self._conn.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?", (status, batch_id)
            )
        self._conn.commit()

    def record_batch_result(
        self,
        custom_id: str,
        result_json: Any = None,
        error: str | None = None,
    ) -> None:
        """Attach a result (JSON string or serialisable value) or an error to an item."""
        self._conn.execute(
            "UPDATE batch_items SET result = ?, error = ? WHERE custom_id = ?",
            (_dumps_result(result_json), error, custom_id),
        )
        self._conn.commit()

    def batch_items_for(self, batch_id: str) -> list[dict[str, Any]]:
        """All items of a batch, payload and result decoded from JSON."""
        cur = self._conn.execute(
            "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY custom_id", (batch_id,)
        )
        rows = []
        for row in cur.fetchall():
            data = dict(row)
            data["payload"] = _loads(data.get("payload"))
            data["result"] = _loads(data.get("result"))
            rows.append(data)
        return rows

    # --- reporting ------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Row counts per table plus per-bucket / per-verdict breakdowns."""
        counts = {
            table: int(self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in (
                "assets",
                "triage",
                "bursts",
                "bws_sets",
                "pairs",
                "scores",
                "decisions",
                "batches",
                "batch_items",
            )
        }
        summary: dict[str, Any] = dict(counts)
        summary["buckets"] = self._group_count("SELECT bucket AS k, COUNT(*) AS n FROM decisions")
        summary["verdicts"] = self._group_count("SELECT verdict AS k, COUNT(*) AS n FROM triage")
        summary["categories"] = self._group_count("SELECT category AS k, COUNT(*) AS n FROM triage")
        summary["burst_groups"] = int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT burst_id) AS n FROM assets WHERE burst_id IS NOT NULL"
            ).fetchone()["n"]
        )
        summary["untriaged"] = int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM assets WHERE id NOT IN (SELECT asset_id FROM triage)"
            ).fetchone()["n"]
        )
        summary["applied"] = int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM decisions WHERE applied_at IS NOT NULL"
            ).fetchone()["n"]
        )
        summary["unapplied"] = summary["decisions"] - summary["applied"]
        summary["open_batches"] = int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM batches WHERE ingested_at IS NULL"
            ).fetchone()["n"]
        )
        return summary

    def _group_count(self, sql: str) -> dict[str, int]:
        """Run a ``k``/``n`` grouping query into a plain dict."""
        cur = self._conn.execute(f"{sql} GROUP BY k ORDER BY k")
        return {r["k"]: int(r["n"]) for r in cur.fetchall() if r["k"] is not None}
