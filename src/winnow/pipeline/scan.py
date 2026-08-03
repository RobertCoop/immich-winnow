"""Stage 0 — pull a date window out of Immich and into the ledger.

Scanning is the only stage that talks to Immich for *reading*: it lists the
image assets in a window, caches a small, metadata-free JPEG per asset under
``settings.cache_dir``, perceptually hashes it, and records everything in the
ledger. It finishes by re-deriving burst groups over the whole ledger (merged
with Immich's own duplicate groups) and stamping a stable burst id on every
member.

Re-running a scan over the same window is cheap: assets whose thumbnail is
already cached and whose hash is already stored are never re-fetched.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from winnow.bursts import AssetLite, group_bursts, merge_duplicate_groups
from winnow.config import Settings
from winnow.images import dhash, prepare_image, to_b64
from winnow.immich import ImmichClient, ImmichError
from winnow.ledger import Ledger

__all__ = [
    "ProgressFn",
    "ScanStats",
    "asset_camera",
    "asset_taken_at",
    "burst_id_for",
    "emit",
    "iso_bound",
    "load_thumb_b64",
    "run_scan",
    "thumb_path",
]

#: Optional progress callback passed to every pipeline stage.
ProgressFn = Callable[[str], None]

#: Length of the hex digest embedded in a generated burst id.
_BURST_ID_CHARS = 12

#: Assets buffered before the ledger is written. Flushing as we go keeps a
#: long scan durable (and its memory flat) when a later page fails.
_FLUSH_EVERY = 200

#: JPEG start-of-image / end-of-image markers, used to spot a truncated file.
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def emit(on_progress: ProgressFn | None, message: str) -> None:
    """Forward a one-line progress message to the caller's callback, if any."""
    if on_progress is not None:
        on_progress(message)


@dataclass
class ScanStats:
    """What a :func:`run_scan` call did.

    Attributes:
        seen: Assets returned by Immich for the window.
        new: Assets recorded in the ledger for the first time.
        thumbs_fetched: Thumbnails downloaded and cached this run.
        thumbs_cached: Thumbnails already present in the cache.
        bursts: Burst groups in the ledger after grouping.
        burst_assets: Assets belonging to one of those groups.
        errors: Assets skipped because their thumbnail could not be prepared.
    """

    seen: int = 0
    new: int = 0
    thumbs_fetched: int = 0
    thumbs_cached: int = 0
    bursts: int = 0
    burst_assets: int = 0
    errors: int = 0


def thumb_path(cache_dir: str | Path, asset_id: str) -> Path:
    """Path of an asset's cached thumbnail inside ``cache_dir``."""
    return Path(cache_dir) / f"{asset_id}.jpg"


def load_thumb_b64(cache_dir: str | Path, asset_id: str) -> str:
    """Read a cached thumbnail and base64-encode it for the Anthropic API.

    Raises:
        OSError: If the thumbnail has not been cached yet.
    """
    return to_b64(thumb_path(cache_dir, asset_id).read_bytes())


def iso_bound(value: str | date | datetime) -> str:
    """Normalise a search bound into the ISO-8601 form Immich expects.

    Bare dates (``"2024-06-01"``) become midnight UTC; datetimes are converted
    to UTC; any other string is passed through untouched.
    """
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if isinstance(value, date):
        return f"{value.isoformat()}T00:00:00.000Z"
    text = value.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return f"{text}T00:00:00.000Z"
    return text


def _parse_dt(value: Any) -> datetime | None:
    """Parse an Immich timestamp into an aware UTC datetime (``None`` if unusable)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def asset_taken_at(asset: dict[str, Any]) -> datetime | None:
    """Best available capture time for an Immich asset, as aware UTC."""
    exif = asset.get("exifInfo") or {}
    for candidate in (
        exif.get("dateTimeOriginal"),
        asset.get("fileCreatedAt"),
        asset.get("localDateTime"),
    ):
        moment = _parse_dt(candidate)
        if moment is not None:
            return moment
    return None


def asset_camera(asset: dict[str, Any]) -> str:
    """``"make|model"`` for an asset, or ``""`` when the EXIF says nothing."""
    exif = asset.get("exifInfo") or {}
    make = (exif.get("make") or "").strip()
    model = (exif.get("model") or "").strip()
    if not make and not model:
        return ""
    return f"{make}|{model}"


def burst_id_for(member_ids: Sequence[str]) -> str:
    """Derive a stable burst id from its members.

    The id depends only on the set of member ids, so re-scanning a library
    reproduces exactly the same burst ids and previously judged bursts stay
    judged.
    """
    digest = hashlib.sha1("\n".join(sorted(member_ids)).encode()).hexdigest()
    return f"b{digest[:_BURST_ID_CHARS]}"


def _is_complete_jpeg(path: Path) -> bool:
    """Cheap sanity check that a cached thumbnail is a whole JPEG.

    A scan interrupted mid-write used to leave a truncated file that the cache
    then trusted forever — erroring the asset on every later run, or shipping
    a corrupt image to the API.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return False
    return len(blob) > 4 and blob.startswith(_JPEG_SOI) and blob.rstrip().endswith(_JPEG_EOI)


def _write_thumb(path: Path, jpeg: bytes) -> None:
    """Write a thumbnail atomically, so an interruption cannot leave a stub."""
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_bytes(jpeg)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _thumbnail_and_hash(
    settings: Settings,
    immich: ImmichClient,
    asset_id: str,
    known_hash: int | None,
    stats: ScanStats,
) -> tuple[Path, int]:
    """Ensure a cached thumbnail exists for ``asset_id`` and return it with its hash."""
    path = thumb_path(settings.cache_dir, asset_id)
    if _is_complete_jpeg(path):
        stats.thumbs_cached += 1
        if known_hash is not None:
            return path, known_hash
        return path, dhash(path.read_bytes())
    raw = immich.fetch_thumbnail(asset_id)
    jpeg = prepare_image(raw, settings.image_edge, settings.jpeg_quality)
    _write_thumb(path, jpeg)
    stats.thumbs_fetched += 1
    return path, dhash(jpeg)


def _duplicate_groups(
    immich: ImmichClient, known_ids: set[str], stats: ScanStats
) -> list[list[str]]:
    """Immich duplicate groups, restricted to assets the ledger knows about."""
    try:
        raw_groups = immich.duplicates()
    except ImmichError:
        stats.errors += 1
        return []
    groups: list[list[str]] = []
    for entry in raw_groups or []:
        members = [
            str(a.get("id"))
            for a in (entry.get("assets") or [])
            if isinstance(a, dict) and str(a.get("id")) in known_ids
        ]
        if len(members) >= 2:
            groups.append(members)
    return groups


def _ledger_lites(ledger: Ledger) -> list[AssetLite]:
    """Every ledger asset that has a usable timestamp, as burst-grouping input."""
    lites: list[AssetLite] = []
    for row in ledger.get_assets():
        moment = _parse_dt(row.get("taken_at"))
        if moment is None:
            continue
        digest = row.get("dhash")
        lites.append(
            AssetLite(
                id=str(row["id"]),
                taken_at=moment,
                camera=row.get("camera") or "",
                dhash=digest if isinstance(digest, int) else None,
            )
        )
    return lites


def run_scan(
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
    taken_after: str | date | datetime,
    taken_before: str | date | datetime,
    *,
    on_progress: ProgressFn | None = None,
) -> ScanStats:
    """Scan a date window of Immich images into the ledger.

    For every image asset in the window this caches a prepared thumbnail
    (skipping downloads for assets already cached), computes its difference
    hash, and upserts the asset row. Afterwards every asset in the ledger is
    re-grouped into bursts — merged with Immich's duplicate groups — and each
    group's members are stamped with a deterministic burst id.

    Args:
        settings: Runtime configuration (cache dir, image size, burst knobs).
        ledger: Open ledger to write into.
        immich: Connected Immich client.
        taken_after: Inclusive lower bound; a bare ``YYYY-MM-DD`` is midnight UTC.
        taken_before: Upper bound, same formats.
        on_progress: Optional callback receiving one short line per asset.

    Returns:
        Counters describing the scan.
    """
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stats = ScanStats()
    known = {str(row["id"]): row for row in ledger.get_assets()}
    rows: list[dict[str, Any]] = []
    scanned_ids: list[str] = []

    def flush() -> None:
        """Persist what has been prepared so far and drop it from memory."""
        if rows:
            ledger.upsert_assets(rows)
            rows.clear()

    try:
        for asset in immich.search_assets(iso_bound(taken_after), iso_bound(taken_before)):
            asset_id = str(asset.get("id") or "")
            if not asset_id or asset.get("type") not in (None, "IMAGE"):
                continue
            stats.seen += 1
            prior = known.get(asset_id)
            prior_hash = prior.get("dhash") if prior else None
            try:
                path, digest = _thumbnail_and_hash(
                    settings,
                    immich,
                    asset_id,
                    prior_hash if isinstance(prior_hash, int) else None,
                    stats,
                )
            except (ImmichError, OSError, ValueError) as exc:
                stats.errors += 1
                emit(on_progress, f"skipped {asset_id}: {exc}")
                continue

            exif = asset.get("exifInfo") or {}
            moment = asset_taken_at(asset)
            rows.append(
                {
                    "id": asset_id,
                    "filename": asset.get("originalFileName"),
                    "taken_at": moment.isoformat() if moment else None,
                    "camera": asset_camera(asset),
                    "width": exif.get("exifImageWidth"),
                    "height": exif.get("exifImageHeight"),
                    "dhash": digest,
                    "thumb_path": str(path),
                    # Immich v3 keeps the star rating on exifInfo; the top-level
                    # field exists in the DTO but stays null even when rated.
                    "immich_rating": exif.get("rating", asset.get("rating")),
                }
            )
            scanned_ids.append(asset_id)
            if prior is None:
                stats.new += 1
            emit(on_progress, f"scanned {asset_id}")
            if len(rows) >= _FLUSH_EVERY:
                flush()
    finally:
        # keep whatever was prepared before a mid-scan API failure
        flush()

    known_ids = set(known) | set(scanned_ids)
    groups = group_bursts(
        _ledger_lites(ledger),
        gap_seconds=settings.burst_gap_seconds,
        max_group=settings.burst_max_group,
        dhash_max_distance=settings.dhash_max_distance,
    )
    merged = merge_duplicate_groups(
        groups,
        _duplicate_groups(immich, known_ids, stats),
        max_group=settings.burst_max_group,
    )
    # Grouping is re-derived over the whole ledger, so last run's stamps must
    # go first: an asset that drops out of a group would otherwise keep a
    # stale burst id and become invisible to every triage queue.
    ledger.clear_bursts()
    for members in merged:
        ledger.assign_burst(burst_id_for(members), members)
        stats.bursts += 1
        stats.burst_assets += len(members)
    emit(on_progress, f"grouped {stats.bursts} bursts over {stats.burst_assets} assets")
    return stats
