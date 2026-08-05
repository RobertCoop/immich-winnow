"""Write-back — turn ledger decisions into reversible Immich changes.

Every decision Winnow makes maps to something a human can undo from the Immich
UI in seconds: a rating, a favorite, a tag, an archive flag, a stack. Nothing
here deletes anything.

The stage is deliberately split in two. :func:`plan` reads the ledger and
returns :class:`Action` objects describing exactly what *would* happen, with no
network access at all; :func:`apply` executes a plan, and defaults to a dry run
so the destructive-looking half is always opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from winnow.config import Settings
from winnow.immich import ImmichClient, ImmichError
from winnow.ledger import Ledger
from winnow.pipeline.scan import ProgressFn, emit

__all__ = [
    "ALL_GROUPS",
    "BUCKET_GROUPS",
    "NONPHOTO_TAGS",
    "TAG_BEST",
    "TAG_BURST_LOSER",
    "TAG_NONPHOTO",
    "TAG_REJECT",
    "Action",
    "ApplyStats",
    "apply",
    "plan",
]

TAG_REJECT = "winnow/reject"
TAG_BEST = "winnow/best"
TAG_BURST_LOSER = "winnow/burst-loser"
TAG_NONPHOTO = "winnow/nonphoto"

#: Per-category tags for things that are not photographs.
NONPHOTO_TAGS: dict[str, str] = {
    "screenshot": "winnow/screenshot",
    "document": "winnow/document",
    "meme": "winnow/meme",
    "other": "winnow/other",
}

#: Decision bucket -> the ``--buckets`` group that switches it on.
BUCKET_GROUPS: dict[str, str] = {
    "reject": "reject",
    "nonphoto": "nonphoto",
    "five_star": "stars",
    "four_star": "stars",
    "three_star": "stars",
    "two_star": "stars",
    "one_star": "stars",
    "album": "stars",
    "burst_loser": "stacks",
    "burst_stack": "stacks",
    "enrich": "enrich",
}

#: Every group name :func:`apply` accepts.
ALL_GROUPS: frozenset[str] = frozenset(BUCKET_GROUPS.values())


@dataclass(frozen=True)
class Action:
    """One reversible change queued against Immich.

    Attributes:
        asset_id: Asset the action is about, or ``None`` for group actions.
        burst_id: Burst the action is about, for stacking.
        bucket: Decision bucket that produced it.
        description: Human-readable one-liner for the plan output.
        api_ops: Ordered primitive operations — ``update_asset``, ``tag`` or
            ``stack`` — each a dict of keyword arguments plus an ``op`` key.
    """

    asset_id: str | None
    burst_id: str | None
    bucket: str
    description: str
    api_ops: list[dict[str, Any]] = field(default_factory=list)

    @property
    def group(self) -> str:
        """The ``--buckets`` group this action belongs to."""
        return BUCKET_GROUPS.get(self.bucket, self.bucket)


@dataclass
class ApplyStats:
    """What an :func:`apply` call did (or would have done).

    Attributes:
        planned: Actions the ledger produced.
        selected: Actions matching the requested groups.
        applied: Actions carried out successfully.
        failed: Actions that raised an Immich error.
        assets_updated: ``PUT /assets/{id}`` calls made.
        assets_tagged: Asset/tag attachments made.
        tags_resolved: Tag names resolved to ids.
        stacks_created: Burst stacks created.
        enriched: Assets whose caption/keywords were written.
        dry_run: Whether this was a rehearsal.
        actions: The selected actions, in execution order.
    """

    planned: int = 0
    selected: int = 0
    applied: int = 0
    failed: int = 0
    assets_updated: int = 0
    assets_tagged: int = 0
    tags_resolved: int = 0
    stacks_created: int = 0
    album_assets: int = 0
    enriched: int = 0
    dry_run: bool = True
    actions: list[Action] = field(default_factory=list)


def _update_op(asset_id: str, **fields: Any) -> dict[str, Any]:
    return {"op": "update_asset", "asset_id": asset_id, **fields}


def _tag_op(asset_id: str, tag: str) -> dict[str, Any]:
    return {"op": "tag", "asset_id": asset_id, "tag": tag}


def _action_for(
    asset_id: str, bucket: str, detail: Any, *, favorite_five: bool = True
) -> Action | None:
    """Build the action for one decision, or ``None`` when it means 'do nothing'."""
    info = detail if isinstance(detail, dict) else {}
    if bucket == "reject":
        # Archive is what actually hides the photo: Immich v3.1 accepts
        # rating=-1 in the request DTO but silently drops it on write (1-5
        # persist fine), so the rating is best-effort forward compatibility,
        # not the mechanism.
        return Action(
            asset_id=asset_id,
            burst_id=None,
            bucket=bucket,
            description=f"reject {asset_id}: archive + tag {TAG_REJECT} (+ rating -1)",
            api_ops=[
                _update_op(asset_id, rating=-1, visibility="archive"),
                _tag_op(asset_id, TAG_REJECT),
            ],
        )
    if bucket == "nonphoto":
        category = str(info.get("category") or "")
        tag = NONPHOTO_TAGS.get(category, TAG_NONPHOTO)
        return Action(
            asset_id=asset_id,
            burst_id=None,
            bucket=bucket,
            description=f"archive {asset_id} ({category or 'non-photo'}) + tag {tag}",
            api_ops=[_update_op(asset_id, visibility="archive"), _tag_op(asset_id, tag)],
        )
    if bucket == "burst_loser":
        return Action(
            asset_id=asset_id,
            burst_id=str(info.get("burst_id") or "") or None,
            bucket=bucket,
            description=f"burst also-ran {asset_id}: tag {TAG_BURST_LOSER}",
            api_ops=[_tag_op(asset_id, TAG_BURST_LOSER)],
        )
    if bucket == "five_star":
        extras = " + favorite" if favorite_five else ""
        update = (
            _update_op(asset_id, rating=5, is_favorite=True)
            if favorite_five
            else _update_op(asset_id, rating=5)
        )
        return Action(
            asset_id=asset_id,
            burst_id=None,
            bucket=bucket,
            description=f"crown {asset_id}: rating 5{extras} + tag {TAG_BEST}",
            api_ops=[update, _tag_op(asset_id, TAG_BEST)],
        )
    if bucket == "four_star":
        return Action(
            asset_id=asset_id,
            burst_id=None,
            bucket=bucket,
            description=f"rate {asset_id}: rating 4",
            api_ops=[_update_op(asset_id, rating=4)],
        )
    if bucket in ("three_star", "two_star", "one_star"):
        stars = {"three_star": 3, "two_star": 2, "one_star": 1}[bucket]
        return Action(
            asset_id=asset_id,
            burst_id=None,
            bucket=bucket,
            description=f"rate {asset_id}: rating {stars}",
            api_ops=[_update_op(asset_id, rating=stars)],
        )
    return None


def _stack_actions(ledger: Ledger, unapplied_only: bool) -> list[Action]:
    """One stack action per judged burst that is still live and unstacked.

    Stacking is tracked on the burst itself (``bursts.applied_at``) rather than
    on its losers' decisions: it is a group action with no asset of its own, so
    riding on the losers would mean a stack is either lost forever (the tag
    succeeded, the stack did not) or created twice (the stack succeeded, the
    tag did not).

    Bursts are also intersected with the *current* grouping. Burst ids are
    content-addressed, so a re-scan that changes a group's membership mints a
    new id and leaves the old verdict row behind; planning from those orphans
    would stack overlapping sets of the same photos.
    """
    actions: list[Action] = []
    groups = ledger.burst_groups()
    for row in ledger.burst_rows():
        burst_id = str(row["burst_id"])
        live = set(groups.get(burst_id, ()))
        if not live:
            continue  # superseded by a later grouping
        if unapplied_only and row.get("applied_at"):
            continue
        winner = str(row.get("winner_id") or "")
        losers = [str(i) for i in (row.get("reject_ids") or []) if str(i) in live]
        if not winner or winner not in live or not losers:
            continue
        members = [winner, *losers]
        actions.append(
            Action(
                asset_id=None,
                burst_id=burst_id,
                bucket="burst_stack",
                description=f"stack {len(members)} frames under {winner}",
                api_ops=[{"op": "stack", "asset_ids": members}],
            )
        )
    return actions


def _enrich_actions(
    ledger: Ledger,
    *,
    write_captions: bool,
    keyword_tags: bool,
    keyword_prefix: str,
) -> list[Action]:
    """Caption/keyword write-backs for every triaged photo not yet enriched.

    Captions only ever land on an EMPTY Immich description (as captured at
    scan time) — a description the user wrote is never overwritten. Keywords
    become ordinary Immich tags, nested under ``keyword_prefix`` when one is
    set, so they browse and filter exactly like hand-made tags.
    """
    if not write_captions and not keyword_tags:
        return []
    actions: list[Action] = []
    for row in ledger.unenriched_rows():
        asset_id = str(row["asset_id"])
        ops: list[dict[str, Any]] = []
        pieces: list[str] = []
        caption = str(row.get("caption") or "").strip()
        if write_captions and caption and not (row.get("immich_description") or "").strip():
            ops.append(_update_op(asset_id, description=caption))
            pieces.append(f"caption {caption!r}")
        if keyword_tags:
            keywords = [str(k) for k in row.get("keywords") or []]
            names = [f"{keyword_prefix}/{k}" if keyword_prefix else k for k in keywords]
            ops.extend(_tag_op(asset_id, name) for name in names)
            if names:
                pieces.append(f"tags {', '.join(names)}")
        if ops:
            actions.append(
                Action(
                    asset_id=asset_id,
                    burst_id=None,
                    bucket="enrich",
                    description=f"enrich {asset_id}: {'; '.join(pieces)}",
                    api_ops=ops,
                )
            )
    return actions


def _album_action(ledger: Ledger, album: str, min_stars: int) -> Action | None:
    """One idempotent group action collecting starred photos into an album.

    Membership comes from *all* star decisions (applied or not): adding an
    already-present asset to an Immich album is a no-op on the server, so the
    action can be replanned every run and newly-configured albums pick up
    photos crowned in earlier runs.
    """
    wanted = {"five_star"} if min_stars >= 5 else {"five_star", "four_star"}
    asset_ids = sorted(
        str(row["asset_id"])
        for row in ledger.decisions()
        if str(row.get("bucket") or "") in wanted
    )
    if not asset_ids:
        return None
    return Action(
        asset_id=None,
        burst_id=None,
        bucket="album",
        description=f"ensure album {album!r} holds {len(asset_ids)} starred photo(s)",
        api_ops=[{"op": "album", "name": album, "asset_ids": asset_ids}],
    )


def plan(
    ledger: Ledger,
    *,
    unapplied_only: bool = True,
    album: str | None = None,
    album_min_stars: int = 5,
    favorite_five: bool = True,
    write_captions: bool = False,
    keyword_tags: bool = False,
    keyword_prefix: str = "kw",
) -> list[Action]:
    """Describe every Immich change the ledger's decisions call for.

    Args:
        ledger: Open ledger.
        unapplied_only: Skip decisions already marked applied (the default),
            which is what makes ``apply`` resumable.
        album: Album name to collect starred photos into; ``None``/empty
            disables the album action.
        album_min_stars: Smallest star band included in the album.
        favorite_five: Whether five-star photos are also favorited.
        write_captions: Write captions to empty Immich descriptions.
        keyword_tags: Write keywords as Immich tags under ``keyword_prefix``.
        keyword_prefix: Parent tag for keywords; ``""`` writes them top-level.

    Returns:
        Actions in execution order: per-asset changes sorted by asset id,
        then one stack per burst, then the album roll-up.
    """
    actions: list[Action] = []
    for row in ledger.decisions(unapplied_only=unapplied_only):
        asset_id = str(row["asset_id"])
        bucket = str(row.get("bucket") or "")
        action = _action_for(asset_id, bucket, row.get("detail"), favorite_five=favorite_five)
        if action is not None:
            actions.append(action)
    actions.extend(_stack_actions(ledger, unapplied_only))
    actions.extend(
        _enrich_actions(
            ledger,
            write_captions=write_captions,
            keyword_tags=keyword_tags,
            keyword_prefix=keyword_prefix,
        )
    )
    if album:
        album_action = _album_action(ledger, album, album_min_stars)
        if album_action is not None:
            actions.append(album_action)
    return actions


def _collect(actions: list[Action]) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Fold per-action ops into one tag->assets map and one asset->fields map."""
    by_tag: dict[str, list[str]] = {}
    updates: dict[str, dict[str, Any]] = {}
    for action in actions:
        for op in action.api_ops:
            if op["op"] == "tag":
                by_tag.setdefault(str(op["tag"]), []).append(str(op["asset_id"]))
            elif op["op"] == "update_asset":
                fields = {k: v for k, v in op.items() if k not in ("op", "asset_id")}
                updates.setdefault(str(op["asset_id"]), {}).update(fields)
    return by_tag, updates


def apply(
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
    buckets: set[str] | None = None,
    dry_run: bool = True,
    *,
    album: str | None = None,
    on_progress: ProgressFn | None = None,
) -> ApplyStats:
    """Execute the plan against Immich.

    Tag names are resolved once up front, tagging is batched per tag, and the
    per-asset updates are merged so an asset is written at most once. Assets
    whose calls all succeeded are marked applied in the ledger, and a created
    stack is stamped on its burst, so a repeat run only retries what failed and
    never stacks the same frames twice.

    Args:
        settings: Runtime configuration — supplies ``best_album``,
            ``best_album_min_stars`` and ``five_star_favorite``.
        ledger: Open ledger.
        immich: Connected Immich client.
        buckets: Groups to act on — any of ``reject``, ``nonphoto``, ``stars``,
            ``stacks``. ``None`` means all of them.
        dry_run: When true (the default) nothing is sent and the plan is
            returned as-is.
        album: Album name override; ``None`` falls back to
            ``settings.best_album`` and ``""`` disables the album for this run.
        on_progress: Optional callback receiving one short line per step.

    Returns:
        Counters plus the selected actions.
    """
    groups = ALL_GROUPS if buckets is None else {str(b) for b in buckets}
    album_name = settings.best_album if album is None else album
    everything = plan(
        ledger,
        album=album_name or None,
        album_min_stars=settings.best_album_min_stars,
        favorite_five=settings.five_star_favorite,
        write_captions=settings.write_captions,
        keyword_tags=settings.keyword_tags,
        keyword_prefix=settings.keyword_tag_prefix,
    )
    selected = [action for action in everything if action.group in groups]
    stats = ApplyStats(
        planned=len(everything),
        selected=len(selected),
        dry_run=dry_run,
        actions=selected,
    )
    if dry_run or not selected:
        emit(on_progress, f"{len(selected)} action(s) planned; nothing written")
        return stats

    by_tag, updates = _collect(selected)
    failed_assets: set[str] = set()
    stacked: list[str] = []

    tag_ids: dict[str, str] = {}
    if by_tag:
        try:
            tag_ids = immich.upsert_tags(sorted(by_tag))
        except ImmichError as exc:
            emit(on_progress, f"tag upsert failed: {exc}")
        stats.tags_resolved = len(tag_ids)

    for action in selected:
        for op in action.api_ops:
            if op["op"] != "stack":
                continue
            try:
                immich.create_stack(list(op["asset_ids"]))
                stats.stacks_created += 1
                stats.applied += 1
                if action.burst_id:
                    stacked.append(action.burst_id)
            except ImmichError as exc:
                stats.failed += 1
                emit(on_progress, f"stack {action.burst_id} failed: {exc}")

    for name, asset_ids in by_tag.items():
        unique = list(dict.fromkeys(asset_ids))
        tag_id = tag_ids.get(name)
        if tag_id is None:
            failed_assets.update(unique)
            emit(on_progress, f"tag {name} could not be resolved")
            continue
        try:
            immich.tag_assets(tag_id, unique)
            stats.assets_tagged += len(unique)
        except ImmichError as exc:
            failed_assets.update(unique)
            emit(on_progress, f"tagging {name} failed: {exc}")

    for asset_id, fields in updates.items():
        try:
            immich.update_asset(asset_id, **fields)
            stats.assets_updated += 1
        except ImmichError as exc:
            failed_assets.add(asset_id)
            emit(on_progress, f"update {asset_id} failed: {exc}")

    for action in selected:
        for op in action.api_ops:
            if op["op"] != "album":
                continue
            name = str(op["name"])
            members = list(op["asset_ids"])
            try:
                album_id = immich.upsert_album(name)
                immich.add_album_assets(album_id, members)
                stats.album_assets += len(members)
                stats.applied += 1
                emit(on_progress, f"album {name!r}: ensured {len(members)} member(s)")
            except ImmichError as exc:
                stats.failed += 1
                emit(on_progress, f"album {name!r} failed: {exc}")

    if stacked:
        ledger.mark_stacked(stacked)
    enrich_ids = {a.asset_id for a in selected if a.bucket == "enrich" and a.asset_id}
    enriched_done = sorted(enrich_ids - failed_assets)
    if enriched_done:
        ledger.mark_enriched(enriched_done)
        stats.enriched = len(enriched_done)
    done = sorted({a.asset_id for a in selected if a.asset_id} - failed_assets)
    if done:
        ledger.mark_applied(done)
    stats.applied += len(done)
    stats.failed += len(failed_assets)
    summary = f"applied {stats.applied} action(s)"
    if stats.failed:
        summary += f", {stats.failed} failed"
    emit(on_progress, summary)
    return stats
