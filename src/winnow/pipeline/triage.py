"""Stage 1 — triage every scanned photo, and settle bursts with one call.

Bursts are resolved first: the frames of a group are shown together in a
single "pick the best" call, the winner is recorded, and the also-rans are
bucketed as ``burst_loser`` without ever being judged individually. Whatever
survives a burst (the winner, plus any frame the model deliberately kept) then
goes through normal single-photo triage along with every ungrouped asset.

Both paths — direct calls and the 50%-cheaper Batch API — share the same
recording logic, so a library can be triaged either way (or half one, half the
other) and the ledger looks the same. Every entry point skips work the ledger
already has, which makes an interrupted run safe to simply repeat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from winnow.config import Settings
from winnow.judge import (
    FATAL_ERRORS,
    ITEM_ERRORS,
    Judge,
    JudgeError,
    batch_status,
    build_burst_request,
    build_triage_request,
    extract_text,
    iter_batch_results,
    parse_message,
    submit_batch,
    to_batch_request,
)
from winnow.ledger import Ledger
from winnow.pipeline.scan import ProgressFn, emit, load_thumb_b64
from winnow.schemas import BurstVerdict, TriageVerdict

__all__ = [
    "BUCKET_MIDDLE",
    "BUCKET_NONPHOTO",
    "BUCKET_REJECT",
    "TriageStats",
    "apply_burst_verdict",
    "apply_triage_verdict",
    "ingest_triage_batch",
    "pending_burst_ids",
    "pending_single_ids",
    "run_triage_direct",
    "submit_triage_batch",
]

BUCKET_REJECT = "reject"
BUCKET_NONPHOTO = "nonphoto"
BUCKET_MIDDLE = "middle"
BUCKET_BURST_LOSER = "burst_loser"

#: Batch kind recorded for triage submissions.
BATCH_KIND = "triage"

#: Conservative ceilings for one Batch API submission. Anthropic allows
#: 100k requests / 256 MB per batch; a 768px thumbnail is ~135 KB once
#: base64-encoded, so a real library blows the size cap long before the count
#: cap. Anything larger is split across several batches.
MAX_BATCH_REQUESTS = 5_000
MAX_BATCH_BYTES = 180 * 1024 * 1024

_TRIAGE_PREFIX = "triage_"
_BURST_PREFIX = "burst_"


@dataclass
class TriageStats:
    """What a triage run judged and decided.

    Attributes:
        bursts_judged: Burst groups resolved by a pick-the-best call.
        burst_losers: Frames bucketed as redundant burst also-rans.
        singles_judged: Photos judged individually.
        rejects: High-confidence rejects bucketed for write-back.
        nonphotos: Screenshots, documents, memes and friends.
        candidates: Photos the model called standouts.
        middles: Photos left in the untouched middle.
        errors: Items skipped after an unusable reply or a missing thumbnail.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens consumed.
    """

    bursts_judged: int = 0
    burst_losers: int = 0
    singles_judged: int = 0
    rejects: int = 0
    nonphotos: int = 0
    candidates: int = 0
    middles: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _bucket_for(verdict: TriageVerdict) -> str:
    """Map a triage verdict onto a write-back bucket.

    Both destructive-looking buckets need the model to be *sure*: a non-photo
    is archived out of the timeline and a reject is rated -1, so a hesitant
    call in either direction stays in the untouched middle. That matters most
    for the catch-all categories (``meme``, ``other``), which would otherwise
    archive wallpapers, illustrations and scans on a shrug.
    """
    if verdict.category != "photo":
        return BUCKET_NONPHOTO if verdict.confidence == "high" else BUCKET_MIDDLE
    if verdict.verdict == "reject" and verdict.confidence == "high":
        return BUCKET_REJECT
    return BUCKET_MIDDLE


def _count_bucket(stats: TriageStats, bucket: str, verdict: TriageVerdict) -> None:
    """Fold one decision into the running counters."""
    stats.singles_judged += 1
    if bucket == BUCKET_REJECT:
        stats.rejects += 1
    elif bucket == BUCKET_NONPHOTO:
        stats.nonphotos += 1
    else:
        stats.middles += 1
    if verdict.verdict == "candidate":
        stats.candidates += 1


def apply_triage_verdict(
    ledger: Ledger,
    asset_id: str,
    verdict: TriageVerdict,
    model: str,
    raw: str | None = None,
) -> str:
    """Record a single-photo verdict and its decision bucket.

    Non-photographs go to ``nonphoto``; a ``reject`` verdict counts only when
    the model was *sure* (confidence ``high``), which keeps the asymmetric
    caution the prompts promise. Everything else lands in the untouched
    ``middle``.

    Returns:
        The bucket the asset was filed under.
    """
    ledger.record_triage(asset_id, verdict, model, raw)
    bucket = _bucket_for(verdict)
    ledger.set_decision(
        asset_id,
        bucket,
        {
            "category": verdict.category,
            "verdict": verdict.verdict,
            "technical_score": verdict.technical_score,
            "confidence": verdict.confidence,
            "reasons": list(verdict.reasons),
        },
    )
    return bucket


def apply_burst_verdict(
    ledger: Ledger,
    burst_id: str,
    members: Sequence[str],
    verdict: BurstVerdict,
    model: str,
) -> tuple[str, list[str]]:
    """Record a burst's winner and bucket its also-rans.

    Args:
        ledger: Ledger to write into.
        burst_id: Group being resolved.
        members: Member asset ids, in the order they were shown to the model.
        verdict: The model's 1-based pick.
        model: Model id that produced the verdict.

    Returns:
        ``(winner_id, loser_ids)``.

    Raises:
        ValueError: If ``best_index`` does not address a member.
    """
    ordered = list(members)
    if not 1 <= verdict.best_index <= len(ordered):
        raise ValueError(f"best_index {verdict.best_index} out of range for {len(ordered)} frames")
    winner = ordered[verdict.best_index - 1]
    losers = [
        ordered[index - 1]
        for index in dict.fromkeys(verdict.reject_indices)
        if 1 <= index <= len(ordered) and ordered[index - 1] != winner
    ]
    ledger.record_burst(burst_id, winner, losers, verdict.note, model)
    for loser in losers:
        ledger.set_decision(
            loser,
            BUCKET_BURST_LOSER,
            {"burst_id": burst_id, "winner_id": winner, "note": verdict.note},
        )
    # Re-judging a regrouped burst can promote a frame that used to lose. Drop
    # its stale bucket, or it stays tagged as an also-ran and is barred from
    # the stage-2 candidate pool forever.
    kept = set(ordered) - set(losers)
    promoted = [
        row["asset_id"]
        for row in ledger.decisions(bucket=BUCKET_BURST_LOSER)
        if row["asset_id"] in kept
    ]
    if promoted:
        ledger.clear_decisions(promoted)
    return winner, losers


def pending_burst_ids(ledger: Ledger) -> list[str]:
    """Burst groups with at least two members, no verdict, and no open batch.

    Excluding groups already sitting in an uningested batch is what stops a
    second ``triage --batch`` from re-billing (and stealing the item rows of)
    the first one.
    """
    groups = ledger.burst_groups()
    inflight = ledger.inflight_custom_ids()
    return [
        bid
        for bid in ledger.unjudged_burst_ids()
        if len(groups.get(bid, ())) >= 2 and f"{_BURST_PREFIX}{bid}" not in inflight
    ]


def pending_single_ids(ledger: Ledger) -> list[str]:
    """Assets still owed an individual triage call.

    That is every ungrouped asset without a verdict, plus the survivors of
    already-resolved bursts: the winner and any frame the model chose not to
    reject. Members of a group that has shrunk below two frames are rescued
    too — a group of one is not a burst, and nothing else would ever judge it.
    Anything already queued in an open batch is held back.
    """
    judged = {row["asset_id"] for row in ledger.triage_rows()}
    groups = ledger.burst_groups()
    inflight = ledger.inflight_custom_ids()
    pending: list[str] = list(ledger.unjudged_asset_ids(exclude_bursts=True))
    for members in groups.values():
        if len(members) < 2:
            pending.extend(members)
    for row in ledger.burst_rows():
        rejected = set(row.get("reject_ids") or ())
        for member in groups.get(row["burst_id"], ()):
            if member not in rejected:
                pending.append(member)
    return [
        aid
        for aid in dict.fromkeys(pending)
        if aid not in judged and f"{_TRIAGE_PREFIX}{aid}" not in inflight
    ]


def _spend(budget: int | None) -> tuple[bool, int | None]:
    """Consume one unit of an optional work budget."""
    if budget is None:
        return True, None
    if budget <= 0:
        return False, budget
    return True, budget - 1


def run_triage_direct(
    settings: Settings,
    ledger: Ledger,
    judge: Judge,
    limit: int | None = None,
    on_progress: ProgressFn | None = None,
) -> TriageStats:
    """Triage everything the ledger still owes a verdict, one call at a time.

    Bursts are resolved first so their winners join the single-photo queue in
    the same run. Anything already recorded is skipped, so re-running after an
    interruption picks up exactly where it stopped.

    Args:
        settings: Runtime configuration (triage model, cache dir).
        ledger: Open ledger.
        judge: Judge wrapping an Anthropic client.
        limit: Optional cap on work items, where one burst group and one
            single photo each count as one item.
        on_progress: Optional callback receiving one short line per item.

    Returns:
        Counters describing the run.
    """
    stats = TriageStats()
    budget = limit
    groups = ledger.burst_groups()

    for burst_id in pending_burst_ids(ledger):
        allowed, budget = _spend(budget)
        if not allowed:
            return stats
        members = list(groups.get(burst_id, ()))
        try:
            images = [load_thumb_b64(settings.cache_dir, member) for member in members]
            result = judge.burst(settings.triage_model, images)
            winner, losers = apply_burst_verdict(
                ledger, burst_id, members, result.verdict, result.model
            )
        except FATAL_ERRORS:
            raise
        except ITEM_ERRORS as exc:
            stats.errors += 1
            emit(on_progress, f"burst {burst_id} failed: {exc}")
            continue
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        stats.bursts_judged += 1
        stats.burst_losers += len(losers)
        emit(on_progress, f"burst {burst_id}: kept {winner}, dropped {len(losers)}")

    for asset_id in pending_single_ids(ledger):
        allowed, budget = _spend(budget)
        if not allowed:
            return stats
        try:
            image = load_thumb_b64(settings.cache_dir, asset_id)
            result = judge.triage(settings.triage_model, image)
        except FATAL_ERRORS:
            raise
        except ITEM_ERRORS as exc:
            stats.errors += 1
            emit(on_progress, f"triage {asset_id} failed: {exc}")
            continue
        bucket = apply_triage_verdict(ledger, asset_id, result.verdict, result.model)
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        _count_bucket(stats, bucket, result.verdict)
        emit(on_progress, f"triage {asset_id}: {bucket}")

    return stats


class _BatchBuilder:
    """Accumulates batch requests and flushes them before the API's limits.

    Anthropic caps a batch at 100k requests and 256 MB, and the images ride
    inline as base64, so a real library has to be split across several jobs.
    Each flush submits and records one batch, then releases its payloads.
    """

    def __init__(
        self,
        ledger: Ledger,
        judge: Judge,
        on_progress: ProgressFn | None,
    ) -> None:
        self.ledger = ledger
        self.judge = judge
        self.on_progress = on_progress
        self.batch_ids: list[str] = []
        self._requests: list[Any] = []
        self._items: dict[str, dict[str, Any]] = {}
        self._bytes = 0

    def add(self, custom_id: str, request: Any, payload: dict[str, Any], size: int) -> None:
        """Queue one request, flushing first when it would breach a limit."""
        if self._requests and (
            len(self._requests) >= MAX_BATCH_REQUESTS or self._bytes + size > MAX_BATCH_BYTES
        ):
            self.flush()
        self._requests.append(request)
        self._items[custom_id] = payload
        self._bytes += size

    def flush(self) -> None:
        """Submit and record whatever is queued."""
        if not self._requests:
            return
        count = len(self._requests)
        batch_id = submit_batch(self.judge.client, self._requests)
        self.ledger.add_batch(batch_id, BATCH_KIND, self._items)
        self.batch_ids.append(batch_id)
        self._requests = []
        self._items = {}
        self._bytes = 0
        emit(self.on_progress, f"submitted batch {batch_id} with {count} requests")


def submit_triage_batch(
    settings: Settings,
    ledger: Ledger,
    judge: Judge,
    limit: int | None = None,
    on_progress: ProgressFn | None = None,
) -> str | None:
    """Submit all outstanding triage work through the Batch API.

    Custom ids follow Winnow's ``kind_id`` convention (``burst_<burst_id>``,
    ``triage_<asset_id>``) because Anthropic forbids ``:`` in a ``custom_id``.
    Only the routing payload is stored in the ledger — never the base64 images.

    Work already sitting in an uningested batch is skipped, so calling this
    twice in a row never double-bills. Anything too large for a single job is
    split across several batches; ``poll``/``ingest`` walk every open batch, so
    only the first id is returned.

    Note that a burst's winner cannot be triaged in the same batch as the burst
    itself: submit again after ingesting to pick the winners up.

    Args:
        settings: Runtime configuration.
        ledger: Open ledger.
        judge: Judge whose ``client`` is used for the submission.
        limit: Optional cap on requests, counting bursts and singles alike.
        on_progress: Optional callback receiving one short line per request.

    Returns:
        The first batch id, or ``None`` when there was nothing left to judge.
    """
    builder = _BatchBuilder(ledger, judge, on_progress)
    budget = limit
    groups = ledger.burst_groups()

    for burst_id in pending_burst_ids(ledger):
        allowed, budget = _spend(budget)
        if not allowed:
            break
        members = list(groups.get(burst_id, ()))
        custom_id = f"{_BURST_PREFIX}{burst_id}"
        try:
            images = [load_thumb_b64(settings.cache_dir, member) for member in members]
            kwargs = build_burst_request(settings.triage_model, images)
            request = to_batch_request(custom_id, kwargs)
        except (JudgeError, OSError) as exc:
            emit(on_progress, f"burst {burst_id} not queued: {exc}")
            continue
        payload = {"kind": "burst", "burst_id": burst_id, "member_ids": members}
        builder.add(custom_id, request, payload, sum(len(image) for image in images))

    for asset_id in pending_single_ids(ledger):
        allowed, budget = _spend(budget)
        if not allowed:
            break
        custom_id = f"{_TRIAGE_PREFIX}{asset_id}"
        try:
            image = load_thumb_b64(settings.cache_dir, asset_id)
            kwargs = build_triage_request(settings.triage_model, image)
            request = to_batch_request(custom_id, kwargs)
        except (JudgeError, OSError) as exc:
            emit(on_progress, f"triage {asset_id} not queued: {exc}")
            continue
        builder.add(custom_id, request, {"kind": "triage", "asset_id": asset_id}, len(image))

    builder.flush()
    if not builder.batch_ids:
        emit(on_progress, "nothing to submit")
        return None
    return builder.batch_ids[0]


def _message_tokens(message: Any) -> tuple[int, int]:
    """Prompt/completion tokens of a batch reply, defaulting to zero."""
    usage = getattr(message, "usage", None)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _ingest_one(
    settings: Settings,
    ledger: Ledger,
    stats: TriageStats,
    custom_id: str,
    message: Any,
    payload: dict[str, Any],
    on_progress: ProgressFn | None,
) -> None:
    """Record one successful batch reply, routed on its custom-id prefix."""
    model = str(getattr(message, "model", None) or settings.triage_model)
    input_tokens, output_tokens = _message_tokens(message)
    raw = extract_text(message)

    if custom_id.startswith(_BURST_PREFIX):
        burst_id = str(payload.get("burst_id") or custom_id[len(_BURST_PREFIX) :])
        members = payload.get("member_ids") or ledger.burst_groups().get(burst_id, [])
        verdict = parse_message(message, BurstVerdict)
        winner, losers = apply_burst_verdict(ledger, burst_id, list(members), verdict, model)
        stats.bursts_judged += 1
        stats.burst_losers += len(losers)
        emit(on_progress, f"burst {burst_id}: kept {winner}, dropped {len(losers)}")
    else:
        asset_id = str(payload.get("asset_id") or custom_id[len(_TRIAGE_PREFIX) :])
        verdict = parse_message(message, TriageVerdict)
        bucket = apply_triage_verdict(ledger, asset_id, verdict, model, raw)
        _count_bucket(stats, bucket, verdict)
        emit(on_progress, f"triage {asset_id}: {bucket}")

    stats.input_tokens += input_tokens
    stats.output_tokens += output_tokens
    ledger.record_batch_result(custom_id, result_json=raw)


def ingest_triage_batch(
    settings: Settings,
    ledger: Ledger,
    judge: Judge,
    batch_id: str | None = None,
    on_progress: ProgressFn | None = None,
) -> TriageStats:
    """Fetch finished batch results and record them exactly like direct calls.

    Args:
        settings: Runtime configuration.
        ledger: Open ledger.
        judge: Judge whose ``client`` is used to read results.
        batch_id: Batch to ingest; ``None`` ingests every open batch that the
            API reports as ``ended``.
        on_progress: Optional callback receiving one short line per result.

    Returns:
        Counters describing everything ingested.
    """
    stats = TriageStats()
    if batch_id is None:
        targets = []
        for row in ledger.open_batches():
            try:
                ended = batch_status(judge.client, row["batch_id"]) == "ended"
            except FATAL_ERRORS:
                raise
            except ITEM_ERRORS as exc:
                emit(on_progress, f"status probe for {row['batch_id']} failed: {exc}")
                continue
            if ended:
                targets.append(row["batch_id"])
    else:
        targets = [batch_id]

    for target in targets:
        payloads = {
            item["custom_id"]: (item.get("payload") or {})
            for item in ledger.batch_items_for(target)
        }
        for custom_id, message, error in iter_batch_results(judge.client, target):
            if error is not None or message is None:
                stats.errors += 1
                ledger.record_batch_result(custom_id, error=error or "missing message")
                emit(on_progress, f"{custom_id} failed: {error}")
                continue
            payload = payloads.get(custom_id) or {}
            try:
                _ingest_one(settings, ledger, stats, custom_id, message, payload, on_progress)
            except FATAL_ERRORS:
                raise
            except ITEM_ERRORS as exc:
                stats.errors += 1
                ledger.record_batch_result(custom_id, error=str(exc))
                emit(on_progress, f"{custom_id} unusable: {exc}")
        ledger.set_batch_status(target, "ingested")

    return stats
