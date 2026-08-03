"""Stage 3 — Swiss-paired head-to-heads over the top of the pool.

The strongest photos from stage 2 are separated by margins too small for a
best-worst set to resolve, so the finals ask the most capable model the
simplest possible question: this one or that one? Pairings are Swiss (closest
scores meet, never a rematch), and every pair is judged **twice with the order
swapped**. Agreement is a real win; disagreement is position bias, and is
recorded as a tie.

Between rounds the Bradley-Terry fit is refreshed over every outcome in the
ledger, so later rounds pair on fresher information. The final ordering is cut
into star bands, which is what the write-back stage turns into ratings.
"""

from __future__ import annotations

from dataclasses import dataclass

from winnow.config import Settings
from winnow.judge import FATAL_ERRORS, ITEM_ERRORS, Judge
from winnow.ledger import Ledger
from winnow.pipeline.rank import collect_pairs, refit_scores
from winnow.pipeline.scan import ProgressFn, emit, load_thumb_b64
from winnow.ranking import TIE, star_bands, swiss_pairs

__all__ = [
    "STAGE",
    "FinalsStats",
    "PairOutcome",
    "judge_pair_twice",
    "run_finals",
]

#: Stage label recorded on every pair the finals judge.
STAGE = "finals"

#: Star value -> decision bucket, for every band the finals can assign.
_STAR_BUCKETS = {5: "five_star", 4: "four_star", 3: "three_star", 2: "two_star", 1: "one_star"}


@dataclass
class FinalsStats:
    """What a :func:`run_finals` call judged and starred.

    Attributes:
        pool: Photos that entered the finals.
        rounds: Swiss rounds actually played.
        pairs_judged: Head-to-heads judged this run (two calls each).
        pairs_skipped: Pairings dropped because they had already been played.
        ties: Head-to-heads recorded as a draw.
        disagreements: Pairs where swapping the order flipped the answer.
        five_star: Photos awarded five stars.
        four_star: Photos awarded four stars.
        three_star: Ranked candidates below the four-star cut (full spectrum).
        two_star: Ordinary keepers rated from triage evidence (full spectrum).
        one_star: Poor-but-kept photos rated from triage evidence (full spectrum).
        protected: Prior five-stars kept at five against a lower fresh ranking.
        errors: Pairs skipped after an unusable reply or a missing thumbnail.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens consumed.
    """

    pool: int = 0
    rounds: int = 0
    pairs_judged: int = 0
    pairs_skipped: int = 0
    ties: int = 0
    disagreements: int = 0
    five_star: int = 0
    four_star: int = 0
    three_star: int = 0
    two_star: int = 0
    one_star: int = 0
    protected: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class PairOutcome:
    """The result of judging one pair in both orders."""

    winner: str
    agreed: bool
    input_tokens: int
    output_tokens: int


def judge_pair_twice(
    settings: Settings,
    judge: Judge,
    a_id: str,
    b_id: str,
) -> PairOutcome:
    """Judge ``a`` against ``b`` twice, swapping which photo is shown first.

    A verdict that survives the swap is a win; a verdict that flips is position
    bias rather than a preference, and comes back as a tie.

    Returns:
        The agreed winner's asset id or ``"tie"``, plus usage for both calls.
    """
    a_b64 = load_thumb_b64(settings.cache_dir, a_id)
    b_b64 = load_thumb_b64(settings.cache_dir, b_id)
    forward = judge.pair(settings.finals_model, a_b64, b_b64)
    reverse = judge.pair(settings.finals_model, b_b64, a_b64)

    first = {"A": a_id, "B": b_id, "tie": TIE}[forward.verdict.winner]
    second = {"A": b_id, "B": a_id, "tie": TIE}[reverse.verdict.winner]
    agreed = first == second
    return PairOutcome(
        winner=first if agreed else TIE,
        agreed=agreed,
        input_tokens=forward.input_tokens + reverse.input_tokens,
        output_tokens=forward.output_tokens + reverse.output_tokens,
    )


def _finals_history(ledger: Ledger) -> set[frozenset[str]]:
    """Pairings already played in the finals, for no-rematch scheduling."""
    history: set[frozenset[str]] = set()
    for row in ledger.pair_rows():
        if row.get("stage") != STAGE:
            continue
        a_id, b_id = row.get("a_id"), row.get("b_id")
        if a_id and b_id and a_id != b_id:
            history.add(frozenset((a_id, b_id)))
    return history


def _pool(ledger: Ledger, size: int) -> list[str]:
    """The top ``size`` scored photos, strongest first."""
    scored = [row for row in ledger.score_rows() if row.get("bt_score") is not None]
    return [str(row["asset_id"]) for row in scored[: max(0, size)]]


def run_finals(
    settings: Settings,
    ledger: Ledger,
    judge: Judge,
    *,
    five_count: int | None = None,
    four_frac: float | None = None,
    allow_demotions: bool = False,
    on_progress: ProgressFn | None = None,
) -> FinalsStats:
    """Play the finals and cut the result into star bands.

    Args:
        settings: Runtime configuration (finals model, pool size, rounds).
        ledger: Open ledger; must already hold stage-2 scores.
        judge: Judge wrapping an Anthropic client.
        five_count: How many photos get five stars; defaults to
            ``settings.five_star_fraction`` of the scored candidates (min 1).
        four_frac: Explicit share of the photos below the five-star cut that
            get four; ``None`` (default) sizes the band from
            ``settings.four_star_fraction`` of the scored candidates.
        allow_demotions: Five stars are sticky by default — a photo that ever
            earned five (from Winnow, or already rated five in Immich itself)
            is never re-scored lower by a later run. Pass ``True`` to let a
            fresh ranking take stars away.
        on_progress: Optional callback receiving one short line per pair.

    Returns:
        Counters describing the run.
    """
    stats = FinalsStats()
    n_scored = sum(1 for row in ledger.score_rows() if row.get("bt_score") is not None)
    five_target = five_count
    if five_target is None:
        five_target = max(1, round(n_scored * settings.five_star_fraction))
    # The Opus pool covers at least twice the crown zone, so every plausible
    # five-star has played head-to-heads before the cut.
    pool = _pool(ledger, max(settings.finals_pool_size, 2 * five_target))
    stats.pool = len(pool)
    if len(pool) < 2:
        emit(on_progress, "finals pool too small to judge")
        return stats

    pool_set = set(pool)
    scores = {
        str(row["asset_id"]): float(row["bt_score"])
        for row in ledger.score_rows()
        if row.get("bt_score") is not None and str(row["asset_id"]) in pool_set
    }
    history = _finals_history(ledger)

    for round_number in range(1, settings.finals_rounds + 1):
        pairings = swiss_pairs(scores, history)
        if not pairings:
            emit(on_progress, f"round {round_number}: no unplayed pairings left")
            break
        for a_id, b_id in pairings:
            key = frozenset((a_id, b_id))
            if key in history:
                stats.pairs_skipped += 1
                continue
            try:
                outcome = judge_pair_twice(settings, judge, a_id, b_id)
            except FATAL_ERRORS:
                raise
            except (*ITEM_ERRORS, KeyError) as exc:
                stats.errors += 1
                emit(on_progress, f"pair {a_id} vs {b_id} failed: {exc}")
                continue
            ledger.record_pair(a_id, b_id, outcome.winner, STAGE, settings.finals_model)
            history.add(key)
            stats.pairs_judged += 1
            stats.input_tokens += outcome.input_tokens
            stats.output_tokens += outcome.output_tokens
            if outcome.winner == TIE:
                stats.ties += 1
            if not outcome.agreed:
                stats.disagreements += 1
            emit(on_progress, f"round {round_number}: {a_id} vs {b_id} -> {outcome.winner}")

        stats.rounds += 1
        ranked = refit_scores(ledger)
        fitted = {item: score for item, score, _rank in ranked}
        scores = {item: fitted.get(item, scores.get(item, 1.0)) for item in pool}

    ranked = refit_scores(ledger, collect_pairs(ledger))
    # Bands cut the GLOBAL ranking, not just the Opus pool — stage-2 evidence
    # is plenty for the four-star band, so star counts scale with the library
    # instead of being capped by the finals pool size.
    order = [item for item, _score, _rank in ranked]
    if not order:
        order = pool

    fives = min(five_target, len(order))
    if four_frac is not None:
        bands = star_bands(order, five_count=fives, four_frac=four_frac)
    else:
        four_target = round(n_scored * settings.four_star_fraction)
        bands = {asset_id: 5 for asset_id in order[:fives]}
        for asset_id in order[fives : fives + four_target]:
            bands[asset_id] = 4
    if settings.full_star_spectrum:
        for asset_id in order:
            bands.setdefault(asset_id, 3)

    prior_five: set[str] = set()
    external_five: set[str] = set()
    if not allow_demotions:
        # Winnow's own earlier awards, applied or not...
        prior_five = {str(r["asset_id"]) for r in ledger.decisions(bucket="five_star")} | {
            str(r["asset_id"]) for r in ledger.score_rows() if r.get("stars") == 5
        }
        # ...plus photos already rated five in Immich itself (by the user, or
        # by a previously applied run) that Winnow never decided on.
        external_five = {
            str(r["id"]) for r in ledger.get_assets() if r.get("immich_rating") == 5
        } - prior_five
        in_order = set(order)
        for asset_id in prior_five:
            if bands.get(asset_id, 0) < 5 and (asset_id in bands or asset_id in in_order):
                bands[asset_id] = 5
                stats.protected += 1
        # Never write a lower rating over an external five — just leave those
        # photos alone this run.
        for asset_id in external_five:
            if bands.get(asset_id, 5) < 5:
                del bands[asset_id]

    # A second finals run plays more rounds and reorders the pool, so photos
    # that have dropped out of the bands must lose last run's award — otherwise
    # write-back keeps crowning a photo the ledger no longer ranks. Sticky
    # five-stars are exempt unless demotions were explicitly allowed.
    demoted = {asset_id for asset_id in order if asset_id not in bands}
    if not allow_demotions:
        demoted -= prior_five | external_five
    for row in ledger.score_rows():
        if row.get("stars") and row["asset_id"] in demoted:
            ledger.set_stars(row["asset_id"], None)
    ledger.clear_decisions(
        [
            row["asset_id"]
            for row in ledger.decisions()
            if row["asset_id"] in demoted
            and row.get("bucket") in _STAR_BUCKETS.values()
        ]
    )

    rank_of = {asset_id: index + 1 for index, asset_id in enumerate(order)}
    band_counts = {5: "five_star", 4: "four_star", 3: "three_star"}
    for asset_id, stars in bands.items():
        ledger.set_stars(asset_id, stars)
        ledger.set_decision(
            asset_id,
            band_counts[stars],
            {"stars": stars, "rank": rank_of.get(asset_id, 0)},
        )
        if stars == 5:
            stats.five_star += 1
        elif stars == 4:
            stats.four_star += 1
        else:
            stats.three_star += 1

    if settings.full_star_spectrum:
        _star_the_middle(ledger, bands, stats)

    protected_note = f" ({stats.protected} kept by sticky five-star)" if stats.protected else ""
    spectrum_note = (
        f", {stats.three_star} at three, {stats.two_star} at two, {stats.one_star} at one"
        if settings.full_star_spectrum
        else ""
    )
    emit(
        on_progress,
        f"starred {stats.five_star} at five, {stats.four_star} at four"
        f"{spectrum_note}{protected_note}",
    )
    return stats


def _star_the_middle(ledger: Ledger, bands: dict[str, int], stats: FinalsStats) -> None:
    """Full-spectrum tail: 2 stars for ordinary keepers, 1 for poor-but-kept.

    Uses only triage evidence — no extra API calls. Photos with any existing
    decision (reject, non-photo, burst loser, or a band above) are skipped, as
    is anything that is not a photograph.
    """
    decided = {str(row["asset_id"]) for row in ledger.decisions()}
    for row in ledger.triage_rows():
        asset_id = str(row["asset_id"])
        if asset_id in bands or asset_id in decided or row.get("category") != "photo":
            continue
        score = int(row.get("technical_score") or 0)
        stars = 2 if score >= 5 else 1
        ledger.set_stars(asset_id, stars)
        ledger.set_decision(
            asset_id,
            _STAR_BUCKETS[stars],
            {"stars": stars, "from": "triage", "technical_score": score},
        )
        if stars == 2:
            stats.two_star += 1
        else:
            stats.one_star += 1
