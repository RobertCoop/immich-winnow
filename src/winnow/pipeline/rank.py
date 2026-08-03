"""Stage 2 — rank the survivors with best-worst scaling.

Triage leaves a pool of candidates that are all, roughly, "good". Ranking them
by asking for a score each is hopeless (models are terrible at absolute
scales), so instead each photo is shown in several small sets and the model
names only the best and the worst of each set. Those two answers imply a
fistful of pairwise outcomes, and a Bradley-Terry fit over every accumulated
outcome turns them into one latent strength per photo.

Sets are designed from a fixed seed, so a re-run proposes exactly the same
sets and skips the ones the ledger has already judged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from winnow.config import Settings
from winnow.judge import FATAL_ERRORS, ITEM_ERRORS, Judge
from winnow.ledger import Ledger
from winnow.pipeline.scan import ProgressFn, emit, load_thumb_b64
from winnow.ranking import bradley_terry, build_bws_sets, bws_to_pairs, rank_scores

__all__ = [
    "DEFAULT_SEED",
    "RankStats",
    "cap_candidates",
    "collect_pairs",
    "refit_scores",
    "run_rank",
]

#: Seed for BWS set design. Fixed so re-runs reproduce the same sets.
DEFAULT_SEED = 20240601


@dataclass
class RankStats:
    """What a :func:`run_rank` call judged and scored.

    Attributes:
        candidates: Photos in the stage-2 pool.
        sets_planned: Best-worst sets the design called for.
        sets_judged: Sets actually sent to the model this run.
        sets_skipped: Sets the ledger had already judged.
        capped: Already-scored photos left out by the anchor cap.
        pairs: Pairwise outcomes the fit was given.
        scored: Photos that came out with a strength.
        errors: Sets skipped after an unusable reply or a missing thumbnail.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens consumed.
    """

    candidates: int = 0
    sets_planned: int = 0
    sets_judged: int = 0
    sets_skipped: int = 0
    capped: int = 0
    pairs: int = 0
    scored: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def collect_pairs(ledger: Ledger) -> list[tuple[str, str, str]]:
    """Every pairwise outcome the ledger holds, as ``(a, b, winner)`` triples.

    Best-worst sets are expanded on the fly (the best beats everyone, everyone
    beats the worst) and head-to-heads recorded by the finals stage are added
    verbatim. Malformed rows are ignored rather than aborting a fit.
    """
    pairs: list[tuple[str, str, str]] = []
    for row in ledger.bws_rows():
        members = row.get("member_ids") or []
        best, worst = row.get("best_id"), row.get("worst_id")
        if not best or not worst:
            continue
        try:
            pairs.extend(bws_to_pairs(members, best, worst))
        except ValueError:
            continue
    for row in ledger.pair_rows():
        a_id, b_id, winner = row.get("a_id"), row.get("b_id"), row.get("winner")
        if a_id and b_id and winner and a_id != b_id:
            pairs.append((a_id, b_id, winner))
    return pairs


def refit_scores(
    ledger: Ledger, pairs: list[tuple[str, str, str]] | None = None
) -> list[tuple[str, float, int]]:
    """Fit Bradley-Terry over recorded outcomes and store strengths and ranks.

    Args:
        ledger: Open ledger; receives the fitted scores.
        pairs: Outcomes to fit; defaults to everything :func:`collect_pairs`
            finds in the ledger.

    Returns:
        ``(asset_id, strength, rank)`` triples, strongest first.
    """
    ranked = rank_scores(bradley_terry(collect_pairs(ledger) if pairs is None else pairs))
    if ranked:
        ledger.upsert_scores({item: (score, rank) for item, score, rank in ranked})
    return ranked


def cap_candidates(
    candidate_ids: Sequence[str], scores: dict[str, float], limit: int | None
) -> list[str]:
    """Every never-scored candidate, plus up to ``limit`` scored anchors.

    New photos always enter scoring. But newcomers judged only against each
    other float on a scale of their own — they connect to the existing ranking
    only through already-scored photos appearing in the same sets. So a
    capped run keeps *all* fresh candidates and mixes in up to ``limit``
    veterans as anchors, chosen evenly across the existing ranking so the two
    scales are pinned together at the top, middle and bottom.

    ``limit`` of ``None`` or ``0`` means every veteran re-enters (no cap).
    """
    fresh = [asset_id for asset_id in candidate_ids if asset_id not in scores]
    veterans = sorted(
        (asset_id for asset_id in candidate_ids if asset_id in scores),
        key=lambda asset_id: -scores[asset_id],
    )
    if limit is None or limit <= 0 or len(veterans) <= limit:
        return fresh + veterans
    if limit == 1:
        return [*fresh, veterans[0]]
    span = len(veterans) - 1
    picks = sorted({round(k * span / (limit - 1)) for k in range(limit)})
    return fresh + [veterans[i] for i in picks]


def run_rank(
    settings: Settings,
    ledger: Ledger,
    judge: Judge,
    *,
    seed: int | None = DEFAULT_SEED,
    limit: int | None = None,
    on_progress: ProgressFn | None = None,
) -> RankStats:
    """Run best-worst scaling over the candidate pool and fit strengths.

    Args:
        settings: Runtime configuration (rank model, set size, appearances).
        ledger: Open ledger.
        judge: Judge wrapping an Anthropic client.
        seed: Seed for the set design; keep it stable across runs so already
            judged sets can be recognised and skipped.
        limit: Cap on how many *already-scored* photos re-enter scoring as
            anchors alongside the newcomers (which always all enter).
            ``None`` or ``0`` means no cap.
        on_progress: Optional callback receiving one short line per set.

    Returns:
        Counters describing the run.
    """
    stats = RankStats()
    all_candidates = [row["asset_id"] for row in ledger.candidates(settings.candidate_score_min)]
    stats.candidates = len(all_candidates)
    score_by_id = {
        str(row["asset_id"]): float(row["bt_score"])
        for row in ledger.score_rows()
        if row.get("bt_score") is not None
    }
    candidate_ids = cap_candidates(all_candidates, score_by_id, limit)
    stats.capped = len(all_candidates) - len(candidate_ids)
    if stats.capped:
        emit(
            on_progress,
            f"anchor cap: {stats.capped} already-scored photo(s) sit out; "
            "their existing scores and pairs are kept",
        )

    sets = build_bws_sets(
        candidate_ids,
        set_size=settings.bws_set_size,
        appearances=settings.bws_appearances,
        seed=seed,
    )
    stats.sets_planned = len(sets)

    already = Counter(tuple(sorted(row.get("member_ids") or [])) for row in ledger.bws_rows())
    per_round = max(1, len(sets) // max(1, settings.bws_appearances))

    for index, members in enumerate(sets):
        key = tuple(sorted(members))
        if already[key] > 0:
            already[key] -= 1
            stats.sets_skipped += 1
            continue
        try:
            images = [load_thumb_b64(settings.cache_dir, member) for member in members]
            result = judge.bws(settings.rank_model, images)
        except FATAL_ERRORS:
            raise
        except ITEM_ERRORS as exc:
            stats.errors += 1
            emit(on_progress, f"set {index + 1} failed: {exc}")
            continue
        best = members[result.verdict.best_index - 1]
        worst = members[result.verdict.worst_index - 1]
        ledger.record_bws(index // per_round + 1, members, best, worst, result.model)
        stats.sets_judged += 1
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        emit(on_progress, f"set {index + 1}/{len(sets)}: best {best}, worst {worst}")

    pairs = collect_pairs(ledger)
    stats.pairs = len(pairs)
    stats.scored = len(refit_scores(ledger, pairs))
    emit(on_progress, f"scored {stats.scored} photos from {stats.pairs} comparisons")
    return stats
