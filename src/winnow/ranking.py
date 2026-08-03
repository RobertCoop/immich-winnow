"""Ranking math: BWS set design, Bradley-Terry fitting, Swiss pairing, star bands.

Everything here is pure standard-library Python (no numpy) and deterministic
given a seed, so the ranking stages stay cheap, portable, and testable without
touching the network.

The stage-2 flow is: :func:`build_bws_sets` designs the best-worst-scaling sets
shown to the model, :func:`bws_to_pairs` expands each verdict into implied
pairwise outcomes, and :func:`bradley_terry` fits latent strengths over all
accumulated pairs. Stage 3 adds head-to-heads scheduled by :func:`swiss_pairs`
and finishes by cutting :func:`star_bands` from the final ordering.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

__all__ = [
    "TIE",
    "bradley_terry",
    "build_bws_sets",
    "bws_to_pairs",
    "rank_scores",
    "star_bands",
    "swiss_pairs",
]

TIE = "tie"
"""Sentinel used in ``(a, b, winner)`` triples to mark a drawn comparison."""

_MIN_STRENGTH = 1e-12
"""Floor applied to strengths so logs and divisions stay well-defined."""


def _balanced_chunk_sizes(n: int, set_size: int) -> list[int]:
    """Split ``n`` members into near-equal chunks of at most ``set_size``.

    Chunks are as even as possible (sizes differ by at most one) and never
    smaller than two members, so no set can degenerate into a single photo.
    """
    groups = max(1, -(-n // set_size))
    groups = min(groups, max(1, n // 2))
    base, extra = divmod(n, groups)
    return [base + (1 if i < extra else 0) for i in range(groups)]


def build_bws_sets(
    ids: Sequence[str],
    *,
    set_size: int = 8,
    appearances: int = 4,
    seed: int | None = None,
) -> list[list[str]]:
    """Design best-worst-scaling sets covering ``ids``.

    Each distinct id appears exactly ``appearances`` times across the returned
    sets, no id appears twice inside one set, and every set holds at least two
    members. When there are more ids than ``set_size``, each pass over the ids
    is shuffled and cut into evenly sized chunks of at most ``set_size``; when
    there are fewer, every pass yields one set containing all of them under a
    fresh shuffle.

    Args:
        ids: Candidate asset ids. Duplicates are collapsed, first occurrence
            wins.
        set_size: Preferred number of photos shown per judging call.
        appearances: Target number of sets each id takes part in.
        seed: Seed for the shuffle; ``None`` uses system randomness.

    Returns:
        A list of sets (each a list of ids). Empty when fewer than two
        distinct ids are supplied, since a comparison needs two photos.
    """
    unique = list(dict.fromkeys(ids))
    if len(unique) < 2:
        return []

    rounds = max(1, appearances)
    size = max(2, set_size)
    rng = random.Random(seed)
    sets: list[list[str]] = []

    if len(unique) <= size:
        for _ in range(rounds):
            shuffled = list(unique)
            rng.shuffle(shuffled)
            sets.append(shuffled)
        return sets

    sizes = _balanced_chunk_sizes(len(unique), size)
    for _ in range(rounds):
        shuffled = list(unique)
        rng.shuffle(shuffled)
        start = 0
        for chunk in sizes:
            sets.append(shuffled[start : start + chunk])
            start += chunk
    return sets


def bws_to_pairs(
    members: Sequence[str], best_id: str, worst_id: str
) -> list[tuple[str, str, str]]:
    """Expand a best-worst verdict into implied pairwise outcomes.

    The best photo is taken to beat every other member and every other member
    to beat the worst photo, which yields ``2 * (k - 1) - 1`` distinct pairs
    for ``k`` members (the best-vs-worst pair is emitted only once).

    Args:
        members: The set that was judged, in the order it was shown.
        best_id: Member chosen as best.
        worst_id: Member chosen as worst.

    Returns:
        ``(a, b, winner)`` triples where ``winner`` is one of ``a`` or ``b``.

    Raises:
        ValueError: If members repeat, either pick is not a member, or the
            best and worst picks are the same photo.
    """
    ordered = list(members)
    if len(set(ordered)) != len(ordered):
        raise ValueError("bws set contains duplicate members")
    if len(ordered) < 2:
        raise ValueError("bws set needs at least two members")
    if best_id not in ordered:
        raise ValueError(f"best_id {best_id!r} is not a member of the set")
    if worst_id not in ordered:
        raise ValueError(f"worst_id {worst_id!r} is not a member of the set")
    if best_id == worst_id:
        raise ValueError("best_id and worst_id must differ")

    pairs: list[tuple[str, str, str]] = [
        (best_id, other, best_id) for other in ordered if other != best_id
    ]
    pairs.extend(
        (other, worst_id, other) for other in ordered if other not in (worst_id, best_id)
    )
    return pairs


def _tally(
    pairs: Iterable[tuple[str, str, str]],
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    """Accumulate win counts and per-opponent comparison counts."""
    order: list[str] = []
    wins: dict[str, float] = {}
    versus: dict[str, dict[str, float]] = {}

    def register(item: str) -> None:
        if item not in wins:
            wins[item] = 0.0
            versus[item] = {}
            order.append(item)

    for a, b, winner in pairs:
        if a == b:
            raise ValueError(f"a photo cannot be compared with itself: {a!r}")
        register(a)
        register(b)
        versus[a][b] = versus[a].get(b, 0.0) + 1.0
        versus[b][a] = versus[b].get(a, 0.0) + 1.0
        if winner == TIE:
            wins[a] += 0.5
            wins[b] += 0.5
        elif winner == a:
            wins[a] += 1.0
        elif winner == b:
            wins[b] += 1.0
        else:
            raise ValueError(f"winner {winner!r} is neither {a!r}, {b!r} nor {TIE!r}")

    return order, wins, versus


def _normalize_geometric(values: dict[str, float]) -> None:
    """Rescale ``values`` in place so their geometric mean is 1.0."""
    if not values:
        return
    logs = [math.log(max(v, _MIN_STRENGTH)) for v in values.values()]
    scale = math.exp(sum(logs) / len(logs))
    for key, value in values.items():
        values[key] = max(value / scale, _MIN_STRENGTH)


def bradley_terry(
    pairs: Iterable[tuple[str, str, str]],
    *,
    iterations: int = 500,
    tol: float = 1e-8,
    prior: float = 0.5,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths from pairwise outcomes via the MM algorithm.

    Each ``(a, b, winner)`` triple is one comparison; ``winner == "tie"``
    counts as half a win for both sides. ``prior`` adds pseudo-comparisons
    against a virtual opponent whose strength is pinned to the (normalized)
    average, giving every photo ``prior`` pseudo-wins and ``prior``
    pseudo-losses. That regularization is what keeps an undefeated photo from
    running off to infinity and a winless one from collapsing to zero.

    Args:
        pairs: Observed comparisons.
        iterations: Maximum minorize-maximize sweeps.
        tol: Stop once the largest relative change in a strength is below this.
        prior: Pseudo-wins against the virtual average opponent. ``0`` disables
            regularization (undefeated photos then only stay finite because of
            the iteration cap).

    Returns:
        Mapping of id to strength, normalized to geometric mean 1.0. Empty when
        no comparisons were supplied.

    Raises:
        ValueError: On a self-comparison or a winner that is not one of the two
            participants (or ``"tie"``).
    """
    order, wins, versus = _tally(pairs)
    if not order:
        return {}

    prior = max(0.0, prior)
    strengths = dict.fromkeys(order, 1.0)

    for _ in range(max(1, iterations)):
        updated: dict[str, float] = {}
        for item in order:
            own = strengths[item]
            denom = (2.0 * prior) / (own + 1.0)
            for opponent, count in versus[item].items():
                denom += count / (own + strengths[opponent])
            updated[item] = max((wins[item] + prior) / denom, _MIN_STRENGTH) if denom else own
        _normalize_geometric(updated)
        delta = max(abs(updated[i] - strengths[i]) / strengths[i] for i in order)
        strengths = updated
        if delta < tol:
            break

    return strengths


def rank_scores(strengths: dict[str, float]) -> list[tuple[str, float, int]]:
    """Order photos by strength, best first, with dense ranks starting at 1.

    Equal strengths (within a small floating-point tolerance) share a rank and
    the next distinct strength takes the immediately following rank. Ids act as
    the tie-breaker for a stable, reproducible ordering.

    Args:
        strengths: Mapping of id to Bradley-Terry strength.

    Returns:
        ``(id, strength, rank)`` triples sorted by descending strength.
    """
    ordered = sorted(strengths.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked: list[tuple[str, float, int]] = []
    rank = 0
    group_score: float | None = None
    for item, score in ordered:
        if group_score is None or not math.isclose(
            score, group_score, rel_tol=1e-9, abs_tol=1e-12
        ):
            rank += 1
            group_score = score
        ranked.append((item, score, rank))
    return ranked


def swiss_pairs(
    scores: dict[str, float],
    history: set[frozenset[str]],
    *,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """Schedule one Swiss round: adjacent by score, never a rematch.

    Photos are sorted by descending score and paired greedily from the top.
    Each photo takes the closest-scoring opponent it has not already faced; if
    every remaining opponent is a rematch, that photo sits the round out, as
    does the odd one when the field has an odd size.

    Args:
        scores: Current strength per photo id.
        history: Already-played pairings as ``frozenset({a, b})`` entries.
        seed: When given, randomizes the order of equally scored photos;
            ``None`` breaks those ties by id for full determinism.

    Returns:
        ``(a, b)`` pairings, highest-scoring pair first.
    """
    if seed is None:
        pool = sorted(scores, key=lambda i: (-scores[i], i))
    else:
        shuffled = list(scores)
        random.Random(seed).shuffle(shuffled)
        pool = sorted(shuffled, key=lambda i: -scores[i])

    pairs: list[tuple[str, str]] = []
    while len(pool) >= 2:
        first = pool.pop(0)
        partner = next(
            (idx for idx, other in enumerate(pool) if frozenset((first, other)) not in history),
            None,
        )
        if partner is None:
            continue
        pairs.append((first, pool.pop(partner)))
    return pairs


def star_bands(
    ranked_ids: Sequence[str], *, five_count: int, four_frac: float = 0.3
) -> dict[str, int]:
    """Cut the final ordering into five- and four-star bands.

    The top ``five_count`` photos earn five stars; the next
    ``floor(four_frac * remainder)`` earn four. Everything below is left out of
    the mapping entirely, so callers can treat "absent" as "no star change".

    Args:
        ranked_ids: Ids ordered best first.
        five_count: How many photos get five stars. Clamped to
            ``[0, len(ranked_ids)]``, so an oversized value stars everything.
        four_frac: Fraction of the photos below the five-star cut that get four
            stars. Clamped to ``[0.0, 1.0]``.

    Returns:
        Mapping of id to star rating (5 or 4) for the starred photos only.
    """
    ids = list(ranked_ids)
    if not ids:
        return {}

    fives = max(0, min(int(five_count), len(ids)))
    frac = min(max(four_frac, 0.0), 1.0)
    fours = int((len(ids) - fives) * frac)

    bands: dict[str, int] = dict.fromkeys(ids[:fives], 5)
    bands.update(dict.fromkeys(ids[fives : fives + fours], 4))
    return bands
