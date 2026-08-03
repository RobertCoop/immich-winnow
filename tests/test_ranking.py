"""Tests for winnow.ranking — set design, Bradley-Terry, Swiss pairing, bands."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from winnow.ranking import (
    TIE,
    bradley_terry,
    build_bws_sets,
    bws_to_pairs,
    rank_scores,
    star_bands,
    swiss_pairs,
)


def ids(n: int, prefix: str = "p") -> list[str]:
    """Return ``n`` synthetic asset ids, zero-padded so sorting is stable."""
    return [f"{prefix}{i:03d}" for i in range(n)]


def assert_set_invariants(
    sets: list[list[str]], members: list[str], *, set_size: int, appearances: int
) -> None:
    """Check the invariants every BWS design must satisfy."""
    assert sets, "expected at least one set"
    counts: Counter[str] = Counter()
    for group in sets:
        assert len(group) >= 2, f"set too small: {group}"
        assert len(group) <= set_size, f"set larger than set_size: {group}"
        assert len(set(group)) == len(group), f"duplicate id inside set: {group}"
        assert set(group) <= set(members), f"unknown ids in set: {group}"
        counts.update(group)
    assert set(counts) == set(members), "every id must appear at least once"
    for member in members:
        assert abs(counts[member] - appearances) <= 1, (
            f"{member} appeared {counts[member]}x, target {appearances}"
        )


# --------------------------------------------------------------------------
# bradley_terry
# --------------------------------------------------------------------------


def test_bt_transitive_round_robin_recovers_order():
    """A beats B beats C, A beats C -> strengths order A > B > C."""
    pairs = [("A", "B", "A"), ("B", "C", "B"), ("A", "C", "A")]
    strengths = bradley_terry(pairs)

    assert set(strengths) == {"A", "B", "C"}
    assert strengths["A"] > strengths["B"] > strengths["C"]
    assert [item for item, _score, _rank in rank_scores(strengths)] == ["A", "B", "C"]
    assert [rank for _item, _score, rank in rank_scores(strengths)] == [1, 2, 3]


def test_bt_pair_order_does_not_matter():
    """Recording B-vs-A instead of A-vs-B yields the same fit."""
    forward = bradley_terry([("A", "B", "A"), ("B", "C", "B"), ("A", "C", "A")])
    flipped = bradley_terry([("B", "A", "A"), ("C", "B", "B"), ("C", "A", "A")])
    for key in forward:
        assert forward[key] == pytest.approx(flipped[key], rel=1e-6)


def test_bt_undefeated_stays_finite_and_ranks_first():
    """The regularizing prior keeps a 100% winner finite, and it still leads."""
    pairs = [
        ("A", "B", "A"),
        ("A", "C", "A"),
        ("A", "D", "A"),
        ("B", "C", "B"),
        ("B", "D", "B"),
        ("C", "D", "C"),
    ]
    strengths = bradley_terry(pairs)

    assert math.isfinite(strengths["A"])
    assert strengths["A"] < 1e6
    assert all(math.isfinite(v) and v > 0 for v in strengths.values())
    ranked = rank_scores(strengths)
    assert ranked[0][0] == "A"
    assert ranked[0][2] == 1
    assert ranked[-1][0] == "D"


def test_bt_winless_item_stays_positive():
    """A photo that never won keeps a small but strictly positive strength."""
    strengths = bradley_terry([("A", "D", "A"), ("B", "D", "B"), ("C", "D", "C")])
    assert strengths["D"] > 0.0
    assert math.isfinite(strengths["D"])
    assert strengths["D"] < strengths["A"]


def test_bt_ties_give_equal_strengths():
    """A single tie splits the win and leaves both photos level."""
    strengths = bradley_terry([("A", "B", TIE)])
    assert strengths["A"] == pytest.approx(strengths["B"])
    assert strengths["A"] == pytest.approx(1.0)


def test_bt_all_tie_round_robin_is_flat():
    pairs = [("A", "B", TIE), ("B", "C", TIE), ("A", "C", TIE)]
    strengths = bradley_terry(pairs)
    assert strengths["A"] == pytest.approx(strengths["B"], rel=1e-9)
    assert strengths["B"] == pytest.approx(strengths["C"], rel=1e-9)


def test_bt_symmetric_split_record_is_equal():
    """Each beating the other once is the same evidence as a tie: level pegging."""
    strengths = bradley_terry([("A", "B", "A"), ("A", "B", "B")])
    assert strengths["A"] == pytest.approx(strengths["B"], rel=1e-9)


def test_bt_tie_sits_between_win_and_loss():
    """A tie against a common opponent beats losing to it and trails beating it."""
    winner = bradley_terry([("A", "X", "A"), ("B", "X", TIE), ("C", "X", "X")])
    assert winner["A"] > winner["B"] > winner["C"]


def test_bt_more_wins_means_more_strength():
    """Evidence accumulates: five wins over B outrank a single win."""
    thin = bradley_terry([("A", "B", "A")])
    thick = bradley_terry([("A", "B", "A")] * 5)
    assert thick["A"] > thin["A"]


def test_bt_normalizes_to_geometric_mean_one():
    pairs = [("A", "B", "A"), ("B", "C", "B"), ("A", "C", "A"), ("C", "D", TIE)]
    strengths = bradley_terry(pairs)
    logs = [math.log(v) for v in strengths.values()]
    assert math.exp(sum(logs) / len(logs)) == pytest.approx(1.0, rel=1e-9)


def test_bt_empty_input_returns_empty_mapping():
    assert bradley_terry([]) == {}
    assert rank_scores({}) == []


def test_bt_prior_shrinks_toward_parity():
    """A bigger prior pulls an undefeated photo back toward the average."""
    pairs = [("A", "B", "A"), ("A", "C", "A")]
    weak = bradley_terry(pairs, prior=0.1)
    strong = bradley_terry(pairs, prior=5.0)
    assert weak["A"] > strong["A"] > 1.0


def test_bt_rejects_bad_winner_and_self_pair():
    with pytest.raises(ValueError, match="neither"):
        bradley_terry([("A", "B", "C")])
    with pytest.raises(ValueError, match="itself"):
        bradley_terry([("A", "A", "A")])


def test_bt_single_iteration_still_returns_normalized_values():
    strengths = bradley_terry([("A", "B", "A")], iterations=1)
    assert set(strengths) == {"A", "B"}
    assert strengths["A"] * strengths["B"] == pytest.approx(1.0, rel=1e-9)


# --------------------------------------------------------------------------
# rank_scores
# --------------------------------------------------------------------------


def test_rank_scores_uses_dense_ranks():
    ranked = rank_scores({"a": 2.0, "b": 1.0, "c": 1.0, "d": 0.5})
    assert ranked == [
        ("a", 2.0, 1),
        ("b", 1.0, 2),
        ("c", 1.0, 2),
        ("d", 0.5, 3),
    ]


def test_rank_scores_breaks_ties_by_id():
    ranked = rank_scores({"z": 1.0, "a": 1.0})
    assert [item for item, _score, _rank in ranked] == ["a", "z"]
    assert {rank for _item, _score, rank in ranked} == {1}


# --------------------------------------------------------------------------
# build_bws_sets
# --------------------------------------------------------------------------


def test_build_bws_sets_needs_two_ids():
    assert build_bws_sets([]) == []
    assert build_bws_sets(["only"]) == []
    assert build_bws_sets(["dup", "dup"]) == []


def test_build_bws_sets_two_ids():
    members = ["a", "b"]
    sets = build_bws_sets(members, set_size=8, appearances=4, seed=7)
    assert len(sets) == 4
    for group in sets:
        assert sorted(group) == ["a", "b"]
    assert_set_invariants(sets, members, set_size=8, appearances=4)


def test_build_bws_sets_fewer_ids_than_set_size():
    members = ids(5)
    sets = build_bws_sets(members, set_size=8, appearances=4, seed=11)
    assert len(sets) == 4
    for group in sets:
        assert sorted(group) == sorted(members)
    assert_set_invariants(sets, members, set_size=8, appearances=4)


def test_build_bws_sets_fifty_ids():
    members = ids(50)
    sets = build_bws_sets(members, set_size=8, appearances=4, seed=13)
    assert_set_invariants(sets, members, set_size=8, appearances=4)
    assert sum(len(group) for group in sets) == 50 * 4


@pytest.mark.parametrize("n", [2, 3, 7, 8, 9, 16, 17, 23, 50, 101])
@pytest.mark.parametrize("set_size", [3, 8])
def test_build_bws_sets_invariants_across_shapes(n, set_size):
    members = ids(n)
    sets = build_bws_sets(members, set_size=set_size, appearances=3, seed=n)
    assert_set_invariants(sets, members, set_size=set_size, appearances=3)


def test_build_bws_sets_is_deterministic_for_a_seed():
    members = ids(37)
    first = build_bws_sets(members, set_size=8, appearances=4, seed=99)
    second = build_bws_sets(members, set_size=8, appearances=4, seed=99)
    other = build_bws_sets(members, set_size=8, appearances=4, seed=100)
    assert first == second
    assert first != other


def test_build_bws_sets_dedupes_input():
    sets = build_bws_sets(["a", "b", "a", "c"], set_size=8, appearances=2, seed=3)
    assert len(sets) == 2
    for group in sets:
        assert sorted(group) == ["a", "b", "c"]


def test_build_bws_sets_never_leaves_an_orphan_member():
    """9 ids with set_size 8 must not produce a one-photo set."""
    members = ids(9)
    sets = build_bws_sets(members, set_size=8, appearances=1, seed=5)
    assert [len(group) for group in sets] == [5, 4]
    assert_set_invariants(sets, members, set_size=8, appearances=1)


# --------------------------------------------------------------------------
# bws_to_pairs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_bws_to_pairs_count_and_winners(k):
    members = ids(k)
    best, worst = members[0], members[-1]
    pairs = bws_to_pairs(members, best, worst)

    assert len(pairs) == 2 * (k - 1) - 1
    keys = [frozenset((a, b)) for a, b, _w in pairs]
    assert len(set(keys)) == len(keys), "no pair may be emitted twice"

    for a, b, winner in pairs:
        assert winner in (a, b)
        assert a != b
        if best in (a, b):
            assert winner == best
        else:
            assert worst in (a, b)
            assert winner != worst


def test_bws_to_pairs_middle_picks():
    members = ["a", "b", "c", "d"]
    pairs = bws_to_pairs(members, "b", "c")
    assert set(pairs) == {
        ("b", "a", "b"),
        ("b", "c", "b"),
        ("b", "d", "b"),
        ("a", "c", "a"),
        ("d", "c", "d"),
    }


def test_bws_to_pairs_two_members():
    assert bws_to_pairs(["a", "b"], "a", "b") == [("a", "b", "a")]


def test_bws_to_pairs_feeds_bradley_terry():
    """The implied pairs put the best photo on top and the worst at the bottom."""
    members = ids(5)
    pairs = bws_to_pairs(members, members[2], members[4])
    ranked = rank_scores(bradley_terry(pairs))
    assert ranked[0][0] == members[2]
    assert ranked[-1][0] == members[4]


def test_bws_to_pairs_rejects_degenerate_input():
    with pytest.raises(ValueError, match="must differ"):
        bws_to_pairs(["a", "b"], "a", "a")
    with pytest.raises(ValueError, match="best_id"):
        bws_to_pairs(["a", "b"], "zz", "b")
    with pytest.raises(ValueError, match="worst_id"):
        bws_to_pairs(["a", "b"], "a", "zz")
    with pytest.raises(ValueError, match="duplicate"):
        bws_to_pairs(["a", "a", "b"], "a", "b")
    with pytest.raises(ValueError, match="at least two"):
        bws_to_pairs(["a"], "a", "a")


# --------------------------------------------------------------------------
# swiss_pairs
# --------------------------------------------------------------------------


def test_swiss_pairs_pairs_adjacent_by_score():
    scores = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0}
    assert swiss_pairs(scores, set()) == [("a", "b"), ("c", "d")]


def test_swiss_pairs_avoids_rematches():
    scores = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0}
    history = {frozenset({"a", "b"})}
    pairs = swiss_pairs(scores, history)
    assert pairs == [("a", "c"), ("b", "d")]
    assert all(frozenset(pair) not in history for pair in pairs)


def test_swiss_pairs_skips_multiple_rematches():
    scores = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0}
    history = {frozenset({"a", "b"}), frozenset({"a", "c"})}
    pairs = swiss_pairs(scores, history)
    assert pairs == [("a", "d"), ("b", "c")]


def test_swiss_pairs_sits_out_a_fully_played_photo():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    history = {frozenset({"a", "b"}), frozenset({"a", "c"})}
    pairs = swiss_pairs(scores, history)
    assert pairs == [("b", "c")]
    assert "a" not in {i for pair in pairs for i in pair}


def test_swiss_pairs_odd_field_leaves_one_out():
    scores = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0}
    pairs = swiss_pairs(scores, set())
    assert pairs == [("a", "b"), ("c", "d")]


def test_swiss_pairs_empty_and_single():
    assert swiss_pairs({}, set()) == []
    assert swiss_pairs({"a": 1.0}, set()) == []


def test_swiss_pairs_full_history_returns_nothing():
    scores = {"a": 2.0, "b": 1.0}
    assert swiss_pairs(scores, {frozenset({"a", "b"})}) == []


def test_swiss_pairs_seed_only_reorders_equal_scores():
    scores = dict.fromkeys(ids(8), 1.0)
    scores[ids(8)[0]] = 9.0
    first = swiss_pairs(scores, set(), seed=1)
    again = swiss_pairs(scores, set(), seed=1)
    assert first == again
    assert first[0][0] == ids(8)[0], "top score still leads its pair"
    assert len(first) == 4


def test_swiss_pairs_no_photo_appears_twice():
    scores = {name: float(10 - i) for i, name in enumerate(ids(10))}
    history = {frozenset({ids(10)[0], ids(10)[1]}), frozenset({ids(10)[2], ids(10)[3]})}
    pairs = swiss_pairs(scores, history)
    flat = [item for pair in pairs for item in pair]
    assert len(flat) == len(set(flat))
    assert all(frozenset(pair) not in history for pair in pairs)


# --------------------------------------------------------------------------
# star_bands
# --------------------------------------------------------------------------


def test_star_bands_empty_input():
    assert star_bands([], five_count=5) == {}
    assert star_bands([], five_count=0) == {}


def test_star_bands_five_count_at_or_above_length():
    members = ids(4)
    assert star_bands(members, five_count=4) == dict.fromkeys(members, 5)
    assert star_bands(members, five_count=99) == dict.fromkeys(members, 5)


def test_star_bands_splits_fives_then_fours():
    members = ids(10)
    bands = star_bands(members, five_count=2, four_frac=0.3)
    assert [bands.get(m) for m in members] == [5, 5, 4, 4, None, None, None, None, None, None]


def test_star_bands_without_fives():
    members = ids(10)
    bands = star_bands(members, five_count=0, four_frac=0.3)
    assert sorted(bands) == members[:3]
    assert set(bands.values()) == {4}


def test_star_bands_zero_four_frac():
    members = ids(6)
    assert star_bands(members, five_count=2, four_frac=0.0) == dict.fromkeys(members[:2], 5)


def test_star_bands_full_four_frac_stars_everything():
    members = ids(6)
    bands = star_bands(members, five_count=1, four_frac=1.0)
    assert len(bands) == 6
    assert bands[members[0]] == 5
    assert all(bands[m] == 4 for m in members[1:])


def test_star_bands_clamps_negative_inputs():
    members = ids(5)
    assert star_bands(members, five_count=-3, four_frac=-1.0) == {}


def test_star_bands_ratings_are_valid():
    members = ids(25)
    bands = star_bands(members, five_count=5, four_frac=0.3)
    assert set(bands.values()) <= {4, 5}
    assert sum(1 for v in bands.values() if v == 5) == 5
    assert sum(1 for v in bands.values() if v == 4) == 6
