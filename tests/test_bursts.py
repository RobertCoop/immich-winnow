"""Tests for winnow.bursts — burst chaining and duplicate-group merging."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from winnow.bursts import AssetLite, group_bursts, merge_duplicate_groups

BASE = datetime(2024, 6, 1, 12, 0, 0)
CAM = "Canon|EOS R5"
OTHER_CAM = "Apple|iPhone 15"


def mk(name: str, offset: float, camera: str = CAM, dh: int | None = None) -> AssetLite:
    """Build an AssetLite ``offset`` seconds after the base timestamp."""
    return AssetLite(id=name, taken_at=BASE + timedelta(seconds=offset), camera=camera, dhash=dh)


def run(count: int, *, step: float = 1.0, prefix: str = "a") -> list[AssetLite]:
    """A run of ``count`` shots ``step`` seconds apart."""
    return [mk(f"{prefix}{i}", i * step) for i in range(count)]


# --------------------------------------------------------------------------
# group_bursts — basics
# --------------------------------------------------------------------------


def test_empty_input() -> None:
    assert group_bursts([]) == []


def test_single_asset_is_not_a_burst() -> None:
    assert group_bursts([mk("a", 0)]) == []


def test_consecutive_shots_form_one_group() -> None:
    assets = [mk("a", 0), mk("b", 2), mk("c", 4)]
    assert group_bursts(assets) == [["a", "b", "c"]]


def test_identical_timestamps_group() -> None:
    assets = [mk("a", 0), mk("b", 0)]
    assert group_bursts(assets) == [["a", "b"]]


def test_group_members_ordered_by_taken_at() -> None:
    assets = [mk("z", 3), mk("m", 1), mk("q", 2)]
    assert group_bursts(assets) == [["m", "q", "z"]]


def test_unsorted_input_is_handled() -> None:
    assets = run(5)
    shuffled = assets[:]
    random.Random(42).shuffle(shuffled)
    assert group_bursts(shuffled) == group_bursts(assets) == [["a0", "a1", "a2", "a3", "a4"]]


def test_far_apart_shots_are_separate_groups() -> None:
    assets = [mk("a", 0), mk("b", 1), mk("c", 300), mk("d", 301)]
    assert group_bursts(assets) == [["a", "b"], ["c", "d"]]


def test_singletons_are_excluded() -> None:
    assets = [mk("a", 0), mk("b", 1), mk("lonely", 500), mk("c", 900), mk("d", 901)]
    assert group_bursts(assets) == [["a", "b"], ["c", "d"]]


# --------------------------------------------------------------------------
# group_bursts — gap boundary
# --------------------------------------------------------------------------


def test_gap_equal_to_limit_is_inside_the_burst() -> None:
    assets = [mk("a", 0), mk("b", 10)]
    assert group_bursts(assets, gap_seconds=10.0) == [["a", "b"]]


def test_gap_just_over_limit_breaks_the_chain() -> None:
    assets = [mk("a", 0), mk("b", 10.001)]
    assert group_bursts(assets, gap_seconds=10.0) == []


def test_gap_seconds_is_per_link_not_per_chain() -> None:
    # Each hop is 8s (<= 10) so a 24s-long chain still counts as one burst.
    assets = [mk("a", 0), mk("b", 8), mk("c", 16), mk("d", 24)]
    assert group_bursts(assets, gap_seconds=10.0) == [["a", "b", "c", "d"]]


def test_zero_gap_only_chains_simultaneous_shots() -> None:
    assets = [mk("a", 0), mk("b", 0), mk("c", 0.5)]
    assert group_bursts(assets, gap_seconds=0.0) == [["a", "b"]]


def test_timezone_aware_timestamps_work() -> None:
    aware = [
        AssetLite("a", datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC), CAM),
        AssetLite("b", datetime(2024, 6, 1, 12, 0, 3, tzinfo=UTC), CAM),
    ]
    assert group_bursts(aware) == [["a", "b"]]


# --------------------------------------------------------------------------
# group_bursts — camera
# --------------------------------------------------------------------------


def test_camera_change_breaks_the_chain() -> None:
    # b sits between a and c in time but comes from another body; sorting by
    # camera reunites a with c and leaves b as a singleton.
    assets = [mk("a", 0, CAM), mk("b", 1, OTHER_CAM), mk("c", 2, CAM)]
    assert group_bursts(assets) == [["a", "c"]]


def test_two_cameras_shooting_in_parallel() -> None:
    assets = [
        mk("a1", 0, OTHER_CAM),
        mk("c1", 0, CAM),
        mk("a2", 1, OTHER_CAM),
        mk("c2", 1, CAM),
    ]
    # OTHER_CAM ("Apple|...") sorts before CAM ("Canon|...").
    assert group_bursts(assets) == [["a1", "a2"], ["c1", "c2"]]


def test_unknown_camera_still_groups() -> None:
    assets = [mk("a", 0, ""), mk("b", 1, "")]
    assert group_bursts(assets) == [["a", "b"]]


def test_unknown_camera_does_not_merge_with_known() -> None:
    assets = [mk("a", 0, ""), mk("b", 1, CAM)]
    assert group_bursts(assets) == []


# --------------------------------------------------------------------------
# group_bursts — dhash
# --------------------------------------------------------------------------


def test_dhash_distance_over_limit_breaks_the_chain() -> None:
    far = (1 << 27) - 1  # 27 set bits => distance 27 from 0
    assets = [mk("a", 0, dh=0), mk("b", 1, dh=0), mk("c", 2, dh=far)]
    assert group_bursts(assets, dhash_max_distance=26) == [["a", "b"]]


def test_dhash_distance_equal_to_limit_is_kept() -> None:
    edge = (1 << 26) - 1  # exactly 26 set bits
    assets = [mk("a", 0, dh=0), mk("b", 1, dh=edge)]
    assert group_bursts(assets, dhash_max_distance=26) == [["a", "b"]]


def test_dhash_check_disabled_by_none() -> None:
    assets = [mk("a", 0, dh=0), mk("b", 1, dh=(1 << 64) - 1)]
    assert group_bursts(assets, dhash_max_distance=None) == [["a", "b"]]


def test_missing_dhash_never_breaks_the_chain() -> None:
    # a and c are visually miles apart, but b has no hash so neither link
    # can be tested and the chain survives.
    assets = [mk("a", 0, dh=0), mk("b", 1, dh=None), mk("c", 2, dh=(1 << 64) - 1)]
    assert group_bursts(assets, dhash_max_distance=26) == [["a", "b", "c"]]


def test_dhash_break_can_split_into_two_bursts() -> None:
    far = (1 << 40) - 1
    assets = [
        mk("a", 0, dh=0),
        mk("b", 1, dh=0),
        mk("c", 2, dh=far),
        mk("d", 3, dh=far),
    ]
    assert group_bursts(assets, dhash_max_distance=26) == [["a", "b"], ["c", "d"]]


# --------------------------------------------------------------------------
# group_bursts — max_group
# --------------------------------------------------------------------------


def test_long_chain_splits_into_consecutive_slices() -> None:
    groups = group_bursts(run(25), max_group=10)
    assert [len(g) for g in groups] == [10, 10, 5]
    assert groups[0] == [f"a{i}" for i in range(10)]
    assert groups[1] == [f"a{i}" for i in range(10, 20)]
    assert groups[2] == [f"a{i}" for i in range(20, 25)]


def test_split_drops_a_trailing_singleton() -> None:
    groups = group_bursts(run(21), max_group=10)
    assert [len(g) for g in groups] == [10, 10]
    assert "a20" not in [asset_id for g in groups for asset_id in g]


def test_chain_exactly_max_group_is_one_slice() -> None:
    assert [len(g) for g in group_bursts(run(10), max_group=10)] == [10]


def test_max_group_of_two() -> None:
    assert group_bursts(run(5), max_group=2) == [["a0", "a1"], ["a2", "a3"]]


def test_max_group_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_group"):
        group_bursts(run(3), max_group=1)


def test_every_id_appears_at_most_once() -> None:
    assets = run(30) + [mk(f"b{i}", 1000 + i, OTHER_CAM) for i in range(7)]
    seen = [asset_id for group in group_bursts(assets, max_group=4) for asset_id in group]
    assert len(seen) == len(set(seen))


# --------------------------------------------------------------------------
# merge_duplicate_groups
# --------------------------------------------------------------------------


def test_merge_without_overlap_keeps_groups() -> None:
    assert merge_duplicate_groups([["a", "b"]], [["c", "d"]]) == [["a", "b"], ["c", "d"]]


def test_merge_empty_inputs() -> None:
    assert merge_duplicate_groups([], []) == []


def test_merge_with_no_duplicates_is_identity() -> None:
    groups = [["a", "b"], ["c", "d", "e"]]
    assert merge_duplicate_groups(groups, []) == groups


def test_merge_joins_overlapping_groups() -> None:
    groups = [["a", "b"], ["c", "d"]]
    dups = [["b", "c"]]
    assert merge_duplicate_groups(groups, dups) == [["a", "b", "c", "d"]]


def test_merge_is_transitive() -> None:
    groups = [["a", "b"], ["c", "d"], ["e", "f"]]
    dups = [["b", "c"], ["d", "e"]]
    assert merge_duplicate_groups(groups, dups) == [["a", "b", "c", "d", "e", "f"]]


def test_merge_pulls_in_new_ids_from_duplicates() -> None:
    assert merge_duplicate_groups([["a", "b"]], [["b", "z"]]) == [["a", "b", "z"]]


def test_merge_keeps_first_seen_member_order() -> None:
    groups = [["b", "a"]]
    dups = [["a", "z"], ["z", "c"]]
    assert merge_duplicate_groups(groups, dups) == [["b", "a", "z", "c"]]


def test_merge_group_order_follows_first_seen() -> None:
    groups = [["m", "n"], ["a", "b"]]
    dups = [["b", "a"]]
    assert merge_duplicate_groups(groups, dups) == [["m", "n"], ["a", "b"]]


def test_merge_is_order_independent_for_dup_groups() -> None:
    groups = [["a", "b"], ["c", "d"], ["e", "f"]]
    forward = merge_duplicate_groups(groups, [["b", "c"], ["d", "e"]])
    backward = merge_duplicate_groups(groups, [["d", "e"], ["b", "c"]])
    assert forward == backward == [["a", "b", "c", "d", "e", "f"]]


def test_merge_collapses_repeated_ids() -> None:
    assert merge_duplicate_groups([["a", "b", "a"]], []) == [["a", "b"]]


def test_merge_drops_singleton_groups() -> None:
    assert merge_duplicate_groups([], [["solo"]]) == []
    assert merge_duplicate_groups([["a", "b"]], [["solo"]]) == [["a", "b"]]


def test_merge_ignores_empty_groups() -> None:
    assert merge_duplicate_groups([[], ["a", "b"]], [[]]) == [["a", "b"]]


def test_merge_is_idempotent() -> None:
    groups = [["a", "b"], ["c", "d"]]
    once = merge_duplicate_groups(groups, [["b", "c"], ["x", "y"]])
    assert merge_duplicate_groups(once, []) == once


def test_merge_handles_a_fully_duplicate_pass() -> None:
    groups = [["a", "b"], ["c", "d"]]
    assert merge_duplicate_groups(groups, groups) == groups


def test_merge_accepts_burst_output() -> None:
    bursts = group_bursts([mk("a", 0), mk("b", 1), mk("c", 100), mk("d", 101)])
    merged = merge_duplicate_groups(bursts, [["b", "c"]])
    assert merged == [["a", "b", "c", "d"]]


def test_merge_caps_group_size() -> None:
    """An Immich duplicate cluster can be huge; every member becomes an image
    in one API request, so merged groups have to be chunked too."""
    big = [f"a{i}" for i in range(25)]
    merged = merge_duplicate_groups([], [big], max_group=10)
    assert [len(group) for group in merged] == [10, 10, 5]
    assert [asset_id for group in merged for asset_id in group] == big


def test_merge_cap_drops_a_trailing_singleton() -> None:
    merged = merge_duplicate_groups([], [["a", "b", "c"]], max_group=2)
    assert merged == [["a", "b"]]


def test_merge_without_a_cap_is_unbounded() -> None:
    big = [f"a{i}" for i in range(25)]
    assert merge_duplicate_groups([], [big]) == [big]


def test_merge_rejects_a_nonsense_cap() -> None:
    with pytest.raises(ValueError):
        merge_duplicate_groups([["a", "b"]], [], max_group=1)
