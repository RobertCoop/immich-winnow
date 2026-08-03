"""Burst detection — grouping near-simultaneous, near-identical shots.

A burst is a run of frames from the same camera taken within a short gap of
each other that still *look* alike. Grouping them lets the pipeline spend one
"pick the best" judgement on the run instead of judging every frame blind.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from winnow.images import hamming

__all__ = ["AssetLite", "group_bursts", "merge_duplicate_groups"]


@dataclass(frozen=True)
class AssetLite:
    """The minimal projection of an Immich asset needed for burst grouping.

    Attributes:
        id: Immich asset id.
        taken_at: Capture time, parsed from ``dateTimeOriginal`` or
            ``fileCreatedAt``. All assets in one call must share tz-awareness.
        camera: ``f"{make}|{model}"``, or ``""`` when unknown. Assets with
            different camera strings are never chained together.
        dhash: Perceptual hash from :func:`winnow.images.dhash`, or ``None``
            when it could not be computed.
    """

    id: str
    taken_at: datetime
    camera: str
    dhash: int | None = None


def _links(
    prev: AssetLite,
    curr: AssetLite,
    gap_seconds: float,
    dhash_max_distance: int | None,
) -> bool:
    """Return True when ``curr`` continues the burst started by ``prev``."""
    if prev.camera != curr.camera:
        return False
    if (curr.taken_at - prev.taken_at).total_seconds() > gap_seconds:
        return False
    if dhash_max_distance is None or prev.dhash is None or curr.dhash is None:
        return True  # nothing to compare: time and camera alone decide
    return hamming(prev.dhash, curr.dhash) <= dhash_max_distance


def _emit(chain: Sequence[AssetLite], max_group: int) -> list[list[str]]:
    """Split a finished chain into consecutive groups of >= 2 ids."""
    if len(chain) < 2:
        return []
    groups: list[list[str]] = []
    for start in range(0, len(chain), max_group):
        window = chain[start : start + max_group]
        if len(window) >= 2:
            groups.append([asset.id for asset in window])
    return groups


def group_bursts(
    assets: Sequence[AssetLite],
    *,
    gap_seconds: float = 10.0,
    max_group: int = 10,
    dhash_max_distance: int | None = 26,
) -> list[list[str]]:
    """Group assets into bursts of near-identical consecutive shots.

    Assets are sorted by ``(camera, taken_at, id)`` and chained while each
    consecutive pair comes from the same camera, is separated by no more than
    ``gap_seconds`` (a gap of exactly ``gap_seconds`` still chains), and — when
    both frames have a hash — differs by no more than ``dhash_max_distance``
    bits. A larger hash distance means the scene changed, so the chain breaks.

    Chains longer than ``max_group`` are split into consecutive slices, and
    only groups of two or more survive; lone shots are not bursts.

    Args:
        assets: Assets to group; order does not matter.
        gap_seconds: Maximum seconds between consecutive frames of a burst.
        max_group: Maximum members per emitted group.
        dhash_max_distance: Maximum tolerated hamming distance between
            consecutive frames, or ``None`` to skip the visual check.

    Returns:
        Groups of asset ids, each group ordered by ``taken_at``.

    Raises:
        ValueError: If ``max_group`` is less than 2.
    """
    if max_group < 2:
        raise ValueError(f"max_group must be >= 2, got {max_group}")

    ordered = sorted(assets, key=lambda a: (a.camera, a.taken_at, a.id))
    groups: list[list[str]] = []
    chain: list[AssetLite] = []
    for asset in ordered:
        if chain and _links(chain[-1], asset, gap_seconds, dhash_max_distance):
            chain.append(asset)
        else:
            groups.extend(_emit(chain, max_group))
            chain = [asset]
    groups.extend(_emit(chain, max_group))
    return groups


def merge_duplicate_groups(
    groups: list[list[str]],
    dup_groups: list[list[str]],
    *,
    max_group: int | None = None,
) -> list[list[str]]:
    """Union burst groups with Immich duplicate groups.

    Any two groups sharing at least one asset id become a single group
    (transitively). Ordering is fully deterministic and independent of the
    union order: members are listed by the order in which they were first
    seen — scanning ``groups`` then ``dup_groups`` left to right — and the
    returned groups follow the first-seen order of their earliest member.
    Repeated ids are collapsed, and merged groups of fewer than two members
    are dropped.

    Args:
        groups: Burst groups from :func:`group_bursts`.
        dup_groups: Duplicate groups reported by Immich.
        max_group: Optional cap on members per group. An Immich duplicate
            cluster can hold hundreds of assets, and every member of a group
            becomes an image in one API request, so merged groups larger than
            this are chunked into consecutive slices (a trailing singleton is
            dropped, exactly as in :func:`group_bursts`).

    Returns:
        The merged groups.

    Raises:
        ValueError: If ``max_group`` is given and is less than 2.
    """
    if max_group is not None and max_group < 2:
        raise ValueError(f"max_group must be >= 2, got {max_group}")
    all_groups = [*groups, *dup_groups]

    order: dict[str, int] = {}
    for group in all_groups:
        for asset_id in group:
            if asset_id not in order:
                order[asset_id] = len(order)

    parent: dict[str, str] = {asset_id: asset_id for asset_id in order}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        # Keep the earliest-seen id as the root so roots are deterministic.
        if order[left_root] <= order[right_root]:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for group in all_groups:
        for other in group[1:]:
            union(group[0], other)

    components: dict[str, list[str]] = {}
    for asset_id in order:  # insertion order == first-seen order
        components.setdefault(find(asset_id), []).append(asset_id)

    merged: list[list[str]] = []
    for members in components.values():
        if max_group is None:
            if len(members) >= 2:
                merged.append(members)
            continue
        for start in range(0, len(members), max_group):
            window = members[start : start + max_group]
            if len(window) >= 2:
                merged.append(window)
    return merged
