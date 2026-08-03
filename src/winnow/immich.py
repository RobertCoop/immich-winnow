"""Thin, typed HTTP client for the Immich API (server v3.x).

Only the endpoints Winnow needs are implemented, and every write it performs
is reversible: ratings, favorites, visibility, tags and stacks. Nothing here
ever deletes an asset.

Example::

    with ImmichClient("http://immich.local:2283", api_key) as immich:
        print(immich.ping()["version"])
        for asset in immich.search_assets("2024-06-01T00:00:00.000Z",
                                          "2024-07-01T00:00:00.000Z"):
            ...
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Any

import httpx

__all__ = ["UNSET", "ImmichClient", "ImmichError"]

#: Sentinel for "argument not supplied" — lets ``None`` be a meaningful value.
UNSET: Any = object()

#: Maximum number of response-body characters embedded in an error message.
_SNIPPET_CHARS = 300

#: Hard cap on pages walked by :meth:`ImmichClient.search_assets`, so a server
#: that keeps echoing the same cursor cannot spin forever.
_MAX_PAGES = 10_000


class ImmichError(RuntimeError):
    """An Immich request failed.

    Attributes:
        status_code: HTTP status of the failing response, or ``None`` when the
            request never produced one (connection error, timeout, ...).
        body: Body snippet of the failing response, if any.
    """

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ImmichClient:
    """Client for the subset of the Immich API that Winnow uses.

    Args:
        base_url: Immich server root, e.g. ``http://immich.local:2283``.
            ``/api`` is appended automatically.
        api_key: Immich API key, sent as the ``x-api-key`` header.
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=f"{self.base_url}/api",
            headers={"x-api-key": api_key, "Accept": "application/json"},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> ImmichClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Perform a request, raising :class:`ImmichError` on failure."""
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # connection errors, timeouts, ...
            raise ImmichError(f"{method} {path} failed: {exc}") from exc
        if response.status_code >= 400:
            snippet = response.text[:_SNIPPET_CHARS]
            raise ImmichError(
                f"{method} {path} -> HTTP {response.status_code}: {snippet}",
                status_code=response.status_code,
                body=snippet,
            )
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        """Perform a request and decode its JSON body.

        A 2xx response that is not JSON at all — a login page from a reverse
        proxy, the wrong port — would otherwise raise a bare decode error that
        escapes every ``except ImmichError`` handler.
        """
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            snippet = response.text[:_SNIPPET_CHARS]
            raise ImmichError(
                f"{method} {path} returned a non-JSON body: {snippet!r}",
                status_code=response.status_code,
                body=snippet,
            ) from exc

    # ------------------------------------------------------------------
    # server
    # ------------------------------------------------------------------
    def ping(self) -> dict:
        """Return the server's ``/server/about`` payload (includes ``version``)."""
        return self._json("GET", "/server/about")

    def my_preferences(self) -> dict:
        """Return the API-key user's preferences (``/users/me/preferences``)."""
        return self._json("GET", "/users/me/preferences")

    # ------------------------------------------------------------------
    # assets
    # ------------------------------------------------------------------
    def search_assets(
        self, taken_after: str, taken_before: str, page_size: int = 250
    ) -> Iterator[dict]:
        """Yield image assets taken within a window, following pagination.

        Args:
            taken_after: ISO-8601 lower bound, e.g. ``2024-06-01T00:00:00.000Z``.
            taken_before: ISO-8601 upper bound.
            page_size: Assets requested per page.

        Yields:
            Raw Immich asset dicts (EXIF included).

        Raises:
            ImmichError: If the server stops making progress — it repeats a
                page cursor, or hands out more than ``_MAX_PAGES`` of them.
        """
        page: Any = 1
        for _ in range(_MAX_PAGES):
            payload = {
                "takenAfter": taken_after,
                "takenBefore": taken_before,
                "size": page_size,
                "page": page,
                "withExif": True,
                "type": "IMAGE",
            }
            body = self._json("POST", "/search/metadata", json=payload)
            bucket = body.get("assets") or {}
            yield from bucket.get("items") or []
            next_page = bucket.get("nextPage")
            if not next_page:
                return
            if next_page == page:
                raise ImmichError(f"search/metadata repeated page cursor {page!r}")
            page = next_page
        raise ImmichError(f"search/metadata exceeded {_MAX_PAGES} pages; refusing to continue")

    def get_asset(self, asset_id: str) -> dict:
        """Return the full asset DTO for ``asset_id``."""
        return self._json("GET", f"/assets/{asset_id}")

    def fetch_thumbnail(self, asset_id: str, size: str = "preview") -> bytes:
        """Return JPEG bytes for an asset thumbnail (``preview`` ~1440px long edge)."""
        response = self._request("GET", f"/assets/{asset_id}/thumbnail", params={"size": size})
        return response.content

    def update_asset(
        self,
        asset_id: str,
        *,
        rating: int | None = UNSET,
        is_favorite: bool = UNSET,
        visibility: str = UNSET,
    ) -> dict:
        """Update an asset, sending only the fields explicitly passed in.

        Args:
            asset_id: Asset to update.
            rating: ``1``-``5``, ``-1`` for rejected, or ``None`` to clear.
            is_favorite: Favorite flag.
            visibility: ``"archive"`` or ``"timeline"``.

        Returns:
            The updated asset DTO.
        """
        payload: dict[str, Any] = {}
        if rating is not UNSET:
            payload["rating"] = rating
        if is_favorite is not UNSET:
            payload["isFavorite"] = is_favorite
        if visibility is not UNSET:
            payload["visibility"] = visibility
        return self._json("PUT", f"/assets/{asset_id}", json=payload)

    def duplicates(self) -> list[dict]:
        """Return Immich's duplicate groups: ``[{duplicateId, assets: [...]}, ...]``."""
        return self._json("GET", "/duplicates")

    # ------------------------------------------------------------------
    # tags
    # ------------------------------------------------------------------
    def list_tags(self) -> list[dict]:
        """Return every tag DTO known to the server."""
        return self._json("GET", "/tags")

    def upsert_tags(self, names: list[str]) -> dict[str, str]:
        """Create tags if missing and map each requested name to its tag id.

        Uses ``PUT /api/tags`` (bulk upsert, nested paths via ``/``). Servers
        that predate that endpoint answer ``404``; those fall back to one
        ``POST /api/tags`` per name followed by a lookup against
        ``GET /api/tags``.

        Args:
            names: Tag values, e.g. ``["winnow/reject", "winnow/best"]``.

        Returns:
            Mapping of requested name to tag id. Names the server never
            reported back are omitted.
        """
        wanted = list(names)
        if not wanted:
            return {}
        try:
            dtos = self._json("PUT", "/tags", json={"tags": wanted})
        except ImmichError as exc:
            if exc.status_code != 404:
                raise
            return self._upsert_tags_fallback(wanted)

        resolved = _match_tags(wanted, dtos)
        if len(resolved) < len(wanted):
            # Bulk upsert answered with a subset (e.g. only newly created
            # tags); fill the gaps from the full tag list.
            resolved = {**_match_tags(wanted, self.list_tags()), **resolved}
        return resolved

    def _upsert_tags_fallback(self, names: Sequence[str]) -> dict[str, str]:
        """Create tags one by one, then resolve ids from ``GET /api/tags``."""
        for name in names:
            try:
                self._request("POST", "/tags", json={"name": name})
            except ImmichError as exc:
                # 400/409 mean "already exists" on older servers — harmless.
                if exc.status_code not in (400, 409):
                    raise
        return _match_tags(names, self.list_tags())

    def tag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        """Add a tag to many assets. A no-op when ``asset_ids`` is empty."""
        if not asset_ids:
            return
        self._request("PUT", f"/tags/{tag_id}/assets", json={"ids": list(asset_ids)})

    def untag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        """Remove a tag from many assets. A no-op when ``asset_ids`` is empty."""
        if not asset_ids:
            return
        self._request("DELETE", f"/tags/{tag_id}/assets", json={"ids": list(asset_ids)})

    def delete_tag(self, tag_id: str) -> None:
        """Delete a tag (assets are untouched)."""
        self._request("DELETE", f"/tags/{tag_id}")

    # ------------------------------------------------------------------
    # stacks
    # ------------------------------------------------------------------
    def create_stack(self, asset_ids: list[str]) -> dict:
        """Stack assets together; the first id becomes the stack primary."""
        return self._json("POST", "/stacks", json={"assetIds": list(asset_ids)})

    def delete_stack(self, stack_id: str) -> None:
        """Remove a stack (its assets stay in the library)."""
        self._request("DELETE", f"/stacks/{stack_id}")

    # ------------------------------------------------------------------
    # albums
    # ------------------------------------------------------------------
    def list_albums(self) -> list[dict]:
        """Return every album DTO the API key can see."""
        return self._json("GET", "/albums")

    def create_album(self, name: str, asset_ids: list[str] | None = None) -> dict:
        """Create an album, optionally seeding it with assets."""
        payload: dict[str, Any] = {"albumName": name}
        if asset_ids:
            payload["assetIds"] = list(asset_ids)
        return self._json("POST", "/albums", json=payload)

    def add_album_assets(self, album_id: str, asset_ids: list[str]) -> list[dict]:
        """Add assets to an album; already-present assets report as duplicates,
        which Immich treats as success-shaped results, so the call is idempotent."""
        return self._json("PUT", f"/albums/{album_id}/assets", json={"ids": list(asset_ids)})

    def upsert_album(self, name: str) -> str:
        """Return the id of the album named ``name``, creating it if needed.

        Matches on exact ``albumName``; if several albums share the name, the
        first one the API returns wins.
        """
        for dto in self.list_albums():
            if str(dto.get("albumName") or "") == name:
                return str(dto["id"])
        return str(self.create_album(name)["id"])

    def delete_album(self, album_id: str) -> None:
        """Delete an album (its assets stay in the library)."""
        self._request("DELETE", f"/albums/{album_id}")


def _match_tags(names: Sequence[str], dtos: Any) -> dict[str, str]:
    """Map requested tag names to ids from a list of tag DTOs.

    Nested tags report the full path in ``value`` and the leaf in ``name``;
    ``value`` matches win over ``name`` matches.
    """
    if not isinstance(dtos, list):
        return {}
    by_value: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for dto in dtos:
        if not isinstance(dto, dict) or "id" not in dto:
            continue
        tag_id = str(dto["id"])
        value = dto.get("value")
        if isinstance(value, str):
            by_value.setdefault(value, tag_id)
        name = dto.get("name")
        if isinstance(name, str):
            by_name.setdefault(name, tag_id)
    resolved: dict[str, str] = {}
    for wanted in names:
        tag_id = by_value.get(wanted) or by_name.get(wanted)
        if tag_id is not None:
            resolved[wanted] = tag_id
    return resolved
