"""Tests for the Immich HTTP client. All traffic is mocked with respx."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from winnow.immich import UNSET, ImmichClient, ImmichError

ROOT = "http://immich.test:2283"
API = f"{ROOT}/api"


@pytest.fixture()
def client() -> Iterator[ImmichClient]:
    with ImmichClient(ROOT, "test-key") as immich:
        yield immich


@pytest.fixture()
def mock_api() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=API, assert_all_called=False) as router:
        yield router


def body_of(route: Any, index: int = 0) -> Any:
    """Decoded JSON body of a recorded request."""
    return json.loads(route.calls[index].request.content)


def asset(asset_id: str, **extra: Any) -> dict:
    """Minimal Immich asset DTO."""
    dto = {
        "id": asset_id,
        "originalFileName": f"{asset_id}.jpg",
        "type": "IMAGE",
        "localDateTime": "2024-06-01T12:00:00.000Z",
    }
    dto.update(extra)
    return dto


# ----------------------------------------------------------------------
# construction / plumbing
# ----------------------------------------------------------------------


def test_base_url_gets_api_suffix_and_strips_trailing_slash() -> None:
    with ImmichClient(f"{ROOT}/", "k") as immich:
        assert immich.base_url == ROOT
        assert str(immich._client.base_url) == f"{API}/"


def test_auth_headers_sent(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.get("/server/about").mock(return_value=httpx.Response(200, json={}))
    client.ping()
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["accept"] == "application/json"
    assert str(request.url) == f"{API}/server/about"


def test_context_manager_closes_transport() -> None:
    immich = ImmichClient(ROOT, "k")
    with immich as entered:
        assert entered is immich
        assert not immich._client.is_closed
    assert immich._client.is_closed


def test_close_is_idempotent(client: ImmichClient) -> None:
    client.close()
    client.close()
    assert client._client.is_closed


# ----------------------------------------------------------------------
# ping
# ----------------------------------------------------------------------


def test_ping_returns_about_payload(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/server/about").mock(
        return_value=httpx.Response(200, json={"version": "v3.1.0", "licensed": False})
    )
    assert client.ping() == {"version": "v3.1.0", "licensed": False}


def test_error_carries_status_and_body(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/server/about").mock(
        return_value=httpx.Response(500, text="boom: upstream exploded")
    )
    with pytest.raises(ImmichError) as excinfo:
        client.ping()
    err = excinfo.value
    assert err.status_code == 500
    assert "500" in str(err)
    assert "boom: upstream exploded" in str(err)
    assert err.body == "boom: upstream exploded"


def test_error_body_snippet_is_truncated(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/server/about").mock(return_value=httpx.Response(401, text="x" * 5000))
    with pytest.raises(ImmichError) as excinfo:
        client.ping()
    assert excinfo.value.status_code == 401
    assert len(excinfo.value.body) == 300
    assert len(str(excinfo.value)) < 500


def test_transport_error_becomes_immich_error(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.get("/server/about").mock(side_effect=httpx.ConnectError("no route to host"))
    with pytest.raises(ImmichError) as excinfo:
        client.ping()
    assert excinfo.value.status_code is None
    assert "no route to host" in str(excinfo.value)


# ----------------------------------------------------------------------
# search / assets
# ----------------------------------------------------------------------


def test_search_assets_paginates_two_pages(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    route = mock_api.post("/search/metadata").mock(
        side_effect=[
            httpx.Response(
                200, json={"assets": {"items": [asset("a1"), asset("a2")], "nextPage": 2}}
            ),
            httpx.Response(200, json={"assets": {"items": [asset("a3")], "nextPage": None}}),
        ]
    )

    found = list(client.search_assets("2024-06-01T00:00:00.000Z", "2024-07-01T00:00:00.000Z", 2))

    assert [a["id"] for a in found] == ["a1", "a2", "a3"]
    assert route.call_count == 2
    first = body_of(route, 0)
    assert first == {
        "takenAfter": "2024-06-01T00:00:00.000Z",
        "takenBefore": "2024-07-01T00:00:00.000Z",
        "size": 2,
        "page": 1,
        "withExif": True,
        "type": "IMAGE",
    }
    assert body_of(route, 1)["page"] == 2


def test_search_assets_coerces_string_next_page_to_number(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    # Immich v3.1 really does return nextPage as a string ("2"), and really
    # does 400 if the next request's `page` is not a number — coerce, never echo.
    route = mock_api.post("/search/metadata").mock(
        side_effect=[
            httpx.Response(200, json={"assets": {"items": [asset("a1")], "nextPage": "2"}}),
            httpx.Response(200, json={"assets": {"items": [], "nextPage": ""}}),
        ]
    )
    assert [a["id"] for a in client.search_assets("A", "B")] == ["a1"]
    assert body_of(route, 1)["page"] == 2


def test_search_assets_rejects_garbage_next_page(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.post("/search/metadata").mock(
        return_value=httpx.Response(
            200, json={"assets": {"items": [asset("a1")], "nextPage": "not-a-number"}}
        )
    )
    with pytest.raises(ImmichError, match="nextPage"):
        list(client.search_assets("A", "B"))


def test_search_assets_default_page_size(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.post("/search/metadata").mock(
        return_value=httpx.Response(200, json={"assets": {"items": [], "nextPage": None}})
    )
    assert list(client.search_assets("A", "B")) == []
    assert body_of(route)["size"] == 250


def test_search_assets_handles_missing_keys(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.post("/search/metadata").mock(return_value=httpx.Response(200, json={}))
    assert list(client.search_assets("A", "B")) == []


def test_search_assets_raises_on_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.post("/search/metadata").mock(return_value=httpx.Response(400, text="bad range"))
    with pytest.raises(ImmichError) as excinfo:
        list(client.search_assets("A", "B"))
    assert excinfo.value.status_code == 400


def test_search_assets_refuses_to_loop_on_a_repeated_cursor(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    """A server that echoes the same cursor would otherwise spin forever."""
    route = mock_api.post("/search/metadata").mock(
        side_effect=[
            httpx.Response(200, json={"assets": {"items": [asset("a1")], "nextPage": 2}}),
            httpx.Response(200, json={"assets": {"items": [asset("a2")], "nextPage": 2}}),
        ]
    )
    with pytest.raises(ImmichError, match="repeated page cursor"):
        list(client.search_assets("A", "B"))
    assert route.call_count == 2


def test_non_json_body_becomes_an_immich_error(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    """Pointing IMMICH_URL at a proxy login page must not escape as a bare
    JSON decode error — every caller only guards ImmichError."""
    mock_api.get("/server/about").mock(
        return_value=httpx.Response(200, html="<html><body>Please sign in</body></html>")
    )
    with pytest.raises(ImmichError) as excinfo:
        client.ping()
    assert excinfo.value.status_code == 200
    assert "non-JSON" in str(excinfo.value)
    assert "sign in" in excinfo.value.body


def test_get_asset(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    assert client.get_asset("abc")["originalFileName"] == "abc.jpg"


def test_get_asset_404_raises(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/assets/nope").mock(return_value=httpx.Response(404, text="Not Found"))
    with pytest.raises(ImmichError) as excinfo:
        client.get_asset("nope")
    assert excinfo.value.status_code == 404


# ----------------------------------------------------------------------
# thumbnails
# ----------------------------------------------------------------------


def test_fetch_thumbnail_returns_bytes(
    mock_api: respx.MockRouter, client: ImmichClient, jpeg_bytes: bytes
) -> None:
    route = mock_api.get("/assets/abc/thumbnail").mock(
        return_value=httpx.Response(200, content=jpeg_bytes)
    )
    data = client.fetch_thumbnail("abc")
    assert data == jpeg_bytes
    assert data[:2] == b"\xff\xd8"
    assert dict(route.calls.last.request.url.params) == {"size": "preview"}


def test_fetch_thumbnail_custom_size(
    mock_api: respx.MockRouter, client: ImmichClient, jpeg_bytes: bytes
) -> None:
    route = mock_api.get("/assets/abc/thumbnail").mock(
        return_value=httpx.Response(200, content=jpeg_bytes)
    )
    client.fetch_thumbnail("abc", size="thumbnail")
    assert route.calls.last.request.url.params["size"] == "thumbnail"


def test_fetch_thumbnail_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/assets/abc/thumbnail").mock(return_value=httpx.Response(404, text="missing"))
    with pytest.raises(ImmichError) as excinfo:
        client.fetch_thumbnail("abc")
    assert excinfo.value.status_code == 404


# ----------------------------------------------------------------------
# update_asset — only explicitly-passed fields go on the wire
# ----------------------------------------------------------------------


def test_update_asset_sends_only_rating(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.put("/assets/abc").mock(
        return_value=httpx.Response(200, json=asset("abc", rating=5))
    )
    result = client.update_asset("abc", rating=5)
    assert result["rating"] == 5
    assert body_of(route) == {"rating": 5}


def test_update_asset_sends_only_favorite(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.put("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    client.update_asset("abc", is_favorite=True)
    assert body_of(route) == {"isFavorite": True}


def test_update_asset_sends_only_visibility(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    route = mock_api.put("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    client.update_asset("abc", visibility="archive")
    assert body_of(route) == {"visibility": "archive"}


def test_update_asset_false_favorite_is_not_dropped(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    route = mock_api.put("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    client.update_asset("abc", is_favorite=False)
    assert body_of(route) == {"isFavorite": False}


def test_update_asset_rating_none_serializes_as_null(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    route = mock_api.put("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    client.update_asset("abc", rating=None)
    raw = route.calls.last.request.content.decode()
    assert json.loads(raw) == {"rating": None}
    assert "null" in raw


def test_update_asset_reject_combo(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.put("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    client.update_asset("abc", rating=-1, is_favorite=False, visibility="timeline")
    assert body_of(route) == {"rating": -1, "isFavorite": False, "visibility": "timeline"}


def test_update_asset_without_fields_sends_empty_body(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    route = mock_api.put("/assets/abc").mock(return_value=httpx.Response(200, json=asset("abc")))
    client.update_asset("abc")
    assert body_of(route) == {}


def test_unset_sentinel_is_not_a_plausible_value() -> None:
    assert UNSET is not None
    assert UNSET is not False


def test_update_asset_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.put("/assets/abc").mock(return_value=httpx.Response(403, text="read-only key"))
    with pytest.raises(ImmichError) as excinfo:
        client.update_asset("abc", rating=5)
    assert excinfo.value.status_code == 403
    assert "read-only key" in str(excinfo.value)


# ----------------------------------------------------------------------
# duplicates
# ----------------------------------------------------------------------


def test_duplicates(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    payload = [
        {"duplicateId": "d1", "assets": [asset("a1"), asset("a2")]},
        {"duplicateId": "d2", "assets": [asset("a3"), asset("a4")]},
    ]
    mock_api.get("/duplicates").mock(return_value=httpx.Response(200, json=payload))
    groups = client.duplicates()
    assert [g["duplicateId"] for g in groups] == ["d1", "d2"]
    assert [a["id"] for a in groups[0]["assets"]] == ["a1", "a2"]


def test_duplicates_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/duplicates").mock(return_value=httpx.Response(503, text="unavailable"))
    with pytest.raises(ImmichError) as excinfo:
        client.duplicates()
    assert excinfo.value.status_code == 503


# ----------------------------------------------------------------------
# tags
# ----------------------------------------------------------------------


def test_list_tags(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.get("/tags").mock(
        return_value=httpx.Response(
            200, json=[{"id": "t1", "name": "reject", "value": "winnow/reject"}]
        )
    )
    assert client.list_tags()[0]["id"] == "t1"


def test_upsert_tags_put_happy_path(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.put("/tags").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "t1", "name": "reject", "value": "winnow/reject"},
                {"id": "t2", "name": "best", "value": "winnow/best"},
            ],
        )
    )
    mapping = client.upsert_tags(["winnow/reject", "winnow/best"])
    assert mapping == {"winnow/reject": "t1", "winnow/best": "t2"}
    assert body_of(route) == {"tags": ["winnow/reject", "winnow/best"]}


def test_upsert_tags_matches_flat_name_when_no_value(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.put("/tags").mock(
        return_value=httpx.Response(200, json=[{"id": "t9", "name": "keepers"}])
    )
    assert client.upsert_tags(["keepers"]) == {"keepers": "t9"}


def test_upsert_tags_empty_makes_no_request(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    assert client.upsert_tags([]) == {}
    assert not mock_api.calls


def test_upsert_tags_fills_gaps_from_list_tags(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    # Bulk upsert only echoes the newly created tag; the other id must come
    # from the full tag listing.
    mock_api.put("/tags").mock(
        return_value=httpx.Response(200, json=[{"id": "t2", "value": "winnow/best"}])
    )
    listing = mock_api.get("/tags").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "t1", "value": "winnow/reject"},
                {"id": "t2", "value": "winnow/best"},
            ],
        )
    )
    mapping = client.upsert_tags(["winnow/reject", "winnow/best"])
    assert mapping == {"winnow/reject": "t1", "winnow/best": "t2"}
    assert listing.call_count == 1


def test_upsert_tags_falls_back_on_404(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    put_route = mock_api.put("/tags").mock(return_value=httpx.Response(404, text="Not Found"))
    post_route = mock_api.post("/tags").mock(
        side_effect=[
            httpx.Response(201, json={"id": "t1", "value": "winnow/reject"}),
            httpx.Response(201, json={"id": "t2", "value": "winnow/best"}),
        ]
    )
    get_route = mock_api.get("/tags").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "t1", "name": "reject", "value": "winnow/reject"},
                {"id": "t2", "name": "best", "value": "winnow/best"},
                {"id": "t3", "name": "unrelated", "value": "other"},
            ],
        )
    )

    mapping = client.upsert_tags(["winnow/reject", "winnow/best"])

    assert mapping == {"winnow/reject": "t1", "winnow/best": "t2"}
    assert put_route.call_count == 1
    assert post_route.call_count == 2
    assert body_of(post_route, 0) == {"name": "winnow/reject"}
    assert body_of(post_route, 1) == {"name": "winnow/best"}
    assert get_route.call_count == 1


def test_upsert_tags_fallback_tolerates_existing_tag(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.put("/tags").mock(return_value=httpx.Response(404))
    mock_api.post("/tags").mock(return_value=httpx.Response(400, text="already exists"))
    mock_api.get("/tags").mock(
        return_value=httpx.Response(200, json=[{"id": "t1", "value": "winnow/reject"}])
    )
    assert client.upsert_tags(["winnow/reject"]) == {"winnow/reject": "t1"}


def test_upsert_tags_fallback_propagates_hard_error(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.put("/tags").mock(return_value=httpx.Response(404))
    mock_api.post("/tags").mock(return_value=httpx.Response(500, text="kaboom"))
    with pytest.raises(ImmichError) as excinfo:
        client.upsert_tags(["winnow/reject"])
    assert excinfo.value.status_code == 500


def test_upsert_tags_omits_unresolvable_names(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.put("/tags").mock(return_value=httpx.Response(200, json=[]))
    mock_api.get("/tags").mock(return_value=httpx.Response(200, json=[]))
    assert client.upsert_tags(["winnow/reject"]) == {}


def test_upsert_tags_non_404_error_propagates(
    mock_api: respx.MockRouter, client: ImmichClient
) -> None:
    mock_api.put("/tags").mock(return_value=httpx.Response(500, text="server on fire"))
    post_route = mock_api.post("/tags")
    with pytest.raises(ImmichError) as excinfo:
        client.upsert_tags(["winnow/reject"])
    assert excinfo.value.status_code == 500
    assert post_route.call_count == 0


def test_tag_assets(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.put("/tags/t1/assets").mock(
        return_value=httpx.Response(200, json=[{"id": "a1", "success": True}])
    )
    assert client.tag_assets("t1", ["a1", "a2"]) is None
    assert body_of(route) == {"ids": ["a1", "a2"]}


def test_tag_assets_empty_is_noop(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    client.tag_assets("t1", [])
    assert not mock_api.calls


def test_tag_assets_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.put("/tags/t1/assets").mock(return_value=httpx.Response(404, text="no such tag"))
    with pytest.raises(ImmichError) as excinfo:
        client.tag_assets("t1", ["a1"])
    assert excinfo.value.status_code == 404


def test_untag_assets(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.delete("/tags/t1/assets").mock(return_value=httpx.Response(200, json=[]))
    assert client.untag_assets("t1", ["a1", "a2"]) is None
    assert body_of(route) == {"ids": ["a1", "a2"]}


def test_untag_assets_empty_is_noop(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    client.untag_assets("t1", [])
    assert not mock_api.calls


def test_delete_tag(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.delete("/tags/t1").mock(return_value=httpx.Response(204))
    assert client.delete_tag("t1") is None
    assert route.call_count == 1


def test_delete_tag_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.delete("/tags/t1").mock(return_value=httpx.Response(400, text="nope"))
    with pytest.raises(ImmichError) as excinfo:
        client.delete_tag("t1")
    assert excinfo.value.status_code == 400


# ----------------------------------------------------------------------
# stacks
# ----------------------------------------------------------------------


def test_create_stack_first_id_is_primary(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.post("/stacks").mock(
        return_value=httpx.Response(
            201, json={"id": "s1", "primaryAssetId": "a1", "assets": [asset("a1"), asset("a2")]}
        )
    )
    stack = client.create_stack(["a1", "a2", "a3"])
    assert stack["id"] == "s1"
    assert body_of(route) == {"assetIds": ["a1", "a2", "a3"]}
    assert body_of(route)["assetIds"][0] == "a1"


def test_create_stack_error(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    mock_api.post("/stacks").mock(return_value=httpx.Response(400, text="need 2+ assets"))
    with pytest.raises(ImmichError) as excinfo:
        client.create_stack(["a1"])
    assert excinfo.value.status_code == 400
    assert "need 2+ assets" in str(excinfo.value)


def test_delete_stack(mock_api: respx.MockRouter, client: ImmichClient) -> None:
    route = mock_api.delete("/stacks/s1").mock(return_value=httpx.Response(204))
    assert client.delete_stack("s1") is None
    assert route.call_count == 1
