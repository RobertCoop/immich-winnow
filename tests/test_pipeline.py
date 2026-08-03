"""End-to-end tests for the Winnow pipeline.

No network and no secrets: Immich is mocked with respx and the judge is a
scripted stub. The stub identifies each photo by the colour of the synthetic
JPEG it is handed, which also proves the pipeline sends the *right* image for
every asset it judges.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
import respx
from PIL import Image

from conftest import make_jpeg
from winnow.config import Settings, load_settings
from winnow.immich import ImmichClient
from winnow.judge import JudgeResult
from winnow.ledger import Ledger
from winnow.pipeline import writeback
from winnow.pipeline.finals import FinalsStats, run_finals
from winnow.pipeline.rank import RankStats, collect_pairs, run_rank
from winnow.pipeline.scan import (
    ScanStats,
    asset_camera,
    asset_taken_at,
    burst_id_for,
    iso_bound,
    run_scan,
    thumb_path,
)
from winnow.pipeline.triage import (
    TriageStats,
    apply_burst_verdict,
    apply_triage_verdict,
    ingest_triage_batch,
    pending_burst_ids,
    pending_single_ids,
    run_triage_direct,
    submit_triage_batch,
)
from winnow.schemas import BurstVerdict, BWSVerdict, PairVerdict, TriageVerdict

ROOT = "http://immich.test:2283"
API = f"{ROOT}/api"

#: Indices 0-2 are one burst; 3-8 are ordinary singles hours apart.
BURST_INDICES = (0, 1, 2)
ASSET_COUNT = 9

#: Relative quality per photo index, used by every stubbed judgment.
QUALITY = {0: 10, 1: 70, 2: 20, 3: 5, 4: 1, 5: 2, 6: 80, 7: 100, 8: 90}

#: The one head-to-head where the stub is position-biased (always picks "A"),
#: which must surface as a tie once the order is swapped.
BIASED_PAIR = frozenset({7, 8})

TRIAGE_SCRIPT: dict[int, dict[str, Any]] = {
    1: {
        "category": "photo",
        "verdict": "candidate",
        "technical_score": 8,
        "reasons": ["burst keeper", "eyes open"],
        "confidence": "medium",
    },
    3: {
        "category": "screenshot",
        "verdict": "neutral",
        "technical_score": 4,
        "reasons": ["app interface"],
        "confidence": "high",
    },
    4: {
        "category": "photo",
        "verdict": "reject",
        "technical_score": 1,
        "reasons": ["severe motion blur"],
        "confidence": "high",
    },
    5: {
        "category": "photo",
        "verdict": "reject",
        "technical_score": 3,
        "reasons": ["soft on the subject"],
        "confidence": "low",
    },
    6: {
        "category": "photo",
        "verdict": "candidate",
        "technical_score": 9,
        "reasons": ["clean rim light"],
        "confidence": "high",
    },
    7: {
        "category": "photo",
        "verdict": "candidate",
        "technical_score": 10,
        "reasons": ["light, moment and framing all land"],
        "confidence": "high",
    },
    8: {
        "category": "photo",
        "verdict": "neutral",
        "technical_score": 8,
        "reasons": ["well exposed"],
        "confidence": "medium",
    },
}


# ----------------------------------------------------------------------
# synthetic Immich library
# ----------------------------------------------------------------------


def aid(index: int) -> str:
    """Deterministic UUID-shaped asset id for a photo index."""
    return f"aaaaaaaa-0000-4000-8000-{index:012d}"


def color_for(index: int) -> tuple[int, int, int]:
    """Solid colour encoding a photo index in its red channel."""
    return (20 + 20 * index, 40, 60)


def index_of(image_b64: str) -> int:
    """Recover a photo index from a base64 JPEG the pipeline sent us."""
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    red = image.getpixel((image.width // 2, image.height // 2))[0]
    index = round(red / 20) - 1
    assert index in QUALITY, f"unrecognised image (red={red})"
    return index


def taken_at_for(index: int) -> str:
    """Capture timestamp: burst frames 2s apart, singles an hour apart."""
    if index in BURST_INDICES:
        return f"2024-06-01T10:00:{2 * index:02d}.000Z"
    return f"2024-06-01T{12 + index - 3:02d}:00:00.000Z"


def asset_dto(index: int) -> dict[str, Any]:
    """Minimal Immich asset DTO for a photo index."""
    make, model = ("Canon", "EOS R6") if index in BURST_INDICES else ("Apple", "iPhone 13")
    taken = taken_at_for(index)
    return {
        "id": aid(index),
        "originalFileName": f"IMG_{index:04d}.jpg",
        "type": "IMAGE",
        "localDateTime": taken,
        "fileCreatedAt": taken,
        "rating": 0,
        "isFavorite": False,
        "exifInfo": {
            "make": make,
            "model": model,
            "exifImageWidth": 4000,
            "exifImageHeight": 3000,
            "dateTimeOriginal": taken,
        },
    }


# ----------------------------------------------------------------------
# scripted judge
# ----------------------------------------------------------------------


def script_triage(image_b64: str) -> TriageVerdict:
    return TriageVerdict(**TRIAGE_SCRIPT[index_of(image_b64)])


def script_burst(images_b64: Sequence[str]) -> BurstVerdict:
    indices = [index_of(image) for image in images_b64]
    best = max(range(len(indices)), key=lambda pos: QUALITY[indices[pos]])
    return BurstVerdict(
        best_index=best + 1,
        reject_indices=[pos + 1 for pos in range(len(indices)) if pos != best],
        note="sharpest frame of the run",
    )


def script_bws(images_b64: Sequence[str]) -> BWSVerdict:
    indices = [index_of(image) for image in images_b64]
    best = max(range(len(indices)), key=lambda pos: QUALITY[indices[pos]])
    worst = min(range(len(indices)), key=lambda pos: QUALITY[indices[pos]])
    return BWSVerdict(best_index=best + 1, worst_index=worst + 1, note="clear extremes")


def script_pair(a_b64: str, b_b64: str) -> PairVerdict:
    left, right = index_of(a_b64), index_of(b_b64)
    if frozenset((left, right)) == BIASED_PAIR:
        return PairVerdict(winner="A", note="position-biased stub")
    winner = "A" if QUALITY[left] > QUALITY[right] else "B"
    return PairVerdict(winner=winner, note="stronger photograph")


def _message(text: str, model: str = "claude-haiku-4-5") -> SimpleNamespace:
    """A canned Anthropic message carrying one JSON text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        model=model,
    )


def _images_of(params: dict[str, Any]) -> list[str]:
    """Pull the base64 images back out of built request params."""
    return [
        block["source"]["data"]
        for block in params["messages"][0]["content"]
        if block.get("type") == "image"
    ]


class StubBatchClient:
    """Stands in for ``anthropic.Anthropic`` batch endpoints."""

    def __init__(self) -> None:
        self.messages = SimpleNamespace(batches=self)
        self.submitted: dict[str, list[dict[str, Any]]] = {}
        self.error_ids: set[str] = set()
        self.status = "ended"
        self._counter = 0

    def create(self, requests: list[dict[str, Any]]) -> SimpleNamespace:
        self._counter += 1
        batch_id = f"msgbatch_{self._counter}"
        self.submitted[batch_id] = list(requests)
        return SimpleNamespace(id=batch_id)

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status=self.status)

    def results(self, batch_id: str) -> Iterator[SimpleNamespace]:
        for request in self.submitted[batch_id]:
            custom_id = request["custom_id"]
            if custom_id in self.error_ids:
                yield SimpleNamespace(
                    custom_id=custom_id,
                    result=SimpleNamespace(type="errored", error="overloaded"),
                )
                continue
            images = _images_of(request["params"])
            if custom_id.startswith("burst_"):
                verdict: Any = script_burst(images)
            else:
                verdict = script_triage(images[0])
            message = _message(verdict.model_dump_json())
            yield SimpleNamespace(
                custom_id=custom_id,
                result=SimpleNamespace(type="succeeded", message=message),
            )


class StubJudge:
    """Scripted stand-in for :class:`winnow.judge.Judge`."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.client = StubBatchClient()

    def triage(self, model: str, image_b64: str) -> JudgeResult:
        self.calls["triage"] += 1
        return JudgeResult(script_triage(image_b64), 11, 7, model)

    def burst(self, model: str, images_b64: list[str]) -> JudgeResult:
        self.calls["burst"] += 1
        return JudgeResult(script_burst(images_b64), 31, 9, model)

    def bws(self, model: str, images_b64: list[str]) -> JudgeResult:
        self.calls["bws"] += 1
        return JudgeResult(script_bws(images_b64), 41, 13, model)

    def pair(self, model: str, a_b64: str, b_b64: str) -> JudgeResult:
        self.calls["pair"] += 1
        return JudgeResult(script_pair(a_b64, b_b64), 21, 5, model)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture()
def dup_groups() -> list[dict[str, Any]]:
    """Mutable payload for ``GET /api/duplicates`` — tests may append to it."""
    return []


@pytest.fixture()
def immich_api(dup_groups: list[dict[str, Any]]) -> Iterator[respx.MockRouter]:
    """A fake Immich server covering every endpoint the pipeline touches."""
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API}/server/about", name="about").mock(
            return_value=httpx.Response(200, json={"version": "v3.1.0"})
        )
        router.post(f"{API}/search/metadata", name="search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "assets": {
                        "items": [asset_dto(i) for i in range(ASSET_COUNT)],
                        "nextPage": None,
                        "count": ASSET_COUNT,
                    }
                },
            )
        )
        for index in range(ASSET_COUNT):
            router.get(f"{API}/assets/{aid(index)}/thumbnail", name=f"thumb{index}").mock(
                return_value=httpx.Response(200, content=make_jpeg(color=color_for(index)))
            )
        router.get(f"{API}/duplicates", name="duplicates").mock(
            side_effect=lambda request: httpx.Response(200, json=dup_groups)
        )
        router.put(f"{API}/tags", name="tag_upsert").mock(side_effect=_upsert_tags)
        router.put(url__regex=re.escape(API) + r"/tags/[^/]+/assets", name="tag_assets").mock(
            return_value=httpx.Response(200, json={})
        )
        router.put(url__regex=re.escape(API) + r"/assets/[0-9a-f-]+$", name="update_asset").mock(
            side_effect=lambda request: httpx.Response(200, json={"id": "ok"})
        )
        router.post(f"{API}/stacks", name="create_stack").mock(
            return_value=httpx.Response(201, json={"id": "stack-1"})
        )
        router.get(f"{API}/albums", name="list_albums").mock(
            return_value=httpx.Response(200, json=[])
        )
        router.post(f"{API}/albums", name="create_album").mock(
            return_value=httpx.Response(201, json={"id": "album-1", "albumName": "Five-Stars"})
        )
        router.put(
            url__regex=re.escape(API) + r"/albums/[^/]+/assets", name="album_assets"
        ).mock(return_value=httpx.Response(200, json=[]))
        yield router


def _upsert_tags(request: httpx.Request) -> httpx.Response:
    """Answer ``PUT /api/tags`` with one DTO per requested tag path."""
    names = json.loads(request.content)["tags"]
    return httpx.Response(
        200,
        json=[
            {"id": f"tag-{i}", "name": name.split("/")[-1], "value": name}
            for i, name in enumerate(names)
        ],
    )


@pytest.fixture()
def settings(settings_env) -> Settings:
    return load_settings(finals_rounds=2)


@pytest.fixture()
def ledger(settings: Settings) -> Iterator[Ledger]:
    with Ledger(settings.db_path) as led:
        yield led


@pytest.fixture()
def immich(immich_api: respx.MockRouter) -> Iterator[ImmichClient]:
    with ImmichClient(ROOT, "test-key") as client:
        yield client


@pytest.fixture()
def judge() -> StubJudge:
    return StubJudge()


@pytest.fixture()
def scanned(settings: Settings, ledger: Ledger, immich: ImmichClient) -> ScanStats:
    return run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")


@pytest.fixture()
def triaged(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> TriageStats:
    return run_triage_direct(settings, ledger, judge)


@pytest.fixture()
def ranked(triaged: TriageStats, settings: Settings, ledger: Ledger, judge: StubJudge) -> RankStats:
    return run_rank(settings, ledger, judge)


@pytest.fixture()
def finaled(ranked: RankStats, settings: Settings, ledger: Ledger, judge: StubJudge) -> FinalsStats:
    return run_finals(settings, ledger, judge, five_count=2, four_frac=0.5)


def buckets_of(ledger: Ledger) -> dict[str, str]:
    """Asset id -> decision bucket."""
    return {row["asset_id"]: row["bucket"] for row in ledger.decisions()}


# ----------------------------------------------------------------------
# scan helpers
# ----------------------------------------------------------------------


def test_iso_bound_normalises_bare_dates() -> None:
    assert iso_bound("2024-06-01") == "2024-06-01T00:00:00.000Z"
    assert iso_bound(date(2024, 6, 1)) == "2024-06-01T00:00:00.000Z"
    assert iso_bound("2024-06-01T09:30:00.000Z") == "2024-06-01T09:30:00.000Z"


def test_iso_bound_converts_datetimes_to_utc() -> None:
    naive = datetime(2024, 6, 1, 9, 30)
    assert iso_bound(naive) == "2024-06-01T09:30:00.000Z"
    aware = datetime(2024, 6, 1, 9, 30, tzinfo=UTC)
    assert iso_bound(aware) == "2024-06-01T09:30:00.000Z"


def test_asset_camera_and_taken_at() -> None:
    dto = asset_dto(0)
    assert asset_camera(dto) == "Canon|EOS R6"
    assert asset_taken_at(dto) == datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
    assert asset_camera({"exifInfo": {}}) == ""
    assert asset_taken_at({"exifInfo": {}}) is None


def test_burst_id_is_stable_and_order_independent() -> None:
    assert burst_id_for(["b", "a"]) == burst_id_for(["a", "b"])
    assert burst_id_for(["a", "b"]) != burst_id_for(["a", "c"])


# ----------------------------------------------------------------------
# scan
# ----------------------------------------------------------------------


def test_scan_populates_ledger(scanned: ScanStats, settings: Settings, ledger: Ledger) -> None:
    assert (scanned.seen, scanned.new, scanned.errors) == (ASSET_COUNT, ASSET_COUNT, 0)
    assert scanned.thumbs_fetched == ASSET_COUNT
    assert scanned.thumbs_cached == 0

    rows = {row["id"]: row for row in ledger.get_assets()}
    assert len(rows) == ASSET_COUNT
    first = rows[aid(0)]
    assert first["filename"] == "IMG_0000.jpg"
    assert first["camera"] == "Canon|EOS R6"
    assert (first["width"], first["height"]) == (4000, 3000)
    assert isinstance(first["dhash"], int)
    assert first["taken_at"].startswith("2024-06-01T10:00:00")


def test_scan_caches_prepared_thumbnails(scanned: ScanStats, settings: Settings) -> None:
    for index in range(ASSET_COUNT):
        path = thumb_path(settings.cache_dir, aid(index))
        assert path.exists()
        image = Image.open(io.BytesIO(path.read_bytes()))
        assert image.format == "JPEG"
        assert not image.getexif()


def test_scan_detects_the_burst(scanned: ScanStats, ledger: Ledger) -> None:
    groups = ledger.burst_groups()
    assert scanned.bursts == 1
    assert scanned.burst_assets == 3
    assert len(groups) == 1
    members = next(iter(groups.values()))
    assert members == [aid(0), aid(1), aid(2)]


def test_scan_leaves_singles_ungrouped(scanned: ScanStats, ledger: Ledger) -> None:
    grouped = {member for members in ledger.burst_groups().values() for member in members}
    for index in range(3, ASSET_COUNT):
        assert aid(index) not in grouped


def test_scan_is_resumable(
    scanned: ScanStats, settings: Settings, ledger: Ledger, immich: ImmichClient
) -> None:
    again = run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")
    assert again.seen == ASSET_COUNT
    assert again.new == 0
    assert again.thumbs_fetched == 0
    assert again.thumbs_cached == ASSET_COUNT
    assert len(ledger.get_assets()) == ASSET_COUNT
    assert len(ledger.burst_groups()) == 1


def test_scan_reports_progress(settings: Settings, ledger: Ledger, immich: ImmichClient) -> None:
    lines: list[str] = []
    run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02", on_progress=lines.append)
    assert len(lines) == ASSET_COUNT + 1
    assert lines[-1].startswith("grouped 1 bursts")


def test_scan_merges_immich_duplicate_groups(
    dup_groups: list[dict[str, Any]],
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    dup_groups.append({"duplicateId": "dup-1", "assets": [{"id": aid(4)}, {"id": aid(5)}]})
    stats = run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")
    groups = ledger.burst_groups()
    assert stats.bursts == 2
    assert sorted(sorted(members) for members in groups.values()) == [
        [aid(0), aid(1), aid(2)],
        [aid(4), aid(5)],
    ]


def test_scan_survives_a_failing_thumbnail(
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    immich_api["thumb4"].mock(return_value=httpx.Response(500, text="boom"))
    stats = run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")
    assert stats.errors == 1
    assert stats.seen == ASSET_COUNT
    assert len(ledger.get_assets()) == ASSET_COUNT - 1
    assert not thumb_path(settings.cache_dir, aid(4)).exists()


# ----------------------------------------------------------------------
# triage — direct
# ----------------------------------------------------------------------


def test_triage_judges_burst_then_singles(triaged: TriageStats, judge: StubJudge) -> None:
    assert triaged.bursts_judged == 1
    assert triaged.burst_losers == 2
    # six ungrouped singles plus the burst winner
    assert triaged.singles_judged == 7
    assert triaged.errors == 0
    assert judge.calls["burst"] == 1
    assert judge.calls["triage"] == 7


def test_triage_records_the_burst_winner(triaged: TriageStats, ledger: Ledger) -> None:
    rows = ledger.burst_rows()
    assert len(rows) == 1
    assert rows[0]["winner_id"] == aid(1)
    assert sorted(rows[0]["reject_ids"]) == [aid(0), aid(2)]
    assert rows[0]["note"]

    triaged_ids = {row["asset_id"] for row in ledger.triage_rows()}
    assert aid(1) in triaged_ids, "burst winner must also be triaged individually"
    assert aid(0) not in triaged_ids
    assert aid(2) not in triaged_ids


def test_triage_buckets_every_asset(triaged: TriageStats, ledger: Ledger) -> None:
    buckets = buckets_of(ledger)
    assert buckets[aid(0)] == "burst_loser"
    assert buckets[aid(2)] == "burst_loser"
    assert buckets[aid(1)] == "middle"
    assert buckets[aid(3)] == "nonphoto"
    assert buckets[aid(4)] == "reject"
    # a reject the model was unsure about stays in the untouched middle
    assert buckets[aid(5)] == "middle"
    assert buckets[aid(6)] == "middle"


def test_triage_stats_match_the_ledger(triaged: TriageStats, ledger: Ledger) -> None:
    counts = Counter(buckets_of(ledger).values())
    assert triaged.rejects == counts["reject"] == 1
    assert triaged.nonphotos == counts["nonphoto"] == 1
    assert triaged.middles == counts["middle"] == 5
    assert triaged.candidates == 3
    assert triaged.input_tokens > 0
    assert triaged.output_tokens > 0


def test_triage_stores_verdict_details(triaged: TriageStats, ledger: Ledger) -> None:
    row = next(r for r in ledger.triage_rows() if r["asset_id"] == aid(4))
    assert row["verdict"] == "reject"
    assert row["confidence"] == "high"
    assert row["technical_score"] == 1
    assert row["reasons"] == ["severe motion blur"]


def test_triage_is_resumable(
    triaged: TriageStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    before = dict(judge.calls)
    again = run_triage_direct(settings, ledger, judge)
    assert dict(judge.calls) == before
    assert (again.bursts_judged, again.singles_judged, again.errors) == (0, 0, 0)
    assert pending_single_ids(ledger) == []


def test_triage_limit_caps_work(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    stats = run_triage_direct(settings, ledger, judge, limit=2)
    assert stats.bursts_judged == 1
    assert stats.singles_judged == 1
    assert judge.calls["triage"] == 1


def test_triage_reports_progress(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    lines: list[str] = []
    run_triage_direct(settings, ledger, judge, on_progress=lines.append)
    assert len(lines) == 8
    assert lines[0].startswith("burst ")


def test_triage_counts_missing_thumbnails_as_errors(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    thumb_path(settings.cache_dir, aid(6)).unlink()
    stats = run_triage_direct(settings, ledger, judge)
    assert stats.errors == 1
    assert stats.singles_judged == 6
    assert aid(6) in pending_single_ids(ledger)


# ----------------------------------------------------------------------
# triage — batch
# ----------------------------------------------------------------------


def test_triage_batch_round_trip(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    batch_id = submit_triage_batch(settings, ledger, judge)
    assert batch_id == "msgbatch_1"

    custom_ids = {item["custom_id"] for item in ledger.batch_items_for(batch_id)}
    assert f"triage_{aid(3)}" in custom_ids
    assert any(cid.startswith("burst_") for cid in custom_ids)
    assert len(custom_ids) == 7  # one burst + six ungrouped singles
    assert all(":" not in cid for cid in custom_ids)
    assert [row["batch_id"] for row in ledger.open_batches()] == [batch_id]

    stats = ingest_triage_batch(settings, ledger, judge, batch_id)
    assert stats.bursts_judged == 1
    assert stats.burst_losers == 2
    assert stats.singles_judged == 6
    assert stats.errors == 0
    assert stats.input_tokens > 0

    buckets = buckets_of(ledger)
    assert buckets[aid(0)] == "burst_loser"
    assert buckets[aid(3)] == "nonphoto"
    assert buckets[aid(4)] == "reject"
    assert ledger.open_batches() == []


def test_triage_batch_picks_up_the_burst_winner_next_round(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    first = submit_triage_batch(settings, ledger, judge)
    ingest_triage_batch(settings, ledger, judge, first)
    assert pending_single_ids(ledger) == [aid(1)]

    second = submit_triage_batch(settings, ledger, judge)
    assert second is not None and second != first
    assert [item["custom_id"] for item in ledger.batch_items_for(second)] == [f"triage_{aid(1)}"]

    ingest_triage_batch(settings, ledger, judge, second)
    assert pending_single_ids(ledger) == []
    assert aid(1) in {row["asset_id"] for row in ledger.triage_rows()}

    assert submit_triage_batch(settings, ledger, judge) is None


def test_triage_batch_ingest_without_id_walks_open_batches(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    submit_triage_batch(settings, ledger, judge)
    stats = ingest_triage_batch(settings, ledger, judge)
    assert stats.singles_judged == 6
    assert ledger.open_batches() == []


def test_triage_batch_leaves_unfinished_batches_alone(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    batch_id = submit_triage_batch(settings, ledger, judge)
    judge.client.status = "in_progress"
    stats = ingest_triage_batch(settings, ledger, judge)
    assert stats.singles_judged == 0
    assert [row["batch_id"] for row in ledger.open_batches()] == [batch_id]


def test_triage_batch_records_failed_items(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    batch_id = submit_triage_batch(settings, ledger, judge)
    judge.client.error_ids.add(f"triage_{aid(3)}")
    stats = ingest_triage_batch(settings, ledger, judge, batch_id)
    assert stats.errors == 1
    assert stats.singles_judged == 5
    failed = next(
        item for item in ledger.batch_items_for(batch_id) if item["custom_id"].endswith(aid(3))
    )
    assert failed["error"] == "errored"
    assert aid(3) not in buckets_of(ledger)


# ----------------------------------------------------------------------
# rank
# ----------------------------------------------------------------------


def test_rank_scores_the_candidate_pool(ranked: RankStats, ledger: Ledger) -> None:
    assert ranked.candidates == 4
    assert ranked.sets_planned == 4  # four appearances, one set each (pool < set size)
    assert ranked.sets_judged == 4
    assert ranked.sets_skipped == 0
    assert ranked.errors == 0
    assert ranked.pairs > 0

    scored = {row["asset_id"]: row for row in ledger.score_rows()}
    assert set(scored) == {aid(1), aid(6), aid(7), aid(8)}
    assert ledger.score_rows()[0]["asset_id"] == aid(7)
    assert scored[aid(7)]["rank"] == 1
    assert scored[aid(7)]["bt_score"] > scored[aid(1)]["bt_score"]


def test_rank_excludes_rejects_and_nonphotos(ranked: RankStats, ledger: Ledger) -> None:
    scored = {row["asset_id"] for row in ledger.score_rows()}
    for index in (0, 2, 3, 4, 5):
        assert aid(index) not in scored


def test_rank_records_sets_and_pairs(ranked: RankStats, ledger: Ledger) -> None:
    rows = ledger.bws_rows()
    assert len(rows) == 4
    for row in rows:
        assert len(row["member_ids"]) == 4
        assert row["best_id"] == aid(7)
        assert row["worst_id"] == aid(1)
        assert row["round"] >= 1
    # best beats 3 others, 2 others beat the worst -> 5 implied pairs per set
    assert len(collect_pairs(ledger)) == 4 * 5


def test_rank_is_resumable(
    ranked: RankStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    before = judge.calls["bws"]
    again = run_rank(settings, ledger, judge)
    assert judge.calls["bws"] == before
    assert again.sets_judged == 0
    assert again.sets_skipped == 4
    assert len(ledger.bws_rows()) == 4


# ----------------------------------------------------------------------
# finals
# ----------------------------------------------------------------------


def test_finals_awards_stars(finaled: FinalsStats, ledger: Ledger) -> None:
    assert finaled.pool == 4
    assert finaled.rounds == 2
    assert finaled.five_star == 2
    assert finaled.four_star == 1

    stars = {row["asset_id"]: row["stars"] for row in ledger.score_rows()}
    assert stars[aid(7)] == 5
    assert sorted(s for s in stars.values() if s) == [4, 5, 5]

    buckets = buckets_of(ledger)
    assert buckets[aid(7)] == "five_star"
    assert Counter(buckets.values())["five_star"] == 2
    assert Counter(buckets.values())["four_star"] == 1


def test_finals_judges_every_pair_twice(finaled: FinalsStats, judge: StubJudge) -> None:
    assert finaled.pairs_judged == 4
    assert judge.calls["pair"] == 2 * finaled.pairs_judged


def test_finals_ties_on_disagreement(finaled: FinalsStats, ledger: Ledger) -> None:
    assert finaled.disagreements == 1
    assert finaled.ties == 1
    rows = [row for row in ledger.pair_rows() if row["winner"] == "tie"]
    assert len(rows) == 1
    assert {rows[0]["a_id"], rows[0]["b_id"]} == {aid(7), aid(8)}
    assert all(row["stage"] == "finals" for row in ledger.pair_rows())


def test_finals_never_replays_a_pair(
    finaled: FinalsStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    again = run_finals(settings, ledger, judge, five_count=2, four_frac=0.5)
    keys = [frozenset((row["a_id"], row["b_id"])) for row in ledger.pair_rows()]
    assert len(keys) == len(set(keys))
    assert again.pairs_judged == 2  # the only two pairings still unplayed

    third = run_finals(settings, ledger, judge, five_count=2, four_frac=0.5)
    assert third.pairs_judged == 0


def test_finals_needs_a_pool(settings: Settings, ledger: Ledger, judge: StubJudge) -> None:
    stats = run_finals(settings, ledger, judge)
    assert stats.pool == 0
    assert stats.pairs_judged == 0
    assert judge.calls["pair"] == 0


def test_finals_default_five_count(
    ranked: RankStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    stats = run_finals(settings, ledger, judge)
    assert stats.five_star == 1  # 5% of four scored candidates, floor of one
    assert stats.four_star == 1  # 15% of four scored candidates rounds to one
    assert stats.three_star == 0  # full spectrum is opt-in


# ----------------------------------------------------------------------
# write-back — plan
# ----------------------------------------------------------------------


def test_plan_lists_every_expected_action(finaled: FinalsStats, ledger: Ledger) -> None:
    actions = writeback.plan(ledger)
    assert Counter(action.bucket for action in actions) == {
        "burst_loser": 2,
        "five_star": 2,
        "reject": 1,
        "nonphoto": 1,
        "four_star": 1,
        "burst_stack": 1,
    }
    assert all(action.description for action in actions)
    assert all(action.api_ops for action in actions)


def test_plan_reject_action(finaled: FinalsStats, ledger: Ledger) -> None:
    action = next(a for a in writeback.plan(ledger) if a.bucket == "reject")
    assert action.asset_id == aid(4)
    assert action.group == "reject"
    # Archive does the hiding; rating -1 rides along for servers that keep it
    # (Immich v3.1 drops -1 silently).
    assert action.api_ops == [
        {"op": "update_asset", "asset_id": aid(4), "rating": -1, "visibility": "archive"},
        {"op": "tag", "asset_id": aid(4), "tag": "winnow/reject"},
    ]


def test_plan_nonphoto_action_uses_the_category_tag(finaled: FinalsStats, ledger: Ledger) -> None:
    action = next(a for a in writeback.plan(ledger) if a.bucket == "nonphoto")
    assert action.asset_id == aid(3)
    assert action.group == "nonphoto"
    assert action.api_ops == [
        {"op": "update_asset", "asset_id": aid(3), "visibility": "archive"},
        {"op": "tag", "asset_id": aid(3), "tag": "winnow/screenshot"},
    ]


def test_plan_five_star_action(finaled: FinalsStats, ledger: Ledger) -> None:
    action = next(a for a in writeback.plan(ledger) if a.asset_id == aid(7))
    assert action.bucket == "five_star"
    assert action.group == "stars"
    assert action.api_ops == [
        {"op": "update_asset", "asset_id": aid(7), "rating": 5, "is_favorite": True},
        {"op": "tag", "asset_id": aid(7), "tag": "winnow/best"},
    ]


def test_plan_stack_action_puts_the_winner_first(finaled: FinalsStats, ledger: Ledger) -> None:
    action = next(a for a in writeback.plan(ledger) if a.bucket == "burst_stack")
    assert action.asset_id is None
    assert action.burst_id
    assert action.group == "stacks"
    assert action.api_ops == [{"op": "stack", "asset_ids": [aid(1), aid(0), aid(2)]}]


def test_plan_ignores_the_middle(finaled: FinalsStats, ledger: Ledger) -> None:
    middles = {a for a, b in buckets_of(ledger).items() if b == "middle"}
    assert middles
    planned = {action.asset_id for action in writeback.plan(ledger)}
    assert not (middles & planned)


# ----------------------------------------------------------------------
# write-back — apply
# ----------------------------------------------------------------------


WRITE_ROUTES = ("tag_upsert", "tag_assets", "update_asset", "create_stack")


def test_apply_dry_run_writes_nothing(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    stats = writeback.apply(settings, ledger, immich, dry_run=True, album="")
    assert stats.dry_run is True
    assert stats.planned == 8
    assert stats.selected == 8
    assert stats.applied == 0
    assert len(stats.actions) == 8
    for name in WRITE_ROUTES:
        assert immich_api[name].call_count == 0
    assert len(ledger.decisions(unapplied_only=True)) == len(ledger.decisions())


def test_apply_live_writes_to_immich(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    stats = writeback.apply(settings, ledger, immich, dry_run=False, album="")
    assert stats.dry_run is False
    assert stats.applied == 8
    assert stats.failed == 0
    assert stats.tags_resolved == 4
    assert stats.assets_tagged == 6
    assert stats.assets_updated == 5
    assert stats.stacks_created == 1

    assert immich_api["tag_upsert"].call_count == 1
    assert json.loads(immich_api["tag_upsert"].calls[0].request.content)["tags"] == [
        "winnow/best",
        "winnow/burst-loser",
        "winnow/reject",
        "winnow/screenshot",
    ]
    assert immich_api["tag_assets"].call_count == 4
    assert immich_api["update_asset"].call_count == 5
    assert immich_api["create_stack"].call_count == 1
    assert json.loads(immich_api["create_stack"].calls[0].request.content) == {
        "assetIds": [aid(1), aid(0), aid(2)]
    }


def test_apply_live_sends_the_expected_asset_payloads(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    writeback.apply(settings, ledger, immich, dry_run=False)
    payloads = {
        call.request.url.path.rsplit("/", 1)[-1]: json.loads(call.request.content)
        for call in immich_api["update_asset"].calls
    }
    assert payloads[aid(4)] == {"rating": -1, "visibility": "archive"}
    assert payloads[aid(3)] == {"visibility": "archive"}
    assert payloads[aid(7)] == {"rating": 5, "isFavorite": True}


def test_apply_marks_decisions_applied_and_is_resumable(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    writeback.apply(settings, ledger, immich, dry_run=False, album="")
    assert writeback.plan(ledger) == []

    calls_before = immich_api["update_asset"].call_count
    again = writeback.apply(settings, ledger, immich, dry_run=False, album="")
    assert again.selected == 0
    assert again.applied == 0
    assert immich_api["update_asset"].call_count == calls_before
    assert ledger.summary()["unapplied"] == ledger.summary()["decisions"] - 7


def test_apply_honours_the_bucket_filter(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    stats = writeback.apply(settings, ledger, immich, {"reject"}, False, album="")
    assert stats.planned == 8
    assert stats.selected == 1
    assert stats.applied == 1
    assert immich_api["create_stack"].call_count == 0
    assert immich_api["update_asset"].call_count == 1
    assert immich_api["tag_assets"].call_count == 1

    remaining = {action.bucket for action in writeback.plan(ledger)}
    assert "reject" not in remaining


def test_apply_stacks_group_covers_burst_tags_and_the_stack(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    stats = writeback.apply(settings, ledger, immich, {"stacks"}, False)
    assert stats.selected == 3
    assert stats.stacks_created == 1
    assert stats.assets_tagged == 2
    assert immich_api["update_asset"].call_count == 0


def test_apply_records_immich_failures(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    immich_api["update_asset"].mock(return_value=httpx.Response(500, text="nope"))
    stats = writeback.apply(settings, ledger, immich, dry_run=False, album="")
    assert stats.failed == 5
    assert stats.applied == 3  # two burst tags plus the stack
    still_pending = {action.asset_id for action in writeback.plan(ledger)}
    assert aid(4) in still_pending


def test_apply_progress_callback(
    finaled: FinalsStats, settings: Settings, ledger: Ledger, immich: ImmichClient
) -> None:
    lines: list[str] = []
    writeback.apply(settings, ledger, immich, dry_run=True, on_progress=lines.append)
    assert lines and "nothing written" in lines[-1]


# ----------------------------------------------------------------------
# write-back — stacks have their own applied-state
# ----------------------------------------------------------------------


def test_apply_does_not_restack_when_tagging_fails(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    """The stack succeeded; only the loser tags failed. Retrying must not
    create the stack a second time."""
    immich_api["tag_assets"].mock(return_value=httpx.Response(500, text="nope"))
    writeback.apply(settings, ledger, immich, {"stacks"}, False)
    assert immich_api["create_stack"].call_count == 1

    # the losers are still pending (their tag failed) so the burst is replanned
    assert {a.bucket for a in writeback.plan(ledger)} >= {"burst_loser"}
    immich_api["tag_assets"].mock(return_value=httpx.Response(200, json={}))
    again = writeback.apply(settings, ledger, immich, {"stacks"}, False)
    assert immich_api["create_stack"].call_count == 1, "stack must not be recreated"
    assert again.assets_tagged == 2


def test_apply_retries_a_stack_that_failed(
    finaled: FinalsStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    """The tags succeeded; only the stack failed. It must not be forgotten."""
    immich_api["create_stack"].mock(return_value=httpx.Response(500, text="nope"))
    stats = writeback.apply(settings, ledger, immich, {"stacks"}, False)
    assert stats.stacks_created == 0
    assert stats.failed == 1

    remaining = [a.bucket for a in writeback.plan(ledger) if a.group == "stacks"]
    assert remaining == ["burst_stack"], "the tags applied; only the stack is still owed"

    immich_api["create_stack"].mock(return_value=httpx.Response(201, json={"id": "stack-1"}))
    retry = writeback.apply(settings, ledger, immich, {"stacks"}, False)
    assert retry.stacks_created == 1
    assert [a for a in writeback.plan(ledger) if a.group == "stacks"] == []


def test_plan_skips_bursts_whose_grouping_is_gone(
    finaled: FinalsStats, ledger: Ledger
) -> None:
    """A re-scan mints a new content-addressed burst id; the orphaned verdict
    row must not plan a second, overlapping stack."""
    ledger.record_burst("b-orphan", aid(1), [aid(0)], "stale", "claude-haiku-4-5")
    stacks = [a for a in writeback.plan(ledger) if a.bucket == "burst_stack"]
    assert len(stacks) == 1
    assert stacks[0].burst_id != "b-orphan"


def test_plan_drops_members_that_left_the_group(finaled: FinalsStats, ledger: Ledger) -> None:
    burst_id = next(iter(ledger.burst_groups()))
    ledger.assign_burst(burst_id, [aid(1), aid(0)])  # aid(2) reassigned elsewhere
    ledger.upsert_assets([{"id": aid(2), "burst_id": None}])
    action = next(a for a in writeback.plan(ledger) if a.bucket == "burst_stack")
    assert action.api_ops == [{"op": "stack", "asset_ids": [aid(1), aid(0)]}]


# ----------------------------------------------------------------------
# re-scanning: regrouped bursts must never strand an asset
# ----------------------------------------------------------------------


def test_rescan_clears_stale_burst_ids(
    dup_groups: list[dict[str, Any]],
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    dup_groups.append({"duplicateId": "dup-1", "assets": [{"id": aid(4)}, {"id": aid(5)}]})
    run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")
    assert len(ledger.burst_groups()) == 2

    # Immich no longer reports the duplicate: 4 and 5 are ordinary singles now.
    dup_groups.clear()
    run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")
    groups = ledger.burst_groups()
    assert sorted(sorted(m) for m in groups.values()) == [[aid(0), aid(1), aid(2)]]

    # and crucially they are visible to triage again
    assert aid(4) in pending_single_ids(ledger)
    assert aid(5) in pending_single_ids(ledger)


def test_pending_singles_rescue_a_group_that_shrank(ledger: Ledger) -> None:
    ledger.upsert_assets([{"id": "solo", "taken_at": "2024-06-01T10:00:00"}])
    ledger.assign_burst("b-lonely", ["solo"])
    # a one-frame "burst" is judged by nothing: not the burst queue (needs 2),
    # and not unjudged_asset_ids (which skips grouped assets)
    assert pending_burst_ids(ledger) == []
    assert pending_single_ids(ledger) == ["solo"]


def test_promoted_burst_member_loses_its_loser_bucket(
    triaged: TriageStats, ledger: Ledger
) -> None:
    burst_id = next(iter(ledger.burst_groups()))
    members = ledger.burst_groups()[burst_id]
    assert buckets_of(ledger)[aid(0)] == "burst_loser"

    # re-judge the same burst, this time keeping frame 0
    apply_burst_verdict(
        ledger,
        burst_id,
        members,
        BurstVerdict(best_index=1, reject_indices=[3], note="changed my mind"),
        "claude-haiku-4-5",
    )
    assert aid(0) not in buckets_of(ledger), "a promoted frame must not stay an also-ran"
    assert buckets_of(ledger)[aid(2)] == "burst_loser"


# ----------------------------------------------------------------------
# triage — batching and confidence gates
# ----------------------------------------------------------------------


def test_triage_batch_does_not_requeue_inflight_work(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    first = submit_triage_batch(settings, ledger, judge)
    assert first is not None

    second = submit_triage_batch(settings, ledger, judge)
    assert second is None, "everything outstanding is already in flight"
    assert len(judge.client.submitted) == 1
    # the first batch keeps its items
    assert len(ledger.batch_items_for(first)) == 7


def test_triage_batch_splits_oversized_submissions(
    scanned: ScanStats,
    settings: Settings,
    ledger: Ledger,
    judge: StubJudge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("winnow.pipeline.triage.MAX_BATCH_REQUESTS", 3)
    first = submit_triage_batch(settings, ledger, judge)
    assert first == "msgbatch_1"
    assert len(judge.client.submitted) == 3  # 7 requests over batches of 3

    queued = {
        item["custom_id"]
        for batch_id in judge.client.submitted
        for item in ledger.batch_items_for(batch_id)
    }
    assert len(queued) == 7
    assert len(ledger.open_batches()) == 3

    stats = ingest_triage_batch(settings, ledger, judge)
    assert stats.singles_judged == 6
    assert stats.bursts_judged == 1
    assert ledger.open_batches() == []


def _api_error(cls: type[Exception]) -> Exception:
    """Build a real anthropic SDK exception without touching the network."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request, json={"error": {"message": "bad image"}})
    return cls("bad image", response=response, body=None)


def test_a_bad_request_skips_one_photo_instead_of_the_run(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    """anthropic.APIError is not an OSError; without it in the handler a
    single corrupt image aborts the whole stage."""
    real_triage = judge.triage
    failed: list[str] = []

    def flaky(model: str, image_b64: str) -> JudgeResult:
        if index_of(image_b64) == 6 and not failed:
            failed.append("once")
            raise _api_error(anthropic.BadRequestError)
        return real_triage(model, image_b64)

    judge.triage = flaky  # type: ignore[method-assign]
    stats = run_triage_direct(settings, ledger, judge)
    assert stats.errors == 1
    assert stats.singles_judged == 6
    assert aid(6) in pending_single_ids(ledger)


def test_an_auth_error_aborts_the_stage(
    scanned: ScanStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    """A bad key cannot be retried per photo — fail loudly and immediately."""

    def unauthorized(model: str, image_b64: str) -> JudgeResult:
        raise _api_error(anthropic.AuthenticationError)

    judge.triage = unauthorized  # type: ignore[method-assign]
    with pytest.raises(anthropic.AuthenticationError):
        run_triage_direct(settings, ledger, judge)


@pytest.mark.parametrize("confidence", ["low", "medium"])
def test_unsure_nonphoto_stays_in_the_middle(ledger: Ledger, confidence: str) -> None:
    verdict = TriageVerdict(
        category="meme",
        verdict="neutral",
        technical_score=5,
        reasons=["might be a wallpaper"],
        confidence=confidence,
    )
    bucket = apply_triage_verdict(ledger, "asset-x", verdict, "claude-haiku-4-5")
    assert bucket == "middle", "archiving needs the model to be sure"


def test_confident_nonphoto_is_still_archived(ledger: Ledger) -> None:
    verdict = TriageVerdict(
        category="meme",
        verdict="neutral",
        technical_score=5,
        reasons=["image macro"],
        confidence="high",
    )
    assert apply_triage_verdict(ledger, "asset-y", verdict, "claude-haiku-4-5") == "nonphoto"


# ----------------------------------------------------------------------
# finals — re-runs must retract what they no longer award
# ----------------------------------------------------------------------


def test_finals_rerun_retracts_stale_stars(
    ranked: RankStats, settings: Settings, ledger: Ledger, judge: StubJudge
) -> None:
    run_finals(settings, ledger, judge, five_count=4, four_frac=0.0)
    assert len([r for r in ledger.score_rows() if r["stars"]]) == 4

    # a stricter cut must demote the rest, not accumulate awards — but only
    # when demotions are explicitly allowed (five stars are sticky by default)
    run_finals(settings, ledger, judge, five_count=1, four_frac=0.0, allow_demotions=True)
    starred = {row["asset_id"]: row["stars"] for row in ledger.score_rows() if row["stars"]}
    assert len(starred) == 1
    buckets = Counter(buckets_of(ledger).values())
    assert buckets["five_star"] == 1
    assert buckets["four_star"] == 0
    assert {a.bucket for a in writeback.plan(ledger)} & {"four_star"} == set()


# ----------------------------------------------------------------------
# scan — cache integrity
# ----------------------------------------------------------------------


def test_scan_refetches_a_truncated_thumbnail(
    scanned: ScanStats,
    immich_api: respx.MockRouter,
    settings: Settings,
    ledger: Ledger,
    immich: ImmichClient,
) -> None:
    path = thumb_path(settings.cache_dir, aid(6))
    path.write_bytes(path.read_bytes()[:200])  # interrupted mid-write
    again = run_scan(settings, ledger, immich, "2024-06-01", "2024-06-02")
    assert again.thumbs_fetched == 1
    assert again.errors == 0
    Image.open(io.BytesIO(path.read_bytes())).verify()


def test_scan_leaves_no_temp_files_behind(scanned: ScanStats, settings: Settings) -> None:
    assert list(settings.cache_dir.glob("*.tmp")) == []


# ----------------------------------------------------------------------
# whole funnel
# ----------------------------------------------------------------------


def test_end_to_end_ledger_summary(finaled: FinalsStats, ledger: Ledger) -> None:
    summary = ledger.summary()
    assert summary["assets"] == ASSET_COUNT
    assert summary["triage"] == 7
    assert summary["bursts"] == 1
    assert summary["bws_sets"] == 4
    assert summary["untriaged"] == 2  # the two burst also-rans, judged as a group
    assert summary["burst_groups"] == 1
    assert summary["buckets"]["reject"] == 1
    assert summary["buckets"]["nonphoto"] == 1
    assert summary["buckets"]["burst_loser"] == 2
    assert summary["buckets"]["five_star"] == 2
    assert summary["buckets"]["four_star"] == 1
    assert summary["applied"] == 0
