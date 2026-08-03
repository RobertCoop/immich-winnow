"""Tests for the self-contained HTML report. No network, no secrets.

The fixture builds a miniature run in a temporary ledger — one burst, one
reject, one screenshot and two starred keepers — with real (tiny) JPEGs in a
cache directory, so the renderer is exercised end to end against the same
ledger API the pipeline writes.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from conftest import make_jpeg
from winnow.ledger import Ledger
from winnow.report import (
    THUMB_MAX_EDGE,
    render_report,
    thumb_data_uri,
    write_html_report,
)
from winnow.schemas import TriageVerdict

#: Assets in the fixture library: three burst frames plus four singles.
BURST_IDS = ("b1", "b2", "b3")
BURST_ID = "bfeedfacefeed"

#: A filename designed to break naive HTML generation.
NASTY_NAME = "<script>alert('xss')</script>.jpg"

#: The one asset deliberately left without a cached thumbnail.
NO_THUMB_ID = "s4"

DATA_URI_PREFIX = "data:image/jpeg;base64,"


def _verdict(**overrides: object) -> TriageVerdict:
    """A triage verdict with sensible defaults."""
    data = {
        "category": "photo",
        "verdict": "neutral",
        "technical_score": 6,
        "reasons": ["nothing remarkable"],
        "confidence": "medium",
    }
    data.update(overrides)
    return TriageVerdict(**data)  # type: ignore[arg-type]


@pytest.fixture()
def seeded(tmp_path: Path) -> Iterator[SimpleNamespace]:
    """A ledger and thumbnail cache describing a small finished run."""
    cache = tmp_path / "cache"
    cache.mkdir()
    ledger = Ledger(tmp_path / "winnow.db")

    rows = [
        {"id": "b1", "filename": "BURST_1.jpg", "taken_at": "2024-06-01T10:00:00+00:00"},
        {"id": "b2", "filename": "BURST_2.jpg", "taken_at": "2024-06-01T10:00:02+00:00"},
        {"id": "b3", "filename": "BURST_3.jpg", "taken_at": "2024-06-01T10:00:04+00:00"},
        {"id": "s1", "filename": "BLURRY.jpg", "taken_at": "2024-06-01T12:00:00+00:00"},
        {"id": "s2", "filename": "Screenshot.png", "taken_at": "2024-06-01T13:00:00+00:00"},
        {"id": "s3", "filename": NASTY_NAME, "taken_at": "2024-06-01T14:00:00+00:00"},
        {"id": NO_THUMB_ID, "filename": "GONE.jpg", "taken_at": "2024-06-01T15:00:00+00:00"},
    ]
    ledger.upsert_assets(rows)
    for index, row in enumerate(rows):
        if row["id"] == NO_THUMB_ID:
            continue
        (cache / f"{row['id']}.jpg").write_bytes(
            make_jpeg(width=640, height=480, color=(30 * index, 60, 90))
        )

    ledger.assign_burst(BURST_ID, list(BURST_IDS))
    ledger.record_burst(
        BURST_ID, "b2", ["b1", "b3"], "second frame is sharpest", "claude-haiku-4-5"
    )
    for loser in ("b1", "b3"):
        ledger.set_decision(loser, "burst_loser", {"burst_id": BURST_ID, "winner_id": "b2"})

    ledger.record_triage("b2", _verdict(verdict="candidate", technical_score=8), "m", None)
    ledger.set_decision("b2", "middle", {"category": "photo"})

    ledger.record_triage(
        "s1",
        _verdict(verdict="reject", technical_score=1, confidence="high", reasons=["motion blur"]),
        "claude-haiku-4-5",
        None,
    )
    ledger.set_decision("s1", "reject", {"category": "photo", "reasons": ["motion blur"]})

    ledger.record_triage(
        "s2",
        _verdict(category="screenshot", technical_score=4, reasons=["app interface"]),
        "claude-haiku-4-5",
        None,
    )
    ledger.set_decision("s2", "nonphoto", {"category": "screenshot"})

    ledger.record_triage(
        "s3",
        _verdict(verdict="candidate", technical_score=10, reasons=["golden light"]),
        "claude-haiku-4-5",
        None,
    )
    ledger.record_triage("s4", _verdict(verdict="candidate", technical_score=9), "m", None)
    ledger.upsert_scores({"s3": (2.5, 1), NO_THUMB_ID: (1.4, 2), "b2": (0.6, 3)})
    ledger.set_stars("s3", 5)
    ledger.set_stars(NO_THUMB_ID, 4)
    ledger.set_decision("s3", "five_star", {"stars": 5, "rank": 1})
    ledger.set_decision(NO_THUMB_ID, "four_star", {"stars": 4, "rank": 2})

    try:
        yield SimpleNamespace(ledger=ledger, cache=cache, tmp=tmp_path)
    finally:
        ledger.close()


@pytest.fixture()
def html(seeded: SimpleNamespace) -> str:
    """The rendered report for the seeded fixture."""
    return render_report(seeded.ledger, seeded.cache)


# ----------------------------------------------------------------------
# document shape
# ----------------------------------------------------------------------


def test_document_is_well_formed(html: str) -> None:
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert '<meta charset="utf-8">' in html
    assert "<title>Winnow report</title>" in html
    assert html.count("<body>") == 1 and html.count("</body>") == 1


def test_every_section_is_present(html: str) -> None:
    for marker in ("summary", "keepers", "rejects", "nonphotos", "bursts"):
        assert f'id="{marker}"' in html, marker


def test_styles_are_inlined(html: str) -> None:
    assert "<style>" in html
    assert "<link" not in html


def test_no_external_assets(html: str) -> None:
    assert "http://" not in html
    assert "https://" not in html
    # Every image is embedded; nothing is fetched from anywhere.
    assert not re.findall(r'<img[^>]+src="(?!data:)', html)


def test_no_script_tags_survive(html: str) -> None:
    assert "<script" not in html.lower()


# ----------------------------------------------------------------------
# content
# ----------------------------------------------------------------------


def test_summary_counts_the_library(html: str) -> None:
    assert "Assets scanned" in html
    assert re.search(r"Assets scanned</td><td class='n'>7</td>", html)
    assert re.search(r"Burst groups</td><td class='n'>1</td>", html)
    assert re.search(r"Rejects</td><td class='n'>1</td>", html)


def test_keepers_show_stars_and_scores(html: str) -> None:
    keepers = _section(html, "keepers")
    assert "★★★★★" in keepers
    assert "★★★★" in keepers
    assert "score 2.500" in keepers
    assert "rank 1" in keepers
    assert "BEST" in keepers


def test_rejects_show_reasons_and_verdict(html: str) -> None:
    rejects = _section(html, "rejects")
    assert "BLURRY.jpg" in rejects
    assert "motion blur" in rejects
    assert "score 1/10" in rejects
    assert "high confidence" in rejects


def test_nonphotos_show_their_category(html: str) -> None:
    nonphotos = _section(html, "nonphotos")
    assert "Screenshot.png" in nonphotos
    assert "screenshot" in nonphotos
    assert "app interface" in nonphotos


def test_burst_group_marks_the_winner(html: str) -> None:
    bursts = _section(html, "bursts")
    assert BURST_ID in bursts
    assert "second frame is sharpest" in bursts
    assert "3 frames" in bursts
    for name in ("BURST_1.jpg", "BURST_2.jpg", "BURST_3.jpg"):
        assert name in bursts
    winner_card = re.search(r'<figure class="card winner">.*?</figure>', bursts, re.S)
    assert winner_card is not None
    assert "BURST_2.jpg" in winner_card.group(0)
    assert "WINNER" in winner_card.group(0)
    assert bursts.count('class="card loser"') == 2


def test_filenames_are_escaped(html: str) -> None:
    assert NASTY_NAME not in html
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;.jpg" in html


def test_missing_thumbnail_falls_back_to_a_placeholder(html: str) -> None:
    assert "no cached thumbnail" in html
    assert html.count("no cached thumbnail") == 1


# ----------------------------------------------------------------------
# thumbnails
# ----------------------------------------------------------------------


def test_thumbnails_are_embedded_for_every_cached_asset(html: str) -> None:
    # Seven cards: 2 keepers + 1 reject + 1 non-photo + 3 burst frames.
    assert html.count("<figure class=") == 7
    assert html.count(DATA_URI_PREFIX) == 6


def test_embedded_thumbnails_are_downscaled(html: str) -> None:
    uris = re.findall(rf'src="{re.escape(DATA_URI_PREFIX)}([^"]+)"', html)
    assert uris
    for payload in uris:
        image = Image.open(io.BytesIO(base64.b64decode(payload)))
        assert max(image.size) <= THUMB_MAX_EDGE
        assert image.format == "JPEG"


def test_thumb_data_uri_prefers_the_cache(seeded: SimpleNamespace) -> None:
    uri = thumb_data_uri(seeded.cache, {"id": "b1"})
    assert uri is not None and uri.startswith(DATA_URI_PREFIX)


def test_thumb_data_uri_falls_back_to_the_stored_path(tmp_path: Path) -> None:
    stored = tmp_path / "elsewhere.jpg"
    stored.write_bytes(make_jpeg())
    uri = thumb_data_uri(tmp_path / "empty-cache", {"id": "zz", "thumb_path": str(stored)})
    assert uri is not None and uri.startswith(DATA_URI_PREFIX)


def test_thumb_data_uri_returns_none_when_missing(tmp_path: Path) -> None:
    assert thumb_data_uri(tmp_path, {"id": "nope"}) is None


def test_thumb_data_uri_returns_none_for_unreadable_bytes(tmp_path: Path) -> None:
    (tmp_path / "broken.jpg").write_bytes(b"not an image at all")
    assert thumb_data_uri(tmp_path, {"id": "broken"}) is None


# ----------------------------------------------------------------------
# writing the file
# ----------------------------------------------------------------------


def test_write_html_report_writes_the_document(seeded: SimpleNamespace) -> None:
    out = seeded.tmp / "report.html"
    path = write_html_report(seeded.ledger, seeded.cache, out)
    assert path == out
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert 'id="summary"' in text


def test_write_html_report_creates_parent_directories(seeded: SimpleNamespace) -> None:
    out = seeded.tmp / "deep" / "nested" / "report.html"
    write_html_report(seeded.ledger, seeded.cache, out)
    assert out.exists()


def test_write_html_report_accepts_str_paths(seeded: SimpleNamespace) -> None:
    out = seeded.tmp / "str-report.html"
    path = write_html_report(seeded.ledger, str(seeded.cache), str(out))
    assert path.exists()


# ----------------------------------------------------------------------
# degenerate ledgers
# ----------------------------------------------------------------------


def test_empty_ledger_renders_empty_states(tmp_path: Path) -> None:
    with Ledger(tmp_path / "empty.db") as ledger:
        html_text = render_report(ledger, tmp_path / "cache")
    assert 'id="summary"' in html_text
    assert "No rejects" in html_text
    assert "No bursts detected." in html_text
    assert "No finalists yet" in html_text
    assert DATA_URI_PREFIX not in html_text


def test_decisions_without_asset_rows_still_render(tmp_path: Path) -> None:
    """A decision for an asset the ledger never scanned must not crash."""
    with Ledger(tmp_path / "orphan.db") as ledger:
        ledger.set_decision("ghost", "reject", {"category": "photo"})
        html_text = render_report(ledger, tmp_path / "cache")
    assert "ghost" in html_text
    assert "no cached thumbnail" in html_text


def _section(html_text: str, marker: str) -> str:
    """Extract one ``<section id="...">`` block for focused assertions."""
    match = re.search(rf'<section id="{marker}">(.*?)</section>', html_text, re.S)
    assert match is not None, f"missing section {marker}"
    return match.group(1)
