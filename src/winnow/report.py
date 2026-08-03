"""Self-contained HTML contact sheet for a Winnow run.

The report is the human checkpoint between judging and write-back: before
anything touches Immich you can look at every photo Winnow proposes to hide,
archive or crown, with the model's own reasons next to it.

The output is a single file with no external assets whatsoever — the CSS is
inlined and every thumbnail is re-encoded small and embedded as a ``data:``
URI — so it can be opened from anywhere, mailed to yourself, or kept as an
audit trail of a run.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from winnow.images import prepare_image, to_b64
from winnow.ledger import Ledger

__all__ = [
    "THUMB_MAX_EDGE",
    "THUMB_QUALITY",
    "render_report",
    "thumb_data_uri",
    "write_html_report",
]

#: Long edge, in pixels, of the thumbnails embedded in the report.
THUMB_MAX_EDGE = 300

#: JPEG quality used for those thumbnails (small file beats pixel-perfect).
THUMB_QUALITY = 60

#: Buckets rendered as their own gallery, in the order they appear.
_STAR_BUCKETS = ("five_star", "four_star")

_STYLE = """
:root {
  --bg: #14161a;
  --panel: #1c1f26;
  --edge: #2b3039;
  --text: #e7e9ee;
  --muted: #99a0ad;
  --gold: #f0b429;
  --bad: #e5484d;
  --cool: #4c8dff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2rem 1.5rem 4rem;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
header { max-width: 1200px; margin: 0 auto 2rem; }
h1 { margin: 0 0 .25rem; font-size: 1.75rem; letter-spacing: -.02em; }
h2 {
  margin: 0 0 1rem;
  font-size: 1.15rem;
  letter-spacing: -.01em;
  border-bottom: 1px solid var(--edge);
  padding-bottom: .5rem;
}
.sub { color: var(--muted); font-size: .9rem; }
section { max-width: 1200px; margin: 0 auto 2.5rem; }
table.summary { border-collapse: collapse; width: 100%; max-width: 560px; }
table.summary td { padding: .3rem .75rem .3rem 0; border-bottom: 1px solid var(--edge); }
table.summary td.n { text-align: right; font-variant-numeric: tabular-nums; color: var(--gold); }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 1rem;
}
figure.card {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--edge);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
figure.card img { width: 100%; height: 170px; object-fit: cover; display: block; background: #000; }
figure.card .noimg {
  height: 170px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  font-size: .8rem;
  background: #0e1013;
}
figcaption { padding: .55rem .65rem .7rem; font-size: .8rem; }
.name { font-weight: 600; word-break: break-all; }
.meta { color: var(--muted); font-variant-numeric: tabular-nums; }
.reasons { color: var(--muted); margin-top: .3rem; }
.badge {
  display: inline-block;
  padding: 0 .4rem;
  border-radius: 999px;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .02em;
}
.card.winner { border-color: var(--gold); }
.card.winner .badge { background: var(--gold); color: #14161a; }
.card.loser { opacity: .72; }
.stars { color: var(--gold); }
.reject .name { color: var(--bad); }
.burst { margin-bottom: 1.75rem; }
.burst h3 { margin: 0 0 .5rem; font-size: .95rem; color: var(--muted); font-weight: 600; }
.empty { color: var(--muted); font-style: italic; }
"""


def _text(value: Any) -> str:
    """HTML-escape any value (``None`` becomes an empty string)."""
    return "" if value is None else html.escape(str(value))


def _thumb_file(cache_dir: str | Path, asset: Mapping[str, Any]) -> Path | None:
    """Locate an asset's cached thumbnail, preferring the canonical cache name."""
    asset_id = str(asset.get("id") or asset.get("asset_id") or "")
    candidate = Path(cache_dir) / f"{asset_id}.jpg"
    if candidate.exists():
        return candidate
    stored = asset.get("thumb_path")
    if stored:
        path = Path(str(stored))
        if path.exists():
            return path
    return None


def thumb_data_uri(cache_dir: str | Path, asset: Mapping[str, Any]) -> str | None:
    """Return a small embedded ``data:`` URI for an asset's thumbnail.

    The cached JPEG is re-encoded down to :data:`THUMB_MAX_EDGE` pixels at
    :data:`THUMB_QUALITY` so a few thousand photos still make a report that
    opens instantly.

    Args:
        cache_dir: Directory holding ``<asset_id>.jpg`` thumbnails.
        asset: Asset row (only ``id``/``thumb_path`` are read).

    Returns:
        The data URI, or ``None`` when no readable thumbnail exists.
    """
    path = _thumb_file(cache_dir, asset)
    if path is None:
        return None
    try:
        small = prepare_image(path.read_bytes(), max_edge=THUMB_MAX_EDGE, quality=THUMB_QUALITY)
    except (OSError, ValueError):
        return None
    return f"data:image/jpeg;base64,{to_b64(small)}"


def _stars(count: Any) -> str:
    """Render a star count as filled stars, or an empty string."""
    try:
        number = int(count)
    except (TypeError, ValueError):
        return ""
    return "★" * max(0, min(5, number))


def _score_line(score_row: Mapping[str, Any] | None) -> str:
    """One-line rank/score/star summary for a card caption."""
    if not score_row:
        return ""
    bits: list[str] = []
    stars = _stars(score_row.get("stars"))
    if stars:
        bits.append(f'<span class="stars">{stars}</span>')
    rank = score_row.get("rank")
    if rank is not None:
        bits.append(f"rank {_text(rank)}")
    score = score_row.get("bt_score")
    if score is not None:
        bits.append(f"score {float(score):.3f}")
    return " &middot; ".join(bits)


def _reasons(triage_row: Mapping[str, Any] | None) -> str:
    """The model's reasons for a verdict, as one escaped sentence."""
    if not triage_row:
        return ""
    reasons = triage_row.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    return _text("; ".join(str(r) for r in reasons))


def _verdict_meta(triage_row: Mapping[str, Any] | None) -> str:
    """``verdict · score 3/10 · high confidence`` for a card caption."""
    if not triage_row:
        return ""
    bits = [str(triage_row.get("verdict") or "")]
    score = triage_row.get("technical_score")
    if score is not None:
        bits.append(f"score {score}/10")
    confidence = triage_row.get("confidence")
    if confidence:
        bits.append(f"{confidence} confidence")
    return _text(" · ".join(bit for bit in bits if bit))


def _card(
    asset: Mapping[str, Any],
    uri: str | None,
    *,
    classes: Sequence[str] = (),
    badge: str = "",
    meta: str = "",
    detail: str = "",
) -> str:
    """Render one photo card: thumbnail plus caption lines."""
    asset_id = str(asset.get("id") or "")
    name = str(asset.get("filename") or asset_id or "unknown")
    css = " ".join(("card", *classes))
    if uri:
        picture = f'<img src="{uri}" alt="{_text(name)}" loading="lazy">'
    else:
        picture = '<div class="noimg">no cached thumbnail</div>'
    parts = [f'<figure class="{css}">', picture, "<figcaption>"]
    badge_html = f' <span class="badge">{_text(badge)}</span>' if badge else ""
    parts.append(f'<div class="name">{_text(name)}{badge_html}</div>')
    if meta:
        parts.append(f'<div class="meta">{meta}</div>')
    if detail:
        parts.append(f'<div class="reasons">{detail}</div>')
    parts.append("</figcaption></figure>")
    return "".join(parts)


def _gallery(cards: Sequence[str], empty: str) -> str:
    """Wrap cards in a responsive grid, or show an empty-state line."""
    if not cards:
        return f'<p class="empty">{_text(empty)}</p>'
    return f'<div class="grid">{"".join(cards)}</div>'


def _summary_table(summary: Mapping[str, Any]) -> str:
    """The headline counters as a compact two-column table."""
    buckets = summary.get("buckets") or {}
    rows: list[tuple[str, Any]] = [
        ("Assets scanned", summary.get("assets", 0)),
        ("Triaged", summary.get("triage", 0)),
        ("Awaiting triage", summary.get("untriaged", 0)),
        ("Burst groups", summary.get("burst_groups", 0)),
        ("Rejects", buckets.get("reject", 0)),
        ("Non-photos", buckets.get("nonphoto", 0)),
        ("Burst also-rans", buckets.get("burst_loser", 0)),
        ("Left in the middle", buckets.get("middle", 0)),
        ("Five star", buckets.get("five_star", 0)),
        ("Four star", buckets.get("four_star", 0)),
        ("Best-worst sets", summary.get("bws_sets", 0)),
        ("Head-to-heads", summary.get("pairs", 0)),
        ("Decisions written back", summary.get("applied", 0)),
        ("Decisions pending", summary.get("unapplied", 0)),
    ]
    cells = "".join(
        f"<tr><td>{_text(label)}</td><td class='n'>{_text(value)}</td></tr>"
        for label, value in rows
    )
    return f'<table class="summary">{cells}</table>'


def _sorted_by_rank(
    asset_ids: Iterable[str],
    scores: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Order asset ids by fitted rank (unranked last, then by id)."""

    def key(asset_id: str) -> tuple[int, float, str]:
        row = scores.get(asset_id) or {}
        rank = row.get("rank")
        if rank is None:
            return (1, 0.0, asset_id)
        return (0, float(rank), asset_id)

    return sorted(asset_ids, key=key)


def render_report(ledger: Ledger, cache_dir: str | Path) -> str:
    """Build the whole contact sheet as one HTML string.

    Args:
        ledger: Open ledger holding the run's decisions.
        cache_dir: Directory of cached ``<asset_id>.jpg`` thumbnails.

    Returns:
        A complete, self-contained HTML document.
    """
    assets = {str(row["id"]): row for row in ledger.get_assets()}
    triage = {str(row["asset_id"]): row for row in ledger.triage_rows()}
    scores = {str(row["asset_id"]): row for row in ledger.score_rows()}
    decisions = {str(row["asset_id"]): row for row in ledger.decisions()}
    groups = ledger.burst_groups()
    bursts = {str(row["burst_id"]): row for row in ledger.burst_rows()}
    summary = ledger.summary()

    def asset_of(asset_id: str) -> dict[str, Any]:
        return dict(assets.get(asset_id) or {"id": asset_id})

    def in_bucket(bucket: str) -> list[str]:
        return [aid for aid, row in decisions.items() if row.get("bucket") == bucket]

    # --- keepers -------------------------------------------------------
    star_cards: list[str] = []
    for bucket in _STAR_BUCKETS:
        for asset_id in _sorted_by_rank(in_bucket(bucket), scores):
            asset = asset_of(asset_id)
            star_cards.append(
                _card(
                    asset,
                    thumb_data_uri(cache_dir, asset),
                    classes=("winner",) if bucket == "five_star" else (),
                    badge="BEST" if bucket == "five_star" else "",
                    meta=_score_line(scores.get(asset_id)),
                    detail=_reasons(triage.get(asset_id)),
                )
            )

    # --- rejects -------------------------------------------------------
    reject_ids = sorted(
        in_bucket("reject"),
        key=lambda aid: (int((triage.get(aid) or {}).get("technical_score") or 0), aid),
    )
    reject_cards = [
        _card(
            asset_of(aid),
            thumb_data_uri(cache_dir, asset_of(aid)),
            classes=("reject",),
            meta=_verdict_meta(triage.get(aid)),
            detail=_reasons(triage.get(aid)),
        )
        for aid in reject_ids
    ]

    # --- non-photos ----------------------------------------------------
    nonphoto_ids = sorted(
        in_bucket("nonphoto"),
        key=lambda aid: (str((triage.get(aid) or {}).get("category") or ""), aid),
    )
    nonphoto_cards = [
        _card(
            asset_of(aid),
            thumb_data_uri(cache_dir, asset_of(aid)),
            meta=_text((triage.get(aid) or {}).get("category") or "non-photo"),
            detail=_reasons(triage.get(aid)),
        )
        for aid in nonphoto_ids
    ]

    # --- bursts --------------------------------------------------------
    burst_blocks: list[str] = []
    for burst_id in sorted(groups):
        members = groups[burst_id]
        verdict = bursts.get(burst_id) or {}
        winner = str(verdict.get("winner_id") or "")
        losers = {str(i) for i in (verdict.get("reject_ids") or [])}
        cards = []
        for member in members:
            asset = asset_of(member)
            is_winner = member == winner
            cards.append(
                _card(
                    asset,
                    thumb_data_uri(cache_dir, asset),
                    classes=("winner",) if is_winner else (("loser",) if member in losers else ()),
                    badge="WINNER" if is_winner else "",
                    meta=_verdict_meta(triage.get(member)),
                    detail=_reasons(triage.get(member)),
                )
            )
        note = _text(verdict.get("note") or "")
        heading = f"{_text(burst_id)} &middot; {len(members)} frames"
        if note:
            heading += f" &middot; {note}"
        burst_blocks.append(
            f'<div class="burst"><h3>{heading}</h3>{_gallery(cards, "no frames")}</div>'
        )

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""<header>
<h1>&#127806; Winnow report</h1>
<div class="sub">{summary.get("assets", 0)} assets &middot; generated {generated}</div>
</header>
<section id="summary"><h2>Summary</h2>{_summary_table(summary)}</section>
<section id="keepers"><h2>Keepers &mdash; four and five star ({len(star_cards)})</h2>
{_gallery(star_cards, "No finalists yet — run the finals stage.")}</section>
<section id="rejects"><h2>Rejects ({len(reject_cards)})</h2>
{_gallery(reject_cards, "No rejects — nothing would be hidden.")}</section>
<section id="nonphotos"><h2>Non-photos ({len(nonphoto_cards)})</h2>
{_gallery(nonphoto_cards, "No screenshots or documents found.")}</section>
<section id="bursts"><h2>Burst groups ({len(burst_blocks)})</h2>
{"".join(burst_blocks) or '<p class="empty">No bursts detected.</p>'}</section>"""

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Winnow report</title>"
        f"<style>{_STYLE}</style></head><body>\n{body}\n</body></html>\n"
    )


def write_html_report(
    ledger: Ledger,
    cache_dir: str | Path,
    out_path: str | Path = "winnow-report.html",
) -> Path:
    """Write the contact-sheet report to ``out_path``.

    Args:
        ledger: Open ledger holding the run's decisions.
        cache_dir: Directory of cached ``<asset_id>.jpg`` thumbnails.
        out_path: Destination file; parent directories are created.

    Returns:
        The path written.
    """
    path = Path(out_path)
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(ledger, cache_dir), encoding="utf-8")
    return path
