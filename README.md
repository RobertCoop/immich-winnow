# 🌾 Winnow

**AI photo culling for [Immich](https://immich.app) — hide the rejects, crown the keepers.**

Winnow runs your photo library through a three-stage judging funnel powered by
Claude vision models, then writes the results back to Immich as **reversible**
ratings, tags, favorites, and stacks. Nothing is ever deleted.

> *winnow (v.): to blow the chaff from the grain; to separate the valuable from
> the worthless.*

## How it works

```
your library ──▶ 1. TRIAGE (Haiku)   every photo: category, quality, verdict
                     │                bursts get a single "pick the best" contest
                     ▼
                 2. RANK (Sonnet)    candidates face off in best-worst sets of 8;
                     │                a Bradley-Terry model turns votes into scores
                     ▼
                 3. FINALS (Opus)    top pool plays Swiss-paired head-to-heads,
                     │                each pair judged twice with order swapped
                     ▼
                 WRITE-BACK          rejects → rating −1 + tag · winners → ★★★★★
                                     + favorite · bursts → stacks · non-photos →
                                     archive. Nothing deleted, everything undoable.
```

Every judgment is stored in a local SQLite ledger, so runs are resumable,
auditable, and cheap to re-do. An HTML contact-sheet report lets you review
every proposed reject before anything touches your server.

### Why the funnel shape?

Culling is a *tail-selection* problem, not a ranking problem: you care about
the obvious losers and the standout keepers, not whether photo #412 beats
photo #487. So Winnow spends pennies triaging everything with a fast model and
reserves the expensive, careful judgments for the photos that might earn a
place on your wall. A 25k-photo library costs roughly **$20–40** in API usage
end to end.

## Install

### As a tool

```bash
uv tool install immich-winnow      # or: pipx install immich-winnow
```

That puts `winnow` on your PATH — drop the `uv run` prefix from every command
below.

### From source

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/robertcoop/immich-winnow && cd immich-winnow
uv sync
cp .env.example .env   # then fill in your keys
```

## Configure

`.env` (never committed):

```ini
IMMICH_URL=http://your-immich:2283
IMMICH_API_KEY=...     # Immich → Account Settings → API Keys
ANTHROPIC_API_KEY=...  # console.anthropic.com
```

See `.env.example` for optional knobs (models per stage, burst sensitivity,
set sizes, thresholds).

## Use

```bash
uv run winnow check                                  # verify both connections
uv run winnow scan --after 2024-06-01 --before 2024-06-08
uv run winnow triage --limit 50                      # try 50 photos first to sanity-check cost
uv run winnow triage                                 # stage 1 (--direct is the default; --batch is 50% off)
uv run winnow poll --ingest                          # if using --batch: fetch finished results
uv run winnow rank                                   # stage 2
uv run winnow finals                                 # stage 3
uv run winnow report --out winnow-report.html        # HTML contact sheet — review it!
uv run winnow apply --dry-run                        # see exactly what would change (-v for the full list)
uv run winnow apply --live --buckets reject,stars    # write back to Immich (asks first; -y to skip)
uv run winnow status                                 # ledger summary any time
```

`winnow --version` prints the version; every command takes `--help`.

Start with a few days of photos to calibrate, then widen the date range.
`scan`/`triage` are incremental — re-running with an overlapping range never
re-judges photos it has already seen.

## What gets written to Immich

Every write-back is findable afterwards by filtering on the `winnow/*` tag it
carries — that is the undo handle. Winnow has no `undo` command; undo is a
bulk edit in the Immich UI over a tag's assets.

| Bucket | Action | Undo |
|---|---|---|
| Confident rejects | rating −1 + tag `winnow/reject` | clear rating / untag |
| Confident non-photos — screenshots, documents, memes, and other (wallpapers, illustrations, renders) | archived + tag `winnow/screenshot`, `winnow/document`, `winnow/meme` or `winnow/other` | unarchive / untag |
| Burst also-rans | stacked under the winner + tag `winnow/burst-loser` | un-stack / untag |
| ★★★★★ finalists | rating 5 + favorite + tag `winnow/best` | clear rating / unfavorite / untag |
| ★★★★ finalists | rating 4 (no tag — they are indistinguishable from your own 4-star ratings) | clear rating |
| Everything else | untouched | — |

Both destructive-looking buckets need a **high-confidence** verdict: an unsure
reject or an unsure "that's a meme" stays in the untouched middle.

## Design notes

- **Asymmetric caution.** Hiding a photo you'd cherish is the only expensive
  mistake, so rejection requires a high-confidence verdict, and `apply` is
  dry-run by default. The judge is told that sentimental value is unknowable —
  uncertain cases stay in the middle, untouched.
- **Position bias** is neutralized by randomized set membership in stage 2 and
  order-swapped duplicate judgments in stage 3 (disagreement = tie).
- **Bursts** are detected via timestamp gaps + perceptual hashes (and Immich's
  own duplicate groups), then judged as a single "pick the best of these"
  contest instead of N independent calls.
- **Batch API support** cuts Anthropic costs 50% for big runs; small runs use
  direct calls for instant feedback.

## Development

```bash
uv run pytest            # full test suite (no network needed)
uv run ruff check .      # lint
```

## License

MIT
