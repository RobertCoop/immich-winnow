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
                                     + favorite · bursts → stacks · screenshots →
                                     archive. All reversible, all tagged.
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

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd winnow
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
uv run winnow triage                                 # stage 1 (add --batch for the 50%-off Batch API)
uv run winnow poll --ingest                          # if using --batch: fetch finished results
uv run winnow rank                                   # stage 2
uv run winnow finals                                 # stage 3
uv run winnow report                                 # HTML contact sheet — review it!
uv run winnow apply --dry-run                        # see exactly what would change
uv run winnow apply --live --buckets reject,stars    # write back to Immich
uv run winnow status                                 # ledger summary any time
```

Start with a few days of photos to calibrate, then widen the date range.
`scan`/`triage` are incremental — re-running with an overlapping range never
re-judges photos it has already seen.

## What gets written to Immich

| Bucket | Action | Undo |
|---|---|---|
| Confident rejects | rating −1 + tag `winnow/reject` | clear rating / untag |
| Screenshots & documents | tag + archived | unarchive |
| Burst also-rans | stacked under the winner + tag | un-stack |
| Finalists | ★★★★ / ★★★★★ + favorite + tag `winnow/best` | clear |
| Everything else | untouched | — |

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
