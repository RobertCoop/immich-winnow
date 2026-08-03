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
                 WRITE-BACK          rejects → archive + tag · winners → ★★★★★
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

### Docker, alongside your Immich stack (recommended for servers)

Winnow ships as a headless container image, deployed the same way as
[immich-power-tools](https://github.com/immich-power-tools/immich-power-tools):
add one service to the compose stack you already run, reusing its `.env`.
There's no web UI and no port. By default the service **stays running and
sweeps your whole library once a week** (`winnow watch`): scan, judge (via
the 50%-off Batch API by default), rank, apply, refresh the report, sleep. Every stage is incremental, so photos it has
already judged cost nothing — each cycle only spends on what's new, and even
backdated imports are picked up because the sweep re-enumerates everything.

```yaml
services:
  # ...your immich services...
  winnow:
    container_name: immich_winnow
    image: ghcr.io/robertcoop/immich-winnow:latest
    command: ["watch"]            # weekly unattended cycle (see below)
    restart: unless-stopped
    env_file:
      - .env                      # reuse the Immich stack's .env
    environment:
      IMMICH_URL: http://immich-server:2283   # Immich by compose service name
    volumes:
      - immich-winnow-data:/data  # ledger, thumbnail cache, reports

volumes:
  immich-winnow-data:
```

Add two lines to the stack's `.env` (`IMMICH_API_KEY=...`,
`ANTHROPIC_API_KEY=...`) and `docker compose up -d winnow`. Heads-up: the
first cycle on a fresh ledger processes your entire library. Tune with e.g.
`command: ["watch", "--every", "3d", "--scoring-limit", "500"]`, keep a
human in the loop with `["watch", "--no-apply"]`, or park the watcher by
adding `profiles: ["winnow"]`.

One-off commands work any time, watcher or not:

```bash
docker compose run --rm winnow check
docker compose run --rm winnow scan --after 2024-06-01 --before 2024-07-01
docker compose run --rm winnow triage --batch
# ...and so on; every CLI command below works the same way.
```

The ledger and reports persist on the `immich-winnow-data` volume (mount a
host directory instead if you want the HTML report easy to open). A sample
[docker-compose.yml](docker-compose.yml) ships in this repo.

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

Winnow is configured entirely through environment variables (a local `.env`
file works too — never commit it).

### Required

| Variable | Where to get it |
|---|---|
| `IMMICH_URL` | Your Immich base URL, e.g. `http://192.168.1.10:2283` — no trailing `/api`. Inside a compose stack, use the service name: `http://immich-server:2283`. |
| `IMMICH_API_KEY` | Immich → **Account Settings → API Keys → New API Key**. Select **All** permissions (or at minimum: read/update assets and thumbnails, manage tags, albums, and stacks — Winnow writes ratings, tags, favorites, archives, stacks, and one album). |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) → API Keys. |

### Common tuning knobs (defaults shown)

| Variable | Default | What it does |
|---|---|---|
| `TRIAGE_MODEL` | `claude-haiku-4-5` | Stage-1 judge (every photo). |
| `RANK_MODEL` | `claude-sonnet-5` | Stage-2 judge (best-worst sets). |
| `FINALS_MODEL` | `claude-opus-5` | Stage-3 judge (head-to-head finals). |
| `DB_PATH` | `winnow.db` | The SQLite ledger — Winnow's memory. Keep it; back it up. |
| `CACHE_DIR` | `.winnow-cache` | Resized thumbnails (~120 KB per photo). |
| `CANDIDATE_SCORE_MIN` | `8` | Triage score floor for entering stage 2. Lower = rank more photos. |
| `SCORING_LIMIT` | `0` | Max already-scored photos re-judged as ranking anchors per run (new photos always score). `0` = unlimited. |
| `BEST_ALBUM` | `Five-Stars` | Album that collects your best photos. `""` disables. |
| `BEST_ALBUM_MIN_STARS` | `5` | `4` also pulls the four-star band into the album. |
| `FIVE_STAR_FAVORITE` | `true` | Also mark five-star photos as Immich favorites. |
| `BURST_GAP_SECONDS` | `10` | Max seconds between frames judged as one burst. |
| `IMAGE_EDGE` | `768` | Long edge (px) of what the judges see. |

### Command flags as env vars (`WINNOW_*`)

Every CLI flag doubles as an env var, which keeps compose files declarative.
The ones you're most likely to set on the watcher service:

| Variable | Default | Flag equivalent |
|---|---|---|
| `WINNOW_EVERY` | `7d` | `watch --every` — cycle cadence (`30m`, `6h`, `7d`, `1w`). |
| `WINNOW_WINDOW_DAYS` | `0` | `--window-days` — `0` sweeps the whole library every cycle. |
| `WINNOW_APPLY` | `true` for `watch`/`run`, `false` elsewhere | `--apply` — write decisions to Immich immediately. |
| `WINNOW_BATCH` | `true` for `watch`/`run`, `false` for `triage` | `--batch` — 50%-off Batch API triage. |
| `WINNOW_ALLOW_DEMOTIONS` | `false` | `finals --allow-demotions` — let re-ranking take five stars away. |

The complete list of every knob lives in [.env.example](.env.example) and,
with comments, in [docker-compose.yml](docker-compose.yml); `winnow
<command> --help` shows the env var for every flag.

## Use

One command does everything:

```bash
uv run winnow run          # scan whole library → batch triage (waits) → rank → finals → apply → report
uv run winnow run --no-apply --after 2024-06-01    # review-first, windowed variant
```

Or stage by stage:

```bash
uv run winnow check                                  # verify both connections
uv run winnow scan --after 2024-06-01 --before 2024-06-08
uv run winnow triage --limit 50                      # try 50 photos first to sanity-check cost
uv run winnow triage --batch --wait                  # stage 1 at 50% off; --wait ingests as batches finish
uv run winnow poll --wait                            # or wait separately (one-shot check: poll --ingest)
uv run winnow rank                                   # stage 2 (--limit caps re-judged anchors)
uv run winnow finals                                 # stage 3 (--allow-demotions to un-stick five-stars)
uv run winnow report --out winnow-report.html        # HTML contact sheet — review it!
uv run winnow apply --dry-run                        # see exactly what would change (-v for the full list)
uv run winnow apply --live --buckets reject,stars    # write back to Immich (asks first; -y to skip)
uv run winnow status                                 # ledger summary any time
```

`triage`, `poll --ingest` and `finals` all take `--apply` to write their
changes to Immich immediately instead of waiting for a reviewed `apply`;
`watch` applies by default (that's its job — use `--no-apply` to review).

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
| Confident rejects | archived + tag `winnow/reject` (+ rating −1 on servers that persist it — Immich v3.1 silently drops −1) | unarchive / untag |
| Confident non-photos — screenshots, documents, memes, and other (wallpapers, illustrations, renders) | archived + tag `winnow/screenshot`, `winnow/document`, `winnow/meme` or `winnow/other` | unarchive / untag |
| Burst also-rans | stacked under the winner + tag `winnow/burst-loser` | un-stack / untag |
| ★★★★★ finalists | rating 5 + favorite (unless `FIVE_STAR_FAVORITE=false`) + tag `winnow/best` | clear rating / unfavorite / untag |
| Starred photos | collected into the `Five-Stars` album (`BEST_ALBUM`; `BEST_ALBUM_MIN_STARS=4` includes ★★★★; `BEST_ALBUM=` disables) | remove from album |
| ★★★★ finalists | rating 4 (no tag — they are indistinguishable from your own 4-star ratings) | clear rating |
| Everything else | untouched | — |

Both destructive-looking buckets need a **high-confidence** verdict: an unsure
reject or an unsure "that's a meme" stays in the untouched middle.

## Design notes

- **Five stars are sticky.** Once a photo earns five stars — from Winnow or
  rated five in Immich by you — later runs never score it lower. A re-ranking
  can only demote it when you pass `finals --allow-demotions`.
- **New photos always get scored; anchors keep the scale honest.** When new
  candidates arrive, they are ranked in sets mixed with already-scored
  "anchor" photos spread across the existing ranking — otherwise newcomers
  would float on a scale of their own. `SCORING_LIMIT` caps how many anchors
  are re-judged per run; newcomers are never capped.
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
