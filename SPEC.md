# Winnow — Implementation Spec (v0.1)

This is the binding contract for all modules. Follow signatures exactly; if you must
deviate, note it in your final report. Existing files `src/winnow/config.py` and
`src/winnow/schemas.py` are DONE — read them, do not modify them.

## Product summary

Winnow culls an Immich photo library in three stages:

1. **Triage** (Haiku): every photo gets `{category, verdict, technical_score, reasons,
   confidence}`. Bursts (near-identical sequences) get a single "pick the best" call
   instead of N independent judgments; the winner is additionally triaged.
2. **Rank** (Sonnet): candidates go through best-worst-scaling sets of ~8; implied
   pairwise outcomes feed a Bradley-Terry fit.
3. **Finals** (Opus): top pool plays Swiss-paired head-to-heads, each pair run twice
   with order swapped; ties on disagreement.

Decisions map to *reversible* Immich write-backs: rating -1 + tag for rejects, tag +
archive for non-photos, stars/favorites for winners, stacks for bursts. Nothing is
ever deleted. All state lives in a local SQLite ledger so every stage is resumable.

## Ground rules

- Python 3.11+, full type hints, docstrings on public API. `ruff` clean
  (line-length 100, rules per pyproject).
- Library code never prints; only `cli.py` does console output (rich).
- Run tests with `uv run --no-sync pytest tests/test_<module>.py -q` (the venv is
  pre-synced; NEVER run plain `uv sync`, `uv add`, or `uv lock` — no-sync avoids
  clobbering a shared venv while other work runs in parallel).
- Never touch `.env`, never hardcode keys, never make live network calls in tests.
- Work in /home/coop/winnow (absolute paths; the session cwd may differ).

## Confirmed Immich API (server v3.1.0, verified live)

Base: `settings.immich_base` + `/api`. Auth header: `x-api-key: <key>`.

- `GET /api/server/about` → `{"version": "v3.1.0", ...}`
- `POST /api/search/metadata` body
  `{"takenAfter": "2024-06-01T00:00:00.000Z", "takenBefore": ..., "size": N,
    "page": N, "withExif": true, "type": "IMAGE"}` →
  `{"assets": {"items": [asset...], "nextPage": 2 | null, "count": N}}`.
  Asset fields used: `id`, `originalFileName`, `localDateTime`, `fileCreatedAt`,
  `type`, `duplicateId`, `rating`, `isFavorite`,
  `exifInfo.{make, model, exifImageWidth, exifImageHeight, dateTimeOriginal}`.
  `nextPage` may be an int or string — treat truthy as "another page exists";
  pass it back verbatim as `page`.
- `GET /api/assets/{id}/thumbnail?size=preview` → JPEG bytes (~1440px long edge).
- `GET /api/assets/{id}` → full asset dto.
- `PUT /api/assets/{id}` body may include `rating` (1-5, -1 rejected, null unrated),
  `isFavorite` (bool), `visibility` ("archive" | "timeline"). Send only provided keys.
- `GET /api/duplicates` → `[{"duplicateId": ..., "assets": [asset...]}, ...]`.
- `GET /api/tags` → `[{"id", "name", "value", ...}]`.
- `PUT /api/tags` body `{"tags": ["winnow/reject"]}` → upsert, returns list of tag dtos
  (nested tag paths use `/`). If this endpoint 404s at runtime, fall back to
  `POST /api/tags {"name": ...}` then match from `GET /api/tags`.
- `PUT /api/tags/{tagId}/assets` body `{"ids": [assetId...]}` → bulk tag.
- `DELETE /api/tags/{tagId}/assets` body `{"ids": [...]}` → bulk untag.
- `DELETE /api/tags/{tagId}` → delete tag.
- `POST /api/stacks` body `{"assetIds": [primary, ...rest]}` → create stack (first id
  becomes primary). `DELETE /api/stacks/{id}` removes it.

## Anthropic API rules (do not deviate)

- Client: `anthropic.Anthropic(api_key=settings.anthropic_api_key)`.
- Models come from settings: triage `claude-haiku-4-5`, rank `claude-sonnet-5`,
  finals `claude-opus-5`. NEVER pass `temperature`, `top_p`, `top_k`, or `thinking`
  (sonnet-5/opus-5 reject sampling params with 400; default thinking is fine).
- `max_tokens`: 1024 for haiku triage/burst; 4096 for sonnet/opus calls (their
  adaptive thinking counts toward max_tokens).
- Image content block:
  `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}`
- Structured output: pass
  `output_config={"format": {"type": "json_schema", "schema": output_schema(ModelCls)}}`
  (helper in `winnow.schemas`). The reply's first `text` block contains JSON; parse
  with `winnow.schemas.parse_verdict(text, ModelCls)`.
- Batch API:
  ```python
  from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
  from anthropic.types.messages.batch_create_params import Request
  batch = client.messages.batches.create(requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**kwargs)), ...])
  b = client.messages.batches.retrieve(batch_id)   # b.processing_status == "ended" when done
  for r in client.messages.batches.results(batch_id):
      r.custom_id; r.result.type  # "succeeded" | "errored" | "canceled" | "expired"
      r.result.message            # on success — same shape as messages.create response
  ```
- The SDK auto-retries 429/5xx. Wrap judge calls so a JSON-parse failure retries the
  request once before raising `JudgeError`.

## Modules

### `src/winnow/immich.py` — owner: agent A

```python
class ImmichError(RuntimeError): ...

class ImmichClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None: ...
    # httpx.Client with base_url=f"{base_url}/api", headers x-api-key + Accept: application/json
    def close(self) -> None: ...
    def __enter__(self) / __exit__(...)   # context manager
    def ping(self) -> dict                        # GET /server/about
    def search_assets(self, taken_after: str, taken_before: str, page_size: int = 250) -> Iterator[dict]
        # paginate POST /search/metadata until nextPage is falsy; yield asset dicts
    def get_asset(self, asset_id: str) -> dict
    def fetch_thumbnail(self, asset_id: str, size: str = "preview") -> bytes
    def update_asset(self, asset_id: str, *, rating: int | None = ..., is_favorite: bool = ...,
                     visibility: str = ...) -> dict
        # sentinel: module-level UNSET = object(); only include explicitly-passed fields
    def duplicates(self) -> list[dict]
    def list_tags(self) -> list[dict]
    def upsert_tags(self, names: list[str]) -> dict[str, str]   # name -> tag id (PUT /tags, fallback per above)
    def tag_assets(self, tag_id: str, asset_ids: list[str]) -> None
    def untag_assets(self, tag_id: str, asset_ids: list[str]) -> None
    def delete_tag(self, tag_id: str) -> None
    def create_stack(self, asset_ids: list[str]) -> dict        # first id = primary
    def delete_stack(self, stack_id: str) -> None
```

Raise `ImmichError` with status + snippet of body on non-2xx. Tests: `respx` mocking
httpx — cover pagination (2 pages), update_asset partial payloads (only passed keys
serialized), error raising, upsert_tags fallback path.

### `src/winnow/images.py` + `src/winnow/bursts.py` — owner: agent B

```python
# images.py
def prepare_image(raw: bytes, max_edge: int = 768, quality: int = 82) -> bytes
    # Pillow: open, ImageOps.exif_transpose, convert RGB, thumbnail to max_edge,
    # save JPEG (metadata stripped by re-encode). Never upscale.
def to_b64(jpeg: bytes) -> str
def dhash(raw: bytes, hash_size: int = 8) -> int      # classic difference hash, pure PIL
def hamming(a: int, b: int) -> int

# bursts.py
@dataclass(frozen=True)
class AssetLite:
    id: str
    taken_at: datetime          # parsed from asset fileCreatedAt/dateTimeOriginal
    camera: str                 # f"{make}|{model}" or "" when unknown
    dhash: int | None = None

def group_bursts(assets: Sequence[AssetLite], *, gap_seconds: float = 10.0,
                 max_group: int = 10, dhash_max_distance: int | None = 26) -> list[list[str]]
    # sort by (camera, taken_at); chain consecutive same-camera shots with
    # gap <= gap_seconds; if both ends of a link have dhash and
    # hamming > dhash_max_distance, break the chain (scene change);
    # split chains longer than max_group; return only groups of >= 2 ids,
    # each group ordered by taken_at.

def merge_duplicate_groups(groups: list[list[str]], dup_groups: list[list[str]]) -> list[list[str]]
    # union-find merge of burst groups with Immich duplicate groups; preserve
    # deterministic ordering (sort members by original first-seen order).
```

Tests: synthetic PIL images (solid colors / gradients) for prepare/dhash; burst
chaining edge cases (gap boundary, camera change, dhash break, max_group split,
singletons excluded, empty input).

### `src/winnow/ledger.py` — owner: agent C

SQLite (stdlib sqlite3), WAL mode, `check_same_thread=False` not needed (single
thread). `class Ledger` with `__init__(path: str | Path)`, `close()`, context
manager. All JSON columns stored as TEXT via json.dumps.

Tables (create if not exist):

```sql
assets(id TEXT PRIMARY KEY, filename TEXT, taken_at TEXT, camera TEXT,
       width INTEGER, height INTEGER, burst_id TEXT, dhash TEXT,
       thumb_path TEXT, immich_rating INTEGER, scanned_at TEXT)
triage(asset_id TEXT PRIMARY KEY REFERENCES assets(id), category TEXT, verdict TEXT,
       technical_score INTEGER, reasons TEXT, confidence TEXT, model TEXT,
       raw TEXT, judged_at TEXT)
bursts(burst_id TEXT PRIMARY KEY, winner_id TEXT, reject_ids TEXT, note TEXT,
       model TEXT, judged_at TEXT)
bws_sets(id INTEGER PRIMARY KEY AUTOINCREMENT, round INTEGER, member_ids TEXT,
         best_id TEXT, worst_id TEXT, model TEXT, judged_at TEXT)
pairs(id INTEGER PRIMARY KEY AUTOINCREMENT, a_id TEXT, b_id TEXT, winner TEXT,
      stage TEXT, model TEXT, judged_at TEXT)          -- winner: asset id or 'tie'
scores(asset_id TEXT PRIMARY KEY, bt_score REAL, rank INTEGER, stars INTEGER,
       updated_at TEXT)
decisions(asset_id TEXT PRIMARY KEY, bucket TEXT, detail TEXT, decided_at TEXT,
          applied_at TEXT)                              -- bucket: reject|nonphoto|middle|four_star|five_star|burst_loser
batches(batch_id TEXT PRIMARY KEY, kind TEXT, status TEXT, submitted_at TEXT,
        ingested_at TEXT)
batch_items(custom_id TEXT PRIMARY KEY, batch_id TEXT, kind TEXT, payload TEXT,
            result TEXT, error TEXT)
```

Methods (all straightforward CRUD; return plain dicts / lists of dicts):
`upsert_assets(rows)`, `assign_burst(burst_id, asset_ids)`, `get_assets(ids=None)`,
`unjudged_asset_ids(exclude_bursts: bool)`, `burst_groups()` -> dict[burst_id, [ids]],
`unjudged_burst_ids()`, `record_triage(asset_id, verdict: TriageVerdict, model, raw)`,
`record_burst(burst_id, winner_id, reject_ids, note, model)`,
`triage_rows()`, `candidates(min_score)` (verdict=='candidate' OR score>=min_score,
category=='photo', not burst losers), `record_bws(round, member_ids, best_id, worst_id, model)`,
`bws_rows()`, `record_pair(a, b, winner, stage, model)`, `pair_rows()`,
`upsert_scores(mapping: dict[str, tuple[float, int]])`, `set_stars(asset_id, stars)`,
`set_decision(asset_id, bucket, detail)`, `decisions(bucket=None, unapplied_only=False)`,
`mark_applied(asset_ids)`, `add_batch(batch_id, kind, items: dict[str, dict])`,
`open_batches()`, `set_batch_status(batch_id, status)`,
`record_batch_result(custom_id, result_json=None, error=None)`,
`batch_items_for(batch_id)`, `summary() -> dict` (counts per table/bucket).

Tests: round-trip every method against a tmp_path db; summary counts; idempotent
upserts.

### `src/winnow/ranking.py` — owner: agent D

```python
def build_bws_sets(ids: Sequence[str], *, set_size: int = 8, appearances: int = 4,
                   seed: int | None = None) -> list[list[str]]
    # Each id appears ~`appearances` times across sets; no id twice in one set;
    # sets have set_size members (last set may be smaller but >= 2). Handle
    # len(ids) < set_size (single set of all, repeated `appearances` times with
    # different shuffles — but never a set with a duplicate id; if len(ids) < 2 -> []).
def bws_to_pairs(members: Sequence[str], best_id: str, worst_id: str) -> list[tuple[str, str, str]]
    # (a, b, winner) triples: best beats every other member; every other member
    # beats worst; no duplicate of the best-vs-worst pair.
def bradley_terry(pairs: Iterable[tuple[str, str, str]], *, iterations: int = 500,
                  tol: float = 1e-8, prior: float = 0.5) -> dict[str, float]
    # MM algorithm, pure python. winner == 'tie' counts half a win each way.
    # `prior` = pseudo-wins against a virtual average opponent (regularization,
    # keeps undefeated/never-winning items finite). Return strengths normalized
    # to geometric mean 1.0.
def rank_scores(strengths: dict[str, float]) -> list[tuple[str, float, int]]
    # sorted desc, dense rank starting at 1.
def swiss_pairs(scores: dict[str, float], history: set[frozenset[str]],
                *, seed: int | None = None) -> list[tuple[str, str]]
    # sort by score desc, pair adjacent, skip pairs already in history (greedy
    # look-ahead), odd item sits out.
def star_bands(ranked_ids: Sequence[str], *, five_count: int, four_frac: float = 0.3)
    -> dict[str, int]   # top five_count -> 5, next four_frac of remainder -> 4
```

Tests: BT on a known tournament (transitive dominance recovers order; tie handling;
regularization keeps finite on 100% winner), set builder invariants (appearance
counts within ±1, no dup in set, all ids covered), swiss no-rematch, star bands.

### `src/winnow/judge.py` — owner: agent E

```python
class JudgeError(RuntimeError): ...

@dataclass
class JudgeResult:            # generic wrapper
    verdict: Any              # TriageVerdict | BurstVerdict | BWSVerdict | PairVerdict
    input_tokens: int
    output_tokens: int
    model: str

TRIAGE_SYSTEM / BURST_SYSTEM / BWS_SYSTEM / PAIR_SYSTEM: str  # curated prompts
# Prompts must: define role (ruthless-but-fair photo culler for a personal library),
# category definitions, verdict semantics (reject = hide-worthy technical failure;
# candidate = standout worth cherishing), scoring rubric, confidence rules
# ("high only when unmistakable"), note that sentimental value is unknowable ->
# uncertain cases lean neutral/low-confidence. Judgments must be decisive JSON only.

def build_triage_request(model: str, image_b64: str) -> dict[str, Any]
def build_burst_request(model: str, images_b64: list[str]) -> dict[str, Any]
def build_bws_request(model: str, images_b64: list[str]) -> dict[str, Any]
def build_pair_request(model: str, image_a_b64: str, image_b_b64: str) -> dict[str, Any]
# Each returns full kwargs for client.messages.create: model, max_tokens, system,
# messages (user content: numbered/lettered image blocks + instruction text),
# output_config with the right schema. Multi-image: precede each image block with
# a text block "Photo 1:", "Photo 2:" / "Photo A:", "Photo B:".

def extract_text(message: Any) -> str      # first text block (iterate content, type=='text')
def parse_message(message: Any, model_cls: type[BaseModel]) -> BaseModel

class Judge:
    def __init__(self, client: Any, *, max_parse_retries: int = 1) -> None: ...
    def _run(self, kwargs: dict, model_cls) -> JudgeResult   # create + parse (+1 retry on parse fail)
    def triage(self, model: str, image_b64: str) -> JudgeResult
    def burst(self, model: str, images_b64: list[str]) -> JudgeResult      # validates indices in range
    def bws(self, model: str, images_b64: list[str]) -> JudgeResult        # validates best != worst, in range
    def pair(self, model: str, a_b64: str, b_b64: str) -> JudgeResult

# Batch helpers:
def to_batch_request(custom_id: str, kwargs: dict) -> Request
def submit_batch(client, requests: list[Request]) -> str                   # returns batch id
def batch_status(client, batch_id: str) -> str                             # processing_status
def iter_batch_results(client, batch_id: str) -> Iterator[tuple[str, Any | None, str | None]]
    # (custom_id, message_or_None, error_kind_or_None)
```

Out-of-range/degenerate verdict indices (e.g. bws best==worst) → retry once, then
JudgeError. Tests: stub client object (`SimpleNamespace`) whose
`messages.create` returns canned message objects (content=[ns(type='text', text=json)],
usage=ns(input_tokens=1, output_tokens=1)); verify request builders (image counts,
labels, schema attached, no temperature/thinking keys), parse robustness (markdown
fences), retry-then-error, batch request construction.

### `src/winnow/pipeline/` + `src/winnow/cli.py` + `src/winnow/report.py` — owner: integrator

- `pipeline/scan.py`: `run_scan(settings, ledger, immich, taken_after, taken_before) -> ScanStats`
  — search assets (IMAGE only), fetch+prepare thumbnails into
  `cache_dir/<asset_id>.jpg` (skip if cached), compute dhash, upsert assets,
  group bursts (merge immich duplicates), assign burst ids.
- `pipeline/triage.py`:
  `run_triage_direct(settings, ledger, judge, limit=None, on_progress=None) -> TriageStats`
  — bursts first (burst call on members, winner recorded, winner also gets individual
  triage; losers get decisions bucket 'burst_loser'), then singles. Skips anything
  already judged (resumable). Writes decisions: nonphoto (category != photo),
  reject (verdict==reject AND confidence=='high'), candidate/middle.
  `submit_triage_batch(...) -> batch_id` / `ingest_triage_batch(...)` — same logic via
  Batch API, using ledger.batch_items custom_id mapping (custom_id formats:
  `triage:<asset_id>`, `burst:<burst_id>`).
- `pipeline/rank.py`: `run_rank(settings, ledger, judge) -> RankStats` — candidates
  from ledger, build_bws_sets, judge each set (skip already-judged sets by member
  hash if re-run), expand to pairs, BT fit, upsert scores.
- `pipeline/finals.py`: `run_finals(settings, ledger, judge) -> FinalsStats` — top
  `finals_pool_size` by score; `finals_rounds` swiss rounds; each pair judged twice
  (A/B then B/A; disagreement -> tie); refit BT over ALL pairs (stage2 + finals),
  update scores; star_bands -> set_stars + decisions five_star/four_star.
- `pipeline/writeback.py`: `plan(ledger) -> list[Action]`, `apply(settings, ledger,
  immich, buckets: set[str], dry_run: bool) -> ApplyStats`. Mapping:
  - nonphoto -> tag `winnow/screenshot` (or per-category), visibility archive
  - reject -> archive + tag `winnow/reject` + best-effort rating -1
    (LIVE FINDING: Immich v3.1.0 echoes rating=-1 in the PUT response but never
    persists it — 1-5 persist fine, 0 is a 400. Archive is what actually hides
    a reject; the -1 is forward compatibility. Ratings live at
    asset.exifInfo.rating on read; the top-level rating field stays null.)
  - burst_loser -> tag `winnow/burst-loser`; stack group with winner primary
  - five_star -> rating 5 + isFavorite + tag `winnow/best`
  - four_star -> rating 4
  Mark applied in ledger. Dry-run returns the plan without calling Immich.
- `report.py`: `write_html_report(ledger, cache_dir, out_path)` — self-contained HTML
  contact sheet: summary stats table; sections per bucket (rejects, nonphotos, five/four
  star, burst groups) with base64-embedded thumbnails (max ~300px, re-encode quality 60),
  scores + reasons captions. No external assets.
- `cli.py`: typer app `winnow` with commands:
  - `check` — ping Immich (print version), verify Anthropic key with a models.list
    call (client.models.list, catch auth error), print settings summary.
  - `scan --after YYYY-MM-DD --before YYYY-MM-DD`
  - `triage [--batch/--direct (default direct)] [--limit N]`
  - `poll [--ingest]` — batch status / ingest finished batches
  - `rank`, `finals`
  - `report [--out winnow-report.html]`
  - `apply [--buckets reject,nonphoto,stars,stacks] [--dry-run/--live]` (default dry-run;
    `--live` required to write)
  - `status` — ledger summary table
  Rich progress bars for scan/triage; cost estimate line after each stage
  (usage tokens * per-model prices from a small PRICES dict).
- Integration tests: fake Immich (respx) + stub judge run a mini end-to-end
  (scan->triage->rank->finals->plan) on synthetic images; CLI smoke via
  `typer.testing.CliRunner` for `status`/`check` (mocked).

## Custom-id conventions (batch mode)

`triage:<asset_id>` / `burst:<burst_id>` / `bws:<set_row_id>` / `pair:<pair_key>` —
ingest routes on the prefix. Anthropic custom_id must match `^[a-zA-Z0-9_-]{1,64}$`;
UUIDs contain `-` only, so `triage_<uuid>` style with `_` separator is REQUIRED
(`:` is NOT allowed). Use `kind + "_" + id`.

## Definition of done (per module)

- `uv run --no-sync pytest tests/test_<module>.py -q` green.
- `uv run --no-sync ruff check src/winnow/<module>.py tests/test_<module>.py` clean.
- No changes outside your owned files (plus you may append to tests/conftest.py
  ONLY if a fixture is genuinely shared — prefer module-local fixtures).
