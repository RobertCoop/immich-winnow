"""``winnow`` — the command line interface.

This is the only module in Winnow that prints. Every command follows the same
shape: build the handful of objects the stage needs, hand them to a pipeline
function with a progress callback, then render the returned stats.

Settings are loaded *inside* each command rather than at import time, so
``winnow --help`` works on a machine with no configuration at all.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer
from anthropic import Anthropic
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from winnow import __version__
from winnow.config import Settings, load_settings
from winnow.immich import ImmichClient, ImmichError
from winnow.judge import Judge, batch_status
from winnow.ledger import Ledger
from winnow.pipeline import writeback
from winnow.pipeline.finals import run_finals
from winnow.pipeline.rank import run_rank
from winnow.pipeline.scan import ProgressFn, run_scan
from winnow.pipeline.triage import ingest_triage_batch, run_triage_direct, submit_triage_batch
from winnow.report import write_html_report

__all__ = [
    "DEFAULT_BUCKETS",
    "PRICES",
    "Session",
    "app",
    "estimate_cost",
    "heading",
    "load_or_exit",
    "parse_buckets",
    "price_for",
    "print_cost",
    "print_stats",
    "progress_reporter",
    "session",
]

#: US dollars per million tokens, ``model prefix -> (input, output)``.
#: Prefix keys keep dated model ids (``claude-haiku-4-5-20251001``) priced.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
}

#: Default value of ``apply --buckets``: everything write-back knows about.
DEFAULT_BUCKETS = "reject,nonphoto,stars,stacks,enrich"

#: Rows of the change table printed before it is truncated. The full list
#: lives in the HTML report, which is the reviewable surface.
MAX_TABLE_ROWS = 25

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="winnow",
    add_completion=False,
    no_args_is_help=True,
    help="AI photo culling for Immich — hide the rejects, crown the keepers.",
)


# --------------------------------------------------------------------------- #
# cost estimation
# --------------------------------------------------------------------------- #


def price_for(model: str) -> tuple[float, float] | None:
    """Per-million-token ``(input, output)`` price for a model id.

    Matching is by longest prefix, so dated snapshots of a model are priced
    like the family they belong to.

    Returns:
        The price pair, or ``None`` for a model Winnow has no price for.
    """
    matches = [(prefix, price) for prefix, price in PRICES.items() if model.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, *, batch: bool = False
) -> float | None:
    """Estimate the US-dollar cost of a stage's token usage.

    Batch API usage is billed at 50% of the standard price.

    Returns:
        The estimate, or ``None`` when the model's price is unknown.
    """
    price = price_for(model)
    if price is None:
        return None
    cost = input_tokens / 1_000_000 * price[0] + output_tokens / 1_000_000 * price[1]
    return cost / 2 if batch else cost


def print_cost(model: str, stats: Any, *, batch: bool = False) -> None:
    """Print the estimated cost of a stage from its stats' token counters."""
    input_tokens = int(getattr(stats, "input_tokens", 0) or 0)
    output_tokens = int(getattr(stats, "output_tokens", 0) or 0)
    suffix = " · batch 50% off" if batch else ""
    usage = f"{input_tokens:,} in / {output_tokens:,} out · {model}{suffix}"
    cost = estimate_cost(model, input_tokens, output_tokens, batch=batch)
    if cost is None:
        console.print(f"[dim]Tokens used: {usage} (no price on file)[/dim]")
        return
    console.print(f"[dim]Estimated cost: ${cost:,.4f}  ({usage})[/dim]")


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #


@dataclass
class Session:
    """The objects a command needs, built once and closed together.

    ``immich`` and ``judge`` are only created for the commands that ask for
    them, so ledger-only commands such as ``status`` and ``report`` open no
    connections at all. They still need a valid configuration: every command
    loads :class:`~winnow.config.Settings`, which requires the two API keys.
    """

    settings: Settings
    ledger: Ledger
    immich: ImmichClient | None = None
    judge: Judge | None = None

    @property
    def api(self) -> ImmichClient:
        """The Immich client (only for sessions opened with ``immich=True``)."""
        if self.immich is None:
            raise RuntimeError("this session was opened without an Immich client")
        return self.immich

    @property
    def claude(self) -> Judge:
        """The judge (only for sessions opened with ``judge=True``)."""
        if self.judge is None:
            raise RuntimeError("this session was opened without a judge")
        return self.judge


def load_or_exit() -> Settings:
    """Load settings, or explain what is missing and exit with status 2."""
    try:
        return load_settings()
    except ValidationError as exc:
        missing = sorted({str(error["loc"][0]).upper() for error in exc.errors() if error["loc"]})
        err_console.print("[bold red]Configuration error.[/bold red]")
        if missing:
            err_console.print(f"Missing or invalid: {', '.join(missing)}")
        err_console.print("Set them in the environment or a .env file (see .env.example).")
        raise typer.Exit(code=2) from exc


@contextmanager
def session(*, immich: bool = False, judge: bool = False) -> Iterator[Session]:
    """Open the ledger — and optionally the two API clients — for one command.

    Args:
        immich: Also build an :class:`~winnow.immich.ImmichClient`.
        judge: Also build a :class:`~winnow.judge.Judge` over an Anthropic client.

    Yields:
        A :class:`Session`; everything it owns is closed on the way out.
    """
    settings = load_or_exit()
    ledger = Ledger(settings.db_path)
    client = ImmichClient(settings.immich_base, settings.immich_api_key) if immich else None
    judge_obj = Judge(Anthropic(api_key=settings.anthropic_api_key)) if judge else None
    try:
        yield Session(settings=settings, ledger=ledger, immich=client, judge=judge_obj)
    finally:
        if client is not None:
            client.close()
        ledger.close()


#: Substrings that mark a progress line as an error worth keeping on screen.
_FAILURE_MARKERS = ("failed", "could not")


@contextmanager
def progress_reporter(label: str) -> Iterator[ProgressFn]:
    """Yield an ``on_progress`` callback backed by a transient rich spinner.

    Ordinary progress lines live and die on the spinner; failure lines are
    also printed permanently, because a transient spinner erases everything
    when it exits and an error nobody saw is an error nobody can act on.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[cyan]{task.completed}[/cyan] done"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(label, total=None)

        def report(message: str) -> None:
            lowered = message.lower()
            if any(marker in lowered for marker in _FAILURE_MARKERS):
                progress.console.print(f"[yellow]! {message}[/yellow]")
            progress.update(task, description=f"{label} · {message}"[:90], advance=1)

        yield report


def _is_number(value: Any) -> bool:
    """True for plain ints (booleans are rendered as yes/no instead)."""
    return isinstance(value, int) and not isinstance(value, bool)


def heading(text: str) -> None:
    """Print a section heading (never interpreted as rich markup)."""
    console.print(f"\n{text}", style="bold", markup=False, highlight=False)


def print_stats(title: str, stats: Any, *, skip: tuple[str, ...] = ()) -> None:
    """Render a stats dataclass as a two-column table under a heading."""
    heading(title)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    for field in fields(stats):
        if field.name in skip:
            continue
        value = getattr(stats, field.name)
        if isinstance(value, (list, dict, set)):
            continue
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif _is_number(value):
            rendered = f"{value:,}"
        else:
            rendered = str(value)
        table.add_row(field.name.replace("_", " "), rendered)
    console.print(table)


def parse_buckets(raw: str) -> set[str]:
    """Parse ``--buckets`` into a validated set of write-back groups."""
    names = {part.strip() for part in raw.split(",") if part.strip()}
    if not names:
        err_console.print("[bold red]--buckets needs at least one group.[/bold red]")
        raise typer.Exit(code=2)
    unknown = sorted(names - set(writeback.ALL_GROUPS))
    if unknown:
        known = ", ".join(sorted(writeback.ALL_GROUPS))
        err_console.print(f"[bold red]Unknown buckets:[/bold red] {', '.join(unknown)}")
        err_console.print(f"Choose from: {known}")
        raise typer.Exit(code=2)
    return names


def _version_callback(value: bool) -> None:
    """Print the version and exit (eager ``--version`` option)."""
    if value:
        console.print(f"winnow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the Winnow version and exit.",
        ),
    ] = False,
) -> None:
    """Cull an Immich library with Claude: triage, rank, finals, write back."""


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command()
def check() -> None:
    """Verify both connections and show the active configuration."""
    settings = load_or_exit()

    heading("Settings")
    table = Table(show_header=False, box=None)
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_row("immich url", settings.immich_base)
    table.add_row("database", str(settings.db_path))
    table.add_row("thumbnail cache", str(settings.cache_dir))
    table.add_row("triage model", settings.triage_model)
    table.add_row("rank model", settings.rank_model)
    table.add_row("finals model", settings.finals_model)
    table.add_row("image edge / quality", f"{settings.image_edge}px / q{settings.jpeg_quality}")
    table.add_row("bws sets", f"{settings.bws_set_size} photos x {settings.bws_appearances}")
    table.add_row("finals", f"top {settings.finals_pool_size}, {settings.finals_rounds} rounds")
    console.print(table)

    ok = True
    with ImmichClient(settings.immich_base, settings.immich_api_key) as immich:
        try:
            about = immich.ping()
            console.print(f"[green]OK[/green] Immich {about.get('version', '?')} reachable")
        except ImmichError as exc:
            ok = False
            err_console.print(f"[bold red]FAIL[/bold red] Immich: {exc}")
        else:
            try:
                prefs = immich.my_preferences()
                if not (prefs.get("ratings") or {}).get("enabled", False):
                    console.print(
                        "[yellow]note[/yellow] Star ratings are hidden in your Immich UI "
                        "(Account Settings → Features → Rating). Winnow still writes them; "
                        "enable the toggle to see finalist stars."
                    )
            except Exception:
                pass

    try:
        models = Anthropic(api_key=settings.anthropic_api_key).models.list()
        count = len(getattr(models, "data", []) or [])
        console.print(f"[green]OK[/green] Anthropic key accepted ({count} models visible)")
    except Exception as exc:  # any SDK/transport failure is a failed check
        ok = False
        err_console.print(f"[bold red]FAIL[/bold red] Anthropic: {exc}")

    if not ok:
        raise typer.Exit(code=1)


@app.command()
def scan(
    after: Annotated[
        str,
        typer.Option("--after", envvar="WINNOW_AFTER", help="Start of the window, YYYY-MM-DD."),
    ],
    before: Annotated[
        str,
        typer.Option("--before", envvar="WINNOW_BEFORE", help="End of the window, YYYY-MM-DD."),
    ],
) -> None:
    """Pull a date window of Immich photos into the local ledger."""
    with session(immich=True) as ses, progress_reporter("Scanning") as report:
        stats = run_scan(ses.settings, ses.ledger, ses.api, after, before, on_progress=report)
    print_stats(f"Scan {after} → {before}", stats)


def _apply_now(ses: Session, groups: set[str]) -> None:
    """Write freshly-made decisions for ``groups`` to Immich right away."""
    with progress_reporter("Applying") as report:
        stats = writeback.apply(
            ses.settings, ses.ledger, ses.api, groups, False, on_progress=report
        )
    print_stats("Write-back", stats, skip=("actions",))


#: Seconds between Batch API status checks while ``--wait``-ing.
BATCH_POLL_INTERVAL = 30.0


def _wait_for_batches(
    ses: Session, *, interval: float = BATCH_POLL_INTERVAL, apply_now: bool = False
) -> None:
    """Block until every open batch has ended, ingesting each as it finishes.

    Safe to interrupt: already-ingested chunks are recorded, and a later
    ``winnow poll --ingest`` (or another ``--wait``) picks up the rest.
    """
    while True:
        open_batches = ses.ledger.open_batches()
        if not open_batches:
            return
        ended = 0
        for row in open_batches:
            try:
                if batch_status(ses.claude.client, str(row["batch_id"])) == "ended":
                    ended += 1
            except Exception:  # a status probe must never abort the wait
                continue
        if ended:
            with progress_reporter("Ingesting") as report:
                stats = ingest_triage_batch(ses.settings, ses.ledger, ses.claude, None, report)
            print_stats("Ingested", stats)
            print_cost(ses.settings.triage_model, stats, batch=True)
            if apply_now:
                _apply_now(ses, {"reject", "nonphoto", "stacks", "enrich"})
            continue
        console.print(
            f"[dim]{len(open_batches)} batch(es) still processing; "
            f"next check in {int(interval)}s…[/dim]"
        )
        time.sleep(interval)


@app.command()
def triage(
    batch: Annotated[
        bool,
        typer.Option(
            "--batch/--direct",
            envvar="WINNOW_BATCH",
            help="Use the 50%-cheaper Batch API, not live calls.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", envvar="WINNOW_LIMIT", help="Stop after this many work items."),
    ] = None,
    apply_now: Annotated[
        bool,
        typer.Option(
            "--apply",
            envvar="WINNOW_APPLY",
            help="Write reject/non-photo/stack changes to Immich immediately, "
            "skipping the separate review-then-apply step.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            envvar="WINNOW_WAIT",
            help="With --batch: stay running, ingest results as each batch "
            "finishes, and only exit when everything has landed.",
        ),
    ] = False,
) -> None:
    """Stage 1 — judge every photo, settling bursts with one call each."""
    with session(immich=apply_now, judge=True) as ses:
        if batch:
            if apply_now and not wait:
                console.print(
                    "[yellow]--apply without --wait does nothing here; "
                    "use `winnow poll --ingest --apply` later, or add --wait.[/yellow]"
                )
            with progress_reporter("Queueing") as report:
                batch_id = submit_triage_batch(ses.settings, ses.ledger, ses.claude, limit, report)
            if batch_id is None and not ses.ledger.open_batches():
                console.print("Nothing left to triage.")
                return
            if batch_id is not None:
                console.print(f"Submitted batch [bold]{batch_id}[/bold].")
            if wait:
                _wait_for_batches(ses, apply_now=apply_now)
                console.print("[green]All batches ingested.[/green]")
            else:
                console.print(
                    "Check on it with [bold]winnow poll[/bold], then [bold]--ingest[/bold] "
                    "(or re-run with [bold]--wait[/bold])."
                )
            return

        with progress_reporter("Triaging") as report:
            stats = run_triage_direct(ses.settings, ses.ledger, ses.claude, limit, report)
        model = ses.settings.triage_model
        if apply_now:
            _apply_now(ses, {"reject", "nonphoto", "stacks", "enrich"})
    print_stats("Triage", stats)
    print_cost(model, stats)


@app.command()
def poll(
    ingest: Annotated[
        bool, typer.Option(
            "--ingest", envvar="WINNOW_INGEST", help="Ingest every finished batch into the ledger."
        )
    ] = False,
    apply_now: Annotated[
        bool,
        typer.Option(
            "--apply",
            envvar="WINNOW_APPLY",
            help="After ingesting, write reject/non-photo/stack changes to Immich immediately.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            envvar="WINNOW_WAIT",
            help="Stay running and ingest every batch as it finishes, then exit.",
        ),
    ] = False,
) -> None:
    """Show open Batch API jobs, and optionally ingest the finished ones."""
    with session(immich=apply_now and (ingest or wait), judge=True) as ses:
        open_batches = ses.ledger.open_batches()
        if not open_batches:
            console.print("No open batches.")
            return

        heading("Open batches")
        table = Table()
        table.add_column("batch id")
        table.add_column("kind")
        table.add_column("submitted")
        table.add_column("status")
        for row in open_batches:
            try:
                status_text = batch_status(ses.claude.client, str(row["batch_id"]))
            except Exception as exc:  # a status probe must never abort the command
                status_text = f"unknown ({exc})"
            table.add_row(
                str(row["batch_id"]),
                str(row.get("kind") or ""),
                str(row.get("submitted_at") or ""),
                status_text,
            )
        console.print(table)

        if wait:
            _wait_for_batches(ses, apply_now=apply_now)
            console.print("[green]All batches ingested.[/green]")
            return

        if not ingest:
            console.print(
                "[dim]Re-run with --ingest once a batch has ended, "
                "or with --wait to block until everything lands.[/dim]"
            )
            return

        with progress_reporter("Ingesting") as report:
            stats = ingest_triage_batch(ses.settings, ses.ledger, ses.claude, None, report)
        model = ses.settings.triage_model
        if apply_now:
            _apply_now(ses, {"reject", "nonphoto", "stacks", "enrich"})
    print_stats("Ingested", stats)
    print_cost(model, stats, batch=True)


@app.command()
def rank(
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Max already-scored photos re-judged as ranking anchors; every "
            "new candidate is always scored (SCORING_LIMIT setting; "
            "0 = unlimited).",
        ),
    ] = None,
) -> None:
    """Stage 2 — rank the candidates with best-worst scaling sets."""
    with session(judge=True) as ses:
        cap = ses.settings.scoring_limit if limit is None else limit
        with progress_reporter("Ranking") as report:
            stats = run_rank(
                ses.settings, ses.ledger, ses.claude, limit=cap or None, on_progress=report
            )
        model = ses.settings.rank_model
    print_stats("Rank", stats)
    print_cost(model, stats)


@app.command()
def finals(
    allow_demotions: Annotated[
        bool,
        typer.Option(
            "--allow-demotions",
            envvar="WINNOW_ALLOW_DEMOTIONS",
            help="Let a fresh ranking take five stars away. By default five-star "
            "awards (Winnow's or Immich's own) are sticky and never lowered.",
        ),
    ] = False,
    apply_now: Annotated[
        bool,
        typer.Option(
            "--apply",
            envvar="WINNOW_APPLY",
            help="Write star/favorite/album changes to Immich immediately, "
            "skipping the separate review-then-apply step.",
        ),
    ] = False,
) -> None:
    """Stage 3 — Swiss head-to-heads over the top pool, then star bands."""
    with session(immich=apply_now, judge=True) as ses:
        with progress_reporter("Finals") as report:
            stats = run_finals(
                ses.settings,
                ses.ledger,
                ses.claude,
                allow_demotions=allow_demotions,
                on_progress=report,
            )
        model = ses.settings.finals_model
        if apply_now:
            _apply_now(ses, {"stars"})
    print_stats("Finals", stats)
    print_cost(model, stats)


@app.command()
def report(
    out: Annotated[
        Path, typer.Option(
            "--out", envvar="WINNOW_REPORT_OUT", help="Where to write the HTML contact sheet."
        )
    ] = Path("winnow-report.html"),
) -> None:
    """Write a self-contained HTML contact sheet of every decision."""
    with session() as ses:
        path = write_html_report(ses.ledger, ses.settings.cache_dir, out)
    size_kb = path.stat().st_size / 1024
    console.print(f"Report written to {path} ({size_kb:,.0f} KB).", markup=False, highlight=False)


def _print_actions(actions: list[writeback.Action], title: str, verbose: bool) -> None:
    """Render the change table, truncated unless ``verbose``."""
    heading(title)
    shown = actions if verbose else actions[:MAX_TABLE_ROWS]
    table = Table()
    table.add_column("group")
    table.add_column("asset")
    table.add_column("change")
    for action in shown:
        target = action.asset_id or action.burst_id or "-"
        table.add_row(action.group, target, action.description)
    console.print(table)
    hidden = len(actions) - len(shown)
    if hidden:
        console.print(
            f"[dim]... and {hidden:,} more. Re-run with --verbose, or see "
            f"[bold]winnow report[/bold] for the full reviewable list.[/dim]"
        )


@app.command()
def apply(
    buckets: Annotated[
        str, typer.Option(
            "--buckets", envvar="WINNOW_BUCKETS", help="Comma-separated groups to write back."
        )
    ] = DEFAULT_BUCKETS,
    live: Annotated[
        bool,
        typer.Option("--live/--dry-run", envvar="WINNOW_LIVE", help="Actually write to Immich."),
    ] = False,
    yes: Annotated[
        bool, typer.Option(
            "--yes", "-y", envvar="WINNOW_YES", help="Skip the confirmation prompt (for scripts)."
        )
    ] = False,
    verbose: Annotated[
        bool, typer.Option(
            "--verbose",
            "-v",
            envvar="WINNOW_VERBOSE",
            help="List every change, not just the first few.",
        )
    ] = False,
    album: Annotated[
        str | None,
        typer.Option(
            "--album",
            help="Override the album for starred photos (BEST_ALBUM setting; "
            "'' disables it for this run).",
        ),
    ] = None,
) -> None:
    """Write decisions back to Immich. Dry-run unless --live is given."""
    groups = parse_buckets(buckets)
    with session(immich=True) as ses:
        album_name = ses.settings.best_album if album is None else album
        planned = [
            a
            for a in writeback.plan(
                ses.ledger,
                album=album_name or None,
                album_min_stars=ses.settings.best_album_min_stars,
                favorite_five=ses.settings.five_star_favorite,
                write_captions=ses.settings.write_captions,
                keyword_tags=ses.settings.keyword_tags,
                keyword_prefix=ses.settings.keyword_tag_prefix,
            )
            if a.group in groups
        ]
        if not planned:
            console.print("Nothing to write back.")
            return

        if live and not yes:
            # --live is irreversible-looking enough to deserve a look first.
            _print_actions(planned, "Changes to write", verbose)
            typer.confirm(
                f"Write {len(planned)} change(s) to {ses.settings.immich_base}?",
                abort=True,
            )

        with progress_reporter("Applying") as report:
            stats = writeback.apply(
                ses.settings,
                ses.ledger,
                ses.api,
                groups,
                not live,
                album=album,
                on_progress=report,
            )

    print_stats("Write-back", stats, skip=("actions",))
    if not live:
        console.print("[yellow]Dry run — nothing was written. Re-run with --live.[/yellow]")
    if stats.actions:
        _print_actions(stats.actions, "Planned changes" if not live else "Changes written", verbose)


_EVERY_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def parse_every(text: str) -> float:
    """Parse an interval like ``90s``, ``30m``, ``6h``, ``7d`` or ``1w`` to seconds.

    A bare number means seconds. Raises ``typer.BadParameter`` on nonsense.
    """
    cleaned = text.strip().lower()
    multiplier = 1.0
    if cleaned and cleaned[-1] in _EVERY_UNITS:
        multiplier = _EVERY_UNITS[cleaned[-1]]
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise typer.BadParameter(
            f"cannot parse interval {text!r} (try 30m, 6h, 7d, 1w)"
        ) from exc
    if value <= 0:
        raise typer.BadParameter("the interval must be positive")
    return value * multiplier


def _run_cycle(
    *,
    after: str,
    before: str,
    batch: bool,
    apply_changes: bool,
    scoring_limit: int | None,
) -> None:
    """One full pass: scan → triage → rank → finals → apply → report."""
    with session(immich=True, judge=True) as ses:
        cap = ses.settings.scoring_limit if scoring_limit is None else scoring_limit
        with progress_reporter("Scanning") as report:
            run_scan(ses.settings, ses.ledger, ses.api, after, before, on_progress=report)
        if batch:
            with progress_reporter("Queueing") as report:
                batch_id = submit_triage_batch(ses.settings, ses.ledger, ses.claude, None, report)
            if batch_id is not None:
                console.print(f"Submitted triage batch [bold]{batch_id}[/bold]; waiting…")
            _wait_for_batches(ses)
        else:
            with progress_reporter("Triaging") as report:
                t_stats = run_triage_direct(ses.settings, ses.ledger, ses.claude, None, report)
            print_cost(ses.settings.triage_model, t_stats)
        with progress_reporter("Ranking") as report:
            r_stats = run_rank(
                ses.settings, ses.ledger, ses.claude, limit=cap or None, on_progress=report
            )
        print_cost(ses.settings.rank_model, r_stats)
        with progress_reporter("Finals") as report:
            f_stats = run_finals(ses.settings, ses.ledger, ses.claude, on_progress=report)
        print_cost(ses.settings.finals_model, f_stats)
        if apply_changes:
            _apply_now(ses, set(parse_buckets(DEFAULT_BUCKETS)))
        report_path = write_html_report(
            ses.ledger, ses.settings.cache_dir, Path("winnow-report.html")
        )
        console.print(f"[dim]Report refreshed: {report_path}[/dim]")


@app.command()
def run(
    after: Annotated[
        str | None,
        typer.Option(
            "--after",
            envvar="WINNOW_AFTER",
            help="Start of the window, YYYY-MM-DD (default: the beginning of time).",
        ),
    ] = None,
    before: Annotated[
        str | None,
        typer.Option(
            "--before",
            envvar="WINNOW_BEFORE",
            help="End of the window, YYYY-MM-DD (default: tomorrow).",
        ),
    ] = None,
    batch: Annotated[
        bool,
        typer.Option(
            "--batch/--direct",
            envvar="WINNOW_BATCH",
            help="Triage via the 50%-cheaper Batch API (default), waiting for results.",
        ),
    ] = True,
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply/--no-apply",
            envvar="WINNOW_APPLY",
            help="Write every decision to Immich at the end (default). "
            "--no-apply leaves them for a reviewed `winnow apply`.",
        ),
    ] = True,
    scoring_limit: Annotated[
        int | None,
        typer.Option(
            "--scoring-limit",
            envvar="WINNOW_SCORING_LIMIT",
            help="Max already-scored photos re-judged as ranking anchors "
            "(SCORING_LIMIT setting; 0 = unlimited).",
        ),
    ] = None,
) -> None:
    """The whole pipeline in one shot: scan → triage → rank → finals → apply → report."""
    started = datetime.now(UTC)
    window_after = after or "1970-01-01"
    window_before = before or (started + timedelta(days=1)).date().isoformat()
    heading(f"Full run — window {window_after} → {window_before}")
    _run_cycle(
        after=window_after,
        before=window_before,
        batch=batch,
        apply_changes=apply_changes,
        scoring_limit=scoring_limit,
    )


@app.command()
def watch(
    every: Annotated[
        str, typer.Option(
            "--every", envvar="WINNOW_EVERY", help="How often to run a cycle: 30m, 6h, 7d, 1w."
        )
    ] = "7d",
    batch: Annotated[
        bool,
        typer.Option(
            "--batch/--direct",
            envvar="WINNOW_BATCH",
            help="Triage via the 50%-cheaper Batch API (default) — an unattended "
            "watcher never minds waiting for the discount.",
        ),
    ] = True,
    window_days: Annotated[
        int,
        typer.Option(
            "--window-days",
            envvar="WINNOW_WINDOW_DAYS",
            help="0 (default) re-enumerates the ENTIRE library every cycle, so "
            "every new photo is picked up even when imported with old dates. "
            "N > 0 restricts each cycle to a trailing taken-date window.",
        ),
    ] = 0,
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply/--no-apply",
            envvar="WINNOW_APPLY",
            help="Write each cycle's decisions straight to Immich (the point of "
            "an unattended watcher). --no-apply just accumulates decisions "
            "for a manual `winnow apply`.",
        ),
    ] = True,
    scoring_limit: Annotated[
        int | None,
        typer.Option(
            "--scoring-limit",
            help="Max already-scored photos re-judged as ranking anchors per "
            "cycle; every new photo is always scored "
            "(SCORING_LIMIT setting; 0 = unlimited).",
        ),
    ] = None,
    once: Annotated[
        bool, typer.Option(
            "--once", envvar="WINNOW_ONCE", help="Run a single cycle and exit (for cron/testing)."
        )
    ] = False,
) -> None:
    """Stay running: scan new photos, judge, rank and apply on an interval.

    This is the container's default mode — a weekly, fully unattended sweep
    of the whole library. Every stage is incremental, so photos the ledger
    already knows cost nothing; only genuinely new material is judged. That
    means the first cycle on a fresh ledger processes the entire library.
    """
    interval = parse_every(every)
    while True:
        cycle_start = datetime.now(UTC)
        if window_days > 0:
            after = (cycle_start - timedelta(days=window_days)).date().isoformat()
        else:
            after = "1970-01-01"
        before = (cycle_start + timedelta(days=1)).date().isoformat()
        heading(f"Cycle {cycle_start:%Y-%m-%d %H:%M} — window {after} → {before}")
        _run_cycle(
            after=after,
            before=before,
            batch=batch,
            apply_changes=apply_changes,
            scoring_limit=scoring_limit,
        )

        if once:
            return
        next_run = datetime.now(UTC) + timedelta(seconds=interval)
        console.print(f"Sleeping until [bold]{next_run:%Y-%m-%d %H:%M} UTC[/bold] ({every}).")
        time.sleep(interval)


@app.command()
def rejudge(
    stage: Annotated[
        str,
        typer.Option(
            "--stage",
            envvar="WINNOW_STAGE",
            help="What to forget: triage (stage 1, incl. burst contests), "
            "rank (stage 2 sets/pairs/scores), finals (stage 3 pairs), or all.",
        ),
    ] = "triage",
    after: Annotated[
        str | None,
        typer.Option(
            "--after", envvar="WINNOW_AFTER", help="Only photos taken on/after this date."
        ),
    ] = None,
    before: Annotated[
        str | None,
        typer.Option(
            "--before", envvar="WINNOW_BEFORE", help="Only photos taken before this date."
        ),
    ] = None,
    bucket: Annotated[
        str | None,
        typer.Option(
            "--bucket",
            envvar="WINNOW_BUCKET",
            help="Only photos currently in this decision bucket (e.g. middle, reject).",
        ),
    ] = None,
    missing_captions: Annotated[
        bool,
        typer.Option(
            "--missing-captions",
            envvar="WINNOW_MISSING_CAPTIONS",
            help="Only verdicts recorded before captions existed — the "
            "back-catalog captioning filter.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", envvar="WINNOW_YES", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Forget judgments so a stage runs again — for new models, new prompts,
    or captioning a back catalog. Touches only the local ledger, never Immich.
    """
    if stage not in ("triage", "rank", "finals", "all"):
        raise typer.BadParameter("--stage must be triage, rank, finals, or all")
    filtered = bool(after or before or bucket or missing_captions)
    if filtered and stage in ("rank", "finals"):
        raise typer.BadParameter(
            "filters only apply to --stage triage/all — rank and finals are "
            "relational and can only be cleared whole"
        )

    with session() as ses:
        targets: list[str] = []
        if stage in ("triage", "all"):
            targets = ses.ledger.select_rejudge_targets(
                after=after, before=before, bucket=bucket, missing_captions=missing_captions
            )
        pieces = []
        if stage in ("triage", "all"):
            pieces.append(f"{len(targets):,} stage-1 verdict(s)")
        if stage in ("rank", "all"):
            pieces.append("every best-worst set, rank pair and fitted score")
        if stage in ("finals", "all"):
            pieces.append("every finals head-to-head")
        summary = " + ".join(pieces)
        if stage in ("triage", "all") and not targets and stage == "triage":
            console.print("Nothing matches — no verdicts to forget.")
            return

        console.print(f"Will forget: {summary}.")
        console.print(
            "[dim]This deletes paid judgments from the local ledger (Immich is "
            "untouched); the next run re-judges and re-spends.[/dim]"
        )
        if not yes:
            typer.confirm("Proceed?", abort=True)

        if stage in ("triage", "all") and targets:
            cleared = ses.ledger.clear_triage(targets)
            console.print(f"Forgot {cleared:,} stage-1 verdict(s).")
        if stage in ("rank", "all"):
            ses.ledger.clear_rank()
            console.print("Forgot stage 2 (sets, pairs, scores).")
        if stage in ("finals", "all"):
            ses.ledger.clear_finals()
            console.print("Forgot stage 3 (head-to-heads).")

    hint = {
        "triage": "winnow triage --batch --wait  (then rank/finals, or winnow run)",
        "rank": "winnow rank",
        "finals": "winnow finals",
        "all": "winnow run",
    }[stage]
    console.print(f"Next: [bold]{hint}[/bold]")


@app.command()
def status() -> None:
    """Show what the ledger currently holds."""
    with session() as ses:
        summary = ses.ledger.summary()
        db_path = ses.settings.db_path

    heading(f"Ledger {db_path}")
    table = Table(show_header=False, box=None)
    table.add_column("metric", style="bold")
    table.add_column("count", justify="right")
    for key in (
        "assets",
        "untriaged",
        "triage",
        "burst_groups",
        "bursts",
        "bws_sets",
        "pairs",
        "scores",
        "decisions",
        "applied",
        "unapplied",
        "open_batches",
    ):
        table.add_row(key.replace("_", " "), f"{int(summary.get(key, 0)):,}")
    console.print(table)

    for label, mapping in (
        ("buckets", summary.get("buckets") or {}),
        ("verdicts", summary.get("verdicts") or {}),
        ("categories", summary.get("categories") or {}),
    ):
        if not mapping:
            continue
        heading(label)
        breakdown = Table(show_header=False, box=None)
        breakdown.add_column("key", style="bold")
        breakdown.add_column("count", justify="right")
        for key, count in sorted(mapping.items()):
            breakdown.add_row(str(key), f"{int(count):,}")
        console.print(breakdown)


if __name__ == "__main__":  # pragma: no cover
    app()
