"""Tests for the ``winnow`` command line interface.

Nothing here touches the network or a real key: Immich is a respx mock and the
Anthropic SDK is replaced by a scripted stub injected at ``winnow.cli.Anthropic``
(which is where every command builds its client).
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from conftest import make_jpeg
from winnow import __version__, cli
from winnow.cli import app, estimate_cost, price_for
from winnow.ledger import Ledger
from winnow.pipeline.finals import FinalsStats
from winnow.pipeline.rank import RankStats
from winnow.schemas import TriageVerdict

ROOT = "http://immich.test:2283"
API = f"{ROOT}/api"

#: Every command the CLI must expose.
COMMANDS = (
    "check",
    "scan",
    "triage",
    "poll",
    "rank",
    "finals",
    "report",
    "apply",
    "status",
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")

TRIAGE_JSON = json.dumps(
    {
        "category": "photo",
        "verdict": "candidate",
        "technical_score": 9,
        "reasons": ["clean light"],
        "confidence": "high",
    }
)


def plain(text: str) -> str:
    """Strip rich's styling so assertions can match plain substrings."""
    return ANSI.sub("", text)


def asset_id(index: int) -> str:
    """Deterministic UUID-shaped asset id."""
    return f"cccccccc-0000-4000-8000-{index:012d}"


# ----------------------------------------------------------------------
# stub Anthropic SDK
# ----------------------------------------------------------------------


@dataclass
class StubState:
    """Everything the fake Anthropic client should do, tweakable per test."""

    reply: str = TRIAGE_JSON
    usage: tuple[int, int] = (1000, 2000)
    processing_status: str = "ended"
    models_error: Exception | None = None
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    batches: dict[str, list[Any]] = field(default_factory=dict)
    clients: list[Any] = field(default_factory=list)


def _message(state: StubState) -> SimpleNamespace:
    """A canned Anthropic message carrying one JSON text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=state.reply)],
        usage=SimpleNamespace(input_tokens=state.usage[0], output_tokens=state.usage[1]),
        model="claude-haiku-4-5",
    )


class StubBatches:
    """Stands in for ``client.messages.batches``."""

    def __init__(self, state: StubState) -> None:
        self.state = state

    def create(self, requests: list[Any]) -> SimpleNamespace:
        batch_id = f"msgbatch_{len(self.state.batches) + 1}"
        self.state.batches[batch_id] = list(requests)
        return SimpleNamespace(id=batch_id)

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status=self.state.processing_status)

    def results(self, batch_id: str) -> Iterator[SimpleNamespace]:
        for request in self.state.batches[batch_id]:
            yield SimpleNamespace(
                custom_id=request["custom_id"],
                result=SimpleNamespace(type="succeeded", message=_message(self.state)),
            )


class StubAnthropic:
    """Minimal stand-in for :class:`anthropic.Anthropic`."""

    def __init__(self, state: StubState, api_key: str | None = None) -> None:
        self.state = state
        self.api_key = api_key
        self.messages = SimpleNamespace(create=self._create, batches=StubBatches(state))
        self.models = SimpleNamespace(list=self._models)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.state.create_calls.append(kwargs)
        return _message(self.state)

    def _models(self) -> SimpleNamespace:
        if self.state.models_error is not None:
            raise self.state.models_error
        return SimpleNamespace(data=[SimpleNamespace(id="claude-haiku-4-5")])


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the console width so table rows never wrap mid-assertion."""
    monkeypatch.setattr(cli, "console", Console(width=200))
    monkeypatch.setattr(cli, "err_console", Console(width=200, stderr=True))


@pytest.fixture()
def bare_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No configuration at all — not even a .env within reach."""
    for name in ("IMMICH_URL", "IMMICH_API_KEY", "ANTHROPIC_API_KEY", "DB_PATH", "CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def stub_sdk(monkeypatch: pytest.MonkeyPatch) -> StubState:
    """Replace the Anthropic constructor the CLI uses with a scripted stub."""
    state = StubState()

    def factory(*_args: Any, api_key: str | None = None, **_kwargs: Any) -> StubAnthropic:
        client = StubAnthropic(state, api_key)
        state.clients.append(client)
        return client

    monkeypatch.setattr(cli, "Anthropic", factory)
    return state


@pytest.fixture()
def immich_api() -> Iterator[respx.MockRouter]:
    """A fake Immich server with every route the CLI can touch."""
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API}/server/about", name="about").mock(
            return_value=httpx.Response(200, json={"version": "v3.1.0"})
        )
        router.post(f"{API}/search/metadata", name="search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "assets": {
                        "items": [_asset_dto(0), _asset_dto(1)],
                        "nextPage": None,
                        "count": 2,
                    }
                },
            )
        )
        router.get(url__regex=re.escape(API) + r"/assets/[^/]+/thumbnail", name="thumb").mock(
            return_value=httpx.Response(200, content=make_jpeg())
        )
        router.get(f"{API}/duplicates", name="duplicates").mock(
            return_value=httpx.Response(200, json=[])
        )
        router.put(f"{API}/tags", name="tag_upsert").mock(
            side_effect=lambda request: httpx.Response(
                200,
                json=[
                    {"id": f"tag-{index}", "name": value.rsplit("/", 1)[-1], "value": value}
                    for index, value in enumerate(json.loads(request.content)["tags"])
                ],
            )
        )
        router.put(url__regex=re.escape(API) + r"/tags/[^/]+/assets", name="tag_assets").mock(
            return_value=httpx.Response(200, json={})
        )
        router.put(url__regex=re.escape(API) + r"/assets/[^/]+$", name="update_asset").mock(
            return_value=httpx.Response(200, json={"id": "ok"})
        )
        router.post(f"{API}/stacks", name="stacks").mock(
            return_value=httpx.Response(201, json={"id": "stack-1"})
        )
        yield router


def _asset_dto(index: int) -> dict[str, Any]:
    """Minimal Immich asset DTO for the scan tests."""
    taken = f"2024-06-01T1{index}:00:00.000Z"
    return {
        "id": asset_id(index),
        "originalFileName": f"IMG_{index:04d}.jpg",
        "type": "IMAGE",
        "localDateTime": taken,
        "fileCreatedAt": taken,
        "rating": 0,
        "exifInfo": {
            "make": "Apple",
            "model": "iPhone 13",
            "exifImageWidth": 4000,
            "exifImageHeight": 3000,
            "dateTimeOriginal": taken,
        },
    }


def _verdict(**overrides: Any) -> TriageVerdict:
    data: dict[str, Any] = {
        "category": "photo",
        "verdict": "neutral",
        "technical_score": 6,
        "reasons": ["fine"],
        "confidence": "medium",
    }
    data.update(overrides)
    return TriageVerdict(**data)


@pytest.fixture()
def seeded(settings_env: Path) -> Iterator[SimpleNamespace]:
    """A ledger with two scanned photos, plus decisions ready for write-back."""
    db_path = settings_env / "test.db"
    cache = settings_env / "cache"
    cache.mkdir(exist_ok=True)

    ledger = Ledger(db_path)
    rows = [
        {"id": asset_id(i), "filename": f"IMG_{i:04d}.jpg", "taken_at": f"2024-06-01T1{i}:00:00"}
        for i in range(6)
    ]
    ledger.upsert_assets(rows)
    for row in rows:
        (cache / f"{row['id']}.jpg").write_bytes(make_jpeg())

    # 0 rejected, 1 screenshot, 2 crowned, 3+4 a judged burst, 5 still unjudged.
    ledger.record_triage(
        asset_id(0),
        _verdict(verdict="reject", technical_score=1, confidence="high", reasons=["blurry"]),
        "claude-haiku-4-5",
        None,
    )
    ledger.set_decision(asset_id(0), "reject", {"category": "photo", "reasons": ["blurry"]})
    ledger.record_triage(asset_id(1), _verdict(category="screenshot"), "claude-haiku-4-5", None)
    ledger.set_decision(asset_id(1), "nonphoto", {"category": "screenshot"})
    ledger.record_triage(asset_id(2), _verdict(verdict="candidate"), "claude-haiku-4-5", None)
    ledger.set_stars(asset_id(2), 5)
    ledger.upsert_scores({asset_id(2): (2.0, 1)})
    ledger.set_decision(asset_id(2), "five_star", {"stars": 5, "rank": 1})
    ledger.assign_burst("bcafe", [asset_id(3), asset_id(4)])
    ledger.record_burst("bcafe", asset_id(3), [asset_id(4)], "sharper", "claude-haiku-4-5")
    ledger.set_decision(asset_id(4), "burst_loser", {"burst_id": "bcafe", "winner_id": asset_id(3)})
    ledger.close()

    yield SimpleNamespace(root=settings_env, db_path=db_path, cache=cache)


def open_ledger(seeded: SimpleNamespace) -> Ledger:
    """Re-open the seeded ledger for post-command assertions."""
    return Ledger(seeded.db_path)


# ----------------------------------------------------------------------
# help, version, configuration
# ----------------------------------------------------------------------


def test_help_lists_every_command_without_configuration(runner: CliRunner, bare_env: Path) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = plain(result.stdout)
    for command in COMMANDS:
        assert command in output, command


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_help_works_without_configuration(
    runner: CliRunner, bare_env: Path, command: str
) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, plain(result.output)


def test_version_option(runner: CliRunner, bare_env: Path) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "winnow" in plain(result.stdout)
    assert __version__ in plain(result.stdout)


def test_version_has_a_single_source_of_truth() -> None:
    """`winnow --version` reads winnow.__version__ and PyPI reads pyproject;
    they must not be able to drift apart."""
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_bytes().decode()
    )
    assert "version" not in pyproject["project"], "pyproject must not hardcode a second version"
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/winnow/__init__.py"


def test_missing_configuration_is_reported_not_traced(runner: CliRunner, bare_env: Path) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2
    message = plain(result.stderr)
    assert "Configuration error" in message
    assert "IMMICH_URL" in message
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_apply_rejects_unknown_buckets(runner: CliRunner, seeded: SimpleNamespace) -> None:
    result = runner.invoke(app, ["apply", "--buckets", "reject,bogus"])
    assert result.exit_code == 2
    assert "bogus" in plain(result.stderr)


# ----------------------------------------------------------------------
# pricing
# ----------------------------------------------------------------------


def test_prices_cover_every_stage_model() -> None:
    assert cli.PRICES["claude-haiku-4-5"] == (1.0, 5.0)
    assert cli.PRICES["claude-sonnet-5"] == (3.0, 15.0)
    assert cli.PRICES["claude-opus-5"] == (5.0, 25.0)


def test_price_for_matches_dated_snapshots() -> None:
    assert price_for("claude-opus-5-20260101") == (5.0, 25.0)
    assert price_for("something-else") is None


def test_estimate_cost_uses_per_million_pricing() -> None:
    assert estimate_cost("claude-sonnet-5", 1_000_000, 200_000) == pytest.approx(6.0)
    assert estimate_cost("claude-haiku-4-5", 0, 0) == 0.0
    assert estimate_cost("mystery-model", 10, 10) is None


# ----------------------------------------------------------------------
# status
# ----------------------------------------------------------------------


def test_status_renders_counts(runner: CliRunner, seeded: SimpleNamespace) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert re.search(r"assets\s+6", output)
    assert re.search(r"untriaged\s+3", output)
    assert re.search(r"triage\s+3", output)
    assert re.search(r"decisions\s+4", output)
    assert re.search(r"burst groups\s+1", output)
    # bucket / verdict breakdowns
    assert re.search(r"five_star\s+1", output)
    assert re.search(r"reject\s+1", output)
    assert re.search(r"screenshot\s+1", output)


def test_status_on_a_fresh_ledger(runner: CliRunner, settings_env: Path) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, plain(result.output)
    assert re.search(r"assets\s+0", plain(result.stdout))


# ----------------------------------------------------------------------
# check
# ----------------------------------------------------------------------


def test_check_reports_both_connections(
    runner: CliRunner, settings_env: Path, immich_api: respx.MockRouter, stub_sdk: StubState
) -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert "v3.1.0" in output
    assert "Anthropic key accepted" in output
    assert "claude-haiku-4-5" in output  # settings summary
    assert immich_api["about"].called


def test_check_fails_when_immich_is_down(
    runner: CliRunner, settings_env: Path, immich_api: respx.MockRouter, stub_sdk: StubState
) -> None:
    immich_api["about"].mock(return_value=httpx.Response(500, text="boom"))
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    assert "FAIL" in plain(result.stderr)


def test_check_fails_when_the_anthropic_key_is_rejected(
    runner: CliRunner, settings_env: Path, immich_api: respx.MockRouter, stub_sdk: StubState
) -> None:
    stub_sdk.models_error = RuntimeError("invalid x-api-key")
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    errors = plain(result.stderr)
    assert "Anthropic" in errors
    assert "invalid x-api-key" in errors


# ----------------------------------------------------------------------
# scan
# ----------------------------------------------------------------------


def test_scan_populates_the_ledger(
    runner: CliRunner, settings_env: Path, immich_api: respx.MockRouter
) -> None:
    result = runner.invoke(app, ["scan", "--after", "2024-06-01", "--before", "2024-06-02"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert re.search(r"seen\s+2", output)
    assert re.search(r"new\s+2", output)

    with Ledger(settings_env / "test.db") as ledger:
        assert len(ledger.get_assets()) == 2
    assert (settings_env / "cache" / f"{asset_id(0)}.jpg").exists()
    assert immich_api["search"].called


def test_scan_sends_the_date_window(
    runner: CliRunner, settings_env: Path, immich_api: respx.MockRouter
) -> None:
    runner.invoke(app, ["scan", "--after", "2024-06-01", "--before", "2024-06-02"])
    body = json.loads(immich_api["search"].calls[0].request.content)
    assert body["takenAfter"] == "2024-06-01T00:00:00.000Z"
    assert body["takenBefore"] == "2024-06-02T00:00:00.000Z"
    assert body["type"] == "IMAGE"


# ----------------------------------------------------------------------
# triage
# ----------------------------------------------------------------------


def test_triage_direct_judges_and_estimates_cost(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState
) -> None:
    result = runner.invoke(app, ["triage"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    # Two photos were still unjudged (the burst winner and one single).
    assert len(stub_sdk.create_calls) == 2
    assert re.search(r"singles judged\s+2", output)
    # 2 calls x (1000 in, 2000 out) at haiku prices: 0.002 + 0.020.
    assert "Estimated cost: $0.0220" in output
    assert "claude-haiku-4-5" in output

    with open_ledger(seeded) as ledger:
        assert len(ledger.triage_rows()) == 5


def test_triage_respects_the_limit(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState
) -> None:
    result = runner.invoke(app, ["triage", "--limit", "1"])
    assert result.exit_code == 0, plain(result.output)
    assert len(stub_sdk.create_calls) == 1


def test_triage_batch_submits_and_reports_the_id(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState
) -> None:
    result = runner.invoke(app, ["triage", "--batch"])
    assert result.exit_code == 0, plain(result.output)
    assert "msgbatch_1" in plain(result.stdout)
    assert not stub_sdk.create_calls  # nothing judged live
    with open_ledger(seeded) as ledger:
        assert [row["batch_id"] for row in ledger.open_batches()] == ["msgbatch_1"]


def test_triage_batch_with_nothing_to_do(
    runner: CliRunner, settings_env: Path, stub_sdk: StubState
) -> None:
    result = runner.invoke(app, ["triage", "--batch"])
    assert result.exit_code == 0, plain(result.output)
    assert "Nothing left to triage" in plain(result.stdout)


# ----------------------------------------------------------------------
# poll
# ----------------------------------------------------------------------


def test_poll_without_batches(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState
) -> None:
    result = runner.invoke(app, ["poll"])
    assert result.exit_code == 0, plain(result.output)
    assert "No open batches" in plain(result.stdout)


def test_poll_lists_open_batches(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState
) -> None:
    runner.invoke(app, ["triage", "--batch"])
    stub_sdk.processing_status = "in_progress"
    result = runner.invoke(app, ["poll"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert "msgbatch_1" in output
    assert "in_progress" in output
    assert "--ingest" in output


def test_poll_ingests_finished_batches(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState
) -> None:
    runner.invoke(app, ["triage", "--batch"])
    result = runner.invoke(app, ["poll", "--ingest"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert re.search(r"singles judged\s+2", output)
    # Same tokens as the direct-mode test, but Batch API usage bills at 50%.
    assert "Estimated cost: $0.0110" in output
    assert "batch 50% off" in output

    with open_ledger(seeded) as ledger:
        assert len(ledger.triage_rows()) == 5
        assert ledger.open_batches() == []


# ----------------------------------------------------------------------
# rank and finals
# ----------------------------------------------------------------------


def test_rank_prints_stats_and_sonnet_cost(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_rank(settings: Any, ledger: Any, judge: Any, **kwargs: Any) -> RankStats:
        captured["model"] = settings.rank_model
        captured["progress"] = kwargs.get("on_progress")
        return RankStats(
            candidates=4, sets_judged=4, pairs=20, input_tokens=1_000_000, output_tokens=200_000
        )

    monkeypatch.setattr(cli, "run_rank", fake_rank)
    result = runner.invoke(app, ["rank"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert re.search(r"candidates\s+4", output)
    assert re.search(r"pairs\s+20", output)
    assert "Estimated cost: $6.0000" in output
    assert "claude-sonnet-5" in output
    assert captured["model"] == "claude-sonnet-5"
    assert callable(captured["progress"])


def test_finals_prints_stats_and_opus_cost(
    runner: CliRunner, seeded: SimpleNamespace, stub_sdk: StubState, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_finals(settings: Any, ledger: Any, judge: Any, **kwargs: Any) -> FinalsStats:
        return FinalsStats(
            pool=8,
            rounds=3,
            pairs_judged=12,
            five_star=2,
            four_star=2,
            input_tokens=100_000,
            output_tokens=40_000,
        )

    monkeypatch.setattr(cli, "run_finals", fake_finals)
    result = runner.invoke(app, ["finals"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert re.search(r"five star\s+2", output)
    assert "Estimated cost: $1.5000" in output
    assert "claude-opus-5" in output


def test_unknown_model_degrades_to_a_token_count(
    runner: CliRunner,
    seeded: SimpleNamespace,
    stub_sdk: StubState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK_MODEL", "some-other-model")
    monkeypatch.setattr(cli, "run_rank", lambda *a, **k: RankStats(input_tokens=5, output_tokens=7))
    result = runner.invoke(app, ["rank"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert "no price on file" in output
    assert "5 in / 7 out" in output


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def test_report_writes_a_self_contained_file(runner: CliRunner, seeded: SimpleNamespace) -> None:
    out = seeded.root / "sheet.html"
    result = runner.invoke(app, ["report", "--out", str(out)])
    assert result.exit_code == 0, plain(result.output)
    assert "Report written" in plain(result.stdout)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert 'id="rejects"' in text
    assert "data:image/jpeg;base64," in text


def test_report_defaults_to_winnow_report_html(runner: CliRunner, seeded: SimpleNamespace) -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0, plain(result.output)
    assert (seeded.root / "winnow-report.html").exists()


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------

WRITE_ROUTES = ("tag_upsert", "tag_assets", "update_asset", "stacks")


def test_apply_defaults_to_a_dry_run(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    result = runner.invoke(app, ["apply"])
    assert result.exit_code == 0, plain(result.output)
    output = plain(result.stdout)
    assert "Dry run" in output
    assert re.search(r"dry run\s+yes", output)

    for name in WRITE_ROUTES:
        assert immich_api[name].call_count == 0, name
    with open_ledger(seeded) as ledger:
        assert ledger.decisions(unapplied_only=True) == ledger.decisions()


def test_dry_run_lists_the_planned_changes(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    output = plain(runner.invoke(app, ["apply"]).stdout)
    assert "winnow/reject" in output
    assert "winnow/best" in output
    assert "winnow/screenshot" in output
    assert "stack 2 frames" in output
    # one action per decision plus the burst stack
    assert re.search(r"selected\s+5", output)


def test_dry_run_truncates_a_long_plan_and_points_at_the_report(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    with open_ledger(seeded) as ledger:
        rows = [{"id": f"bulk-{i:04d}"} for i in range(60)]
        ledger.upsert_assets(rows)
        for row in rows:
            ledger.set_decision(row["id"], "reject", {"category": "photo"})

    output = plain(runner.invoke(app, ["apply", "--buckets", "reject"]).stdout)
    assert "and 36 more" in output
    assert "winnow report" in output
    # the summary comes first, so it is not buried under the table
    assert output.index("Write-back") < output.index("Planned changes")

    verbose = plain(runner.invoke(app, ["apply", "--buckets", "reject", "--verbose"]).stdout)
    assert "more" not in verbose.split("Planned changes")[1]


def test_apply_buckets_filter_narrows_the_plan(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    output = plain(runner.invoke(app, ["apply", "--buckets", "reject"]).stdout)
    assert re.search(r"selected\s+1", output)
    assert "winnow/reject" in output
    assert "winnow/best" not in output


def test_apply_live_prompts_before_writing_anything(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    result = runner.invoke(app, ["apply", "--live", "--buckets", "reject"], input="n\n")
    assert result.exit_code == 1
    output = plain(result.output)
    assert "Write 1 change(s) to http://immich.test:2283?" in output
    # the plan is shown *before* the prompt, and nothing is written on abort
    assert "winnow/reject" in output
    for name in WRITE_ROUTES:
        assert immich_api[name].call_count == 0, name


def test_apply_live_proceeds_when_confirmed(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    result = runner.invoke(app, ["apply", "--live", "--buckets", "reject"], input="y\n")
    assert result.exit_code == 0, plain(result.output)
    assert immich_api["update_asset"].call_count == 1


def test_apply_live_writes_to_immich(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    result = runner.invoke(app, ["apply", "--live", "--yes", "--buckets", "reject"])
    assert result.exit_code == 0, plain(result.output)
    assert "Dry run" not in plain(result.stdout)
    assert "Changes written" in plain(result.stdout)

    assert immich_api["tag_upsert"].call_count == 1
    assert immich_api["tag_assets"].call_count == 1
    assert immich_api["update_asset"].call_count == 1
    body = json.loads(immich_api["update_asset"].calls[0].request.content)
    assert body == {"rating": -1, "visibility": "archive"}

    with open_ledger(seeded) as ledger:
        applied = [row["asset_id"] for row in ledger.decisions() if row["applied_at"]]
        assert applied == [asset_id(0)]


def test_apply_live_is_resumable(
    runner: CliRunner, seeded: SimpleNamespace, immich_api: respx.MockRouter
) -> None:
    runner.invoke(app, ["apply", "--live", "--yes", "--buckets", "reject"])
    calls_before = immich_api["update_asset"].call_count
    output = plain(runner.invoke(app, ["apply", "--live", "--yes", "--buckets", "reject"]).stdout)
    assert immich_api["update_asset"].call_count == calls_before
    assert "Nothing to write back" in output
