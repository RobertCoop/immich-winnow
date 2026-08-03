"""The Winnow pipeline: scan, triage, rank, finals, write-back.

Each stage is a plain function taking ``(settings, ledger, ...)`` and returning
a small stats dataclass, and each one skips whatever the ledger already holds —
so any stage can be interrupted and simply run again. Nothing in here prints;
callers pass an ``on_progress`` callback if they want to show progress.
"""

from __future__ import annotations

from winnow.pipeline.finals import FinalsStats, run_finals
from winnow.pipeline.rank import RankStats, collect_pairs, refit_scores, run_rank
from winnow.pipeline.scan import (
    ProgressFn,
    ScanStats,
    load_thumb_b64,
    run_scan,
    thumb_path,
)
from winnow.pipeline.triage import (
    TriageStats,
    ingest_triage_batch,
    run_triage_direct,
    submit_triage_batch,
)
from winnow.pipeline.writeback import Action, ApplyStats, apply, plan

__all__ = [
    "Action",
    "ApplyStats",
    "FinalsStats",
    "ProgressFn",
    "RankStats",
    "ScanStats",
    "TriageStats",
    "apply",
    "collect_pairs",
    "ingest_triage_batch",
    "load_thumb_b64",
    "plan",
    "refit_scores",
    "run_finals",
    "run_rank",
    "run_scan",
    "run_triage_direct",
    "submit_triage_batch",
    "thumb_path",
]
