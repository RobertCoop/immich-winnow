"""Tests for verdict models and schema/parse helpers."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from winnow.schemas import (
    BurstVerdict,
    BWSVerdict,
    PairVerdict,
    TriageVerdict,
    output_schema,
    parse_verdict,
)

TRIAGE_JSON = {
    "category": "photo",
    "verdict": "candidate",
    "technical_score": 9,
    "reasons": ["sharp", "great_composition"],
    "confidence": "high",
}


def test_triage_roundtrip():
    v = TriageVerdict.model_validate(TRIAGE_JSON)
    assert v.verdict == "candidate"
    assert v.technical_score == 9


@pytest.mark.parametrize("score", [-1, 11, 100])
def test_triage_score_bounds(score):
    bad = dict(TRIAGE_JSON, technical_score=score)
    with pytest.raises(ValidationError):
        TriageVerdict.model_validate(bad)


def test_output_schema_is_strict_and_constraint_free():
    schema = output_schema(TriageVerdict)
    assert schema["additionalProperties"] is False
    dumped = json.dumps(schema)
    # numeric range constraints are rejected by the API; must not appear
    assert "minimum" not in dumped
    assert "maximum" not in dumped


def test_output_schema_all_models():
    for cls in (TriageVerdict, BurstVerdict, BWSVerdict, PairVerdict):
        schema = output_schema(cls)
        assert schema.get("type") == "object"
        assert schema["additionalProperties"] is False


def test_parse_verdict_plain():
    v = parse_verdict(json.dumps(TRIAGE_JSON), TriageVerdict)
    assert v.category == "photo"


def test_parse_verdict_with_markdown_fences():
    text = "Here you go:\n```json\n" + json.dumps(TRIAGE_JSON) + "\n```\nDone."
    v = parse_verdict(text, TriageVerdict)
    assert v.technical_score == 9


def test_parse_verdict_with_braces_in_strings():
    payload = dict(TRIAGE_JSON, reasons=['odd "brace}" reason'])
    v = parse_verdict("prefix " + json.dumps(payload) + " suffix", TriageVerdict)
    assert v.reasons == ['odd "brace}" reason']


def test_parse_verdict_no_json_raises():
    with pytest.raises(ValueError):
        parse_verdict("no json here at all", PairVerdict)


def test_pair_verdict_values():
    assert PairVerdict.model_validate({"winner": "tie", "note": ""}).winner == "tie"
    with pytest.raises(ValidationError):
        PairVerdict.model_validate({"winner": "C", "note": ""})


def test_burst_verdict_one_based():
    with pytest.raises(ValidationError):
        BurstVerdict.model_validate({"best_index": 0, "reject_indices": [], "note": ""})
