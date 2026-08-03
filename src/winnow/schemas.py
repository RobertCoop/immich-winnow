"""Pydantic models for judge verdicts, plus JSON-schema helpers.

These models are the contract between Claude and the pipeline. Their JSON
schemas are sent to the API as structured-output formats, so they must stay
free of schema features the API rejects (numeric min/max constraints,
recursion). Range checks are enforced by validators instead, which do not
leak into the generated JSON schema.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, field_validator

Category = Literal["photo", "screenshot", "document", "meme", "other"]
VerdictKind = Literal["reject", "neutral", "candidate"]
Confidence = Literal["low", "medium", "high"]


class TriageVerdict(BaseModel):
    """Stage-1 verdict for a single photo."""

    category: Category
    verdict: VerdictKind
    technical_score: int
    reasons: list[str]
    confidence: Confidence

    @field_validator("technical_score")
    @classmethod
    def _score_range(cls, v: int) -> int:
        if not 0 <= v <= 10:
            raise ValueError(f"technical_score must be 0-10, got {v}")
        return v


class BurstVerdict(BaseModel):
    """Pick-the-best verdict for a burst of near-identical shots.

    Indices are 1-based, matching the labels shown to the model.
    """

    best_index: int
    reject_indices: list[int]
    note: str

    @field_validator("best_index")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("best_index is 1-based and must be >= 1")
        return v


class BWSVerdict(BaseModel):
    """Best-worst-scaling verdict over a set of photos (1-based indices)."""

    best_index: int
    worst_index: int
    note: str

    @field_validator("best_index", "worst_index")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("indices are 1-based and must be >= 1")
        return v


class PairVerdict(BaseModel):
    """Head-to-head verdict between photo A and photo B."""

    winner: Literal["A", "B", "tie"]
    note: str


#: Annotations pydantic emits that structured outputs has no use for. The API
#: rejects unsupported keywords outright, so they are stripped rather than
#: merely ignored.
_DROPPED_KEYWORDS = ("default", "title")


def _strictify(node: Any) -> None:
    """Recursively make a generated schema safe for structured outputs.

    Every object node gets ``additionalProperties: false``, and the annotations
    in :data:`_DROPPED_KEYWORDS` are removed wherever they appear.
    """
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for keyword in _DROPPED_KEYWORDS:
            node.pop(keyword, None)
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def output_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a verdict model, suitable for structured outputs.

    All object nodes get ``additionalProperties: false`` (required by the
    API), and ``default``/``title`` annotations are dropped.
    """
    schema = model_cls.model_json_schema()
    _strictify(schema)
    return schema


def parse_verdict(text: str, model_cls: type[BaseModel]) -> Any:
    """Parse a model's JSON reply into a verdict instance.

    Tolerates surrounding prose or markdown fences by extracting the first
    balanced JSON object from the text.
    """
    text = text.strip()
    try:
        return model_cls.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in reply: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return model_cls.model_validate(json.loads(text[start : i + 1]))
    raise ValueError(f"unbalanced JSON object in reply: {text[:200]!r}")
