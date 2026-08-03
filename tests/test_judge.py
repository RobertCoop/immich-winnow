"""Tests for winnow.judge — stubbed clients only, never a network call."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from winnow.judge import (
    BURST_SYSTEM,
    BWS_SYSTEM,
    CUSTOM_ID_PATTERN,
    MAX_BURST_IMAGES,
    PAIR_SYSTEM,
    RANK_EFFORT,
    RANK_MAX_TOKENS,
    TRIAGE_MAX_TOKENS,
    TRIAGE_SYSTEM,
    Judge,
    JudgeError,
    batch_status,
    build_burst_request,
    build_bws_request,
    build_pair_request,
    build_triage_request,
    extract_text,
    iter_batch_results,
    parse_message,
    submit_batch,
    to_batch_request,
)
from winnow.schemas import (
    BurstVerdict,
    BWSVerdict,
    PairVerdict,
    TriageVerdict,
    output_schema,
)

FORBIDDEN_KEYS = ("temperature", "top_p", "top_k", "thinking")

TRIAGE_JSON = json.dumps(
    {
        "category": "photo",
        "verdict": "neutral",
        "technical_score": 6,
        "reasons": ["ordinary light", "subject sharp"],
        "confidence": "medium",
    }
)
BURST_JSON = json.dumps({"best_index": 2, "reject_indices": [1, 3], "note": "2 is sharpest"})
BWS_JSON = json.dumps({"best_index": 1, "worst_index": 3, "note": "1 has the light"})
PAIR_JSON = json.dumps({"winner": "A", "note": "A is sharper"})


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


def make_message(text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> Any:
    """A canned Anthropic-shaped message whose first text block is ``text``."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class StubMessages:
    """Records ``create`` kwargs and hands back canned replies in order."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("messages.create called more times than expected")
        return self.replies.pop(0)


def make_client(*texts: str) -> Any:
    """Stub client whose ``messages.create`` returns one message per text."""
    return SimpleNamespace(messages=StubMessages([make_message(t) for t in texts]))


class StubBatches:
    def __init__(
        self,
        *,
        batch_id: str = "msgbatch_1",
        processing_status: str = "ended",
        results: list[Any] | None = None,
    ) -> None:
        self.batch_id = batch_id
        self.processing_status = processing_status
        self._results = results or []
        self.created: list[list[Any]] = []
        self.retrieved: list[str] = []
        self.results_for: list[str] = []

    def create(self, *, requests: list[Any]) -> Any:
        self.created.append(requests)
        return SimpleNamespace(id=self.batch_id, processing_status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        self.retrieved.append(batch_id)
        return SimpleNamespace(id=batch_id, processing_status=self.processing_status)

    def results(self, batch_id: str) -> list[Any]:
        self.results_for.append(batch_id)
        return self._results


def make_batch_client(**kwargs: Any) -> Any:
    batches = StubBatches(**kwargs)
    return SimpleNamespace(messages=SimpleNamespace(batches=batches)), batches


def walk_keys(node: Any) -> list[str]:
    """Every dict key anywhere inside ``node``."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(key)
            found.extend(walk_keys(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            found.extend(walk_keys(item))
    return found


def image_blocks(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    content = kwargs["messages"][0]["content"]
    return [b for b in content if b["type"] == "image"]


def text_blocks(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    content = kwargs["messages"][0]["content"]
    return [b for b in content if b["type"] == "text"]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prompt", [TRIAGE_SYSTEM, BURST_SYSTEM, BWS_SYSTEM, PAIR_SYSTEM]
)
def test_prompts_are_substantial_and_demand_json(prompt: str) -> None:
    assert len(prompt) > 400
    assert "JSON" in prompt
    assert "PERSONAL" in prompt


def test_triage_prompt_covers_the_rubric_and_confidence_rules() -> None:
    for term in ("screenshot", "document", "meme", "reject", "candidate", "neutral"):
        assert term in TRIAGE_SYSTEM
    assert '"high" ONLY' in TRIAGE_SYSTEM
    assert "SENTIMENTAL VALUE IS UNKNOWABLE" in TRIAGE_SYSTEM
    # anchored 0-10 scale
    for anchor in ("0-2", "3-4", "5-6", "7", "8", "9", "10"):
        assert anchor in TRIAGE_SYSTEM


@pytest.mark.parametrize("prompt", [BURST_SYSTEM, BWS_SYSTEM, PAIR_SYSTEM])
def test_comparison_prompts_name_the_criteria(prompt: str) -> None:
    lowered = prompt.lower()
    assert "sharp" in lowered
    assert "composition" in lowered
    assert "moment" in lowered


def test_burst_prompt_mentions_eyes_and_expressions() -> None:
    assert "eyes" in BURST_SYSTEM.lower()
    assert "expression" in BURST_SYSTEM.lower()


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #


def test_triage_request_shape() -> None:
    kwargs = build_triage_request("claude-haiku-4-5", "AAAA")
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == TRIAGE_MAX_TOKENS
    assert kwargs["system"] == TRIAGE_SYSTEM
    assert kwargs["messages"][0]["role"] == "user"
    images = image_blocks(kwargs)
    assert len(images) == 1
    assert images[0]["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "AAAA",
    }
    assert kwargs["output_config"] == {
        "format": {"type": "json_schema", "schema": output_schema(TriageVerdict)}
    }


def test_triage_request_has_no_image_label() -> None:
    kwargs = build_triage_request("m", "AAAA")
    labels = [b["text"] for b in text_blocks(kwargs)]
    assert not any(t.startswith("Photo ") for t in labels)


@pytest.mark.parametrize("count", [2, 3, 5, 10])
def test_burst_request_labels_every_image(count: int) -> None:
    images = [f"img{i}" for i in range(count)]
    kwargs = build_burst_request("claude-haiku-4-5", images)
    assert kwargs["max_tokens"] == TRIAGE_MAX_TOKENS
    assert kwargs["system"] == BURST_SYSTEM
    blocks = kwargs["messages"][0]["content"]
    # label, image, label, image, ..., trailing instruction
    assert len(blocks) == 2 * count + 1
    for i in range(count):
        assert blocks[2 * i] == {"type": "text", "text": f"Photo {i + 1}:"}
        assert blocks[2 * i + 1]["type"] == "image"
        assert blocks[2 * i + 1]["source"]["data"] == images[i]
    assert blocks[-1]["type"] == "text"
    assert str(count) in blocks[-1]["text"]
    assert kwargs["output_config"]["format"]["schema"] == output_schema(BurstVerdict)


def test_bws_request_uses_rank_budget_and_numeric_labels() -> None:
    images = [f"img{i}" for i in range(8)]
    kwargs = build_bws_request("claude-sonnet-5", images)
    assert kwargs["max_tokens"] == RANK_MAX_TOKENS
    assert kwargs["system"] == BWS_SYSTEM
    assert len(image_blocks(kwargs)) == 8
    labels = [b["text"] for b in text_blocks(kwargs) if b["text"].startswith("Photo ")]
    assert labels == [f"Photo {i}:" for i in range(1, 9)]
    assert kwargs["output_config"]["format"]["schema"] == output_schema(BWSVerdict)


def test_pair_request_uses_letter_labels() -> None:
    kwargs = build_pair_request("claude-opus-5", "aaa", "bbb")
    assert kwargs["max_tokens"] == RANK_MAX_TOKENS
    assert kwargs["system"] == PAIR_SYSTEM
    blocks = kwargs["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Photo A:"}
    assert blocks[1]["source"]["data"] == "aaa"
    assert blocks[2] == {"type": "text", "text": "Photo B:"}
    assert blocks[3]["source"]["data"] == "bbb"
    assert len(image_blocks(kwargs)) == 2
    assert kwargs["output_config"]["format"]["schema"] == output_schema(PairVerdict)


@pytest.mark.parametrize(
    "kwargs",
    [
        build_triage_request("m", "a"),
        build_burst_request("m", ["a", "b", "c"]),
        build_bws_request("m", ["a", "b", "c"]),
        build_pair_request("m", "a", "b"),
    ],
)
def test_requests_never_carry_forbidden_sampling_params(kwargs: dict[str, Any]) -> None:
    keys = walk_keys(kwargs)
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in keys


@pytest.mark.parametrize(
    "kwargs",
    [
        build_triage_request("m", "a"),
        build_burst_request("m", ["a", "b"]),
        build_bws_request("m", ["a", "b"]),
        build_pair_request("m", "a", "b"),
    ],
)
def test_requests_declare_a_json_schema(kwargs: dict[str, Any]) -> None:
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["type"] == "object"
    assert fmt["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("builder", [build_burst_request, build_bws_request])
def test_multi_image_builders_reject_degenerate_sets(builder: Any) -> None:
    with pytest.raises(JudgeError):
        builder("m", ["only-one"])
    with pytest.raises(JudgeError):
        builder("m", [])


def test_burst_request_refuses_an_over_limit_group() -> None:
    images = ["img"] * (MAX_BURST_IMAGES + 1)
    with pytest.raises(JudgeError, match="per-request limit"):
        build_burst_request("m", images)
    # exactly at the limit is still allowed
    assert len(image_blocks(build_burst_request("m", ["img"] * MAX_BURST_IMAGES))) == (
        MAX_BURST_IMAGES
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        build_bws_request("claude-sonnet-5", ["a", "b"]),
        build_pair_request("claude-opus-5", "a", "b"),
    ],
)
def test_deliberative_requests_ask_for_low_effort(kwargs: dict[str, Any]) -> None:
    # these models think by default and bill it against max_tokens
    assert kwargs["output_config"]["effort"] == RANK_EFFORT
    assert kwargs["max_tokens"] >= 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        build_triage_request("claude-haiku-4-5", "a"),
        build_burst_request("claude-haiku-4-5", ["a", "b"]),
    ],
)
def test_haiku_requests_never_send_effort(kwargs: dict[str, Any]) -> None:
    # Haiku 4.5 rejects output_config.effort outright
    assert "effort" not in kwargs["output_config"]


def test_requests_are_json_serialisable() -> None:
    # the SDK serialises the payload; anything non-JSON would blow up at runtime
    json.dumps(build_burst_request("m", ["a", "b"]))
    json.dumps(build_pair_request("m", "a", "b"))


# --------------------------------------------------------------------------- #
# extract_text / parse_message
# --------------------------------------------------------------------------- #


def test_extract_text_returns_first_text_block() -> None:
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text="first"),
            SimpleNamespace(type="text", text="second"),
        ]
    )
    assert extract_text(message) == "first"


def test_extract_text_accepts_dict_blocks() -> None:
    message = SimpleNamespace(content=[{"type": "text", "text": "hello"}])
    assert extract_text(message) == "hello"


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(content=[]),
        SimpleNamespace(content=None),
        SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={})]),
    ],
)
def test_extract_text_raises_without_a_text_block(message: Any) -> None:
    with pytest.raises(JudgeError, match="no text content block"):
        extract_text(message)


def test_parse_message_tolerates_markdown_fences() -> None:
    fenced = f"Here you go:\n```json\n{TRIAGE_JSON}\n```\nHope that helps!"
    verdict = parse_message(make_message(fenced), TriageVerdict)
    assert isinstance(verdict, TriageVerdict)
    assert verdict.category == "photo"
    assert verdict.technical_score == 6


def test_parse_message_raises_judge_error_on_garbage() -> None:
    with pytest.raises(JudgeError, match="could not parse TriageVerdict"):
        parse_message(make_message("no json here at all"), TriageVerdict)


def test_parse_message_raises_judge_error_on_schema_violation() -> None:
    bad = json.dumps(
        {
            "category": "photo",
            "verdict": "neutral",
            "technical_score": 42,
            "reasons": [],
            "confidence": "medium",
        }
    )
    with pytest.raises(JudgeError):
        parse_message(make_message(bad), TriageVerdict)


@pytest.mark.parametrize(
    ("payload", "model_cls"),
    [
        (TRIAGE_JSON, TriageVerdict),
        (BURST_JSON, BurstVerdict),
        (BWS_JSON, BWSVerdict),
        (PAIR_JSON, PairVerdict),
    ],
)
def test_parse_message_round_trips_every_verdict_model(
    payload: str, model_cls: type
) -> None:
    verdict = parse_message(make_message(payload), model_cls)
    assert isinstance(verdict, model_cls)


def test_parse_message_rejects_unknown_enum_value() -> None:
    bad = json.dumps(
        {
            "category": "photo",
            "verdict": "maybe",
            "technical_score": 5,
            "reasons": [],
            "confidence": "medium",
        }
    )
    with pytest.raises(JudgeError):
        parse_message(make_message(bad), TriageVerdict)


# --------------------------------------------------------------------------- #
# Judge — happy paths
# --------------------------------------------------------------------------- #


def test_triage_returns_verdict_usage_and_model() -> None:
    client = make_client(TRIAGE_JSON)
    result = Judge(client).triage("claude-haiku-4-5", "AAAA")
    assert isinstance(result.verdict, TriageVerdict)
    assert result.verdict.confidence == "medium"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "claude-haiku-4-5"
    assert len(client.messages.calls) == 1
    assert client.messages.calls[0]["system"] == TRIAGE_SYSTEM


def test_burst_returns_verdict_and_sends_all_images() -> None:
    client = make_client(BURST_JSON)
    result = Judge(client).burst("claude-haiku-4-5", ["a", "b", "c"])
    assert isinstance(result.verdict, BurstVerdict)
    assert result.verdict.best_index == 2
    assert result.verdict.reject_indices == [1, 3]
    assert len(image_blocks(client.messages.calls[0])) == 3


def test_bws_returns_verdict() -> None:
    client = make_client(BWS_JSON)
    result = Judge(client).bws("claude-sonnet-5", ["a", "b", "c"])
    assert isinstance(result.verdict, BWSVerdict)
    assert (result.verdict.best_index, result.verdict.worst_index) == (1, 3)
    assert result.model == "claude-sonnet-5"


@pytest.mark.parametrize("winner", ["A", "B", "tie"])
def test_pair_accepts_every_outcome(winner: str) -> None:
    client = make_client(json.dumps({"winner": winner, "note": "n"}))
    result = Judge(client).pair("claude-opus-5", "a", "b")
    assert isinstance(result.verdict, PairVerdict)
    assert result.verdict.winner == winner


def test_usage_is_read_from_the_message() -> None:
    message = make_message(TRIAGE_JSON, input_tokens=1234, output_tokens=77)
    client = SimpleNamespace(messages=StubMessages([message]))
    result = Judge(client).triage("m", "a")
    assert (result.input_tokens, result.output_tokens) == (1234, 77)


def test_missing_usage_degrades_to_zero() -> None:
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text=TRIAGE_JSON)])
    client = SimpleNamespace(messages=StubMessages([message]))
    result = Judge(client).triage("m", "a")
    assert (result.input_tokens, result.output_tokens) == (0, 0)


# --------------------------------------------------------------------------- #
# Judge — retry behaviour
# --------------------------------------------------------------------------- #


def test_invalid_json_then_valid_succeeds_with_two_calls() -> None:
    client = make_client("sorry, I can't do that", TRIAGE_JSON)
    result = Judge(client).triage("m", "a")
    assert isinstance(result.verdict, TriageVerdict)
    assert len(client.messages.calls) == 2
    assert client.messages.calls[0] == client.messages.calls[1]


def test_two_invalid_replies_raise_judge_error() -> None:
    client = make_client("nope", "still nope")
    with pytest.raises(JudgeError, match="no usable TriageVerdict after 2 attempts"):
        Judge(client).triage("m", "a")
    assert len(client.messages.calls) == 2


def test_refusal_fails_immediately_without_a_second_call() -> None:
    refused = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(input_tokens=9, output_tokens=0),
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )
    client = SimpleNamespace(messages=StubMessages([refused, make_message(TRIAGE_JSON)]))
    with pytest.raises(JudgeError, match="declined to judge"):
        Judge(client).triage("m", "a")
    assert len(client.messages.calls) == 1, "a refusal must not be retried verbatim"


def test_truncated_reply_fails_immediately_without_a_second_call() -> None:
    truncated = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"category": "pho')],
        usage=SimpleNamespace(input_tokens=9, output_tokens=1024),
        stop_reason="max_tokens",
    )
    client = SimpleNamespace(messages=StubMessages([truncated, make_message(TRIAGE_JSON)]))
    with pytest.raises(JudgeError, match="max_tokens"):
        Judge(client).triage("m", "a")
    assert len(client.messages.calls) == 1


def test_end_turn_stop_reason_is_untouched() -> None:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=TRIAGE_JSON)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = SimpleNamespace(messages=StubMessages([message]))
    assert isinstance(Judge(client).triage("m", "a").verdict, TriageVerdict)


def test_zero_retries_means_a_single_attempt() -> None:
    client = make_client("nope")
    with pytest.raises(JudgeError, match="after 1 attempts"):
        Judge(client, max_parse_retries=0).triage("m", "a")
    assert len(client.messages.calls) == 1


def test_extra_retries_are_honoured() -> None:
    client = make_client("nope", "nope", TRIAGE_JSON)
    result = Judge(client, max_parse_retries=2).triage("m", "a")
    assert isinstance(result.verdict, TriageVerdict)
    assert len(client.messages.calls) == 3


def test_bws_best_equals_worst_triggers_a_retry() -> None:
    degenerate = json.dumps({"best_index": 2, "worst_index": 2, "note": "oops"})
    client = make_client(degenerate, BWS_JSON)
    result = Judge(client).bws("m", ["a", "b", "c"])
    assert (result.verdict.best_index, result.verdict.worst_index) == (1, 3)
    assert len(client.messages.calls) == 2


def test_bws_best_equals_worst_twice_raises() -> None:
    degenerate = json.dumps({"best_index": 2, "worst_index": 2, "note": "oops"})
    client = make_client(degenerate, degenerate)
    with pytest.raises(JudgeError, match="no usable BWSVerdict"):
        Judge(client).bws("m", ["a", "b", "c"])
    assert len(client.messages.calls) == 2


def test_bws_out_of_range_index_is_rejected() -> None:
    out_of_range = json.dumps({"best_index": 9, "worst_index": 1, "note": "x"})
    client = make_client(out_of_range, BWS_JSON)
    result = Judge(client).bws("m", ["a", "b", "c"])
    assert result.verdict.best_index == 1
    assert len(client.messages.calls) == 2


def test_bws_zero_index_is_rejected_by_the_schema() -> None:
    client = make_client(json.dumps({"best_index": 0, "worst_index": 1, "note": "x"}))
    with pytest.raises(JudgeError):
        Judge(client, max_parse_retries=0).bws("m", ["a", "b"])


def test_burst_out_of_range_best_index_is_rejected() -> None:
    bad = json.dumps({"best_index": 7, "reject_indices": [1], "note": "x"})
    client = make_client(bad, bad)
    with pytest.raises(JudgeError, match="no usable BurstVerdict"):
        Judge(client).burst("m", ["a", "b", "c"])


def test_burst_out_of_range_reject_index_is_rejected() -> None:
    bad = json.dumps({"best_index": 1, "reject_indices": [2, 99], "note": "x"})
    client = make_client(bad, BURST_JSON)
    result = Judge(client).burst("m", ["a", "b", "c"])
    assert result.verdict.best_index == 2
    assert len(client.messages.calls) == 2


def test_burst_best_index_inside_reject_indices_is_rejected() -> None:
    bad = json.dumps({"best_index": 2, "reject_indices": [1, 2], "note": "x"})
    client = make_client(bad, bad)
    with pytest.raises(JudgeError):
        Judge(client).burst("m", ["a", "b", "c"])


def test_burst_allows_keeping_a_second_frame() -> None:
    lenient = json.dumps({"best_index": 1, "reject_indices": [2], "note": "3 differs"})
    client = make_client(lenient)
    result = Judge(client).burst("m", ["a", "b", "c"])
    assert result.verdict.reject_indices == [2]


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "custom_id",
    [
        "triage_9f8c1c2e-4b1a-4f6d-9a1e-0d1f2a3b4c5d",
        "burst_abc123",
        "bws_42",
        "pair_a1-b2_c3",
        "x",
        "z" * 64,
    ],
)
def test_valid_custom_ids_are_accepted(custom_id: str) -> None:
    assert CUSTOM_ID_PATTERN.match(custom_id)
    request = to_batch_request(custom_id, build_triage_request("m", "a"))
    assert request["custom_id"] == custom_id
    assert request["params"]["model"] == "m"
    assert request["params"]["max_tokens"] == TRIAGE_MAX_TOKENS
    assert request["params"]["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "custom_id",
    ["triage:abc", "", "z" * 65, "has space", "emoji_\U0001f600", "slash/id", "dot.id"],
)
def test_invalid_custom_ids_are_rejected(custom_id: str) -> None:
    assert not CUSTOM_ID_PATTERN.match(custom_id)
    with pytest.raises(JudgeError, match="invalid custom_id"):
        to_batch_request(custom_id, build_triage_request("m", "a"))


def test_batch_request_params_carry_no_forbidden_keys() -> None:
    request = to_batch_request("bws_1", build_bws_request("m", ["a", "b"]))
    keys = walk_keys(dict(request))
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in keys


def test_submit_batch_returns_the_batch_id() -> None:
    client, batches = make_batch_client(batch_id="msgbatch_xyz")
    requests = [
        to_batch_request("triage_1", build_triage_request("m", "a")),
        to_batch_request("triage_2", build_triage_request("m", "b")),
    ]
    assert submit_batch(client, requests) == "msgbatch_xyz"
    assert batches.created == [requests]


def test_submit_batch_rejects_an_empty_list() -> None:
    client, batches = make_batch_client()
    with pytest.raises(JudgeError, match="empty batch"):
        submit_batch(client, [])
    assert batches.created == []


def test_batch_status_reads_processing_status() -> None:
    client, batches = make_batch_client(processing_status="in_progress")
    assert batch_status(client, "msgbatch_1") == "in_progress"
    assert batches.retrieved == ["msgbatch_1"]


def test_iter_batch_results_splits_successes_from_failures() -> None:
    ok_message = make_message(TRIAGE_JSON)
    rows = [
        SimpleNamespace(
            custom_id="triage_1",
            result=SimpleNamespace(type="succeeded", message=ok_message),
        ),
        SimpleNamespace(
            custom_id="triage_2",
            result=SimpleNamespace(type="errored", error=SimpleNamespace(type="overloaded")),
        ),
        SimpleNamespace(custom_id="burst_3", result=SimpleNamespace(type="expired")),
        SimpleNamespace(custom_id="burst_4", result=SimpleNamespace(type="canceled")),
    ]
    client, batches = make_batch_client(results=rows)
    out = list(iter_batch_results(client, "msgbatch_1"))
    assert out[0] == ("triage_1", ok_message, None)
    # the API's own error type rides along so a re-submit can tell a permanent
    # invalid_request from a transient server error
    assert out[1] == ("triage_2", None, "errored:overloaded")
    assert out[2] == ("burst_3", None, "expired")
    assert out[3] == ("burst_4", None, "canceled")
    assert batches.results_for == ["msgbatch_1"]


def test_iter_batch_results_keeps_the_permanent_error_type() -> None:
    rows = [
        SimpleNamespace(
            custom_id="triage_9",
            result=SimpleNamespace(type="errored", error=SimpleNamespace(type="invalid_request")),
        )
    ]
    client, _ = make_batch_client(results=rows)
    assert list(iter_batch_results(client, "msgbatch_1")) == [
        ("triage_9", None, "errored:invalid_request")
    ]


def test_batch_results_can_be_parsed_with_parse_message() -> None:
    rows = [
        SimpleNamespace(
            custom_id="triage_1",
            result=SimpleNamespace(type="succeeded", message=make_message(TRIAGE_JSON)),
        )
    ]
    client, _ = make_batch_client(results=rows)
    (custom_id, message, error), = list(iter_batch_results(client, "msgbatch_1"))
    assert error is None
    assert custom_id == "triage_1"
    verdict = parse_message(message, TriageVerdict)
    assert isinstance(verdict, TriageVerdict)
