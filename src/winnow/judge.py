"""Claude-backed judging: prompts, request builders, and result parsing.

This module owns every prompt Winnow sends and every response it reads back.
It never talks to the network itself — callers pass in an ``anthropic.Anthropic``
client (or any object exposing the same ``messages.create`` /
``messages.batches`` surface), which keeps the whole module trivially testable.

Request builders return plain ``dict`` kwargs so the same payload can be sent
either synchronously (``client.messages.create(**kwargs)``) or as part of a
Batch API submission (:func:`to_batch_request`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel

from winnow.prompts import PromptLibrary, PromptTask
from winnow.schemas import (
    BurstVerdict,
    BWSVerdict,
    PairVerdict,
    TriageVerdict,
    output_schema,
    parse_verdict,
)

__all__ = [
    "BURST_SYSTEM",
    "BWS_SYSTEM",
    "FATAL_ERRORS",
    "ITEM_ERRORS",
    "MAX_BURST_IMAGES",
    "PAIR_SYSTEM",
    "RANK_EFFORT",
    "TRIAGE_SYSTEM",
    "Judge",
    "JudgeError",
    "JudgeResult",
    "batch_status",
    "build_burst_request",
    "build_bws_request",
    "build_pair_request",
    "build_triage_request",
    "extract_text",
    "iter_batch_results",
    "parse_message",
    "submit_batch",
    "to_batch_request",
]

#: ``max_tokens`` for the cheap stage-1 calls (Haiku triage and burst picks).
#: Haiku 4.5 does not think by default, so the whole budget is answer text.
TRIAGE_MAX_TOKENS = 1024
#: ``max_tokens`` for the deliberative stage-2/3 calls (Sonnet / Opus).
#: Those models run adaptive thinking by default and ``max_tokens`` caps
#: thinking *plus* the answer, so this needs real headroom above the handful
#: of tokens the JSON verdict itself costs.
RANK_MAX_TOKENS = 8192
#: ``output_config.effort`` for stage 2/3. These are one-line verdicts, not
#: deep reasoning tasks, and the default (``high``) would spend most of the
#: budget thinking. Never sent to Haiku, which rejects the field.
RANK_EFFORT = "low"

#: Hard ceiling on images in one burst request (the Messages API caps a
#: request at 100 images).
MAX_BURST_IMAGES = 100

#: Anthropic's constraint on Batch API custom ids.
CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_JPEG_MEDIA_TYPE = "image/jpeg"

#: Stop reasons that mean "this exact request will never work": retrying an
#: identical payload just doubles the bill.
_TERMINAL_STOP_REASONS: frozenset[str] = frozenset({"refusal", "max_tokens"})


class JudgeError(RuntimeError):
    """Raised when Claude's reply cannot be turned into a usable verdict."""


#: Errors a per-item pipeline handler should count and skip: one bad photo
#: must never abort a whole stage. ``anthropic.APIError`` is **not** an
#: ``OSError``, so the SDK classes have to be listed explicitly.
ITEM_ERRORS: tuple[type[BaseException], ...] = (
    JudgeError,
    OSError,
    ValueError,
    anthropic.APIStatusError,
    anthropic.APIConnectionError,
)

#: Errors that are never per-item: retrying the next photo cannot help, so
#: these propagate and end the stage.
FATAL_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
)


@dataclass
class JudgeResult:
    """A parsed verdict plus the usage accounting for the call that produced it."""

    verdict: Any
    input_tokens: int
    output_tokens: int
    model: str


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_DEFAULT_PROMPTS = PromptLibrary.default()

#: Default rendered system prompts (loaded from packaged Jinja2 templates).
TRIAGE_SYSTEM = _DEFAULT_PROMPTS.render("triage", model="", image_count=1)
BURST_SYSTEM = _DEFAULT_PROMPTS.render("burst", model="", image_count=2)
BWS_SYSTEM = _DEFAULT_PROMPTS.render("bws", model="", image_count=2)
PAIR_SYSTEM = _DEFAULT_PROMPTS.render("pair", model="", image_count=2)

_TRIAGE_INSTRUCTION = (
    "Judge this photograph: category, verdict, technical_score, reasons, confidence. "
    "Reply with JSON only."
)


def _burst_instruction(count: int) -> str:
    return (
        f"These {count} photos are one burst, shown in capture order as Photo 1 through "
        f"Photo {count}. Pick the single best frame and list the redundant ones. "
        "Reply with JSON only."
    )


def _bws_instruction(count: int) -> str:
    return (
        f"These {count} photos are unrelated, labeled Photo 1 through Photo {count}. "
        "Name the best and the worst of the set; they must be different photos. "
        "Reply with JSON only."
    )


_PAIR_INSTRUCTION = (
    "Which is the stronger photograph, A or B? Answer 'tie' only if they are truly "
    "inseparable. Reply with JSON only."
)


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #


def _image_block(image_b64: str) -> dict[str, Any]:
    """A single base64 JPEG image content block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _JPEG_MEDIA_TYPE,
            "data": image_b64,
        },
    }


def _labeled_images(
    images_b64: Sequence[str], labels: Sequence[str]
) -> list[dict[str, Any]]:
    """Interleave ``Photo <label>:`` text blocks with their image blocks."""
    content: list[dict[str, Any]] = []
    for label, image_b64 in zip(labels, images_b64, strict=True):
        content.append({"type": "text", "text": f"Photo {label}:"})
        content.append(_image_block(image_b64))
    return content


def _output_config(model_cls: type[BaseModel], effort: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {"format": {"type": "json_schema", "schema": output_schema(model_cls)}}
    if effort is not None:
        config["effort"] = effort
    return config


def _request(
    *,
    model: str,
    max_tokens: int,
    system: str,
    content: list[dict[str, Any]],
    model_cls: type[BaseModel],
    effort: str | None = None,
) -> dict[str, Any]:
    """Assemble ``messages.create`` kwargs.

    Deliberately omits ``temperature``, ``top_p``, ``top_k`` and ``thinking``:
    the stage-2/3 models reject sampling parameters outright, and their default
    adaptive thinking is what we want. That thinking is billed against
    ``max_tokens`` alongside the answer text, so stage 2/3 also pass a low
    ``effort`` — these are one-line verdicts, not deep reasoning tasks — and a
    budget with headroom. Haiku gets neither: it does not think by default and
    rejects ``effort`` outright.
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "output_config": _output_config(model_cls, effort),
    }


def build_triage_request(
    model: str, image_b64: str, *, system: str | None = None
) -> dict[str, Any]:
    """Kwargs for a single-photo stage-1 triage call."""
    return _request(
        model=model,
        max_tokens=TRIAGE_MAX_TOKENS,
        system=TRIAGE_SYSTEM if system is None else system,
        content=[_image_block(image_b64), {"type": "text", "text": _TRIAGE_INSTRUCTION}],
        model_cls=TriageVerdict,
    )


def build_burst_request(
    model: str, images_b64: list[str], *, system: str | None = None
) -> dict[str, Any]:
    """Kwargs for a pick-the-best call over a burst of near-identical frames.

    Raises:
        JudgeError: if the group is degenerate (fewer than two frames) or
            larger than :data:`MAX_BURST_IMAGES`, which the API would reject.
    """
    if len(images_b64) < 2:
        raise JudgeError(f"a burst needs at least 2 photos, got {len(images_b64)}")
    if len(images_b64) > MAX_BURST_IMAGES:
        raise JudgeError(
            f"a burst of {len(images_b64)} photos exceeds the {MAX_BURST_IMAGES}-image "
            "per-request limit"
        )
    labels = [str(i) for i in range(1, len(images_b64) + 1)]
    content = _labeled_images(images_b64, labels)
    content.append({"type": "text", "text": _burst_instruction(len(images_b64))})
    return _request(
        model=model,
        max_tokens=TRIAGE_MAX_TOKENS,
        system=BURST_SYSTEM if system is None else system,
        content=content,
        model_cls=BurstVerdict,
    )


def build_bws_request(
    model: str, images_b64: list[str], *, system: str | None = None
) -> dict[str, Any]:
    """Kwargs for a best-worst-scaling call over a set of unrelated photos."""
    if len(images_b64) < 2:
        raise JudgeError(f"a BWS set needs at least 2 photos, got {len(images_b64)}")
    labels = [str(i) for i in range(1, len(images_b64) + 1)]
    content = _labeled_images(images_b64, labels)
    content.append({"type": "text", "text": _bws_instruction(len(images_b64))})
    return _request(
        model=model,
        max_tokens=RANK_MAX_TOKENS,
        system=BWS_SYSTEM if system is None else system,
        content=content,
        model_cls=BWSVerdict,
        effort=RANK_EFFORT,
    )


def build_pair_request(
    model: str, image_a_b64: str, image_b_b64: str, *, system: str | None = None
) -> dict[str, Any]:
    """Kwargs for a finals head-to-head between photo A and photo B."""
    content = _labeled_images([image_a_b64, image_b_b64], ["A", "B"])
    content.append({"type": "text", "text": _PAIR_INSTRUCTION})
    return _request(
        model=model,
        max_tokens=RANK_MAX_TOKENS,
        system=PAIR_SYSTEM if system is None else system,
        content=content,
        model_cls=PairVerdict,
        effort=RANK_EFFORT,
    )


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def _block_field(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def extract_text(message: Any) -> str:
    """Return the text of the first ``text`` content block in ``message``.

    Raises:
        JudgeError: if the reply contains no text block.
    """
    for block in getattr(message, "content", None) or []:
        if _block_field(block, "type") == "text":
            return _block_field(block, "text") or ""
    raise JudgeError("reply contained no text content block")


def parse_message(message: Any, model_cls: type[BaseModel]) -> BaseModel:
    """Parse a Claude reply into ``model_cls``, tolerating fences and stray prose.

    Raises:
        JudgeError: if no valid instance of ``model_cls`` can be recovered.
    """
    text = extract_text(message)
    try:
        return parse_verdict(text, model_cls)
    except ValueError as exc:  # includes pydantic ValidationError
        raise JudgeError(f"could not parse {model_cls.__name__} from reply: {exc}") from exc


def _check_stop_reason(message: Any) -> None:
    """Reject a reply that stopped for a reason a retry cannot fix.

    A ``refusal`` or a ``max_tokens`` truncation would otherwise surface as
    "reply contained no text content block" and be retried with a byte-identical
    request, doubling the cost of a call that is guaranteed to fail again.

    Raises:
        JudgeError: on a terminal stop reason.
    """
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason not in _TERMINAL_STOP_REASONS:
        return
    if stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        suffix = f" (category {category})" if category else ""
        raise JudgeError(f"the model declined to judge this image{suffix}")
    raise JudgeError("reply hit max_tokens before the verdict was complete")


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage", None)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _check_indices(indices: Sequence[int], count: int, what: str) -> None:
    for index in indices:
        if not 1 <= index <= count:
            raise JudgeError(f"{what} {index} is out of range 1..{count}")


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


class Judge:
    """Runs judging calls against a Claude client, retrying unparseable replies."""

    def __init__(
        self,
        client: Any,
        *,
        max_parse_retries: int = 1,
        prompts: PromptLibrary | None = None,
    ) -> None:
        """Wrap ``client`` (anything with ``messages.create``).

        Args:
            client: An ``anthropic.Anthropic`` instance or compatible stub.
            max_parse_retries: Extra attempts after a parse or validation
                failure. The default of 1 means two calls at most.
            prompts: Optional compiled :class:`~winnow.prompts.PromptLibrary`
                used to render system prompts dynamically per request.
        """
        self.client = client
        self.max_parse_retries = max_parse_retries
        self.prompts = prompts

    def _resolve_system(
        self,
        task: PromptTask,
        model: str,
        image_count: int,
        system: str | None,
        prompt_context: Mapping[str, Any] | None,
    ) -> str | None:
        if system is not None:
            return system
        if self.prompts is not None:
            return self.prompts.render(
                task,
                model=model,
                image_count=image_count,
                **(prompt_context or {}),
            )
        return None

    def _run(
        self,
        kwargs: dict[str, Any],
        model_cls: type[BaseModel],
        validate: Callable[[Any], None] | None = None,
    ) -> JudgeResult:
        """Issue the request, parse the reply, and retry once on failure.

        Args:
            kwargs: Full ``messages.create`` kwargs from a request builder.
            model_cls: Verdict model the reply must satisfy.
            validate: Optional extra check applied to the parsed verdict; it
                should raise :class:`JudgeError` for degenerate answers.

        Raises:
            JudgeError: if every attempt fails to yield a usable verdict, or
                immediately (without retrying) when the reply stopped for a
                reason an identical request cannot recover from.
        """
        last_error: Exception | None = None
        for _ in range(self.max_parse_retries + 1):
            message = self.client.messages.create(**kwargs)
            _check_stop_reason(message)  # outside the retry: never re-sent
            try:
                verdict = parse_message(message, model_cls)
                if validate is not None:
                    validate(verdict)
            except JudgeError as exc:
                last_error = exc
                continue
            input_tokens, output_tokens = _usage(message)
            return JudgeResult(
                verdict=verdict,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=kwargs["model"],
            )
        raise JudgeError(
            f"no usable {model_cls.__name__} after {self.max_parse_retries + 1} attempts: "
            f"{last_error}"
        ) from last_error

    def triage(
        self,
        model: str,
        image_b64: str,
        *,
        system: str | None = None,
        prompt_context: Mapping[str, Any] | None = None,
    ) -> JudgeResult:
        """Judge a single photo; ``result.verdict`` is a ``TriageVerdict``."""
        resolved_system = self._resolve_system("triage", model, 1, system, prompt_context)
        return self._run(
            build_triage_request(model, image_b64, system=resolved_system),
            TriageVerdict,
        )

    def burst(
        self,
        model: str,
        images_b64: list[str],
        *,
        system: str | None = None,
        prompt_context: Mapping[str, Any] | None = None,
    ) -> JudgeResult:
        """Pick the best frame of a burst; ``result.verdict`` is a ``BurstVerdict``."""
        count = len(images_b64)

        def validate(verdict: BurstVerdict) -> None:
            _check_indices([verdict.best_index], count, "best_index")
            _check_indices(verdict.reject_indices, count, "reject index")
            if verdict.best_index in verdict.reject_indices:
                raise JudgeError(
                    f"best_index {verdict.best_index} also appears in reject_indices"
                )

        resolved_system = self._resolve_system("burst", model, count, system, prompt_context)
        return self._run(
            build_burst_request(model, images_b64, system=resolved_system),
            BurstVerdict,
            validate,
        )

    def bws(
        self,
        model: str,
        images_b64: list[str],
        *,
        system: str | None = None,
        prompt_context: Mapping[str, Any] | None = None,
    ) -> JudgeResult:
        """Name best and worst of a set; ``result.verdict`` is a ``BWSVerdict``."""
        count = len(images_b64)

        def validate(verdict: BWSVerdict) -> None:
            _check_indices([verdict.best_index], count, "best_index")
            _check_indices([verdict.worst_index], count, "worst_index")
            if verdict.best_index == verdict.worst_index:
                raise JudgeError(
                    f"best_index and worst_index are both {verdict.best_index}"
                )

        resolved_system = self._resolve_system("bws", model, count, system, prompt_context)
        return self._run(
            build_bws_request(model, images_b64, system=resolved_system),
            BWSVerdict,
            validate,
        )

    def pair(
        self,
        model: str,
        a_b64: str,
        b_b64: str,
        *,
        system: str | None = None,
        prompt_context: Mapping[str, Any] | None = None,
    ) -> JudgeResult:
        """Judge A vs B; ``result.verdict`` is a ``PairVerdict``."""
        resolved_system = self._resolve_system("pair", model, 2, system, prompt_context)
        return self._run(
            build_pair_request(model, a_b64, b_b64, system=resolved_system),
            PairVerdict,
        )


def resolve_prompts(judge: Any, settings: Any) -> PromptLibrary:
    """Return ``judge.prompts`` when available, otherwise load from ``settings``."""
    if isinstance(judge, Judge) and judge.prompts is not None:
        return judge.prompts
    return PromptLibrary.from_settings(settings)


# --------------------------------------------------------------------------- #
# Batch API helpers
# --------------------------------------------------------------------------- #


def to_batch_request(custom_id: str, kwargs: dict[str, Any]) -> Request:
    """Wrap builder kwargs as a Batch API request.

    Raises:
        JudgeError: if ``custom_id`` violates Anthropic's
            ``^[a-zA-Z0-9_-]{1,64}$`` constraint (note that ``:`` is not
            allowed, hence Winnow's ``kind_id`` convention).
    """
    if not CUSTOM_ID_PATTERN.match(custom_id):
        raise JudgeError(
            f"invalid custom_id {custom_id!r}: must match ^[a-zA-Z0-9_-]{{1,64}}$"
        )
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**kwargs))


def submit_batch(client: Any, requests: list[Request]) -> str:
    """Submit ``requests`` as one batch and return the batch id."""
    if not requests:
        raise JudgeError("cannot submit an empty batch")
    batch = client.messages.batches.create(requests=requests)
    return str(getattr(batch, "id", None) or batch["id"])


def batch_status(client: Any, batch_id: str) -> str:
    """Return the batch's ``processing_status`` (``"ended"`` when complete)."""
    batch = client.messages.batches.retrieve(batch_id)
    return str(getattr(batch, "processing_status", None) or batch["processing_status"])


def iter_batch_results(
    client: Any, batch_id: str
) -> Iterator[tuple[str, Any | None, str | None]]:
    """Yield ``(custom_id, message_or_None, error_kind_or_None)`` per batch row.

    ``error_kind`` is the result type for anything that did not succeed —
    ``"errored"``, ``"canceled"`` or ``"expired"`` — suffixed with the API's
    own error type when it has one (``"errored:invalid_request"``). That
    distinction is what tells a later re-submit whether the request needs
    fixing or was merely unlucky.
    """
    for row in client.messages.batches.results(batch_id):
        result = row.result
        kind = getattr(result, "type", None)
        if kind == "succeeded":
            yield row.custom_id, result.message, None
            continue
        detail = getattr(getattr(result, "error", None), "type", None)
        yield row.custom_id, None, f"{kind}:{detail}" if detail else str(kind)
