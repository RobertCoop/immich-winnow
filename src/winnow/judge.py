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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel

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

#: Shared closing line, appended to every system prompt.
_JSON_ONLY = (
    "Reply with JSON matching the schema. JSON only — no prose, no markdown "
    "fences, no commentary before or after.\n"
)

TRIAGE_SYSTEM = """\
You are the culling eye of a working photographer, reviewing someone's PERSONAL
photo library: family, friends, pets, travel, ordinary days. Your job is to be
ruthless but fair. Hide the failures, flag the standouts, and leave the large
ordinary middle alone.

CATEGORIES — judge what the image IS, not how useful it is.
- photo: an actual photograph, made by pointing a camera or phone at the world.
- screenshot: a capture of a screen — app UI, web page, chat thread, map, ticket.
- document: a photo or scan whose point is the information on it — receipts,
  forms, whiteboards, book pages, labels, business cards, signage.
- meme: image macros, reaction images, saved social graphics, wallpapers, and
  anything else downloaded rather than taken.
- other: illustrations, renders, generated images, or anything fitting none of
  the above.
Screenshots, documents and memes are NOT photographs.

A non-photo category means the image will be ARCHIVED — hidden from the main
timeline. Apply the same caution you would to a reject: when you are not sure
an image is a screenshot, document, meme or other, call it a photo. Archiving
something its owner made, was sent, or scanned on purpose is the expensive
mistake, and a scanned drawing or a shared family picture is a photograph as
far as this decision is concerned.

VERDICTS
- reject: the frame failed badly enough that its owner would want it hidden.
  Badly blurred or smeared by motion; hopelessly under- or over-exposed with no
  recoverable detail; an accidental shot (pocket, floor, ceiling, lens cap, a
  thumb over the lens); eyes closed or mid-blink on EVERY person in the frame;
  or otherwise unusable. A merely mediocre photo is NOT a reject.
- candidate: a standout worth cherishing. Sharp where it matters, well composed,
  and emotionally engaging — a photo its owner would print, share, or hang.
- neutral: everything else. Most photographs are neutral. That is the correct
  answer far more often than either extreme.

TECHNICAL_SCORE — craft only, 0 to 10, ignoring how much you like the subject:
  0-2  unusable: hopeless blur, black or blown-out frame, accidental capture
  3-4  poor: subject soft, badly lit, careless or distracting framing
  5-6  ordinary: correctly exposed and acceptably sharp, and nothing more
  7    good: clean focus, decent light, deliberate composition
  8    very good: real strength in light or composition
  9    excellent: several strengths at once — a photograph, not a snapshot
  10   exceptional: light, moment and composition all land together. Rare.

CONFIDENCE
Use "high" ONLY when the call is unmistakable: a plainly ruined frame, an
obvious screenshot, an undeniably beautiful image. Use "medium" when the
evidence is clear but a thoughtful person could disagree. Use "low" when you
are essentially guessing.

SENTIMENTAL VALUE IS UNKNOWABLE TO YOU. A dim, crooked snapshot may be the only
surviving picture of someone who has died; a flawless landscape may mean
nothing at all. When a photo is technically weak but holds people, pets, or a
moment that clearly mattered to somebody, lean neutral with low or medium
confidence instead of rejecting it. Reserve high-confidence rejects for frames
whose failure is purely technical and total.

REASONS
Give one to three short, concrete phrases naming what you actually see:
"subject motion blur", "backlit, face in shadow", "both children looking away",
"clean rim light on the dog". Do not hedge and do not restate the verdict.

CAPTION AND KEYWORDS
Also describe the photo for search. caption: one plain factual sentence, at
most ~120 characters, naming the subject, setting and action — never start
with "A photo of" or "An image of". keywords: three to six lowercase words or
short phrases a person would search for (subjects, scene, activity, notable
objects; e.g. "beach", "golden retriever", "birthday cake"). No hashtags, no
duplicates of the caption verbatim, no quality judgments.

""" + _JSON_ONLY

BURST_SYSTEM = """\
You are picking the single best frame out of a burst: a run of near-identical
photographs of the same subject, taken seconds apart, from someone's PERSONAL
photo library. One frame survives as the keeper; the rest are hidden as
redundant duplicates.

The photos are labeled "Photo 1", "Photo 2", and so on, in the order they were
taken. Compare them against each other — not against some ideal — in this order
of importance:

1. SHARPNESS ON THE SUBJECT. The eyes, the face, the point of interest. A frame
   that is sharp where it matters beats one that is merely sharp somewhere.
2. EXPRESSIONS AND EYES. Eyes open, no mid-blink, no mid-word grimace. With
   several people in frame, favor the shot where the most of them look good at
   the same time.
3. THE MOMENT. Peak action, the fuller laugh, the gesture completed rather than
   half-formed.
4. COMPOSITION AND FRAMING. Cleaner edges, less clutter, better balance, nothing
   important cropped off.
5. EXPOSURE AND COLOR, as a tiebreaker.

best_index is the 1-based label of the keeper.
reject_indices lists the 1-based labels of the other photos in the burst —
normally every one of them except the keeper. Leave a second frame out of
reject_indices only when it is genuinely a different picture worth keeping on
its own merits (a different subject, a different moment), never merely because
it was a close second.

note is one sentence naming what decided it, e.g. "3 is the only frame with
both children's eyes open".

Be decisive: exactly one best_index, chosen even when the frames are close.

""" + _JSON_ONLY

BWS_SYSTEM = """\
You are ranking photographs from someone's PERSONAL photo library by naming the
two extremes of a set: the single best photo and the single worst.

The photos are labeled "Photo 1", "Photo 2", and so on. They are usually
unrelated — often different subjects, days and places — so weigh each on its
own merits and then compare:

- Is the subject sharp and clearly rendered?
- Do the light and exposure serve the picture, or fight it?
- Is the composition deliberate — clean edges, balance, nothing important cut
  off or competing for attention?
- Does the moment land? Expression, gesture, timing, atmosphere.
- Would its owner stop scrolling on it?

Judge the photograph, not the subject: a beautifully made picture of something
dull beats a clumsy picture of something charming.

best_index is the one you would put in front of someone first.
worst_index is the one you would drop first.
Both are 1-based labels within this set, and they MUST be different photos.

Be decisive. When two are close, choose one anyway and say why in a single
sentence in note. A tie is not an available answer.

""" + _JSON_ONLY

PAIR_SYSTEM = """\
You are judging a head-to-head between two photographs from someone's PERSONAL
photo library, shown as "Photo A" and "Photo B".

Both have already survived earlier rounds, so assume both are good. Decide which
is the stronger photograph:

- Sharpness where it counts — the eyes, the face, the subject.
- Light: its quality and direction, and whether it shapes the subject or
  flattens it.
- Composition: framing, balance, clean edges, an uncluttered background.
- The moment: expression, gesture, timing, the feeling it carries.
- Overall impact — which one you would still remember an hour from now.

winner is "A", "B", or "tie". Use "tie" ONLY when the two are genuinely
inseparable in quality; a small but real preference is a win, not a tie.

note is one sentence naming the deciding difference, e.g. "B — the light on her
face is softer and the background is far less busy".

""" + _JSON_ONLY

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


def build_triage_request(model: str, image_b64: str) -> dict[str, Any]:
    """Kwargs for a single-photo stage-1 triage call."""
    return _request(
        model=model,
        max_tokens=TRIAGE_MAX_TOKENS,
        system=TRIAGE_SYSTEM,
        content=[_image_block(image_b64), {"type": "text", "text": _TRIAGE_INSTRUCTION}],
        model_cls=TriageVerdict,
    )


def build_burst_request(model: str, images_b64: list[str]) -> dict[str, Any]:
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
        system=BURST_SYSTEM,
        content=content,
        model_cls=BurstVerdict,
    )


def build_bws_request(model: str, images_b64: list[str]) -> dict[str, Any]:
    """Kwargs for a best-worst-scaling call over a set of unrelated photos."""
    if len(images_b64) < 2:
        raise JudgeError(f"a BWS set needs at least 2 photos, got {len(images_b64)}")
    labels = [str(i) for i in range(1, len(images_b64) + 1)]
    content = _labeled_images(images_b64, labels)
    content.append({"type": "text", "text": _bws_instruction(len(images_b64))})
    return _request(
        model=model,
        max_tokens=RANK_MAX_TOKENS,
        system=BWS_SYSTEM,
        content=content,
        model_cls=BWSVerdict,
        effort=RANK_EFFORT,
    )


def build_pair_request(model: str, image_a_b64: str, image_b_b64: str) -> dict[str, Any]:
    """Kwargs for a finals head-to-head between photo A and photo B."""
    content = _labeled_images([image_a_b64, image_b_b64], ["A", "B"])
    content.append({"type": "text", "text": _PAIR_INSTRUCTION})
    return _request(
        model=model,
        max_tokens=RANK_MAX_TOKENS,
        system=PAIR_SYSTEM,
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

    def __init__(self, client: Any, *, max_parse_retries: int = 1) -> None:
        """Wrap ``client`` (anything with ``messages.create``).

        Args:
            client: An ``anthropic.Anthropic`` instance or compatible stub.
            max_parse_retries: Extra attempts after a parse or validation
                failure. The default of 1 means two calls at most.
        """
        self.client = client
        self.max_parse_retries = max_parse_retries

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

    def triage(self, model: str, image_b64: str) -> JudgeResult:
        """Judge a single photo; ``result.verdict`` is a ``TriageVerdict``."""
        return self._run(build_triage_request(model, image_b64), TriageVerdict)

    def burst(self, model: str, images_b64: list[str]) -> JudgeResult:
        """Pick the best frame of a burst; ``result.verdict`` is a ``BurstVerdict``."""
        count = len(images_b64)

        def validate(verdict: BurstVerdict) -> None:
            _check_indices([verdict.best_index], count, "best_index")
            _check_indices(verdict.reject_indices, count, "reject index")
            if verdict.best_index in verdict.reject_indices:
                raise JudgeError(
                    f"best_index {verdict.best_index} also appears in reject_indices"
                )

        return self._run(build_burst_request(model, images_b64), BurstVerdict, validate)

    def bws(self, model: str, images_b64: list[str]) -> JudgeResult:
        """Name best and worst of a set; ``result.verdict`` is a ``BWSVerdict``."""
        count = len(images_b64)

        def validate(verdict: BWSVerdict) -> None:
            _check_indices([verdict.best_index], count, "best_index")
            _check_indices([verdict.worst_index], count, "worst_index")
            if verdict.best_index == verdict.worst_index:
                raise JudgeError(
                    f"best_index and worst_index are both {verdict.best_index}"
                )

        return self._run(build_bws_request(model, images_b64), BWSVerdict, validate)

    def pair(self, model: str, a_b64: str, b_b64: str) -> JudgeResult:
        """Judge A vs B; ``result.verdict`` is a ``PairVerdict``."""
        return self._run(build_pair_request(model, a_b64, b_b64), PairVerdict)


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
