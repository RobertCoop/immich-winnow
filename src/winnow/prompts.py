"""Jinja2 system prompt loading, rendering, validation, and export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import jinja2
from jinja2 import (
    ChoiceLoader,
    FileSystemLoader,
    PackageLoader,
    PrefixLoader,
    StrictUndefined,
    Template,
    TemplateError,
)
from jinja2.sandbox import SandboxedEnvironment

from winnow.config import (
    DEFAULT_FINALS_MODEL,
    DEFAULT_RANK_MODEL,
    DEFAULT_TRIAGE_MODEL,
    Settings,
)

__all__ = [
    "DEFAULT_TEMPLATE_FILES",
    "PROMPT_TASKS",
    "PromptError",
    "PromptLibrary",
    "PromptTask",
    "build_prompt_context",
    "photo_context",
    "safe_settings_dict",
    "sample_context",
]

PromptTask = Literal["triage", "burst", "bws", "pair"]

PROMPT_TASKS: tuple[PromptTask, ...] = ("triage", "burst", "bws", "pair")

DEFAULT_TEMPLATE_FILES: dict[str, str] = {
    "triage": "triage.j2",
    "burst": "burst.j2",
    "bws": "bws.j2",
    "pair": "pair.j2",
    "_json_only": "_json_only.j2",
}

_PHOTO_FIELDS: tuple[str, ...] = (
    "id",
    "filename",
    "taken_at",
    "camera",
    "width",
    "height",
    "immich_rating",
    "immich_description",
    "burst_id",
)

# Explicit allowlist so API keys, URLs, and credentials are never reachable inside templates.
_SAFE_SETTINGS_FIELDS: tuple[str, ...] = (
    "triage_model",
    "rank_model",
    "finals_model",
    "image_edge",
    "jpeg_quality",
    "burst_gap_seconds",
    "burst_max_group",
    "dhash_max_distance",
    "bws_set_size",
    "bws_appearances",
    "finals_rounds",
    "finals_pool_size",
    "five_star_fraction",
    "four_star_fraction",
    "full_star_spectrum",
    "candidate_score_min",
    "scoring_limit",
    "best_album",
    "best_album_min_stars",
    "five_star_favorite",
    "write_captions",
    "keyword_tags",
    "keyword_tag_prefix",
)


class PromptError(RuntimeError):
    """Raised when a system prompt template cannot be loaded or rendered."""


def photo_context(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract safe photo metadata fields from an asset row (or ``None``)."""
    if row is None:
        return {field: None for field in _PHOTO_FIELDS}
    return {field: row.get(field) for field in _PHOTO_FIELDS}


def safe_settings_dict(settings: Any | None) -> dict[str, Any]:
    """Return a copy of allowlisted non-secret settings fields for templates."""
    defaults = {name: Settings.model_fields[name].default for name in _SAFE_SETTINGS_FIELDS}
    if settings is None:
        return defaults
    return {
        name: getattr(settings, name, defaults[name])
        for name in _SAFE_SETTINGS_FIELDS
    }


def build_prompt_context(
    task: PromptTask,
    *,
    model: str,
    image_count: int,
    settings: Any | None = None,
    batch_size: int | None = None,
    total_photos: int | None = None,
    library_size: int | None = None,
    photos: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the Jinja2 variable context for rendering one system prompt."""
    photo_list = [photo_context(p) for p in photos] if photos else []
    resolved_batch = image_count if batch_size is None else int(batch_size)
    resolved_total = image_count if total_photos is None else int(total_photos)
    resolved_library = resolved_total if library_size is None else int(library_size)
    return {
        "task": task,
        "model": model,
        "image_count": image_count,
        "batch_size": resolved_batch,
        "total_photos": resolved_total,
        "library_size": resolved_library,
        "photos": photo_list,
        "photo": photo_list[0] if task == "triage" and photo_list else None,
        "photo_a": photo_list[0] if task == "pair" and len(photo_list) >= 1 else None,
        "photo_b": photo_list[1] if task == "pair" and len(photo_list) >= 2 else None,
        "settings": safe_settings_dict(settings),
    }


def _sample_photo(index: int) -> dict[str, Any]:
    return {
        "id": f"00000000-0000-0000-0000-{index:012d}",
        "filename": f"DSC_{index:04d}.JPG",
        "taken_at": f"2024-06-01T12:00:{index:02d}+00:00",
        "camera": "NIKON CORPORATION|NIKON Z 9",
        "width": 6000,
        "height": 4000,
        "immich_rating": None,
        "immich_description": None,
        "burst_id": "burst001" if index <= 3 else None,
    }


def sample_context(task: PromptTask, settings: Any | None = None) -> dict[str, Any]:
    """Build a realistic sample context for validating or previewing ``task``."""
    safe = safe_settings_dict(settings)
    if task == "triage":
        return build_prompt_context(
            "triage",
            model=str(safe.get("triage_model") or DEFAULT_TRIAGE_MODEL),
            image_count=1,
            settings=settings,
            batch_size=25,
            total_photos=30,
            library_size=100,
            photos=[_sample_photo(1)],
        )
    if task == "burst":
        count = 3
        return build_prompt_context(
            "burst",
            model=str(safe.get("triage_model") or DEFAULT_TRIAGE_MODEL),
            image_count=count,
            settings=settings,
            batch_size=5,
            total_photos=15,
            library_size=100,
            photos=[_sample_photo(i) for i in range(1, count + 1)],
        )
    if task == "bws":
        count = int(safe.get("bws_set_size") or 8)
        return build_prompt_context(
            "bws",
            model=str(safe.get("rank_model") or DEFAULT_RANK_MODEL),
            image_count=count,
            settings=settings,
            batch_size=10,
            total_photos=20,
            library_size=100,
            photos=[_sample_photo(i) for i in range(1, count + 1)],
        )
    return build_prompt_context(
        "pair",
        model=str(safe.get("finals_model") or DEFAULT_FINALS_MODEL),
        image_count=2,
        settings=settings,
        batch_size=12,
        total_photos=24,
        library_size=100,
        photos=[_sample_photo(1), _sample_photo(2)],
    )


def _pkg_loader() -> PackageLoader:
    return PackageLoader("winnow", "templates")


def _new_environment(loader: jinja2.BaseLoader) -> SandboxedEnvironment:
    return SandboxedEnvironment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


@cache
def _default_template(task: PromptTask) -> Template:
    pkg = _pkg_loader()
    loader = ChoiceLoader([PrefixLoader({"default": pkg}), pkg])
    return _new_environment(loader).get_template(DEFAULT_TEMPLATE_FILES[task])


def _load_override_template(task: PromptTask, override: Path | str) -> tuple[Template, Path]:
    path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise PromptError(f"{task} prompt template not found or not a file: {override}")
    pkg = _pkg_loader()
    # Keep PrefixLoader({"default": ...}) first so "default/..." never collides
    # with a "default" subdirectory in a user's bind mount.
    loader = ChoiceLoader(
        [
            PrefixLoader({"default": pkg}),
            FileSystemLoader(str(path.parent)),
            pkg,
        ]
    )
    env = _new_environment(loader)
    try:
        template = env.get_template(path.name)
    except (TemplateError, OSError) as exc:
        raise PromptError(f"failed to load {task} prompt template {path}: {exc}") from exc
    return template, path


class PromptLibrary:
    """Compiled system prompt templates for Winnow's four judge tasks."""

    def __init__(
        self,
        templates: Mapping[PromptTask, Template],
        sources: Mapping[PromptTask, Path | None],
        settings: Any | None = None,
    ) -> None:
        self._templates = dict(templates)
        self._sources = dict(sources)
        self.settings = settings

    @classmethod
    @cache
    def default(cls) -> PromptLibrary:
        """Return a cached library using only packaged default templates."""
        templates = {task: _default_template(task) for task in PROMPT_TASKS}
        sources: dict[PromptTask, Path | None] = {task: None for task in PROMPT_TASKS}
        return cls(templates, sources, settings=None)

    @classmethod
    def from_settings(cls, settings: Any | None) -> PromptLibrary:
        """Load active templates from ``settings``, compile them, and fail-fast validate."""
        templates: dict[PromptTask, Template] = {}
        sources: dict[PromptTask, Path | None] = {}
        for task in PROMPT_TASKS:
            override = (
                getattr(settings, f"{task}_prompt_path", None) if settings is not None else None
            )
            if override:
                template, resolved = _load_override_template(task, override)
                templates[task] = template
                sources[task] = resolved
            else:
                templates[task] = _default_template(task)
                sources[task] = None
        library = cls(templates, sources, settings=settings)
        library.validate_all()
        return library

    def source(self, task: PromptTask) -> Path | None:
        """Return the custom template path for ``task``, or ``None`` if using default."""
        return self._sources.get(task)

    def render(
        self,
        task: PromptTask,
        *,
        model: str,
        image_count: int,
        batch_size: int | None = None,
        total_photos: int | None = None,
        library_size: int | None = None,
        photos: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Render the system prompt for ``task`` using the compiled template."""
        if task not in self._templates:
            raise PromptError(f"unknown prompt task: {task!r}")
        ctx = build_prompt_context(
            task,
            model=model,
            image_count=image_count,
            settings=self.settings,
            batch_size=batch_size,
            total_photos=total_photos,
            library_size=library_size,
            photos=photos,
        )
        src = self._sources.get(task)
        label = str(src) if src is not None else f"default ({DEFAULT_TEMPLATE_FILES[task]})"
        try:
            return self._templates[task].render(ctx)
        except TemplateError as exc:
            raise PromptError(f"failed to render {task} prompt template ({label}): {exc}") from exc

    def validate_all(self) -> None:
        """Test-render every task template with realistic sample data to catch errors early."""
        for task in PROMPT_TASKS:
            ctx = sample_context(task, self.settings)
            src = self._sources.get(task)
            label = str(src) if src is not None else f"default ({DEFAULT_TEMPLATE_FILES[task]})"
            try:
                self._templates[task].render(ctx)
            except TemplateError as exc:
                raise PromptError(
                    f"invalid {task} prompt template ({label}): {exc}"
                ) from exc

    @staticmethod
    def export_defaults(dest: Path | str, *, force: bool = False) -> list[Path]:
        """Write packaged default ``.j2`` files into ``dest`` as editable starter templates."""
        target_dir = Path(dest)
        if target_dir.exists() and not target_dir.is_dir():
            raise PromptError(f"export target is not a directory: {target_dir}")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PromptError(f"could not create export directory {target_dir}: {exc}") from exc

        pkg_root = files("winnow").joinpath("templates")
        filenames = list(DEFAULT_TEMPLATE_FILES.values())
        if not force:
            existing = [target_dir / name for name in filenames if (target_dir / name).exists()]
            if existing:
                raise PromptError(
                    f"refusing to overwrite existing template(s) without --force: {existing[0]}"
                )

        written: list[Path] = []
        for name in filenames:
            content = pkg_root.joinpath(name).read_text(encoding="utf-8")
            out_path = target_dir / name
            try:
                out_path.write_text(content, encoding="utf-8")
            except OSError as exc:
                raise PromptError(f"failed to write {out_path}: {exc}") from exc
            written.append(out_path)
        return written
