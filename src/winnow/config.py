"""Runtime configuration, loaded from environment variables and `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TRIAGE_MODEL = "claude-haiku-4-5"
DEFAULT_RANK_MODEL = "claude-sonnet-5"
DEFAULT_FINALS_MODEL = "claude-opus-5"

Effort = Literal["none", "low", "medium", "high"]
StageName = Literal["triage", "rank", "finals"]

EFFORT_THINKING_TOKENS: dict[Effort, int] = {
    "none": 0,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


class Settings(BaseSettings):
    """All knobs for a Winnow run.

    Required values (no defaults): ``IMMICH_URL``, ``IMMICH_API_KEY``, and
    either ``ANTHROPIC_API_KEY`` or ``OPENAI_BASE_URL``. Everything else has
    sensible defaults and can be overridden via environment variables of the
    same (upper-cased) name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- connections ---
    immich_url: str
    immich_api_key: str
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = Field(default=None, validate_default=True)

    @field_validator("anthropic_api_key")
    @classmethod
    def _require_llm_credentials(cls, v: str | None, info: ValidationInfo) -> str | None:
        if not v and not info.data.get("openai_base_url"):
            raise ValueError("ANTHROPIC_API_KEY is required unless OPENAI_BASE_URL is set")
        return v

    @property
    def use_openai(self) -> bool:
        """True when configured to talk to an OpenAI-compatible endpoint."""
        return bool(self.openai_base_url)

    # --- storage ---
    db_path: Path = Path("winnow.db")
    cache_dir: Path = Path(".winnow-cache")

    # --- models per stage ---
    triage_model: str = DEFAULT_TRIAGE_MODEL
    rank_model: str = DEFAULT_RANK_MODEL
    finals_model: str = DEFAULT_FINALS_MODEL

    # --- thinking effort & token budgets per stage ---
    triage_effort: Effort = "low"
    rank_effort: Effort = "medium"
    finals_effort: Effort = "high"
    # Optional explicit cap on thinking tokens per stage (0 disables thinking).
    # When None, derived from the stage's effort via EFFORT_THINKING_TOKENS.
    triage_thinking_tokens: int | None = None
    rank_thinking_tokens: int | None = None
    finals_thinking_tokens: int | None = None
    # Reserved output token budget for the JSON verdict (added to thinking_tokens
    # when computing the total API max_tokens so thinking never starves output).
    triage_max_tokens: int = 1024
    rank_max_tokens: int = 1024
    finals_max_tokens: int = 1024

    @field_validator("triage_thinking_tokens", "rank_thinking_tokens", "finals_thinking_tokens")
    @classmethod
    def _non_negative_thinking(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("thinking token budget must be >= 0")
        return v

    @field_validator("triage_max_tokens", "rank_max_tokens", "finals_max_tokens")
    @classmethod
    def _min_output_tokens(cls, v: int) -> int:
        if v < 128:
            raise ValueError("reserved output max_tokens must be >= 128")
        return v

    def thinking_tokens_for(self, stage: StageName) -> int:
        """Resolved thinking token budget for ``stage``."""
        explicit: int | None = getattr(self, f"{stage}_thinking_tokens")
        if explicit is not None:
            return explicit
        effort: Effort = getattr(self, f"{stage}_effort")
        return EFFORT_THINKING_TOKENS[effort]

    def effort_for(self, stage: StageName) -> Effort:
        """Resolved effort label for ``stage`` (``'none'`` when thinking budget is 0)."""
        if self.thinking_tokens_for(stage) == 0:
            return "none"
        effort: Effort = getattr(self, f"{stage}_effort")
        return "low" if effort == "none" else effort

    def api_max_tokens_for(self, stage: StageName) -> int:
        """Total ``max_tokens`` sent to the API (thinking budget + reserved answer budget)."""
        output_budget = int(getattr(self, f"{stage}_max_tokens"))
        return self.thinking_tokens_for(stage) + output_budget

    def stage_request_kwargs(self, stage: StageName) -> dict[str, int | str | None]:
        """Keyword overrides (``max_tokens``, ``effort``, ``thinking_tokens``) for ``stage``."""
        effort = self.effort_for(stage)
        if self.use_openai:
            return {
                "max_tokens": self.api_max_tokens_for(stage),
                "effort": effort,
                "thinking_tokens": self.thinking_tokens_for(stage),
            }
        model = str(getattr(self, f"{stage}_model"))
        anth_effort = None if (effort == "none" or model.startswith("claude-haiku")) else effort
        max_tokens = (
            self.api_max_tokens_for(stage)
            if anth_effort is not None
            else int(getattr(self, f"{stage}_max_tokens"))
        )
        return {
            "max_tokens": max_tokens,
            "effort": anth_effort,
            "thinking_tokens": None,
        }

    # --- image preparation ---
    image_edge: int = 768
    jpeg_quality: int = 82

    # --- burst detection ---
    burst_gap_seconds: float = 10.0
    burst_max_group: int = 10
    dhash_max_distance: int = 26

    # --- stage 2: best-worst scaling ---
    bws_set_size: int = 8
    bws_appearances: int = 4

    # --- stage 3: finals ---
    finals_rounds: int = 3
    # Floor for how many top photos play Opus head-to-heads; grows
    # automatically to twice the five-star target.
    finals_pool_size: int = 50

    # --- star bands (fractions of the *scored candidate* pool) ---
    # Five stars: Opus-refined crown. Four stars: next band, straight from
    # the ranking evidence. On a library where ~8% of photos become
    # candidates, the defaults work out to roughly the top 0.4% / next 1.2%
    # of the whole library.
    five_star_fraction: float = 0.05
    four_star_fraction: float = 0.15
    # Also assign 3 stars (remaining ranked candidates), 2 stars (ordinary
    # keepers) and 1 star (poor but kept) — the full spectrum. Off by
    # default: it rates nearly every photo, ending the "middle is untouched"
    # guarantee, so switch it on deliberately.
    full_star_spectrum: bool = False

    # --- thresholds ---
    candidate_score_min: int = 8
    # Max already-scored photos re-judged as ranking anchors per run; every
    # new photo always enters scoring. 0 = unlimited.
    scoring_limit: int = 0

    # --- write-back behavior ---
    # Album to collect the best photos into ("" disables). Created if missing.
    best_album: str = "Five-Stars"
    # Minimum star band that lands in the album (5 = five-star only, 4 = both).
    best_album_min_stars: int = 5
    # Whether five-star photos are also marked as Immich favorites.
    five_star_favorite: bool = True
    # Write each photo's caption into its Immich description — only when the
    # description is empty, so your own words are never overwritten.
    write_captions: bool = True
    # Also write keywords as Immich tags under keyword_tag_prefix. Off by
    # default: a big library mints hundreds of tags in the tag browser.
    keyword_tags: bool = False
    keyword_tag_prefix: str = "kw"

    @property
    def immich_base(self) -> str:
        """Immich base URL without a trailing slash."""
        return self.immich_url.rstrip("/")


def load_settings(_env_file: Path | str | None = None, **overrides: object) -> Settings:
    """Load settings from the environment / .env, applying overrides."""
    if _env_file is not None:
        return Settings(_env_file=_env_file, **overrides)  # type: ignore[arg-type]
    return Settings(**overrides)  # type: ignore[arg-type]
