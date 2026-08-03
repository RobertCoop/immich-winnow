"""Runtime configuration, loaded from environment variables and `.env`."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TRIAGE_MODEL = "claude-haiku-4-5"
DEFAULT_RANK_MODEL = "claude-sonnet-5"
DEFAULT_FINALS_MODEL = "claude-opus-5"


class Settings(BaseSettings):
    """All knobs for a Winnow run.

    Required values (no defaults): ``IMMICH_URL``, ``IMMICH_API_KEY``,
    ``ANTHROPIC_API_KEY``. Everything else has sensible defaults and can be
    overridden via environment variables of the same (upper-cased) name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- connections ---
    immich_url: str
    immich_api_key: str
    anthropic_api_key: str

    # --- storage ---
    db_path: Path = Path("winnow.db")
    cache_dir: Path = Path(".winnow-cache")

    # --- models per stage ---
    triage_model: str = DEFAULT_TRIAGE_MODEL
    rank_model: str = DEFAULT_RANK_MODEL
    finals_model: str = DEFAULT_FINALS_MODEL

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

    @property
    def immich_base(self) -> str:
        """Immich base URL without a trailing slash."""
        return self.immich_url.rstrip("/")


def load_settings(**overrides: object) -> Settings:
    """Load settings from the environment / .env, applying overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]
