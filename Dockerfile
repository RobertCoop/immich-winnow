# ---- build: resolve the locked environment with uv -------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---- runtime: slim image, non-root, /data volume for ledger + reports ------
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Winnow" \
      org.opencontainers.image.description="AI photo culling for Immich — hide the rejects, crown the keepers." \
      org.opencontainers.image.source="https://github.com/robertcoop/immich-winnow" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd -r winnow && useradd -r -g winnow -d /data -s /usr/sbin/nologin winnow \
    && mkdir -p /data && chown winnow:winnow /data

# The venv keeps its build-time path so entry-point shebangs stay valid.
COPY --from=build /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    DB_PATH=/data/winnow.db \
    CACHE_DIR=/data/cache \
    PYTHONUNBUFFERED=1

WORKDIR /data
USER winnow
VOLUME /data

ENTRYPOINT ["winnow"]
CMD ["--help"]
