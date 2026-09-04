FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY alembic.ini ./
COPY migrations ./migrations
COPY . .

EXPOSE 3000

# Migrations own the schema: bring a fresh database up to head, then serve.
CMD ["sh", "-c", "uv run alembic upgrade head && uv run python app.py"]
