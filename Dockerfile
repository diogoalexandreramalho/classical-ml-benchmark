# Reproducible container for the data-science pipeline.
#
# Build:
#   docker build -t data-science .
#
# Reproduce the full report (writes to ./artifacts on the host):
#   docker run --rm -v "$(pwd)/artifacts:/app/artifacts" data-science make reproduce
#
# Drop into a shell:
#   docker run --rm -it -v "$(pwd)/artifacts:/app/artifacts" data-science bash

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# System dependencies:
# - curl + ca-certificates: install uv, fetch UCI data
# - git: pre-commit / version metadata
# - texlive-* + latexmk: compile reports/report.tex via `make report`
# - libgomp1: required by xgboost wheels
# - unrar-free / tar: extract the Parkinson's UCI zip (contains .rar)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        latexmk \
    && rm -rf /var/lib/apt/lists/*

# uv (Python env manager — matches the host workflow)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Install dependencies first (cached layer when only source changes)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Copy the rest of the project + install it
COPY . .
RUN uv sync --frozen

# Default: print make targets. Override with the docker run command.
CMD ["make", "help"]
