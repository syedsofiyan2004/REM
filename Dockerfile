# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
ARG APP_BUILD=unknown
ENV APP_BUILD=${APP_BUILD}

# System deps (keep minimal)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app

EXPOSE 8000

# Workers configurable with UVICORN_WORKERS (default 2). Keep-alive a bit higher for ALB.
ENV UVICORN_WORKERS=2
CMD ["sh", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --timeout-keep-alive 75"]
