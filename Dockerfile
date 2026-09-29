# syntax=docker/dockerfile:1
# Enable BuildKit for cache mounts: DOCKER_BUILDKIT=1 docker compose build

# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + built frontend ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for OpenCV / EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — heavy layer; cache pip downloads across rebuilds
COPY requirements.txt .
COPY backend/requirements.txt ./backend/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r backend/requirements.txt

# Application source (small — changes often; keep after deps for cache hits)
COPY core/ ./core/
COPY backend/ ./backend/

# DB and uploads are bind-mounted at runtime (see docker-compose.yml)
RUN mkdir -p /app/data/uploads

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/api/config/status || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
