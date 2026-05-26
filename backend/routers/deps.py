"""FastAPI dependency helpers — inject AppState into route handlers."""

from fastapi import Request
from backend.models.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
