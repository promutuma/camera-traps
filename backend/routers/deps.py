"""FastAPI dependency helpers — inject AppState into route handlers."""

from fastapi import Request
from backend.models.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_current_user(request: Request) -> str:
    """Return the signed-in username from the session cookie, or anonymous."""
    username = request.session.get("username")
    if isinstance(username, str) and username.strip():
        return username.strip()
    return "anonymous"
