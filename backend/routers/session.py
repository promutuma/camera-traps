"""Username-only session endpoints (no password)."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/session", tags=["session"])


class SessionCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)


class SessionResponse(BaseModel):
    username: str | None
    created_at: str | None = None


@router.get("", response_model=SessionResponse)
def get_session(request: Request):
    username = request.session.get("username")
    return SessionResponse(
        username=username,
        created_at=request.session.get("created_at"),
    )


@router.post("", response_model=SessionResponse)
def set_session(request: Request, body: SessionCreate):
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    request.session["username"] = username
    request.session["created_at"] = datetime.now(timezone.utc).isoformat()
    return SessionResponse(username=username, created_at=request.session["created_at"])


@router.delete("")
def clear_session(request: Request):
    request.session.clear()
    return {"ok": True}
