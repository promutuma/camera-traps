"""
In-memory background job tracker for HITL retrain runs.
Mirrors backend/services/job_manager.py's Job/JobManager shape so the two
systems stay consistent, but tracks retrain-specific fields (method, dataset
size, validation metrics, activated model version) instead of per-image
upload state.

Each job progresses: queued -> running -> done | error
Only one retrain job may be queued/running at a time (enforced by the router,
mirroring the upload semaphore in backend/routers/images.py).
"""

from __future__ import annotations
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

_TTL_SECONDS = int(os.environ.get("RETRAIN_JOB_TTL_HOURS", "24")) * 3600


@dataclass
class RetrainJob:
    job_id: str
    status: str = "queued"   # queued | running | done | error
    current_step: str = "Queued"
    method: Optional[str] = None       # correction_layer | full_finetune | export_only | none
    dataset_size: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_version: Optional[str] = None
    activated: bool = False
    triggered_by: str = "anonymous"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


class RetrainJobManager:
    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._jobs: Dict[str, RetrainJob] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _evict_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                jid for jid, job in self._jobs.items()
                if job.status in ("done", "error")
                and job.finished_at is not None
                and (now - job.finished_at) > self._ttl
            ]
            for jid in expired:
                self._jobs.pop(jid, None)

    def create(self, triggered_by: str = "anonymous") -> RetrainJob:
        self._evict_expired()
        job_id = str(uuid.uuid4())
        job = RetrainJob(job_id=job_id, triggered_by=triggered_by)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[RetrainJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> Optional[RetrainJob]:
        """Most recently created job (queued/running/done/error), if any."""
        with self._lock:
            if not self._jobs:
                return None
            return max(self._jobs.values(), key=lambda j: j.created_at)

    def is_active(self) -> bool:
        """True if any job is currently queued or running (singleton guard)."""
        with self._lock:
            return any(j.status in ("queued", "running") for j in self._jobs.values())

    def all(self) -> list:
        with self._lock:
            return list(self._jobs.values())


# Singleton used across the app
retrain_job_manager = RetrainJobManager()
