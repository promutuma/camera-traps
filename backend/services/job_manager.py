"""
In-memory background job tracker for long-running image processing tasks.
Each job progresses: queued → running → done | error

Completed/failed jobs are evicted after JOB_TTL_HOURS (default 2 h).
Their temp directories are deleted at eviction time.
"""

from __future__ import annotations
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict

_TTL_SECONDS = int(os.environ.get("JOB_TTL_HOURS", "2")) * 3600


@dataclass
class Job:
    job_id: str
    status: str = "queued"   # queued | running | done | error
    total: int = 0
    completed: int = 0
    results: List[Any] = field(default_factory=list)
    scrub_audit: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    temp_dir: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


class JobManager:
    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._jobs: Dict[str, Job] = {}
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            jid for jid, job in self._jobs.items()
            if job.status in ("done", "error")
            and job.finished_at is not None
            and (now - job.finished_at) > self._ttl
        ]
        for jid in expired:
            self._delete_temp(self._jobs[jid])
            del self._jobs[jid]

    def _delete_temp(self, job: Job) -> None:
        if job.temp_dir:
            shutil.rmtree(job.temp_dir, ignore_errors=True)
            job.temp_dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self) -> Job:
        self._evict_expired()
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        self._evict_expired()
        return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._delete_temp(self._jobs[job_id])
            del self._jobs[job_id]
            return True
        return False


# Singleton used across the app
job_manager = JobManager()
