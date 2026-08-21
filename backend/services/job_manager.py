"""
In-memory background job tracker for long-running image processing tasks.
Each job progresses: queued → running → done | error

Completed/failed jobs are evicted after JOB_TTL_HOURS (default 2 h).
Their temp directories are deleted at eviction time.
"""

from __future__ import annotations
import os
import shutil
import threading
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
    # Per-image model events — appended during processing, read by SSE stream.
    # List append and slice are GIL-atomic in CPython, but callers that need a
    # consistent snapshot should use job_manager's lock or read under their own
    # synchronisation.
    model_events: List[Dict] = field(default_factory=list)
    # SHA-256 hashes computed from raw bytes at upload time (filename → hex).
    # Carried into _run_processing so the post-processing step skips a disk re-read.
    file_hashes: Dict[str, str] = field(default_factory=dict)


class JobManager:
    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._jobs: Dict[str, Job] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove finished jobs older than TTL. Filesystem cleanup runs outside
        the lock so it doesn't block concurrent readers."""
        now = time.time()
        to_delete: List[Job] = []
        with self._lock:
            expired = [
                jid for jid, job in self._jobs.items()
                if job.status in ("done", "error")
                and job.finished_at is not None
                and (now - job.finished_at) > self._ttl
            ]
            for jid in expired:
                to_delete.append(self._jobs.pop(jid))
        for job in to_delete:
            self._delete_temp(job)

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
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        self._evict_expired()
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        job = None
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs.pop(job_id)
        if job is not None:
            self._delete_temp(job)
            return True
        return False


# Singleton used across the app
job_manager = JobManager()
