"""
In-memory background job tracker for long-running image processing tasks.
Each job progresses: queued → running → done | error
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict


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


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    def create(self) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


# Singleton used across the app
job_manager = JobManager()
