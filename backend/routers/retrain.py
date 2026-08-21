"""
HITL retraining.
POST /api/retrain/run              trigger a retrain run from accumulated corrections
GET  /api/retrain/status           poll the latest run (JSON)
GET  /api/retrain/stream           real-time SSE for the latest run
GET  /api/retrain/job/{id}         poll a specific run
GET  /api/retrain/history          past runs (persisted, survives restarts)
GET  /api/retrain/preview          how many new corrections are available right now
POST /api/retrain/activate/{id}    roll back/promote a previous run's model
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import RetrainJobStatus, RetrainTriggerRequest
from backend.routers.deps import get_state
from backend.services.retrain_job_manager import retrain_job_manager, RetrainJob

router = APIRouter(prefix="/retrain", tags=["retrain"])


def _to_status(job: RetrainJob) -> RetrainJobStatus:
    return RetrainJobStatus(
        job_id=job.job_id,
        status=job.status,
        current_step=job.current_step,
        method=job.method,
        dataset_size=job.dataset_size,
        metrics=job.metrics,
        model_version=job.model_version,
        activated=job.activated,
        triggered_by=job.triggered_by,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


def _run_retrain(job_id: str, state: AppState) -> None:
    job = retrain_job_manager.get(job_id)
    if not job or not state.retrain_engine:
        return
    try:
        state.retrain_engine.run(job, state.speciesnet_model)
    finally:
        if state.db_manager:
            state.db_manager.save_retrain_run(job)
        # Apply the newly-activated correction layer immediately, no restart needed.
        if job.status == "done" and job.activated and state.speciesnet_model:
            artifact = state.retrain_engine.load_artifact(job.job_id)
            if artifact:
                state.speciesnet_model.set_correction_layer(artifact)
            if state.db_manager:
                state.db_manager.activate_retrain_run(job.job_id)


@router.get("/preview")
def preview(state: AppState = Depends(get_state)):
    """How many un-trained-on corrections exist right now."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    rows = state.db_manager.get_correction_training_data()
    species = sorted({r["corrected_species"] for r in rows if r.get("corrected_species")})
    return {"available_corrections": len(rows), "distinct_species": len(species)}


@router.post("/run")
def run(
    req: RetrainTriggerRequest,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_state),
):
    if not state.retrain_engine:
        raise HTTPException(status_code=503, detail="Retrain engine not ready")
    if retrain_job_manager.is_active():
        raise HTTPException(status_code=409, detail="A retrain run is already in progress")

    job = retrain_job_manager.create(triggered_by=req.reviewer_id)
    background_tasks.add_task(_run_retrain, job.job_id, state)
    return {"job_id": job.job_id, "status": "started"}


@router.get("/status", response_model=RetrainJobStatus)
def status():
    job = retrain_job_manager.latest()
    if not job:
        raise HTTPException(status_code=404, detail="No retrain runs yet")
    return _to_status(job)


@router.get("/job/{job_id}", response_model=RetrainJobStatus)
def job_status(job_id: str):
    job = retrain_job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _to_status(job)


@router.get("/stream")
async def stream(cursor: int = 0):
    """SSE stream of the latest retrain run's progress."""
    async def event_gen():
        last_step = None
        while True:
            job = retrain_job_manager.latest()
            if not job:
                yield _sse({"type": "progress", "status": "error", "error": "No retrain runs yet"})
                return

            if job.current_step != last_step:
                last_step = job.current_step
                yield _sse({
                    "type": "progress",
                    "job_id": job.job_id,
                    "status": job.status,
                    "current_step": job.current_step,
                    "error": job.error,
                })

            if job.status in ("done", "error"):
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/history")
def history(state: AppState = Depends(get_state), limit: int = 50):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    return state.db_manager.load_recent_retrain_runs(limit=limit)


@router.post("/activate/{job_id}")
def activate(job_id: str, state: AppState = Depends(get_state)):
    """Roll back to (or re-promote) a previous retrain run's model."""
    if not state.db_manager or not state.retrain_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    artifact = state.retrain_engine.load_artifact(job_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="No saved model for that run")
    if state.speciesnet_model:
        state.speciesnet_model.set_correction_layer(artifact)
    state.db_manager.activate_retrain_run(job_id)
    return {"ok": True, "activated": job_id}


@router.post("/deactivate")
def deactivate(state: AppState = Depends(get_state)):
    """Turn off the correction layer entirely, reverting to raw SpeciesNet output."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    if state.speciesnet_model:
        state.speciesnet_model.set_correction_layer(None)
    state.db_manager.activate_retrain_run("")  # deactivates all (no matching job_id)
    return {"ok": True}
