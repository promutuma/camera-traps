"""
Tab 1 — Upload & Process
POST /api/images/upload            save files immediately, return job_id
POST /api/images/process/{id}      start background AI pipeline
GET  /api/images/job/{id}          poll progress (JSON)
GET  /api/images/job/{id}/stream   real-time SSE (progress + per-model events)
GET  /api/images/results/{id}      get finished results
GET  /api/images/stored/{filename} serve persistent image file
GET  /api/images/file/{id}/{name}  serve temp image file
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_ALLOWED_MAGIC: List[tuple] = [
    (b"\xff\xd8\xff",),
    (b"\x89PNG\r\n\x1a\n",),
    (b"II\x2a\x00", b"MM\x00\x2a"),
    (b"BM",),
    (b"GIF87a", b"GIF89a"),
    (b"RIFF",),
]


def _is_allowed_image(data: bytes) -> bool:
    """Return True if data starts with a known image magic-byte sequence."""
    for signatures in _ALLOWED_MAGIC:
        if any(data.startswith(sig) for sig in signatures):
            if data.startswith(b"RIFF") and data[8:12] != b"WEBP":
                continue
            return True
    return False


from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import JobStatus
from backend.routers.deps import get_state
from backend.services.job_manager import job_manager

router = APIRouter(prefix="/images", tags=["images"])

_job_semaphore = threading.Semaphore(1)
_PARALLEL_IMAGES = max(1, (os.cpu_count() or 4) // 2)

db_path = os.environ.get("DB_PATH", "wildlife_data.db")
UPLOADS_DIR = Path(db_path).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}


def _safe_filename(name: str) -> str:
    name = Path(name or "upload").name
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "upload"


def _unique_storage_name(base_name: str, file_hash: str, batch_names: set) -> str:
    """Return a collision-safe filename within this batch and on disk."""
    safe = _safe_filename(base_name)
    if safe not in batch_names and not (UPLOADS_DIR / safe).is_file():
        return safe
    stem = Path(safe).stem or "upload"
    ext = Path(safe).suffix or ".jpg"
    suffix = file_hash[:8]
    candidate = f"{stem}_{suffix}{ext}"
    counter = 1
    while candidate in batch_names or (UPLOADS_DIR / candidate).is_file():
        candidate = f"{stem}_{suffix}_{counter}{ext}"
        counter += 1
    return candidate


def _ingest_image_bytes(
    job,
    fname: str,
    contents: bytes,
    state: AppState,
    batch_names: set,
    batch_hashes: set,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Validate, deduplicate, and persist one image for a job.
    Returns (dest_path, safe_name, skip_reason).
    skip_reason is 'rejected', 'duplicate', or None on success.
    """
    if len(contents) > _MAX_UPLOAD_BYTES:
        return None, None, "rejected"
    if not _is_allowed_image(contents):
        return None, None, "rejected"

    file_hash = hashlib.sha256(contents).hexdigest()
    if file_hash in batch_hashes:
        return None, None, "duplicate"
    if state.db_manager and state.db_manager.image_hash_exists(file_hash):
        return None, None, "duplicate"

    batch_hashes.add(file_hash)
    safe_name = _unique_storage_name(fname, file_hash, batch_names)
    batch_names.add(safe_name)
    job.file_hashes[safe_name] = file_hash

    persistent = UPLOADS_DIR / safe_name
    with open(persistent, "wb") as f:
        f.write(contents)

    return str(persistent), safe_name, None


@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...), state: AppState = Depends(get_state)):
    """Save uploaded files; return job_id immediately so SSE can attach."""
    job = job_manager.create()
    duplicates = []
    rejected = []
    batch_hashes: set = set()
    batch_names: set = set()

    for upload in files:
        contents = await upload.read()
        fname = upload.filename or "upload"
        dest, safe_name, skip = _ingest_image_bytes(
            job, fname, contents, state, batch_names, batch_hashes,
        )
        if skip == "rejected":
            rejected.append(fname if len(contents) <= _MAX_UPLOAD_BYTES else f"{fname} (too large)")
            continue
        if skip == "duplicate":
            duplicates.append(fname)
            continue
        job.image_paths.append(dest)

    job.total = len(job.image_paths)

    if not job.image_paths:
        job_manager.delete(job.job_id)
        response: dict = {"file_count": 0}
        if duplicates:
            response["duplicates_skipped"] = duplicates
        if rejected:
            response["rejected"] = rejected
        return response

    response = {"job_id": job.job_id, "file_count": len(job.image_paths)}
    if duplicates:
        response["duplicates_skipped"] = duplicates
    if rejected:
        response["rejected"] = rejected
    return response


@router.post("/pipeline/start")
def start_pipeline(
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_state),
    expected_total: int = Query(..., ge=1, le=50000),
    station_id: Optional[str] = None,
    camera_id: Optional[str] = None,
):
    """Create a pipelined job and start the analysis worker before uploads finish."""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="AI models not loaded yet")

    job = job_manager.create()
    job.pipeline = True
    job.expected_total = expected_total
    job.total = expected_total
    job._work_queue = queue.Queue()

    if station_id and camera_id and state.station_manager:
        try:
            state.station_manager.set_current_camera(station_id, camera_id)
        except Exception:
            logger.warning("Could not sync camera assignment for %s/%s", station_id, camera_id)

    background_tasks.add_task(_run_pipeline, job.job_id, state, station_id or None, camera_id or None)
    return {"job_id": job.job_id, "status": "started", "expected_total": expected_total}


@router.post("/pipeline/{job_id}/file")
async def pipeline_upload_file(
    job_id: str,
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
):
    """Upload a single file into a running pipeline job; analysis begins immediately."""
    job = job_manager.get(job_id)
    if not job or not job.pipeline or not job._work_queue:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    if job.uploads_finished:
        raise HTTPException(status_code=409, detail="Uploads already closed for this job")
    if job.status in ("done", "done_with_errors", "error"):
        raise HTTPException(status_code=409, detail=f"Job already {job.status}")

    contents = await file.read()
    fname = file.filename or "upload"

    with job._pipeline_lock:
        dest, safe_name, skip = _ingest_image_bytes(
            job,
            fname,
            contents,
            state,
            job._job_names,
            job._job_hashes,
        )
        if skip == "rejected":
            reason = "too large" if len(contents) > _MAX_UPLOAD_BYTES else "unsupported type"
            return {"ok": False, "skipped": True, "reason": reason, "filename": fname}
        if skip == "duplicate":
            return {"ok": False, "skipped": True, "reason": "duplicate", "filename": fname}

        idx = len(job.image_paths)
        job.image_paths.append(dest)
        job.uploaded += 1

    job._work_queue.put(("image", idx, dest, safe_name))
    return {"ok": True, "skipped": False, "filename": fname, "safe_name": safe_name, "index": idx}


@router.post("/pipeline/{job_id}/finish")
def pipeline_finish_uploads(job_id: str):
    """Signal that no more files will be uploaded for this pipeline job."""
    job = job_manager.get(job_id)
    if not job or not job.pipeline:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    if job.uploads_finished:
        return {"ok": True, "uploaded": job.uploaded, "already_finished": True}

    job.uploads_finished = True
    if job._work_queue:
        job._work_queue.put(("poison",))
    return {"ok": True, "uploaded": job.uploaded, "already_finished": False}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _scrubbed_fallback_path(safe_name: str) -> Optional[Path]:
    """Return scrubbed copy path when the original is missing."""
    scrubbed = UPLOADS_DIR / "scrubbed" / safe_name
    return scrubbed if scrubbed.is_file() else None


def _resolve_image_path(
    image_id: int,
    filename: str,
    file_hash: Optional[str],
) -> Optional[Path]:
    """Resolve the on-disk file for a DB image row."""
    safe_name = _safe_filename(filename)
    candidate = UPLOADS_DIR / safe_name
    if candidate.is_file():
        if file_hash:
            try:
                if _sha256_file(candidate) == file_hash:
                    return candidate
            except OSError:
                pass
        else:
            return candidate

    if file_hash:
        suffix = file_hash[:8]
        try:
            for path in UPLOADS_DIR.iterdir():
                if not path.is_file() or path.name.startswith("."):
                    continue
                if suffix not in path.stem:
                    continue
                try:
                    if _sha256_file(path) == file_hash:
                        return path
                except OSError:
                    continue
        except OSError:
            pass

    if candidate.is_file():
        return candidate
    return _scrubbed_fallback_path(safe_name)


def _display_file_for_image(state: AppState, record: dict, original: Path) -> Path:
    """Return scrubbed copy when enabled, or scrubbed-only fallback."""
    safe_name = _safe_filename(record["filename"])
    if state.scrubber and state.config.enable_scrubbing:
        return state.scrubber.resolve_display_file(
            original,
            safe_name,
            scrubbing_enabled=True,
        )
    if not original.is_file():
        scrubbed = _scrubbed_fallback_path(safe_name)
        if scrubbed:
            return scrubbed
        if state.scrubber:
            scrubbed = state.scrubber.scrubbed_path_for_basename(safe_name)
            if scrubbed:
                return scrubbed
    return original


@router.get("/stored/{filename}")
def serve_stored_image(filename: str, state: AppState = Depends(get_state)):
    safe_name = _safe_filename(filename)
    file_path = UPLOADS_DIR / safe_name
    if not file_path.is_file():
        scrubbed = _scrubbed_fallback_path(safe_name)
        if not scrubbed:
            raise HTTPException(status_code=404, detail="Image not found")
        file_path = scrubbed
    elif state.scrubber and state.config.enable_scrubbing:
        file_path = state.scrubber.resolve_display_file(
            file_path, safe_name, scrubbing_enabled=True,
        )
    return FileResponse(str(file_path), headers=_CACHE_HEADERS)


@router.get("/stored-by-id/{image_id}")
def serve_stored_image_by_id(image_id: int, state: AppState = Depends(get_state)):
    """Serve image file resolved from database image PK."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    record = state.db_manager.get_image_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")
    original = _resolve_image_path(image_id, record["filename"], record.get("file_hash"))
    if not original:
        raise HTTPException(status_code=404, detail="Image file not found")
    display_path = _display_file_for_image(state, record, original)
    return FileResponse(str(display_path), headers=_CACHE_HEADERS)


def _serve_thumbnail_resolved(cache_key: str, original: Path, w: int) -> FileResponse:
    from PIL import Image as _PILImage

    if not original.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    thumb_dir = UPLOADS_DIR / "thumbs" / str(w)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{cache_key}.jpg"

    if not thumb_path.is_file():
        try:
            img = _PILImage.open(original)
            img.thumbnail((w, w), _PILImage.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(thumb_path, "JPEG", quality=82, optimize=True)
        except Exception:
            return FileResponse(str(original), headers=_CACHE_HEADERS)

    return FileResponse(str(thumb_path), media_type="image/jpeg", headers=_CACHE_HEADERS)


@router.get("/thumb/{filename}")
def serve_thumbnail(
    filename: str,
    w: int = Query(800, ge=64, le=2560),
    state: AppState = Depends(get_state),
):
    """Return a cached JPEG thumbnail; uses scrubbed copy when scrubbing is enabled."""
    safe_name = _safe_filename(filename)
    original = UPLOADS_DIR / safe_name
    if not original.is_file():
        scrubbed = _scrubbed_fallback_path(safe_name)
        if not scrubbed:
            raise HTTPException(status_code=404, detail="Image not found")
        original = scrubbed
    elif state.scrubber and state.config.enable_scrubbing:
        original = state.scrubber.resolve_display_file(
            original, safe_name, scrubbing_enabled=True,
        )
    cache_key = safe_name
    if state.scrubber and state.config.enable_scrubbing:
        scrubbed = state.scrubber.scrubbed_path_for_basename(safe_name)
        if scrubbed:
            cache_key = f"scrub_{safe_name}"
    elif not (UPLOADS_DIR / safe_name).is_file() and _scrubbed_fallback_path(safe_name):
        cache_key = f"scrub_{safe_name}"
    return _serve_thumbnail_resolved(cache_key, original, w)


@router.get("/thumb-by-id/{image_id}")
def serve_thumbnail_by_id(
    image_id: int,
    w: int = Query(800, ge=64, le=2560),
    state: AppState = Depends(get_state),
):
    """Serve thumbnail resolved from database image PK."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    record = state.db_manager.get_image_record(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")
    original = _resolve_image_path(image_id, record["filename"], record.get("file_hash"))
    if not original:
        raise HTTPException(status_code=404, detail="Image file not found")
    display_path = _display_file_for_image(state, record, original)
    safe_name = _safe_filename(record["filename"])
    cache_key = f"{image_id}_{safe_name}"
    if state.scrubber and state.config.enable_scrubbing:
        scrubbed = state.scrubber.scrubbed_path_for_basename(safe_name)
        if scrubbed:
            cache_key = f"{image_id}_scrub_{safe_name}"
    return _serve_thumbnail_resolved(cache_key, display_path, w)


@router.get("/file/{job_id}/{filename}")
def serve_image(job_id: str, filename: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_name = _safe_filename(filename)
    for path in job.image_paths:
        if os.path.basename(path) == safe_name and Path(path).is_file():
            return FileResponse(str(path))
    file_path = UPLOADS_DIR / safe_name
    if file_path.is_file():
        return FileResponse(str(file_path))
    scrubbed = _scrubbed_fallback_path(safe_name)
    if scrubbed:
        return FileResponse(str(scrubbed))
    raise HTTPException(status_code=404, detail="File not found")


def _build_processor(state: AppState, station_id: Optional[str], camera_id: Optional[str]):
    """Shared setup for batch and pipeline workers."""
    cfg = state.config
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.animal_detector import AnimalDetector
    from core.image_processor import ImageProcessor

    if state.md_model:
        state.md_model.set_confidence_threshold(cfg.detection_confidence)
    if state.dn_model:
        state.dn_model.brightness_threshold = cfg.brightness_threshold

    animal_detector = AnimalDetector(
        megadetector=state.md_model,
        confidence_threshold=cfg.detection_confidence,
        speciesnet=state.speciesnet_model,
    )
    processor = ImageProcessor(
        ocr_processor=state.ocr_model,
        animal_detector=animal_detector,
        day_night_classifier=state.dn_model,
        ocr_enabled=cfg.enable_ocr,
        detection_enabled=cfg.enable_detection,
        day_night_enabled=cfg.enable_day_night,
        ocr_strip_percent=cfg.ocr_strip_height,
    )

    station_lat: Optional[float] = None
    station_lon: Optional[float] = None
    if station_id and state.station_manager:
        try:
            stations_df = state.station_manager.get_stations()
            match = stations_df[stations_df["station_id"] == station_id]
            if not match.empty:
                lat = match.iloc[0].get("gps_lat")
                lon = match.iloc[0].get("gps_lon")
                if lat is not None and lon is not None and not (lat == 0 and lon == 0):
                    station_lat = float(lat)
                    station_lon = float(lon)
        except Exception as exc:
            logger.warning("Could not resolve station coordinates for %s: %s", station_id, exc)

    return processor, cfg, station_lat, station_lon


def _save_image_result(job, job_id: str, state: AppState, cfg, image_name: str, result_list: list) -> None:
    if not state.db_manager or not result_list:
        return
    import pandas as pd

    try:
        saved = state.db_manager.save_results(pd.DataFrame(result_list))
        img_id = saved.get("image_id") if isinstance(saved, dict) else None
        if img_id:
            img_id = int(img_id)
            for r in result_list:
                r["image_id"] = img_id
        if state.file_manager and img_id:
            best = max(
                result_list,
                key=lambda r: r.get("detection_confidence", 0.0) or 0.0,
            )
            state.file_manager.update_image_with_file_info(
                img_id,
                str(UPLOADS_DIR / image_name),
                float(best.get("detection_confidence", 0.0) or 0.0),
                detected_animal=best.get("detected_animal", ""),
                precomputed_hash=job.file_hashes.get(image_name),
            )
        if cfg.enable_scrubbing and state.scrubber and result_list:
            persistent_path = str(UPLOADS_DIR / image_name)
            for r in result_list:
                r["filepath"] = persistent_path
                if "primary_label" not in r and r.get("detected_animal"):
                    r["primary_label"] = r["detected_animal"]
            try:
                audit_entries = state.scrubber.scrub_batch(pd.DataFrame(result_list))
                if audit_entries:
                    job.scrub_audit.extend(_json_safe(audit_entries))
                    if cfg.delete_original_after_scrub:
                        for entry in audit_entries:
                            if entry.get("scrubbed") != "Yes":
                                continue
                            scrubbed_path = entry.get("scrubbed_path")
                            if not scrubbed_path or not Path(scrubbed_path).is_file():
                                continue
                            original_path = UPLOADS_DIR / image_name
                            if original_path.is_file():
                                try:
                                    original_path.unlink()
                                    logger.info(
                                        "Deleted original after scrub (job %s): %s",
                                        job_id,
                                        image_name,
                                    )
                                except OSError as del_exc:
                                    logger.warning(
                                        "Could not delete original after scrub %s: %s",
                                        image_name,
                                        del_exc,
                                    )
            except Exception as scrub_exc:
                logger.warning("Privacy scrub failed for %s: %s", image_name, scrub_exc)
    except Exception as db_exc:
        logger.error("Failed to persist result for %s (job %s): %s", image_name, job_id, db_exc)


def _apply_result_row_metadata(result_list: list, cfg, station_id: Optional[str], camera_id: Optional[str]) -> None:
    for r in result_list:
        if station_id:
            r["station_id"] = station_id
        elif not r.get("station_id"):
            r["station_id"] = cfg.default_station_id
        if camera_id:
            r["camera_id"] = camera_id


def _handle_image_result(
    job,
    job_id: str,
    state: AppState,
    cfg,
    idx: int,
    image_path: str,
    result_list: list,
    results_by_idx: dict,
    image_errors: list,
    write_lock: threading.Lock,
    station_id: Optional[str],
    camera_id: Optional[str],
) -> None:
    image_events = result_list[0].pop("_model_events", []) if result_list else []
    image_name = os.path.basename(image_path)

    if any(r.get("detected_animal") == "Error" for r in result_list):
        err_msg = result_list[0].get("processing_status", "Processing error")
        with write_lock:
            image_errors.append(f"{image_name}: {err_msg}")
            job.model_events.append({
                "type": "model_event",
                "image": image_name,
                "image_index": idx,
                "model": "Result",
                "error": True,
                "message": err_msg,
            })
            job.completed += 1
        return

    _apply_result_row_metadata(result_list, cfg, station_id, camera_id)

    with write_lock:
        _save_image_result(job, job_id, state, cfg, image_name, result_list)
        for ev in image_events:
            job.model_events.append({
                "type": "model_event",
                "image": image_name,
                "image_index": idx,
                **ev,
            })
        results_by_idx[idx] = result_list
        job.completed += 1


def _run_processing(
    job_id: str,
    state: AppState,
    station_id: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> None:
    job = job_manager.get(job_id)
    if not job:
        return

    with _job_semaphore:
        job.status = "running"
        try:
            processor, cfg, station_lat, station_lon = _build_processor(state, station_id, camera_id)
            results_by_idx: dict = {}
            image_errors: list = []
            write_lock = threading.Lock()

            def _process_one(args: tuple) -> tuple:
                idx, path = args
                result = processor.process_single_image(
                    path,
                    station_latitude=station_lat,
                    station_longitude=station_lon,
                )
                return idx, path, result if isinstance(result, list) else [result]

            with ThreadPoolExecutor(max_workers=_PARALLEL_IMAGES) as image_pool:
                futures = {
                    image_pool.submit(_process_one, (idx, path)): (idx, path)
                    for idx, path in enumerate(job.image_paths)
                }
                for future in as_completed(futures):
                    idx, image_path = futures[future]
                    try:
                        _, _, result_list = future.result()
                    except Exception as exc:
                        logger.error("Image %s failed: %s", image_path, exc)
                        with write_lock:
                            image_errors.append(f"{os.path.basename(image_path)}: {exc}")
                            job.completed += 1
                        continue

                    _handle_image_result(
                        job, job_id, state, cfg, idx, image_path, result_list,
                        results_by_idx, image_errors, write_lock, station_id, camera_id,
                    )

            results = []
            for idx in range(len(job.image_paths)):
                results.extend(results_by_idx.get(idx, []))

            if image_errors:
                job.error = f"{len(image_errors)} image(s) failed: " + "; ".join(image_errors)
            job.results = _json_safe(results)
            job.status = "done_with_errors" if image_errors else "done"

        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        finally:
            job.finished_at = time.time()
            if state.db_manager:
                try:
                    state.db_manager.save_job(job)
                except Exception as persist_exc:
                    logger.error("Could not persist job metadata for %s: %s", job_id, persist_exc)


def _run_pipeline(
    job_id: str,
    state: AppState,
    station_id: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> None:
    """Consume images from a queue as uploads arrive; analyse concurrently with uploading."""
    job = job_manager.get(job_id)
    if not job or not job._work_queue:
        return

    with _job_semaphore:
        job.status = "running"
        try:
            processor, cfg, station_lat, station_lon = _build_processor(state, station_id, camera_id)
            results_by_idx: dict = {}
            image_errors: list = []
            write_lock = threading.Lock()
            work_q = job._work_queue
            poison_seen = False
            pending: dict = {}

            def _process_one(args: tuple) -> tuple:
                idx, path = args
                result = processor.process_single_image(
                    path,
                    station_latitude=station_lat,
                    station_longitude=station_lon,
                )
                return idx, path, result if isinstance(result, list) else [result]

            def _handle_done(future) -> None:
                idx, image_path, _safe_name = pending.pop(future)
                try:
                    _, _, result_list = future.result()
                except Exception as exc:
                    logger.error("Image %s failed: %s", image_path, exc)
                    with write_lock:
                        image_errors.append(f"{os.path.basename(image_path)}: {exc}")
                        job.completed += 1
                    return

                _handle_image_result(
                    job, job_id, state, cfg, idx, image_path, result_list,
                    results_by_idx, image_errors, write_lock, station_id, camera_id,
                )

            with ThreadPoolExecutor(max_workers=_PARALLEL_IMAGES) as image_pool:
                while True:
                    for fut in list(pending.keys()):
                        if fut.done():
                            _handle_done(fut)

                    if poison_seen and not pending:
                        try:
                            item = work_q.get_nowait()
                        except queue.Empty:
                            break
                        if item[0] != "poison":
                            _, idx, path, safe_name = item
                            fut = image_pool.submit(_process_one, (idx, path))
                            pending[fut] = (idx, path, safe_name)
                        continue

                    if poison_seen and not pending and work_q.empty():
                        break

                    try:
                        item = work_q.get(timeout=0.3)
                    except queue.Empty:
                        continue

                    if item[0] == "poison":
                        poison_seen = True
                        continue

                    _, idx, path, safe_name = item
                    fut = image_pool.submit(_process_one, (idx, path))
                    pending[fut] = (idx, path, safe_name)

                for fut in list(pending.keys()):
                    if fut.done():
                        _handle_done(fut)

            results = []
            for idx in range(len(job.image_paths)):
                results.extend(results_by_idx.get(idx, []))

            job.total = len(job.image_paths)
            if image_errors:
                job.error = f"{len(image_errors)} image(s) failed: " + "; ".join(image_errors)
            job.results = _json_safe(results)
            job.status = "done_with_errors" if image_errors else "done"

        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        finally:
            job.finished_at = time.time()
            if state.db_manager:
                try:
                    state.db_manager.save_job(job)
                except Exception as persist_exc:
                    logger.error("Could not persist job metadata for %s: %s", job_id, persist_exc)


@router.post("/process/{job_id}")
def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_state),
    station_id: Optional[str] = None,
    camera_id: Optional[str] = None,
):
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="AI models not loaded yet")
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if station_id and camera_id and state.station_manager:
        try:
            state.station_manager.set_current_camera(station_id, camera_id)
        except Exception:
            logger.warning("Could not sync camera assignment for %s/%s", station_id, camera_id)
    background_tasks.add_task(_run_processing, job_id, state, station_id or None, camera_id or None)
    return {"ok": True, "job_id": job_id}


@router.get("/job/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, state: AppState = Depends(get_state)):
    job = job_manager.get(job_id)
    if job:
        return JobStatus(
            job_id=job.job_id,
            status=job.status,
            total=job.total,
            completed=job.completed,
            uploaded=job.uploaded if job.pipeline else None,
            error=job.error,
        )
    if state.db_manager:
        row = state.db_manager.get_job(job_id)
        if row:
            return JobStatus(
                job_id=row["job_id"],
                status=row["status"],
                total=int(row.get("total") or 0),
                completed=int(row.get("completed") or 0),
                error=row.get("error"),
            )
    raise HTTPException(status_code=404, detail="Job not found")


@router.get("/job/{job_id}/stream")
async def stream_job_progress(job_id: str, cursor: int = Query(0, ge=0)):
    """
    Server-Sent Events stream.

    Yields two event types:
      {"type": "model_event", "image": "...", "model": "...", ...}
      {"type": "progress",    "status": "...", "total": N, "completed": M, ...}
    """

    async def event_gen():
        last = cursor
        while True:
            job = job_manager.get(job_id)
            if not job:
                yield _sse({"type": "error", "error": "Job not found"})
                break

            while last < len(job.model_events):
                yield _sse(job.model_events[last])
                last += 1

            yield _sse({
                "type": "progress",
                "job_id": job.job_id,
                "status": job.status,
                "total": job.total,
                "completed": job.completed,
                "uploaded": job.uploaded if job.pipeline else None,
                "error": job.error,
            })

            if job.status in ("done", "done_with_errors", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _json_safe(obj):
    """Convert numpy/pandas scalars to native Python types for JSON responses."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return obj


@router.get("/results/{job_id}")
def get_job_results(job_id: str, state: AppState = Depends(get_state)):
    job = job_manager.get(job_id)
    if job:
        if job.status not in ("done", "done_with_errors"):
            raise HTTPException(status_code=202, detail=f"Job status: {job.status}")
        return _json_safe({
            "job_id": job_id,
            "results": job.results,
            "scrub_audit": job.scrub_audit,
            "total": job.total,
        })
    if state.db_manager:
        row = state.db_manager.get_job(job_id)
        if row and row["status"] in ("done", "done_with_errors"):
            return _json_safe({
                "job_id": job_id,
                "results": [],
                "scrub_audit": [],
                "total": int(row.get("total") or 0),
                "persisted_only": True,
            })
    raise HTTPException(status_code=404, detail="Job not found")
