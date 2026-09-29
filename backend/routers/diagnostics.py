"""Tab 5 — Deep Inspection: OCR + MegaDetector raw candidates + SpeciesNet predictions."""

import os
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.post("/inspect")
async def inspect_image(file: UploadFile = File(...), state: AppState = Depends(get_state)):
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1] or ".jpg"
    )
    tmp.write(await file.read())
    tmp.close()

    try:
        result: dict = {}

        # OCR
        if state.ocr_model:
            try:
                result["ocr"] = state.ocr_model.process_image(tmp.name)
            except Exception as e:
                result["ocr"] = {"error": str(e)}

        # MegaDetector raw candidates
        megadetector_candidates = []
        if state.md_model:
            try:
                candidates = state.md_model.detect_all_candidates(tmp.name)
                for c in (candidates if isinstance(candidates, list) else [candidates]):
                    c["model"] = "MDv5a"
                    megadetector_candidates.append(c)
            except Exception as e:
                pass
        result["megadetector"] = megadetector_candidates

        # Google SpeciesNet top-5
        if state.speciesnet_model:
            try:
                from PIL import Image
                img = Image.open(tmp.name).convert("RGB")
                predictions = state.speciesnet_model.classify_crop(img, top_k=5)
                result["speciesnet"] = [
                    {"label": label, "score": round(float(score), 4)}
                    for label, score in predictions
                ]
            except Exception as e:
                result["speciesnet"] = {"error": str(e)}

        return result

    finally:
        os.unlink(tmp.name)
