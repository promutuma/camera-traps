from fastapi import APIRouter, Depends
from backend.models.schemas import ConfigResponse, ConfigUpdate
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", response_model=ConfigResponse)
def get_config(state: AppState = Depends(get_state)):
    cfg = state.config
    return ConfigResponse(**cfg.__dict__)


@router.patch("", response_model=ConfigResponse)
def update_config(update: ConfigUpdate, state: AppState = Depends(get_state)):
    cfg = state.config
    for key, value in update.model_dump(exclude_none=True).items():
        setattr(cfg, key, value)

    # Propagate relevant values to already-loaded services
    if state.md_model and update.detection_confidence is not None:
        state.md_model.set_confidence_threshold(cfg.detection_confidence)
    if state.dn_model and update.brightness_threshold is not None:
        state.dn_model.brightness_threshold = cfg.brightness_threshold
    if state.scrubber and update.blur_strength is not None:
        state.scrubber.blur_strength = cfg.blur_strength
    if state.independence_engine and update.independence_window is not None:
        state.independence_engine.window_minutes = cfg.independence_window

    return ConfigResponse(**cfg.__dict__)


@router.get("/status")
def model_status(state: AppState = Depends(get_state)):
    return {
        "models_loaded": state.models_loaded,
        "error": state.models_error,
    }
