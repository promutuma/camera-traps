from fastapi import APIRouter, Depends, BackgroundTasks
from backend.models.schemas import ConfigResponse, ConfigUpdate
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/config", tags=["config"])


def _speciesnet_loaded(state: AppState) -> bool:
    sn = state.speciesnet_model
    return bool(sn and getattr(sn, "classifier", None))


def _speciesnet_error(state: AppState):
    if state.config.enable_low_spec:
        return state.speciesnet_error or "SpeciesNet disabled (Low-Spec mode)"
    sn = state.speciesnet_model
    if sn and sn.load_error:
        return sn.load_error
    return state.speciesnet_error


@router.get("", response_model=ConfigResponse)
def get_config(state: AppState = Depends(get_state)):
    cfg = state.config
    return ConfigResponse(**cfg.__dict__)


@router.patch("", response_model=ConfigResponse)
def update_config(
    update: ConfigUpdate,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_state),
):
    cfg = state.config
    prev_low_spec = cfg.enable_low_spec
    for key, value in update.model_dump(exclude_none=True).items():
        setattr(cfg, key, value)

    if state.md_model and update.detection_confidence is not None:
        state.md_model.set_confidence_threshold(cfg.detection_confidence)
    if state.dn_model and update.brightness_threshold is not None:
        state.dn_model.brightness_threshold = cfg.brightness_threshold
    if state.scrubber and update.blur_strength is not None:
        state.scrubber.blur_strength = cfg.blur_strength
    if state.independence_engine and update.independence_window is not None:
        state.independence_engine.window_minutes = cfg.independence_window

    if update.enable_low_spec is not None and update.enable_low_spec != prev_low_spec:
        from backend.services.model_loader import reload_speciesnet
        background_tasks.add_task(reload_speciesnet, state)

    return ConfigResponse(**cfg.__dict__)


@router.get("/status")
def model_status(state: AppState = Depends(get_state)):
    md_loaded = bool(state.md_model and getattr(state.md_model, "model", None))
    md_error = getattr(state.md_model, "load_error", None) if state.md_model else None
    sn_loaded = _speciesnet_loaded(state)
    sn_error = _speciesnet_error(state) if not sn_loaded else None
    ocr_loaded = state.ocr_model is not None
    dn_loaded = state.dn_model is not None
    models_partial = (
        state.models_loaded
        and not state.config.enable_low_spec
        and (state.speciesnet_loading or not sn_loaded)
    )
    return {
        "models_loaded": state.models_loaded,
        "models_partial": models_partial,
        "error": state.models_error,
        "md": {"loaded": md_loaded, "error": md_error},
        "speciesnet": {
            "loaded": sn_loaded,
            "loading": state.speciesnet_loading,
            "error": sn_error,
        },
        "ocr": {"loaded": ocr_loaded},
        "dn": {"loaded": dn_loaded},
    }
