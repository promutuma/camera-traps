"""Tab 3 — Analysis Statistics."""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/stats", tags=["statistics"])


@router.get("/summary")
def get_summary(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        return {
            "total_images": 0,
            "animals_identified": 0,
            "day_count": 0,
            "night_count": 0,
            "species_distribution": [],
            "day_night_distribution": [],
            "confidence_series": [],
            "hourly_distribution": [],
        }

    animals = df[df["primary_label"] == "Animal"] if "primary_label" in df.columns else df
    species_dist = (
        animals["detected_animal"].value_counts().reset_index()
        .rename(columns={"detected_animal": "species", "count": "count"})
        .to_dict(orient="records")
    ) if "detected_animal" in animals.columns else []

    day_night_dist = (
        df["day_night"].value_counts().reset_index()
        .rename(columns={"day_night": "label", "count": "count"})
        .to_dict(orient="records")
    ) if "day_night" in df.columns else []

    conf_series = (
        df["detection_confidence"].fillna(0).tolist()
    ) if "detection_confidence" in df.columns else []

    # Hourly activity distribution (0-23 hours)
    hourly_counts = [0] * 24
    if "capture_time" in df.columns:
        for t in df["capture_time"].dropna():
            try:
                h = int(str(t).split(":")[0])
                if 0 <= h < 24:
                    hourly_counts[h] += 1
            except Exception:
                pass
    hourly_dist = [{"hour": f"{h:02d}:00", "count": count} for h, count in enumerate(hourly_counts)]

    return {
        "total_images": len(df),
        "animals_identified": len(animals),
        "day_count": int((df["day_night"] == "Day").sum()) if "day_night" in df.columns else 0,
        "night_count": int((df["day_night"] == "Night").sum()) if "day_night" in df.columns else 0,
        "species_distribution": species_dist,
        "day_night_distribution": day_night_dist,
        "confidence_series": conf_series,
        "hourly_distribution": hourly_dist,
    }
