"""Tab 3 — Analysis Statistics."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/stats", tags=["statistics"])


def _unique_images_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per physical image — dedupe by id (image PK), not filename."""
    if df.empty or "id" not in df.columns:
        return df
    return df.drop_duplicates("id")


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

    unique_df = _unique_images_df(df)

    # Exclude Empty detections for animal-level stats
    non_empty = df[
        df["detected_animal"].str.lower().ne("empty") if "detected_animal" in df.columns else df.index.notna()
    ]
    animals = non_empty[non_empty["primary_label"] == "Animal"] if "primary_label" in non_empty.columns else non_empty

    species_dist = (
        animals["detected_animal"].value_counts().reset_index()
        .rename(columns={"detected_animal": "species", "count": "count"})
        .to_dict(orient="records")
    ) if "detected_animal" in animals.columns else []

    # Day/Night distribution by unique image id
    if "day_night" in unique_df.columns:
        img_day_night = unique_df[["id", "day_night"]].drop_duplicates("id")
        day_night_dist = (
            img_day_night["day_night"].value_counts().reset_index()
            .rename(columns={"day_night": "label", "count": "count"})
            .to_dict(orient="records")
        )
    else:
        day_night_dist = []

    conf_series = (
        non_empty["detection_confidence"].fillna(0).tolist()
    ) if "detection_confidence" in non_empty.columns else []

    # Hourly activity — one entry per unique image id
    hourly_counts = [0] * 24
    if "capture_time" in unique_df.columns and "id" in unique_df.columns:
        for _, row in unique_df[["id", "capture_time"]].dropna(subset=["capture_time"]).iterrows():
            try:
                h = int(str(row["capture_time"]).split(":")[0])
                if 0 <= h < 24:
                    hourly_counts[h] += 1
            except Exception:
                pass
    elif "capture_time" in df.columns:
        for t in df["capture_time"].dropna():
            try:
                h = int(str(t).split(":")[0])
                if 0 <= h < 24:
                    hourly_counts[h] += 1
            except Exception:
                pass
    hourly_dist = [{"hour": f"{h:02d}:00", "count": count} for h, count in enumerate(hourly_counts)]

    total_images = int(unique_df["id"].nunique()) if "id" in unique_df.columns else len(unique_df)
    if "id" in animals.columns:
        animal_images = int(animals["id"].nunique())
    else:
        animal_images = len(animals)

    if "day_night" in unique_df.columns:
        day_count = int((unique_df["day_night"] == "Day").sum())
        night_count = int((unique_df["day_night"] == "Night").sum())
    else:
        day_count = 0
        night_count = 0

    return {
        "total_images": total_images,
        "animals_identified": animal_images,
        "day_count": day_count,
        "night_count": night_count,
        "species_distribution": species_dist,
        "day_night_distribution": day_night_dist,
        "confidence_series": conf_series,
        "hourly_distribution": hourly_dist,
    }


@router.get("/export")
def export_statistics(
    metric: str = "summary",
    format: str = "csv",
    state: AppState = Depends(get_state)
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

    df = state.db_manager.get_history_df()
    unique_df = _unique_images_df(df)

    if df.empty:
        if metric == "summary":
            export_df = pd.DataFrame(columns=["metric", "value"])
        elif metric == "species":
            export_df = pd.DataFrame(columns=["species", "count"])
        elif metric == "daynight":
            export_df = pd.DataFrame(columns=["label", "count"])
        elif metric == "hourly":
            export_df = pd.DataFrame(columns=["hour", "count"])
        elif metric == "confidence":
            export_df = pd.DataFrame(columns=["index", "confidence"])
        else:
            raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")
    else:
        non_empty = df[
            df["detected_animal"].str.lower().ne("empty") if "detected_animal" in df.columns else df.index.notna()
        ]
        animals = non_empty[non_empty["primary_label"] == "Animal"] if "primary_label" in non_empty.columns else non_empty

        if metric == "summary":
            total_images = int(unique_df["id"].nunique()) if "id" in unique_df.columns else len(unique_df)
            animal_images = int(animals["id"].nunique()) if "id" in animals.columns else len(animals)
            if "day_night" in unique_df.columns:
                day_count = int((unique_df["day_night"] == "Day").sum())
                night_count = int((unique_df["day_night"] == "Night").sum())
            else:
                day_count = night_count = 0
            export_df = pd.DataFrame([
                {"metric": "Total Images", "value": total_images},
                {"metric": "Animals Identified", "value": animal_images},
                {"metric": "Day Images", "value": day_count},
                {"metric": "Night Images", "value": night_count},
            ])
        elif metric == "species":
            if "detected_animal" in animals.columns:
                export_df = (
                    animals["detected_animal"].value_counts().reset_index()
                    .rename(columns={"detected_animal": "species", "count": "count"})
                )
            else:
                export_df = pd.DataFrame(columns=["species", "count"])
        elif metric == "daynight":
            if "day_night" in unique_df.columns:
                img_day_night = unique_df[["id", "day_night"]].drop_duplicates("id")
                export_df = (
                    img_day_night["day_night"].value_counts().reset_index()
                    .rename(columns={"day_night": "label", "count": "count"})
                )
            elif "day_night" in df.columns:
                export_df = (
                    df["day_night"].value_counts().reset_index()
                    .rename(columns={"day_night": "label", "count": "count"})
                )
            else:
                export_df = pd.DataFrame(columns=["label", "count"])
        elif metric == "hourly":
            hourly_counts = [0] * 24
            if "capture_time" in unique_df.columns and "id" in unique_df.columns:
                for _, row in unique_df[["id", "capture_time"]].dropna(subset=["capture_time"]).iterrows():
                    try:
                        h = int(str(row["capture_time"]).split(":")[0])
                        if 0 <= h < 24:
                            hourly_counts[h] += 1
                    except Exception:
                        pass
            elif "capture_time" in df.columns:
                for t in df["capture_time"].dropna():
                    try:
                        h = int(str(t).split(":")[0])
                        if 0 <= h < 24:
                            hourly_counts[h] += 1
                    except Exception:
                        pass
            export_df = pd.DataFrame([
                {"hour": f"{h:02d}:00", "count": count}
                for h, count in enumerate(hourly_counts)
            ])
        elif metric == "confidence":
            if "detection_confidence" in non_empty.columns:
                conf_list = non_empty["detection_confidence"].fillna(0).tolist()
                export_df = pd.DataFrame([
                    {"index": i, "confidence": v}
                    for i, v in enumerate(conf_list)
                ])
            else:
                export_df = pd.DataFrame(columns=["index", "confidence"])
        else:
            raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")

    is_excel = format.lower() in ("excel", "xlsx")
    filename = f"statistics_{metric}.xlsx" if is_excel else f"statistics_{metric}.csv"

    if is_excel:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Statistics")
        out.seek(0)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        content = out.getvalue()
    else:
        media_type = "text/csv"
        content = export_df.to_csv(index=False)

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8") if isinstance(content, str) else content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
