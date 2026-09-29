"""
Human-readable, image-grouped export formatting for detection results.
One row per image in CSV/Excel primary sheets; full per-detection detail in nested JSON and a detail sheet.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

NON_WILDLIFE = {"empty", "blank", "person", "vehicle", "human", "error", "unidentified", "n/a", "none", "unknown"}


def slugify_filename_part(text: Any, max_len: int = 40) -> str:
    """Make a filesystem-safe slug for export filenames."""
    value = str(text or "").strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value)
    return value[:max_len].strip("-") or "all"


def export_counts(raw_df: pd.DataFrame) -> Tuple[int, int]:
    stats = compute_export_stats(raw_df)
    return stats["image_count"], stats["detection_count"]


def compute_export_stats(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize export dataset for filenames and metadata blocks."""
    empty = {
        "image_count": 0,
        "detection_count": 0,
        "wildlife_count": 0,
        "non_wildlife_count": 0,
        "species_count": 0,
        "station_count": 0,
        "day_count": 0,
        "night_count": 0,
        "capture_date_from": None,
        "capture_date_to": None,
        "top_species": [],
        "stations": [],
    }
    if raw_df is None or raw_df.empty:
        return empty

    image_col = "image_id" if "image_id" in raw_df.columns else "id"
    species_col = "_canonical_species" if "detected_animal" in raw_df.columns else None
    if species_col:
        raw_df = raw_df.copy()
        raw_df[species_col] = raw_df["detected_animal"].map(canonical_species_name)

    wildlife_mask = (
        raw_df[species_col].map(lambda s: _record_type(s) == "Wildlife")
        if species_col
        else pd.Series([False] * len(raw_df))
    )
    wildlife = raw_df[wildlife_mask] if species_col else raw_df.iloc[0:0]


    day_night = raw_df["day_night"].astype(str).str.strip().str.lower() if "day_night" in raw_df.columns else pd.Series(dtype=str)
    day_count = int((day_night == "day").sum()) if not day_night.empty else 0
    night_count = int((day_night == "night").sum()) if not day_night.empty else 0

    stations = pd.Series(dtype=str)
    if "station_id" in raw_df.columns:
        stations = raw_df["station_id"].dropna().astype(str).str.strip()
        stations = stations[stations != ""]

    capture_dates: List[str] = []
    if image_col in raw_df.columns:
        for _, row in raw_df.drop_duplicates(subset=[image_col]).iterrows():
            capture_date = _clean_str(row.get("capture_date"))
            if capture_date:
                capture_dates.append(capture_date)
    capture_dates.sort()

    top_species: List[str] = []
    if species_col and not wildlife.empty:
        top_species = [
            str(s) for s in wildlife[species_col].value_counts().head(3).index.tolist()
            if _clean_str(s)
        ]

    return {
        "image_count": int(raw_df[image_col].nunique()) if image_col in raw_df.columns else len(raw_df),
        "detection_count": len(raw_df),
        "wildlife_count": int(wildlife_mask.sum()),
        "non_wildlife_count": int((~wildlife_mask).sum()),
        "species_count": int(wildlife[species_col].nunique()) if species_col and not wildlife.empty else 0,
        "station_count": int(stations.nunique()) if not stations.empty else 0,
        "day_count": day_count,
        "night_count": night_count,
        "capture_date_from": capture_dates[0] if capture_dates else None,
        "capture_date_to": capture_dates[-1] if capture_dates else None,
        "top_species": top_species,
        "stations": sorted(stations.unique().tolist())[:12],
    }


def _filter_label_parts(filters: Optional[Dict[str, Any]]) -> List[str]:
    if not filters:
        return []
    parts: List[str] = []
    station = filters.get("station")
    species = filters.get("species")
    day_night = filters.get("day_night")
    min_conf = filters.get("min_conf")
    max_conf = filters.get("max_conf")
    if station:
        parts.append(f"station-{slugify_filename_part(station, 16)}")
    if species:
        parts.append(f"species-{slugify_filename_part(species, 16)}")
    if day_night:
        parts.append(slugify_filename_part(day_night, 10))
    if min_conf is not None:
        parts.append(f"conf-min-{int(float(min_conf) * 100)}pct")
    if max_conf is not None:
        parts.append(f"conf-max-{int(float(max_conf) * 100)}pct")
    return parts


def build_export_filename(
    *,
    export_kind: str,
    file_format: str,
    project_name: Optional[str] = None,
    project_area: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a concise download filename under 50-55 characters.

    Examples:
      viumbelens_gambella_results_20260910_1315.xlsx
      viumbelens_gambella_station-010_20260910_1315.xlsx
      viumbelens_gambella_lion_20260910_1315.xlsx
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    project_slug = slugify_filename_part(project_name or "project", 12)
    ext = file_format.lstrip(".").lower()

    filter_tag = ""
    if filters:
        if filters.get("station"):
            filter_tag = slugify_filename_part(str(filters["station"]), 11)
        elif filters.get("species"):
            filter_tag = slugify_filename_part(str(filters["species"]), 11)
        elif filters.get("day_night"):
            filter_tag = slugify_filename_part(str(filters["day_night"]), 8)

    segments: List[str] = ["viumbelens", project_slug]
    if filter_tag:
        segments.append(filter_tag)
    else:
        kind_slug = slugify_filename_part(export_kind, 8)
        segments.append(kind_slug if kind_slug else "results")

    segments.append(timestamp)
    filename = f"{'_'.join(segments)}.{ext}"
    if len(filename) > 55:
        filename = f"viumbelens_{project_slug[:10]}_{timestamp}.{ext}"
    return filename




def build_export_metadata_rows(
    *,
    export_kind: str,
    file_format: str,
    filename: str,
    project_name: str,
    project_area: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Human-readable key/value rows for CSV headers, JSON export_info, and Excel info sheet."""
    stats = stats or {}
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows: List[Dict[str, str]] = [
        {"Field": "Export file", "Value": filename},
        {"Field": "Generated at (UTC)", "Value": generated_at},
        {"Field": "Export type", "Value": export_kind.replace("-", " ").title()},
        {"Field": "Format", "Value": file_format.upper()},
        {"Field": "Layout", "Value": "One row per image with nested/full detection detail"},
        {"Field": "Project", "Value": project_name},
    ]
    if project_area:
        rows.append({"Field": "Survey area", "Value": project_area})

    rows.extend([
        {"Field": "Total images", "Value": str(stats.get("image_count", 0))},
        {"Field": "Total detections", "Value": str(stats.get("detection_count", 0))},
        {"Field": "Wildlife detections", "Value": str(stats.get("wildlife_count", 0))},
        {"Field": "Blank / person / vehicle detections", "Value": str(stats.get("non_wildlife_count", 0))},
        {"Field": "Unique wildlife species", "Value": str(stats.get("species_count", 0))},
        {"Field": "Unique stations", "Value": str(stats.get("station_count", 0))},
        {"Field": "Day detections", "Value": str(stats.get("day_count", 0))},
        {"Field": "Night detections", "Value": str(stats.get("night_count", 0))},
    ])

    date_from = stats.get("capture_date_from")
    date_to = stats.get("capture_date_to")
    if date_from and date_to:
        rows.append({
            "Field": "Capture date range",
            "Value": date_from if date_from == date_to else f"{date_from} to {date_to}",
        })

    top_species = stats.get("top_species") or []
    if top_species:
        rows.append({"Field": "Top species", "Value": ", ".join(top_species)})

    stations = stats.get("stations") or []
    if stations:
        rows.append({"Field": "Stations included", "Value": ", ".join(stations)})

    if filters:
        rows.append({"Field": "Filters applied", "Value": "Yes"})
        for key, value in filters.items():
            label = str(key).replace("_", " ").title()
            if key in {"min_conf", "max_conf"}:
                rows.append({"Field": label, "Value": f"{float(value) * 100:.0f}%"})
            else:
                rows.append({"Field": label, "Value": str(value)})
    else:
        rows.append({"Field": "Filters applied", "Value": "None (full dataset)"})

    rows.append({
        "Field": "Notes",
        "Value": (
            "Includes blank, person, and vehicle records. "
            "Bounding boxes use normalized image percentages. "
            "Excel/JSON include full per-detection model breakdown and raw output."
        ),
    })
    return rows


def build_export_info_dict(
    *,
    export_kind: str,
    file_format: str,
    filename: str,
    project_name: str,
    project_area: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stats = stats or {}
    return {
        "export_file": filename,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_type": export_kind,
        "format": file_format,
        "layout": "image-centric",
        "project": {
            "name": project_name,
            "survey_area": project_area,
        },
        "totals": {
            "images": stats.get("image_count", 0),
            "detections": stats.get("detection_count", 0),
            "wildlife_detections": stats.get("wildlife_count", 0),
            "non_wildlife_detections": stats.get("non_wildlife_count", 0),
            "unique_wildlife_species": stats.get("species_count", 0),
            "unique_stations": stats.get("station_count", 0),
            "day_detections": stats.get("day_count", 0),
            "night_detections": stats.get("night_count", 0),
        },
        "coverage": {
            "capture_date_from": stats.get("capture_date_from"),
            "capture_date_to": stats.get("capture_date_to"),
            "top_species": stats.get("top_species") or [],
            "stations": stats.get("stations") or [],
        },
        "filters_applied": filters or {},
        "filtered_export": bool(filters),
        "notes": (
            "Image-centric export: each image groups all its detections. "
            "Includes blank, person, and vehicle records. Bounding boxes use normalized image percentages."
        ),
    }


def content_disposition_attachment(filename: str) -> str:
    """Return a Content-Disposition header value safe for download filenames."""
    safe = filename.replace('"', "").replace("\n", "").replace("\r", "")
    return f'attachment; filename="{safe}"'


def _parse_json(raw: Any) -> Any:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, dict)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# Canonical rollups: merge alternate candidates / subspecies into the native survey species
SPECIES_ROLLUPS: Dict[str, str] = {
    "kinda baboon": "olive baboon",
    "chacma baboon": "olive baboon",
    "baboon species": "olive baboon",
    "yellow baboon": "olive baboon",
    "human": "Person",
    "person": "Person",
    "vehicle": "Vehicle",
}

# Non-native or low-confidence false-positive candidates to consolidate into Unidentified Wildlife
NON_NATIVE_CANDIDATES: set[str] = {
    "woodchuck", "wild turkey", "coyote", "virginia opossum", "american bison",
    "western gray kangaroo", "south american coati", "white-tailed deer",
    "red deer", "european roe deer", "sika deer", "reeves' muntjac", "muntjac species",
    "himalayan marmot", "giant anteater", "eastern gray squirrel", "striped skunk",
    "nutria", "tayra", "takin", "western mountain coati", "american black vulture",
    "eurasian otter", "north american river otter", "brush-tailed rock wallaby",
    "macaque species", "milne-edwards' macaque", "southern plains gray langur",
    "hatinh langur", "black-and-gold howler monkey", "yellow-throated marten",
}


def canonical_species_name(species: Any) -> str:
    """Resolve raw classifier predictions to canonical regional species and consolidate candidate variants."""
    s = _clean_str(species)
    lower = s.lower()
    if lower in SPECIES_ROLLUPS:
        return SPECIES_ROLLUPS[lower]
    if lower in NON_NATIVE_CANDIDATES:
        return "Unidentified Wildlife"
    return s


def _record_type(species: Any) -> str:
    label = str(species or "").strip().lower()
    if not label or label == "unidentified wildlife":
        return "Unidentified"
    if label in NON_WILDLIFE or "person" in label or "vehicle" in label:
        return label.title()
    return "Wildlife"



def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _pct(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        v = float(value)
        return round(v * 100, 1) if v <= 1 else round(v, 1)
    except (TypeError, ValueError):
        return None


def _format_bbox(raw: Any) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[List[float]]]:
    parsed = _parse_json(raw)
    if not isinstance(parsed, list) or len(parsed) != 4:
        return "", None, None, None, None, None
    try:
        left, top, width, height = [float(x) for x in parsed]
        text = f"{left * 100:.1f}%, {top * 100:.1f}%, {width * 100:.1f}%, {height * 100:.1f}%"
        return (
            text,
            round(left * 100, 2),
            round(top * 100, 2),
            round(width * 100, 2),
            round(height * 100, 2),
            [left, top, width, height],
        )
    except (TypeError, ValueError):
        return "", None, None, None, None, None


def _format_candidates(raw: Any, limit: int = 8) -> str:
    parsed = _parse_json(raw)
    if not isinstance(parsed, list):
        return ""
    parts: List[str] = []
    for item in parsed[:limit]:
        if not isinstance(item, dict):
            continue
        name = item.get("common_name") or item.get("display") or item.get("label") or ""
        if not name:
            continue
        conf = item.get("confidence")
        if isinstance(conf, (int, float)):
            parts.append(f"{name} ({float(conf) * 100:.0f}%)")
        else:
            parts.append(str(name))
    return "; ".join(parts)


def _format_taxonomy(raw: Any) -> str:
    parsed = _parse_json(raw)
    if isinstance(parsed, list):
        return " > ".join(str(x) for x in parsed if str(x).strip())
    return _clean_str(raw)


def _format_model_breakdown(raw: Any) -> str:
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return _clean_str(raw)
    parts: List[str] = []
    for model, items in parsed.items():
        if isinstance(items, list):
            labels = []
            for item in items[:5]:
                if isinstance(item, dict):
                    label = item.get("label") or item.get("common_name") or "?"
                    conf = item.get("conf") or item.get("confidence")
                    if isinstance(conf, (int, float)):
                        labels.append(f"{label} ({float(conf) * 100:.0f}%)")
                    else:
                        labels.append(str(label))
                else:
                    labels.append(str(item))
            if labels:
                parts.append(f"{model}: {', '.join(labels)}")
        else:
            parts.append(f"{model}: {items}")
    return " | ".join(parts)


def _json_pretty(raw: Any) -> str:
    parsed = _parse_json(raw)
    if parsed is None:
        return _clean_str(raw)
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return _clean_str(raw)


def _confidence_bits(r: pd.Series) -> str:
    conf_pct = _pct(r.get("detection_confidence"))
    md = _pct(r.get("md_confidence"))
    sn = _pct(r.get("speciesnet_confidence"))
    parts: List[str] = []
    if conf_pct is not None:
        parts.append(f"{conf_pct:.0f}%")
    if md is not None:
        parts.append(f"MD {md:.0f}%")
    if sn is not None:
        parts.append(f"SN {sn:.0f}%")
    return " / ".join(parts)


def _detection_id_label(r: pd.Series, index: int) -> str:
    det_id = r.get("detection_id")
    if det_id is not None and pd.notna(det_id):
        return f"#{int(det_id)}"
    return f"#{index}"


def _species_display_name(r: pd.Series) -> str:
    species = canonical_species_name(r.get("detected_animal")) or "Unknown"
    sci = _clean_str(r.get("scientific_name"))
    if sci and sci.lower() != species.lower() and species != "Unidentified Wildlife":
        return f"{species} ({sci})"
    return species


def _species_list_line(r: pd.Series, index: int = 1) -> str:
    """Compact per-detection species line for the Species List column."""
    name = _species_display_name(r)
    conf = _confidence_bits(r)
    species = canonical_species_name(r.get("detected_animal"))
    rtype = _record_type(species)
    method = _clean_str(r.get("detection_method"))

    bits = [f"{_detection_id_label(r, index)} {name}"]
    if conf:
        bits.append(conf)
    bits.append(rtype)
    if method:
        bits.append(method)
    return " · ".join(bits)


def _detection_summary_line(r: pd.Series, index: int = 1) -> str:
    """Detailed per-detection summary for the Detections Summary column."""
    species = canonical_species_name(r.get("detected_animal")) or "Unknown"
    name = _species_display_name(r)
    conf = _confidence_bits(r)
    rtype = _record_type(species)

    method = _clean_str(r.get("detection_method"))
    bbox_text, *_ = _format_bbox(r.get("bbox"))
    taxonomy = _format_taxonomy(r.get("taxonomy_hierarchy"))
    model_breakdown = _format_model_breakdown(r.get("model_breakdown"))
    ide = _clean_str(r.get("ide_id"))

    bits = [f"{index}. [{_detection_id_label(r, index)}] {name}"]
    if conf:
        bits.append(f"confidence {conf}")
    bits.append(rtype)
    if method:
        bits.append(f"method {method}")
    if bbox_text:
        bits.append(f"bbox {bbox_text}")
    if taxonomy:
        bits.append(f"taxonomy {taxonomy}")
    if model_breakdown:
        bits.append(f"models {model_breakdown}")
    if ide:
        bits.append(f"IDE {ide}")
    return " — ".join(bits)



def _format_species_list_for_image(grp: pd.DataFrame) -> str:
    lines = [_species_list_line(r, i + 1) for i, (_, r) in enumerate(grp.iterrows())]
    return "\n".join(lines)


def _format_detections_summary_for_image(grp: pd.DataFrame) -> str:
    lines = [_detection_summary_line(r, i + 1) for i, (_, r) in enumerate(grp.iterrows())]
    return "\n".join(lines)


def _species_detail_items(grp: pd.DataFrame) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, (_, r) in enumerate(grp.iterrows(), start=1):
        species = _clean_str(r.get("detected_animal"))
        bbox_text, bx, by, bw, bh, bbox_raw = _format_bbox(r.get("bbox"))
        item: Dict[str, Any] = {
            "detection_id": int(r["detection_id"]) if r.get("detection_id") is not None and pd.notna(r.get("detection_id")) else None,
            "label": _detection_id_label(r, i),
            "species": species,
            "scientific_name": _clean_str(r.get("scientific_name")),
            "display_name": _species_display_name(r),
            "record_type": _record_type(species),
            "confidence_percent": _pct(r.get("detection_confidence")),
            "megadetector_confidence_percent": _pct(r.get("md_confidence")),
            "speciesnet_confidence_percent": _pct(r.get("speciesnet_confidence")),
            "detection_method": _clean_str(r.get("detection_method")),
            "top_candidates": _format_candidates(r.get("top_candidates")),
            "taxonomy": _format_taxonomy(r.get("taxonomy_hierarchy")),
            "ide_id": _clean_str(r.get("ide_id")),
            "summary_line": _species_list_line(r, i),
        }
        if bbox_text:
            item["bounding_box"] = {
                "description": bbox_text,
                "left_percent": bx,
                "top_percent": by,
                "width_percent": bw,
                "height_percent": bh,
                "normalized": bbox_raw,
            }
        items.append(item)
    return items


def _build_detection_record(r: pd.Series) -> Dict[str, Any]:
    species = _clean_str(r.get("detected_animal"))
    conf = r.get("detection_confidence")
    conf_pct = _pct(conf)
    bbox_text, bx, by, bw, bh, bbox_raw = _format_bbox(r.get("bbox"))
    det_id = r.get("detection_id")

    record: Dict[str, Any] = {
        "detection_id": int(det_id) if det_id is not None and pd.notna(det_id) else None,
        "record_type": _record_type(species),
        "species": species,
        "scientific_name": _clean_str(r.get("scientific_name")),
        "confidence": round(float(conf), 4) if conf_pct is not None and pd.notna(conf) else None,
        "confidence_percent": conf_pct,
        "megadetector_confidence_percent": _pct(r.get("md_confidence")),
        "speciesnet_confidence_percent": _pct(r.get("speciesnet_confidence")),
        "detection_method": _clean_str(r.get("detection_method")),
        "ide_id": _clean_str(r.get("ide_id")),
        "taxonomy": _format_taxonomy(r.get("taxonomy_hierarchy")),
        "taxonomy_hierarchy": _parse_json(r.get("taxonomy_hierarchy")),
        "top_model_candidates": _format_candidates(r.get("top_candidates")),
        "top_candidates": _parse_json(r.get("top_candidates")),
        "model_breakdown": _parse_json(r.get("model_breakdown")),
        "model_breakdown_summary": _format_model_breakdown(r.get("model_breakdown")),
        "raw_model_output": _parse_json(r.get("raw_model_output")) or _clean_str(r.get("raw_model_output")),
    }
    if bbox_text:
        record["bounding_box"] = {
            "description": bbox_text,
            "left_percent": bx,
            "top_percent": by,
            "width_percent": bw,
            "height_percent": bh,
            "normalized": bbox_raw,
        }
    return record


def _image_columns() -> List[str]:
    return [
        "Image ID", "Filename", "File Hash", "Station", "Camera",
        "Capture Date", "Capture Time", "Capture DateTime",
        "Day / Night", "Temperature", "Brightness", "Notes", "Processed At",
        "Detection Count", "Detection IDs", "Species List", "Record Types",
        "Detections Summary", "Lowest Confidence %", "Highest Confidence %",
    ]


def _detection_detail_columns() -> List[str]:
    return [
        "Detection ID", "Image ID", "Filename",
        "Record Type", "Species", "Scientific Name",
        "Confidence", "Confidence %",
        "MegaDetector Confidence %", "SpeciesNet Confidence %",
        "Detection Method", "IDE ID",
        "Bounding Box", "BBox Left %", "BBox Top %", "BBox Width %", "BBox Height %",
        "Top Model Candidates", "Taxonomy",
        "Model Breakdown", "Raw Model Output",
        "Station", "Camera", "Capture DateTime", "Day / Night",
        "File Hash", "Brightness", "Notes", "Processed At",
    ]


def _group_raw_by_image(df: pd.DataFrame) -> List[Tuple[Any, pd.DataFrame]]:
    if df is None or df.empty:
        return []
    image_col = "image_id" if "image_id" in df.columns else "id"
    groups: List[Tuple[Any, pd.DataFrame]] = []
    for image_id, grp in df.groupby(image_col, sort=False):
        groups.append((image_id, grp.reset_index(drop=True)))
    return groups


def prepare_images_export(df: pd.DataFrame) -> pd.DataFrame:
    """One row per image with aggregated detection summaries."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_image_columns())

    rows: List[Dict[str, Any]] = []
    for image_id, grp in _group_raw_by_image(df):
        first = grp.iloc[0]
        capture_date = _clean_str(first.get("capture_date"))
        capture_time = _clean_str(first.get("capture_time"))
        capture_dt = f"{capture_date} {capture_time}".strip()

        detections = [_build_detection_record(r) for _, r in grp.iterrows()]
        conf_pcts = [d["confidence_percent"] for d in detections if d.get("confidence_percent") is not None]
        record_types = list(dict.fromkeys(
            d.get("record_type") or ""
            for d in detections
            if d.get("record_type")
        ))

        det_ids = [
            int(r.get("detection_id"))
            for _, r in grp.iterrows()
            if r.get("detection_id") is not None and pd.notna(r.get("detection_id"))
        ]

        rows.append({
            "Image ID": int(image_id) if pd.notna(image_id) else "",
            "Filename": _clean_str(first.get("filename")),
            "File Hash": _clean_str(first.get("file_hash")),
            "Station": _clean_str(first.get("station_id")),
            "Camera": _clean_str(first.get("camera_id")),
            "Capture Date": capture_date,
            "Capture Time": capture_time,
            "Capture DateTime": capture_dt,
            "Day / Night": _clean_str(first.get("day_night")),
            "Temperature": _clean_str(first.get("temperature")),
            "Brightness": _clean_str(first.get("brightness")),
            "Notes": _clean_str(first.get("user_notes")),
            "Processed At": _clean_str(first.get("processed_at")),
            "Detection Count": len(grp),
            "Detection IDs": ", ".join(str(x) for x in det_ids),
            "Species List": _format_species_list_for_image(grp),
            "Record Types": ", ".join(record_types),
            "Detections Summary": _format_detections_summary_for_image(grp),
            "Lowest Confidence %": min(conf_pcts) if conf_pcts else "",
            "Highest Confidence %": max(conf_pcts) if conf_pcts else "",
        })

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["Capture Date", "Capture Time", "Filename"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False, False, True])
    return out.reset_index(drop=True)


def prepare_detections_detail_export(df: pd.DataFrame) -> pd.DataFrame:
    """Full per-detection detail — used on the Detections sheet and for complete data access."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_detection_detail_columns())

    image_col = "image_id" if "image_id" in df.columns else "id"
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        species = _clean_str(r.get("detected_animal"))
        capture_date = _clean_str(r.get("capture_date"))
        capture_time = _clean_str(r.get("capture_time"))
        capture_dt = f"{capture_date} {capture_time}".strip()
        conf = r.get("detection_confidence")
        conf_pct = _pct(conf)
        bbox_text, bx, by, bw, bh, _ = _format_bbox(r.get("bbox"))
        det_id = r.get("detection_id")

        rows.append({
            "Detection ID": int(det_id) if det_id is not None and pd.notna(det_id) else "",
            "Image ID": int(r[image_col]) if image_col in r and pd.notna(r.get(image_col)) else "",
            "Filename": _clean_str(r.get("filename")),
            "Record Type": _record_type(species),
            "Species": species,
            "Scientific Name": _clean_str(r.get("scientific_name")),
            "Confidence": round(float(conf), 4) if conf_pct is not None and pd.notna(conf) else "",
            "Confidence %": conf_pct if conf_pct is not None else "",
            "MegaDetector Confidence %": _pct(r.get("md_confidence")) or "",
            "SpeciesNet Confidence %": _pct(r.get("speciesnet_confidence")) or "",
            "Detection Method": _clean_str(r.get("detection_method")),
            "IDE ID": _clean_str(r.get("ide_id")),
            "Bounding Box": bbox_text,
            "BBox Left %": bx if bx is not None else "",
            "BBox Top %": by if by is not None else "",
            "BBox Width %": bw if bw is not None else "",
            "BBox Height %": bh if bh is not None else "",
            "Top Model Candidates": _format_candidates(r.get("top_candidates")),
            "Taxonomy": _format_taxonomy(r.get("taxonomy_hierarchy")),
            "Model Breakdown": _format_model_breakdown(r.get("model_breakdown")),
            "Raw Model Output": _json_pretty(r.get("raw_model_output")),
            "Station": _clean_str(r.get("station_id")),
            "Camera": _clean_str(r.get("camera_id")),
            "Capture DateTime": capture_dt,
            "Day / Night": _clean_str(r.get("day_night")),
            "File Hash": _clean_str(r.get("file_hash")),
            "Brightness": _clean_str(r.get("brightness")),
            "Notes": _clean_str(r.get("user_notes")),
            "Processed At": _clean_str(r.get("processed_at")),
        })

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["Capture DateTime", "Filename", "Detection ID"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False, True, True])
    return out.reset_index(drop=True)


# Backward-compatible alias — primary export is now image-grouped.
def prepare_detections_export(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_images_export(df)


def summary_by_species(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Species", "Record Type", "Detections", "% of Total", "Unique Images"])
    image_col = "image_id" if "image_id" in raw_df.columns else "id"
    work = raw_df.copy()
    work["_species"] = work["detected_animal"].map(canonical_species_name)
    work["_rtype"] = work["_species"].map(_record_type)
    total_dets = len(work)
    rows = []
    for (species, rtype), grp in work.groupby(["_species", "_rtype"], dropna=False):
        c = len(grp)
        pct = f"{(c / total_dets * 100):.1f}%" if total_dets > 0 else "0.0%"
        rows.append({
            "Species": species,
            "Record Type": rtype,
            "Detections": c,
            "% of Total": pct,
            "Unique Images": grp[image_col].nunique() if image_col in grp.columns else c,
        })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["_is_wild"] = df_out["Record Type"].map(lambda t: 0 if t == "Wildlife" else 1)
        df_out = df_out.sort_values(["_is_wild", "Detections", "Species"], ascending=[True, False, True]).drop(columns=["_is_wild"])
    return df_out.reset_index(drop=True)


def summary_by_station(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary by station matrix: Station totals plus distinct numeric columns
    for each wildlife species detected, followed by an overall Total row.
    Each species and its count is cleanly placed in its own separate cell.
    """
    base_cols = ["Station", "Images", "Total Detections", "Wildlife Detections", "Non-Wildlife Detections", "Total Species"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=base_cols)

    image_col = "image_id" if "image_id" in raw_df.columns else "id"
    work = raw_df.copy()
    work["_station"] = work["station_id"].map(_clean_str)
    work["_species"] = work["detected_animal"].map(canonical_species_name)
    work["_rtype"] = work["_species"].map(_record_type)

    wildlife_df = work[work["_rtype"] == "Wildlife"]
    wildlife_species_list = [
        s for s in wildlife_df["_species"].value_counts().index.tolist() if s
    ]

    rows = []
    for station, grp in work.groupby("_station", dropna=False):
        st_wild = grp[grp["_rtype"] == "Wildlife"]
        st_non_wild = grp[grp["_rtype"] != "Wildlife"]
        st_name = station if station else "(none)"
        sp_counts = st_wild["_species"].value_counts()

        row_dict: Dict[str, Any] = {
            "Station": st_name,
            "Images": grp[image_col].nunique() if image_col in grp.columns else len(grp),
            "Total Detections": len(grp),
            "Wildlife Detections": len(st_wild),
            "Non-Wildlife Detections": len(st_non_wild),
            "Total Species": len(sp_counts),
        }
        for sp in wildlife_species_list:
            row_dict[sp] = int(sp_counts.get(sp, 0))
        rows.append(row_dict)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Total Detections", ascending=False).reset_index(drop=True)
        total_row: Dict[str, Any] = {
            "Station": "Total (All Stations)",
            "Images": work[image_col].nunique() if image_col in work.columns else len(work),
            "Total Detections": len(work),
            "Wildlife Detections": len(wildlife_df),
            "Non-Wildlife Detections": len(work) - len(wildlife_df),
            "Total Species": len(wildlife_species_list),
        }
        for sp in wildlife_species_list:
            total_row[sp] = int((wildlife_df["_species"] == sp).sum())
        df_out = pd.concat([df_out, pd.DataFrame([total_row])], ignore_index=True)

    return df_out


def species_by_station(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabular station-by-species breakdown where Station, Species, and Count
    are isolated in distinct individual cells for Excel sorting and filtering.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Station", "Species", "Record Type", "Detections", "Unique Images"])
    image_col = "image_id" if "image_id" in raw_df.columns else "id"
    work = raw_df.copy()
    work["_station"] = work["station_id"].map(_clean_str)
    work["_species"] = work["detected_animal"].map(canonical_species_name)
    work["_rtype"] = work["_species"].map(_record_type)

    rows = []
    for (st, sp, rt), grp in work.groupby(["_station", "_species", "_rtype"], dropna=False):
        rows.append({
            "Station": st or "(none)",
            "Species": sp,
            "Record Type": rt,
            "Detections": len(grp),
            "Unique Images": grp[image_col].nunique() if image_col in grp.columns else len(grp),
        })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["_is_wild"] = df_out["Record Type"].map(lambda t: 0 if t == "Wildlife" else 1)
        df_out = df_out.sort_values(["Station", "_is_wild", "Detections"], ascending=[True, True, False]).drop(columns=["_is_wild"])
    return df_out.reset_index(drop=True)




def build_json_export(
    raw_df: pd.DataFrame,
    export_df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    export_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    image_col = "image_id" if "image_id" in raw_df.columns else "id"
    images: List[Dict[str, Any]] = []

    for image_id, grp in _group_raw_by_image(raw_df):
        first = grp.iloc[0]
        capture_date = _clean_str(first.get("capture_date"))
        capture_time = _clean_str(first.get("capture_time"))
        detections = [_build_detection_record(r) for _, r in grp.iterrows()]
        conf_pcts = [d["confidence_percent"] for d in detections if d.get("confidence_percent") is not None]

        images.append({
            "image_id": int(image_id) if pd.notna(image_id) else None,
            "filename": _clean_str(first.get("filename")),
            "file_hash": _clean_str(first.get("file_hash")),
            "station": _clean_str(first.get("station_id")),
            "camera": _clean_str(first.get("camera_id")),
            "capture_date": capture_date,
            "capture_time": capture_time,
            "capture_datetime": f"{capture_date} {capture_time}".strip(),
            "day_night": _clean_str(first.get("day_night")),
            "temperature": _clean_str(first.get("temperature")),
            "brightness": _clean_str(first.get("brightness")),
            "notes": _clean_str(first.get("user_notes")),
            "processed_at": _clean_str(first.get("processed_at")),
            "detection_count": len(detections),
            "species_list": _format_species_list_for_image(grp),
            "species_details": _species_detail_items(grp),
            "detections_summary": _format_detections_summary_for_image(grp),
            "lowest_confidence_percent": min(conf_pcts) if conf_pcts else None,
            "highest_confidence_percent": max(conf_pcts) if conf_pcts else None,
            "detections": detections,
        })

    unique_images = raw_df[image_col].nunique() if not raw_df.empty and image_col in raw_df.columns else 0
    info = export_info or {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": int(unique_images),
        "total_detections": len(raw_df) if raw_df is not None else 0,
        "filters_applied": filters or {},
    }
    return {
        "export_info": info,
        "summary": {
            "by_species": summary_by_species(raw_df).to_dict(orient="records"),
            "by_station": summary_by_station(raw_df).to_dict(orient="records"),
            "species_by_station": species_by_station(raw_df).to_dict(orient="records"),
        },
        "images": images,
    }


def detections_csv_bytes(
    export_df: pd.DataFrame,
    metadata_rows: Optional[List[Dict[str, str]]] = None,
) -> bytes:
    buf = StringIO()
    if metadata_rows:
        buf.write("# ViumbeLens camera-trap export\n")
        for row in metadata_rows:
            buf.write(f"# {row['Field']}: {row['Value']}\n")
        buf.write("#\n")
    export_df.to_csv(buf, index=False)
    return "\ufeff".encode("utf-8") + buf.getvalue().encode("utf-8")


def detections_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def detections_excel_bytes(
    images_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    metadata_rows: Optional[List[Dict[str, str]]] = None,
) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if metadata_rows:
            pd.DataFrame(metadata_rows).to_excel(writer, index=False, sheet_name="Export Info")
        images_df.to_excel(writer, index=False, sheet_name="Images")
        detections_df.to_excel(writer, index=False, sheet_name="Detections")
        summary_by_species(raw_df).to_excel(writer, index=False, sheet_name="Summary by Species")
        summary_by_station(raw_df).to_excel(writer, index=False, sheet_name="Summary by Station")
        species_by_station(raw_df).to_excel(writer, index=False, sheet_name="Species by Station")
        _style_excel_workbook(writer.book)
    buf.seek(0)
    return buf.read()



def _style_excel_workbook(workbook) -> None:
    try:
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="166534")

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        if sheet.max_row >= 1:
            for cell in sheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for col_idx, column_cells in enumerate(sheet.columns, start=1):
            max_len = 0
            column = get_column_letter(col_idx)
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, min(len(value), 60))
            sheet.column_dimensions[column].width = max(12, min(max_len + 2, 42))

        wrap_cols = {
            "Species List", "Detections Summary", "Top Model Candidates", "Model Breakdown",
            "Raw Model Output", "Value", "Notes", "Stations included",
        }
        header_names = {cell.value for cell in sheet[1]} if sheet.max_row >= 1 else set()
        wrap = bool(header_names & wrap_cols)
        for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=wrap)
