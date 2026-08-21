"""
Ensemble fusion for the wildlife detection pipeline.

Detection fusion: NMS across MDv5a bounding boxes.
Classification: SpeciesNet only (65 M camera-trap images, full taxonomy).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

# IoU threshold for considering two boxes the same object
_NMS_IOU = 0.50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou(a: List[float], b: List[float]) -> float:
    """IoU between two [x, y, w, h] normalised boxes (top-left origin)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    return inter / (area_a + area_b - inter)


def _parse_snet_meta(label: str) -> Dict:
    """
    Parse a SpeciesNet label string into structured metadata.

    SpeciesNet labels are JSON-encoded dicts with keys:
      display, common_name, scientific_name, hierarchy, id
    Falls back to a plain display dict when the label isn't JSON.
    """
    if label.startswith("{"):
        try:
            return json.loads(label)
        except Exception:
            pass
    return {"display": label.strip(), "common_name": label.strip()}




# ---------------------------------------------------------------------------
# Detection fusion
# ---------------------------------------------------------------------------

def nms_merge_detections(
    cands_a: List[Dict],
    cands_b: List[Dict],
    source_a: str = "MDv5a",
    source_b: str = "MDv1000",
    iou_threshold: float = _NMS_IOU,
) -> List[Dict]:
    """
    Merge two candidate lists from different detectors via greedy NMS.

    Each candidate dict must have: label, conf, bbox ([x,y,w,h] normalised).
    Returns merged list; each entry gains a 'sources' list field.
    """
    tagged_a = [dict(c, sources=[source_a]) for c in cands_a]
    tagged_b = [dict(c, sources=[source_b]) for c in cands_b]
    all_cands = tagged_a + tagged_b

    if not all_cands:
        return []

    all_cands.sort(key=lambda x: x["conf"], reverse=True)

    merged: List[Dict] = []
    suppressed: set = set()

    for i, cand in enumerate(all_cands):
        if i in suppressed:
            continue

        keep = dict(cand)
        bbox_i = keep.get("bbox") or []

        for j in range(i + 1, len(all_cands)):
            if j in suppressed:
                continue
            other = all_cands[j]
            bbox_j = other.get("bbox") or []
            if bbox_i and bbox_j and _iou(bbox_i, bbox_j) >= iou_threshold:
                # Same object seen by both detectors — merge
                keep["sources"] = list(set(keep["sources"] + other["sources"]))
                # Keep the higher-confidence box's geometry
                if other["conf"] > keep["conf"]:
                    keep["bbox"] = bbox_j
                    keep["conf"] = other["conf"]
                suppressed.add(j)

        suppressed.add(i)
        merged.append(keep)

    return merged



def fuse_species_speciesnet_first(
    speciesnet: List[Tuple[str, float]],
) -> Dict:
    """
    SpeciesNet classification fusion — returns the top prediction and all candidates.

    SpeciesNet is trained on 65 M camera-trap images and outputs precise species
    names with full taxonomy (id, common_name, scientific_name, hierarchy).
    Up to top_k=20 results are passed in; all are returned as all_candidates.

    Returns
    -------
    dict with:
      species        — top-1 common name ("African Lion")
      confidence     — top-1 confidence score
      speciesnet_top — (name, confidence) tuple for the top prediction
      all_candidates — list of (name, confidence) for all predictions
    """
    if not speciesnet:
        return {
            "species": "Unknown",
            "confidence": 0.0,
            "speciesnet_top": None,
            "all_candidates": [],
        }

    def _display(label: str) -> str:
        meta = _parse_snet_meta(label)
        return meta.get("display") or meta.get("common_name") or label.strip()

    sn_top_raw = speciesnet[0]
    sn_display = _display(sn_top_raw[0])
    sn_confidence = sn_top_raw[1]

    all_candidates = [(_display(label), round(conf, 4)) for label, conf in speciesnet]

    return {
        "species": sn_display,
        "confidence": round(min(sn_confidence, 1.0), 4),
        "speciesnet_top": (sn_display, round(sn_confidence, 4)),
        "all_candidates": all_candidates,
    }
