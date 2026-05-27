"""
Ensemble fusion for the multi-model wildlife detection pipeline.

Detection fusion: NMS across MDv5a + MDv1000 bounding boxes.
Classification fusion: weighted score merge across BioClip + SpeciesNet.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# IoU threshold for considering two boxes the same object
_NMS_IOU = 0.50

# Agreement bonus when both classifiers pick the same top-1 species
_AGREEMENT_BONUS = 0.08

# Classifier weights [bioclip, speciesnet].
# SpeciesNet trained on 65 M camera-trap images → slightly higher weight.
_DEFAULT_WEIGHTS: Tuple[float, float] = (0.45, 0.55)


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


# ---------------------------------------------------------------------------
# Classification fusion
# ---------------------------------------------------------------------------

def fuse_species(
    bioclip: List[Tuple[str, float]],
    speciesnet: List[Tuple[str, float]],
    weights: Tuple[float, float] = _DEFAULT_WEIGHTS,
) -> Dict:
    """
    Merge species predictions from BioClip and SpeciesNet.

    Returns:
        species       – display name of the top prediction
        confidence    – fused confidence score (0–1)
        agreement     – 'High' | 'Medium' | 'Low'
        bioclip_top   – (label, conf) from BioClip or None
        speciesnet_top – (label, conf) from SpeciesNet or None
        all_candidates – top-5 fused (label, conf) pairs
    """
    w_bio, w_snet = weights
    import json

    def _clean_snet_label(label: str) -> str:
        if label.startswith("{") and label.endswith("}"):
            try:
                return json.loads(label).get("display", label)
            except Exception:
                pass
        return label

    bc_top: Optional[Tuple[str, float]] = bioclip[0] if bioclip else None
    sn_top_raw: Optional[Tuple[str, float]] = speciesnet[0] if speciesnet else None
    sn_top: Optional[Tuple[str, float]] = (
        (_clean_snet_label(sn_top_raw[0]), sn_top_raw[1]) if sn_top_raw else None
    )

    # Accumulate weighted scores; use lower-cased name as key
    scores: Dict[str, float] = {}
    name_map: Dict[str, str] = {}  # lower → original casing

    for label, score in bioclip:
        key = label.lower().strip()
        scores[key] = scores.get(key, 0.0) + score * w_bio
        name_map.setdefault(key, label)

    for label, score in speciesnet:
        clean_label = _clean_snet_label(label)
        key = clean_label.lower().strip()
        scores[key] = scores.get(key, 0.0) + score * w_snet
        name_map.setdefault(key, clean_label)

    if not scores:
        return {
            "species": "Unknown",
            "confidence": 0.0,
            "agreement": "Low",
            "bioclip_top": None,
            "speciesnet_top": None,
            "all_candidates": [],
        }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_key, top_score = ranked[0]

    # Determine agreement and apply bonus
    agreement = "Low"
    if bc_top and sn_top:
        bc_key = bc_top[0].lower().strip()
        sn_key = sn_top[0].lower().strip()
        if bc_key == sn_key or bc_key in sn_key or sn_key in bc_key:
            top_score = min(1.0, top_score + _AGREEMENT_BONUS)
            agreement = "High"
        elif any(word in sn_key for word in bc_key.split() if len(word) > 3):
            top_score = min(1.0, top_score + _AGREEMENT_BONUS * 0.5)
            agreement = "Medium"
    elif bc_top or sn_top:
        agreement = "Medium"

    display_name = name_map.get(top_key, top_key.title())

    return {
        "species": display_name,
        "confidence": round(min(top_score, 1.0), 4),
        "agreement": agreement,
        "bioclip_top": bc_top,
        "speciesnet_top": sn_top,
        "all_candidates": [
            (name_map.get(k, k.title()), round(v, 4))
            for k, v in ranked[:5]
        ],
    }
