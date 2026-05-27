"""
Ensemble fusion for the multi-model wildlife detection pipeline.

Detection fusion: NMS across MDv5a + MDv1000 bounding boxes.
Classification fusion: weighted score merge across BioClip + SpeciesNet.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

# IoU threshold for considering two boxes the same object
_NMS_IOU = 0.50

# Agreement bonus when both classifiers pick the same top-1 species
_AGREEMENT_BONUS = 0.08

# Classifier weights [bioclip, speciesnet].
# SpeciesNet trained on 65 M camera-trap images → higher weight overall.
_DEFAULT_WEIGHTS: Tuple[float, float] = (0.40, 0.60)

# Night-time weights: SpeciesNet was explicitly trained on nocturnal/IR camera
# trap images; BioCLIP has no such specialisation.
_NIGHT_WEIGHTS: Tuple[float, float] = (0.25, 0.75)


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


def _taxonomy_agreement(bc_name: str, sn_meta: Dict) -> str:
    """
    Compare a BioCLIP label against SpeciesNet's full taxonomy metadata.

    Returns 'High', 'Medium', or 'Low'.

    High   – common name / display name match (exact or contained)
    Medium – shared meaningful word OR BioCLIP word found in the taxonomic
             hierarchy (family / genus level agreement)
    Low    – no detectable overlap
    """
    bc = bc_name.lower().strip()

    sn_display  = sn_meta.get("display", "").lower().strip()
    sn_common   = sn_meta.get("common_name", "").lower().strip()
    sn_hierarchy: List[str] = [h.lower() for h in sn_meta.get("hierarchy", [])]

    # ── High: name-level agreement ─────────────────────────────────────────
    for sn_name in (sn_display, sn_common):
        if not sn_name:
            continue
        if bc == sn_name or bc in sn_name or sn_name in bc:
            return "High"

    # Word-level overlap on meaningful tokens (len > 3)
    bc_words = {w for w in bc.split() if len(w) > 3}
    for sn_name in (sn_display, sn_common):
        sn_words = {w for w in sn_name.split() if len(w) > 3}
        if bc_words & sn_words:
            return "High"

    # ── Medium: taxonomy hierarchy match ──────────────────────────────────
    if bc_words and sn_hierarchy:
        hierarchy_str = " ".join(sn_hierarchy)
        if any(word in hierarchy_str for word in bc_words):
            return "Medium"

    return "Low"


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
    is_night: bool = False,
) -> Dict:
    """
    Merge species predictions from BioClip and SpeciesNet.

    Parameters
    ----------
    bioclip     : list of (label, confidence) from BioCLIP
    speciesnet  : list of (label, confidence) from SpeciesNet
                  Labels may be JSON-encoded dicts with taxonomy metadata.
    weights     : (w_bioclip, w_speciesnet) — overridden at night.
    is_night    : when True, applies night-time weights that heavily favour
                  SpeciesNet (trained on nocturnal/IR camera-trap imagery).

    Returns
    -------
    dict with keys: species, confidence, agreement, bioclip_top,
                    speciesnet_top, all_candidates
    """
    # Dynamic weights: SpeciesNet dominates at night
    if is_night:
        weights = _NIGHT_WEIGHTS
    w_bio, w_snet = weights

    def _display(label: str) -> str:
        """Extract human-readable name from a SpeciesNet JSON label."""
        meta = _parse_snet_meta(label)
        return meta.get("display") or meta.get("common_name") or label.strip()

    bc_top: Optional[Tuple[str, float]] = bioclip[0] if bioclip else None
    sn_top_raw: Optional[Tuple[str, float]] = speciesnet[0] if speciesnet else None
    sn_top: Optional[Tuple[str, float]] = (
        (_display(sn_top_raw[0]), sn_top_raw[1]) if sn_top_raw else None
    )

    # Accumulate weighted scores; use lower-cased display name as key
    scores: Dict[str, float] = {}
    name_map: Dict[str, str] = {}  # lower → original casing

    for label, score in bioclip:
        key = label.lower().strip()
        scores[key] = scores.get(key, 0.0) + score * w_bio
        name_map.setdefault(key, label)

    for label, score in speciesnet:
        display = _display(label)
        key = display.lower().strip()
        scores[key] = scores.get(key, 0.0) + score * w_snet
        name_map.setdefault(key, display)

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

    # ── Taxonomy-aware agreement ───────────────────────────────────────────
    agreement = "Low"
    if bc_top and sn_top_raw:
        sn_meta = _parse_snet_meta(sn_top_raw[0])
        agreement = _taxonomy_agreement(bc_top[0], sn_meta)
        if agreement == "High":
            top_score = min(1.0, top_score + _AGREEMENT_BONUS)
        elif agreement == "Medium":
            top_score = min(1.0, top_score + _AGREEMENT_BONUS * 0.5)
    elif bc_top or sn_top_raw:
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
