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
# BioClip is a general CLIP model not trained on camera-trap imagery, so we
# trust it less than SpeciesNet for this domain.
_DEFAULT_WEIGHTS: Tuple[float, float] = (0.05, 0.95)
_SPECIESNET_FIRST_WEIGHTS: Tuple[float, float] = (0.0, 1.0)  # SpeciesNet only (BioClip disabled)

# Night-time weights: SpeciesNet was explicitly trained on nocturnal/IR camera
# trap images; BioCLIP has no such specialisation and is nearly blind to IR.
_NIGHT_WEIGHTS: Tuple[float, float] = (0.02, 0.98)


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
    Compare a BioCLIP species name against SpeciesNet's full taxonomy metadata.
    Fallback used when no structured BioCLIP taxonomy is available.

    High   – common/display name match (exact or contained)
    Medium – shared meaningful word OR species word in hierarchy
    Low    – no detectable overlap
    """
    bc = bc_name.lower().strip()
    sn_display  = sn_meta.get("display", "").lower().strip()
    sn_common   = sn_meta.get("common_name", "").lower().strip()
    sn_hierarchy: List[str] = [h.lower() for h in sn_meta.get("hierarchy", [])]

    for sn_name in (sn_display, sn_common):
        if not sn_name:
            continue
        if bc == sn_name or bc in sn_name or sn_name in bc:
            return "High"

    bc_words = {w for w in bc.split() if len(w) > 3}
    for sn_name in (sn_display, sn_common):
        sn_words = {w for w in sn_name.split() if len(w) > 3}
        if bc_words & sn_words:
            return "High"

    if bc_words and sn_hierarchy:
        hierarchy_str = " ".join(sn_hierarchy)
        if any(word in hierarchy_str for word in bc_words):
            return "Medium"

    return "Low"


def _taxonomy_agreement_structured(bc_tax: Dict, sn_meta: Dict, bc_family_override: Optional[str] = None) -> str:
    """
    Taxonomy-aware agreement using BioCLIP's full taxonomic path.

    Compares BioCLIP's family / genus / species against SpeciesNet's hierarchy
    and common/display names — much more reliable than word matching.

    High   – genus match in hierarchy, OR species common-name match
    Medium – family match in hierarchy (same ecological group)
    Low    – no shared taxon
    """
    sn_hierarchy = [h.lower() for h in sn_meta.get("hierarchy", [])]
    sn_common    = sn_meta.get("common_name", "").lower().strip()
    sn_display   = sn_meta.get("display", "").lower().strip()

    bc_species = bc_tax.get("species", "").lower()
    bc_genus   = bc_tax.get("genus", "").lower()
    bc_family  = bc_tax.get("family", "").lower()
    bc_order   = bc_tax.get("order", "").lower()

    # High: species name matches common/display
    for sn_name in (sn_common, sn_display):
        if not sn_name:
            continue
        if bc_species == sn_name or bc_species in sn_name or sn_name in bc_species:
            return "High"
        bc_words = {w for w in bc_species.split() if len(w) > 3}
        sn_words = {w for w in sn_name.split() if len(w) > 3}
        if bc_words & sn_words:
            return "High"

    # High: genus appears in SpeciesNet's taxonomy hierarchy
    if bc_genus and bc_genus in sn_hierarchy:
        return "High"

    # Medium: family matches (same ecological group)
    if bc_family and bc_family in sn_hierarchy:
        return "Medium"

    # Medium: independent family-level BioCLIP prediction also matches
    if bc_family_override and bc_family_override.lower() in sn_hierarchy:
        return "Medium"

    # Medium: order matches (broad agreement, e.g. both Carnivora)
    if bc_order and bc_order in sn_hierarchy:
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
    bioclip_taxonomy: Optional[Dict] = None,
    speciesnet_bypass_threshold: float = 0.0,
) -> Dict:
    """
    Merge species predictions from BioClip and SpeciesNet.

    Parameters
    ----------
    bioclip          : list of (label, confidence) from BioCLIP
    speciesnet       : list of (label, confidence) from SpeciesNet
                       Labels may be JSON-encoded dicts with taxonomy metadata.
    weights          : (w_bioclip, w_speciesnet) — overridden at night.
    is_night         : when True, applies night-time weights that heavily favour
                       SpeciesNet (trained on nocturnal/IR camera-trap imagery).
    bioclip_taxonomy : structured dict from BioClipClassifier.predict_taxonomy().
                       When present, enables genus/family-level agreement
                       instead of word matching.

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

    # Option 3: bypass fusion entirely when SpeciesNet is sufficiently confident
    if (
        speciesnet_bypass_threshold > 0.0
        and sn_top_raw is not None
        and sn_top_raw[1] >= speciesnet_bypass_threshold
    ):
        sn_display = _display(sn_top_raw[0])
        sn_meta = _parse_snet_meta(sn_top_raw[0])
        agreement = "High"
        if bc_top and bioclip_taxonomy:
            try:
                agreement = _taxonomy_agreement_structured(
                    bioclip_taxonomy.get("top", {}),
                    sn_meta,
                    bc_family_override=bioclip_taxonomy.get("family_prediction"),
                )
            except Exception:
                pass
        elif bc_top:
            agreement = _taxonomy_agreement(bc_top[0], sn_meta)
        conf = min(1.0, sn_top_raw[1] + (_AGREEMENT_BONUS if agreement == "High" else 0.0))
        return {
            "species": sn_display,
            "confidence": round(conf, 4),
            "agreement": agreement,
            "bioclip_top": bc_top,
            "speciesnet_top": (sn_display, sn_top_raw[1]),
            "all_candidates": [(sn_display, sn_top_raw[1])]
            + [(_display(l), s) for l, s in speciesnet[1:5]],
        }
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
        if bioclip_taxonomy and bioclip_taxonomy.get("top"):
            agreement = _taxonomy_agreement_structured(
                bioclip_taxonomy["top"],
                sn_meta,
                bc_family_override=bioclip_taxonomy.get("family_prediction"),
            )
        else:
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


def fuse_species_speciesnet_first(
    speciesnet: List[Tuple[str, float]],
    bioclip: Optional[List[Tuple[str, float]]] = None,
    speciesnet_bypass_threshold: float = 0.0,
) -> Dict:
    """
    SpeciesNet-first fusion: returns ALL SpeciesNet outputs, no BioClip weighting.

    This is the recommended approach since:
    - SpeciesNet trained on 65M camera-trap images (domain-specific)
    - BioClip is a general CLIP model (outputs generic names: "cat", "dog", etc.)
    - SpeciesNet outputs precise species names with taxonomy

    Parameters
    ----------
    speciesnet : list of ALL (label, confidence) from SpeciesNet
    bioclip    : optional, for agreement scoring only (not used in weighting)
    speciesnet_bypass_threshold : ignored (always use SpeciesNet)

    Returns
    -------
    dict with:
      - species: top SpeciesNet prediction
      - confidence: SpeciesNet top-1 confidence
      - agreement: Low (we're bypassing BioClip)
      - speciesnet_top: top SpeciesNet prediction
      - all_candidates: ALL SpeciesNet outputs (not just top-5)
      - bioclip_top: None (not used)
    """
    if not speciesnet:
        return {
            "species": "Unknown",
            "confidence": 0.0,
            "agreement": "Low",
            "bioclip_top": None,
            "speciesnet_top": None,
            "all_candidates": [],
        }

    def _display(label: str) -> str:
        """Extract human-readable name from SpeciesNet JSON label."""
        meta = _parse_snet_meta(label)
        return meta.get("display") or meta.get("common_name") or label.strip()

    # Top prediction from SpeciesNet
    sn_top_raw = speciesnet[0]
    sn_display = _display(sn_top_raw[0])
    sn_confidence = sn_top_raw[1]

    # Agreement score: check if BioClip agrees (optional)
    agreement = "Low"  # We're not weighting BioClip
    if bioclip and bioclip[0]:
        bc_top_name = bioclip[0][0]
        sn_meta = _parse_snet_meta(sn_top_raw[0])
        agreement = _taxonomy_agreement(bc_top_name, sn_meta)

    # Return ALL SpeciesNet outputs, not just top-5
    all_candidates = [(_display(label), round(conf, 4)) for label, conf in speciesnet]

    return {
        "species": sn_display,
        "confidence": round(min(sn_confidence, 1.0), 4),
        "agreement": agreement,
        "bioclip_top": bioclip[0] if bioclip else None,
        "speciesnet_top": (sn_display, round(sn_confidence, 4)),
        "all_candidates": all_candidates,  # ← ALL outputs, not top-5
    }
