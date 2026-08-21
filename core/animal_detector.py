"""
Wildlife Identification Pipeline
─────────────────────────────────────────────
Stage 1 – Detection    MDv5a → candidates
Stage 2 – Classify     SpeciesNet → [(species, conf), ...] with full taxonomy

Each result dict includes a '_model_events' list for real-time SSE display.
"""

from __future__ import annotations

import json
import os
import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from .ensemble_engine import nms_merge_detections, fuse_species_speciesnet_first


def _parse_sn_label(label: str) -> Dict:
    """Parse a SpeciesNet JSON-encoded taxonomy label into a plain dict."""
    if isinstance(label, str) and label.startswith('{'):
        try:
            return json.loads(label)
        except Exception:
            pass
    return {'display': str(label).strip(), 'common_name': str(label).strip()}

# MegaDetector (Official Package)
try:
    from megadetector.detection import run_detector
    MD_AVAILABLE = True
except ImportError as exc:
    MD_AVAILABLE = False
    print(f"Warning: megadetector not installed. Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# MegaDetector wrapper (supports any model version string)
# ─────────────────────────────────────────────────────────────────────────────

class MegaDetectorWrapper:
    """Wrapper for a single MegaDetector model (MDv5a or any version string)."""

    CLASS_MAP = {"1": "Animal", "2": "Person", "3": "Vehicle"}

    def __init__(
        self,
        model_version: str = "MDV5a",
        confidence_threshold: float = 0.2,
        low_spec: bool = False,
    ):
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.low_spec = low_spec
        self.model = None
        self.load_error: Optional[str] = None
        self._load_model()

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold

    def _load_model(self) -> None:
        if not MD_AVAILABLE:
            self.load_error = "megadetector library not found"
            return
        try:
            print(f"Loading MegaDetector {self.model_version} (low_spec={self.low_spec})…")
            self.model = run_detector.load_detector(self.model_version)

            if self.low_spec and self.model:
                try:
                    import torch
                    target = getattr(self.model, "model", None)
                    if target is None and isinstance(self.model, torch.nn.Module):
                        target = self.model
                    if target is not None:
                        q = torch.quantization.quantize_dynamic(
                            target,
                            {torch.nn.Linear, torch.nn.LSTM},
                            dtype=torch.qint8,
                        )
                        if hasattr(self.model, "model"):
                            self.model.model = q
                        else:
                            self.model = q
                        print(f"  {self.model_version}: INT8 quantization applied.")
                except Exception as q_err:
                    print(f"  {self.model_version}: quantization failed ({q_err}), using fp32.")

            print(f"  {self.model_version} loaded successfully.")
        except Exception as exc:
            print(f"Error loading {self.model_version}: {exc}")
            self.load_error = str(exc)

    def get_status(self) -> Dict:
        return {"loaded": self.model is not None, "error": self.load_error}

    def detect_all_candidates(
        self,
        image_path: str,
        _capture_raw: bool = False,
    ) -> Union[List[Dict], Tuple[List[Dict], List[Dict]]]:
        """Run inference; return detections above threshold.

        When _capture_raw=True, returns (filtered, raw_all) where raw_all
        contains every box the model produced, regardless of threshold.
        This avoids running the model twice when raw output is needed.
        """
        if self.model is None:
            return ([], []) if _capture_raw else []
        try:
            image = Image.open(image_path)
            result = self.model.generate_detections_one_image(image)
            filtered: List[Dict] = []
            raw_all: List[Dict] = []
            for det in result.get("detections", []):
                conf = float(det["conf"])
                label = self.CLASS_MAP.get(det["category"], "Unknown")
                bbox = det.get("bbox")
                if bbox:
                    raw_all.append({
                        "category": label,
                        "confidence": round(conf, 4),
                        "bbox": bbox,
                    })
                    if conf >= self.confidence_threshold:
                        filtered.append({
                            "label": label,
                            "conf": conf,
                            "bbox": bbox,
                        })
            return (filtered, raw_all) if _capture_raw else filtered
        except Exception as exc:
            print(f"{self.model_version} inference error: {exc}")
            return ([], []) if _capture_raw else []

    # Legacy single-result interface kept for backward compat
    def detect_primary(
        self, image_path: str
    ) -> Tuple[str, float, Optional[List[float]]]:
        cands = self.detect_all_candidates(image_path)
        if not cands:
            return "Empty", 0.0, None
        best = max(cands, key=lambda x: x["conf"])
        return best["label"], best["conf"], best["bbox"]

    def detect_all(self, image_path: str) -> Any:
        if self.model is None:
            return {"detections": []}
        try:
            return self.model.generate_detections_one_image(Image.open(image_path))
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AnimalDetector:
    """Orchestrates the MDv5a + SpeciesNet wildlife ID pipeline."""

    def __init__(
        self,
        megadetector: Optional[MegaDetectorWrapper],
        confidence_threshold: float = 0.2,
        speciesnet=None,  # SpeciesNetWrapper | None
        speciesnet_bypass_threshold: float = 0.60,
        use_speciesnet_first: bool = True,
        **kwargs,  # absorb any legacy keyword args
    ):
        self.megadetector = megadetector
        self.speciesnet = speciesnet
        self._speciesnet_bypass_threshold = speciesnet_bypass_threshold
        self._use_speciesnet_first = use_speciesnet_first

        if self.megadetector:
            self.megadetector.set_confidence_threshold(confidence_threshold)
        self._threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify(self, crop: Image.Image) -> List[Tuple[str, float]]:
        """Run SpeciesNet on an animal crop. Returns (label, confidence) pairs."""
        if not self.speciesnet or self.speciesnet.classifier is None:
            return []
        return self.speciesnet.classify_crop(crop)

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_crop(
        img_cv2: np.ndarray, bbox: List[float], pad: float = 0.10
    ) -> Optional[Image.Image]:
        """Convert normalised bbox → padded crop as PIL RGB image."""
        h, w = img_cv2.shape[:2]
        x, y, bw, bh = bbox

        x_px = int(x * w)
        y_px = int(y * h)
        w_px = int(bw * w)
        h_px = int(bh * h)

        pad_w = int(w_px * pad)
        pad_h = int(h_px * pad)

        x1 = max(0, x_px - pad_w)
        y1 = max(0, y_px - pad_h)
        x2 = min(w, x_px + w_px + pad_w)
        y2 = min(h, y_px + h_px + pad_h)

        crop = img_cv2[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: str, is_night: bool = False) -> List[Dict]:
        """
        Full pipeline for one image.

        Parameters
        ----------
        image_path : path to the image file
        is_night   : passed through to the ensemble engine (reserved for future
                     night-time weight adjustments).

        Returns a list of result dicts — one per detected subject.
        The first result also carries '_model_events' for SSE streaming.
        """
        _empty = {
            "detected_animal": "Empty",
            "primary_label": "Empty",
            "species_label": "N/A",
            "detection_confidence": 0.0,
            "bbox": None,
            "method": "MDv5a",
            "secondary_method": None,
            "agreement": None,
            "model_breakdown": {},
        }

        if not self.megadetector:
            result = dict(_empty)
            result["_model_events"] = []
            result["raw_model_output"] = {"megadetector": {"all_boxes": [], "confidence_threshold": None}, "speciesnet": None}
            return [result]

        # ── Stage 1: Detection ──────────────────────────────────────────
        # _capture_raw=True returns (filtered_candidates, all_raw_boxes) in one
        # model pass so we never run MegaDetector twice.
        candidates, raw_md_all = self.megadetector.detect_all_candidates(image_path, _capture_raw=True)
        merged = nms_merge_detections(candidates, [])

        _raw_md_block = {
            "all_boxes": raw_md_all,
            "confidence_threshold": self._threshold,
        }

        model_events: List[Dict] = [
            {
                "model": "MDv5a",
                "detections": [
                    {"label": c["label"], "conf": round(c["conf"], 3)}
                    for c in candidates
                ],
            },
            {
                "model": "Detection",
                "merged_count": len(merged),
                "sources_used": ["MDv5a"],
            },
        ]

        if not merged:
            result = dict(_empty)
            result["_model_events"] = model_events
            result["raw_model_output"] = {"megadetector": _raw_md_block, "speciesnet": None}
            return [result]

        # ── Stage 2: Classify each animal crop ──────────────────────────
        img_cv2: Optional[np.ndarray] = None
        final_results: List[Dict] = []
        is_first = True

        for cand in merged:
            label = cand["label"]
            conf = float(cand["conf"])
            bbox = cand["bbox"]
            sources = cand.get("sources", ["MDv5a"])

            base = dict(_empty)
            base["primary_label"] = label
            base["detected_animal"] = label
            base["detection_confidence"] = conf
            base["bbox"] = bbox

            if label != "Animal":
                if is_first:
                    base["_model_events"] = model_events
                    is_first = False
                base["raw_model_output"] = {"megadetector": _raw_md_block, "speciesnet": None}
                final_results.append(base)
                continue

            # Load image once
            try:
                if img_cv2 is None:
                    img_cv2 = cv2.imread(image_path)
                if img_cv2 is None:
                    if is_first:
                        base["_model_events"] = model_events
                        is_first = False
                    final_results.append(base)
                    continue

                crop = self._extract_crop(img_cv2, bbox)
                if crop is None:
                    if is_first:
                        base["_model_events"] = model_events
                        is_first = False
                    final_results.append(base)
                    continue

                # Stage 2: SpeciesNet classification
                sn_results = self._classify(crop)

                ev_sn = {
                    "model": "SpeciesNet",
                    "top5": [[s, round(c, 3)] for s, c in sn_results[:5]],
                } if sn_results else {"model": "SpeciesNet", "top5": [], "skipped": True}

                fusion = fuse_species_speciesnet_first(sn_results)
                top_species = fusion["species"]
                top_conf = fusion["confidence"]

                ev_result = {
                    "model": "Result",
                    "species": top_species,
                    "confidence": top_conf,
                    "all_candidates": fusion["all_candidates"],
                }

                carries_events = is_first
                if is_first:
                    model_events += [ev_sn, ev_result]
                    is_first = False

                species_label = ", ".join(
                    f"{s} {c:.2f}" for s, c in fusion["all_candidates"][:3]
                )

                species_data = [
                    {
                        "species_label": f"{s} {c:.2f}",
                        "detected_animal": s,
                        "detection_confidence": c,
                        "primary_label": "Animal",
                        "detection_method": self._method_label(),
                        "bbox": bbox,
                    }
                    for s, c in fusion["all_candidates"]
                ]

                # Build raw SpeciesNet output: all predictions with full taxonomy
                raw_sn = [
                    {
                        "id": (m := _parse_sn_label(lbl)).get("id"),
                        "common_name": m.get("common_name") or m.get("display") or str(lbl),
                        "scientific_name": m.get("scientific_name") or None,
                        "hierarchy": m.get("hierarchy") or [],
                        "confidence": round(float(conf_sn), 6),
                    }
                    for lbl, conf_sn in sn_results
                ]

                base.update({
                    "primary_label": "Animal",
                    "detected_animal": top_species,
                    "species_label": species_label,
                    "species_data": species_data,
                    "detection_confidence": top_conf,
                    "md_confidence": conf,
                    "md_bbox": bbox,
                    "md_category": "1",
                    "speciesnet_confidence": sn_results[0][1] if sn_results else 0.0,
                    "sn_raw_results": sn_results,  # full ranked list with raw JSON labels
                    "method": self._method_label(),
                    "secondary_method": "SpeciesNet",
                    "model_breakdown": {
                        "MDv5a": [c for c in candidates if c.get("label") == "Animal"],
                    },
                    "raw_model_output": {
                        "megadetector": _raw_md_block,
                        "speciesnet": raw_sn,
                    },
                    "_model_events": model_events if carries_events else [],
                })
                final_results.append(base)

            except Exception as exc:
                print(f"Classification error for {label}: {exc}")
                if is_first:
                    base["_model_events"] = model_events
                    is_first = False
                base["raw_model_output"] = {"megadetector": _raw_md_block, "speciesnet": None}
                final_results.append(base)

        if not final_results:
            result = dict(_empty)
            result["_model_events"] = model_events
            result["raw_model_output"] = {"megadetector": _raw_md_block, "speciesnet": None}
            return [result]

        # Ensure exactly one result carries the image-level model events
        if is_first and final_results:
            final_results[0]["_model_events"] = model_events

        return final_results

    def _method_label(self) -> str:
        parts = ["MDv5a"]
        if self.speciesnet and self.speciesnet.classifier is not None:
            parts.append("SpeciesNet")
        return " + ".join(parts)


# Backward-compat alias
EnsembleDetector = AnimalDetector
