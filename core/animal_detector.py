"""
Multi-Model Wildlife Identification Pipeline
─────────────────────────────────────────────
Stage 1 – Detection   (parallel)
  MDv5a    → candidates_a
  MDv1000  → candidates_b          (skipped in low-spec / if not loaded)
  NMS merge → merged_candidates

Stage 2 – Crop + Classify   (parallel per animal crop)
  BioClip     → [(species, conf), ...]
  SpeciesNet  → [(species, conf), ...]   (skipped if not loaded)
  Ensemble fusion → final species + agreement score

Each result dict includes a '_model_events' list for real-time SSE display.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

from .bioclip_classifier import BioClipClassifier
from .ensemble_engine import nms_merge_detections, fuse_species

# MegaDetector (Official Package)
try:
    from megadetector.detection import run_detector
    MD_AVAILABLE = True
except ImportError as exc:
    MD_AVAILABLE = False
    print(f"Warning: megadetector not installed. Error: {exc}")

# Thread pool shared across detector calls
_executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────────────────────────────────────
# MegaDetector wrapper (supports any model version string)
# ─────────────────────────────────────────────────────────────────────────────

class MegaDetectorWrapper:
    """Wrapper for a single MegaDetector model (MDv5a, MDv1000-redwood, …)."""

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

    def detect_all_candidates(self, image_path: str) -> List[Dict]:
        """Run inference; return all detections above threshold."""
        if self.model is None:
            return []
        try:
            image = Image.open(image_path)
            result = self.model.generate_detections_one_image(image)
            out = []
            for det in result.get("detections", []):
                if det["conf"] >= self.confidence_threshold:
                    label = self.CLASS_MAP.get(det["category"], "Unknown")
                    out.append({
                        "label": label,
                        "conf": float(det["conf"]),
                        "bbox": det["bbox"],
                    })
            return out
        except Exception as exc:
            print(f"{self.model_version} inference error: {exc}")
            return []

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
    """
    Orchestrates the full multi-model wildlife ID pipeline.

    Parameters
    ----------
    megadetector : MegaDetectorWrapper
        Primary detector (MDv5a).
    bioclip : BioClipClassifier
        Zero-shot species classifier.
    confidence_threshold : float
        Applied to both detectors.
    megadetector_v1000 : MegaDetectorWrapper | None
        Secondary detector (MDv1000-redwood). None → single-detector mode.
    speciesnet : SpeciesNetWrapper | None
        Google SpeciesNet classifier. None → BioClip-only mode.
    """

    WILDLIFE_CLASSES = [
        # ── Bovids & antelope ──
        "plains zebra", "Grevy's zebra", "mountain zebra",
        "African elephant", "African bush elephant", "African forest elephant",
        "lion", "leopard", "cheetah", "serval", "caracal", "African wildcat",
        "African wild dog", "spotted hyena", "striped hyena", "brown hyena", "aardwolf",
        "giraffe", "reticulated giraffe", "Masai giraffe",
        "African buffalo", "common eland", "greater kudu", "lesser kudu",
        "bushbuck", "nyala", "sitatunga", "bongo",
        "impala", "Thomson's gazelle", "Grant's gazelle", "springbok",
        "gerenuk", "dibatag", "dik-dik", "steenbok", "oribi",
        "common duiker", "blue duiker", "red duiker",
        "blue wildebeest", "black wildebeest", "topi", "hartebeest",
        "roan antelope", "sable antelope", "gemsbok", "East African oryx",
        "white rhinoceros", "black rhinoceros",
        "common warthog", "bushpig", "giant forest hog", "red river hog",
        "common hippopotamus", "pygmy hippopotamus",
        "Nile lechwe", "waterbuck", "reedbuck", "lechwe",
        "White-eared kob", "Uganda kob",
        # ── Primates ──
        "olive baboon", "chacma baboon", "hamadryas baboon", "gelada",
        "vervet monkey", "patas monkey", "colobus monkey",
        "mandrill", "drill", "chimpanzee", "bonobo",
        "western gorilla", "eastern gorilla", "mountain gorilla",
        "ring-tailed lemur", "sifaka",
        # ── Carnivores ──
        "black-backed jackal", "side-striped jackal", "golden jackal",
        "bat-eared fox", "Cape fox", "Ethiopian wolf",
        "honey badger", "African civet", "common genet",
        "spotted-necked otter", "Cape clawless otter",
        "banded mongoose", "dwarf mongoose", "meerkat", "Egyptian mongoose",
        "African golden cat",
        # ── Other mammals ──
        "aardvark",
        "ground pangolin", "giant pangolin", "tree pangolin",
        "Cape porcupine", "African crested porcupine",
        "Cape hare", "springhare",
        "African wild ass", "rock hyrax",
        # ── Reptiles ──
        "Nile crocodile", "dwarf crocodile", "Nile monitor", "rock python",
        "leopard tortoise",
        # ── Birds commonly camera-trapped ──
        "ostrich", "kori bustard", "southern ground hornbill",
        "helmeted guineafowl", "crested francolin", "secretary bird",
        "marabou stork",
        # ── Fallback generic terms ──
        "antelope", "gazelle", "monkey", "baboon", "crocodile",
        "rhinoceros", "hippopotamus", "wildebeest", "hyena",
        "animal", "human", "person", "vehicle",
    ]

    def __init__(
        self,
        megadetector: Optional[MegaDetectorWrapper],
        bioclip: Optional[BioClipClassifier],
        confidence_threshold: float = 0.2,
        megadetector_v1000: Optional[MegaDetectorWrapper] = None,
        speciesnet=None,  # SpeciesNetWrapper | None
    ):
        self.megadetector = megadetector
        self.megadetector_v1000 = megadetector_v1000
        self.bioclip = bioclip
        self.speciesnet = speciesnet

        if self.megadetector:
            self.megadetector.set_confidence_threshold(confidence_threshold)
        if self.megadetector_v1000:
            self.megadetector_v1000.set_confidence_threshold(confidence_threshold)
        self._threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_parallel(self, image_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Run MDv5a and MDv1000 concurrently; return (cands_a, cands_b)."""
        if not self.megadetector_v1000:
            return self.megadetector.detect_all_candidates(image_path), []

        future_a = _executor.submit(
            self.megadetector.detect_all_candidates, image_path
        )
        future_b = _executor.submit(
            self.megadetector_v1000.detect_all_candidates, image_path
        )
        return future_a.result(), future_b.result()

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_parallel(
        self, crop: Image.Image
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Run BioClip and SpeciesNet concurrently; return (bc, sn)."""
        if not self.speciesnet or self.speciesnet.classifier is None:
            bc = self.bioclip.predict_list(crop, threshold=self._threshold) if self.bioclip else []
            return bc, []

        future_bc = _executor.submit(
            self.bioclip.predict_list, crop, self._threshold
        )
        future_sn = _executor.submit(self.speciesnet.classify_crop, crop)
        return future_bc.result(), future_sn.result()

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
        is_night   : when True, the ensemble engine applies night-time weights
                     that favour SpeciesNet over BioCLIP (SpeciesNet was trained
                     on nocturnal/IR camera-trap images; BioCLIP was not).

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
            return [result]

        # ── Stage 1: Detection ──────────────────────────────────────────
        cands_a, cands_b = self._detect_parallel(image_path)
        merged = nms_merge_detections(cands_a, cands_b)

        model_events: List[Dict] = [
            {
                "model": "MDv5a",
                "detections": [
                    {"label": c["label"], "conf": round(c["conf"], 3)}
                    for c in cands_a
                ],
            },
        ]
        if self.megadetector_v1000:
            model_events.append({
                "model": "MDv1000",
                "detections": [
                    {"label": c["label"], "conf": round(c["conf"], 3)}
                    for c in cands_b
                ],
            })
        model_events.append({
            "model": "Detection",
            "merged_count": len(merged),
            "sources_used": (
                ["MDv5a", "MDv1000"] if self.megadetector_v1000 else ["MDv5a"]
            ),
        })

        if not merged:
            result = dict(_empty)
            result["_model_events"] = model_events
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

                # Stage 2a: Parallel classification
                bc_results, sn_results = self._classify_parallel(crop)

                # BioClip fallback if nothing above threshold
                if not bc_results and self.bioclip:
                    bc_results = self.bioclip.predict_list(crop, threshold=0.0, top_k=1)

                # Classification events
                ev_bc = {
                    "model": "BioClip",
                    "top5": [[s, round(c, 3)] for s, c in bc_results[:5]],
                }
                ev_sn = {
                    "model": "SpeciesNet",
                    "top5": [[s, round(c, 3)] for s, c in sn_results[:5]],
                } if sn_results else {"model": "SpeciesNet", "top5": [], "skipped": True}

                # Stage 2b: Fuse (pass night context for dynamic weighting)
                fusion = fuse_species(bc_results, sn_results, is_night=is_night)
                top_species = fusion["species"]
                top_conf = fusion["confidence"]
                agreement = fusion["agreement"]

                ev_result = {
                    "model": "Result",
                    "species": top_species,
                    "confidence": top_conf,
                    "agreement": agreement,
                    "all_candidates": fusion["all_candidates"],
                }

                carries_events = is_first
                if is_first:
                    model_events += [ev_bc, ev_sn, ev_result]
                    is_first = False

                # Build species label string
                species_label = ", ".join(
                    f"{s} {c:.2f}" for s, c in fusion["all_candidates"][:3]
                )

                # Structured per-species data (for DB / results page)
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

                base.update({
                    "primary_label": "Animal",
                    "detected_animal": top_species,
                    "species_label": species_label,
                    "species_data": species_data,
                    "detection_confidence": top_conf,
                    "md_confidence": conf,
                    "md_bbox": bbox,
                    "md_category": "1",
                    "bioclip_confidence": bc_results[0][1] if bc_results else 0.0,
                    "speciesnet_confidence": sn_results[0][1] if sn_results else 0.0,
                    "agreement": agreement,
                    "method": self._method_label(),
                    "secondary_method": "BioClip+SpeciesNet",
                    "model_breakdown": {
                        "MDv5a": [c for c in cands_a if c.get("label") == "Animal"],
                        "MDv1000": [c for c in cands_b if c.get("label") == "Animal"],
                        "BioClip": bc_results[:5],
                        "SpeciesNet": sn_results[:5],
                        "Fusion": {
                            "species": top_species,
                            "confidence": top_conf,
                            "agreement": agreement,
                        },
                    },
                    "_model_events": model_events if carries_events else [],
                })
                final_results.append(base)

            except Exception as exc:
                print(f"Classification error for {label}: {exc}")
                if is_first:
                    base["_model_events"] = model_events
                    is_first = False
                final_results.append(base)

        if not final_results:
            result = dict(_empty)
            result["_model_events"] = model_events
            return [result]

        # Ensure exactly one result carries the image-level model events
        if is_first and final_results:
            final_results[0]["_model_events"] = model_events

        return final_results

    def _method_label(self) -> str:
        parts = ["MDv5a"]
        if self.megadetector_v1000:
            parts.append("MDv1000")
        parts.append("BioClip")
        if self.speciesnet and self.speciesnet.classifier is not None:
            parts.append("SpeciesNet")
        return " + ".join(parts)


# Backward-compat alias
EnsembleDetector = AnimalDetector
