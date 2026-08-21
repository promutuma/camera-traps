"""
Wrapper around the SpeciesNet classifier (Google CameraTrapAI).

Install: pip install speciesnet
Model download: requires Kaggle credentials on first run.
  export KAGGLE_USERNAME=<your_username>
  export KAGGLE_KEY=<your_api_key>

SpeciesNet uses EfficientNetV2-M trained on 65M+ camera-trap images,
classifying into 2 000+ labels (species, higher taxa, blank, vehicle, human).
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Tuple

from PIL import Image


# Default SpeciesNet model (PyTorch variant, v4.0.2a)
_DEFAULT_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"


class SpeciesNetWrapper:
    """
    Thin wrapper for SpeciesNet's per-crop classifier.

    Usage:
        wrapper = SpeciesNetWrapper(lat=-1.0, lng=37.0, country="KEN")
        results = wrapper.classify_crop(pil_image)
        # → [("panthera leo", 0.82), ("felidae", 0.09), ...]
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        low_spec: bool = False,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        country: Optional[str] = None,
    ):
        self.model_name = model_name
        self.low_spec = low_spec
        self.lat = lat
        self.lng = lng
        self.country = country
        self.classifier = None
        self.load_error: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from speciesnet.classifier import SpeciesNetClassifier

            # None → auto-select CUDA / MPS / CPU; "cpu" forces CPU
            device = "cpu" if self.low_spec else None
            self.classifier = SpeciesNetClassifier(
                model_name=self.model_name,
                device=device,
            )
            print("SpeciesNet classifier loaded successfully.")

        except ImportError:
            self.load_error = (
                "speciesnet package not installed. "
                "Run: pip install speciesnet"
            )
            print(f"Warning: {self.load_error}")

        except Exception as exc:
            # Common causes: missing Kaggle credentials, network failure
            self.load_error = str(exc)
            print(
                f"Warning: SpeciesNet failed to load ({exc}). "
                "Ensure KAGGLE_USERNAME and KAGGLE_KEY environment variables are set."
            )

    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        geo: Dict = {}
        if self.lat is not None:
            geo["lat"] = self.lat
        if self.lng is not None:
            geo["lng"] = self.lng
        if self.country is not None:
            geo["country"] = self.country
        return {
            "loaded": self.classifier is not None,
            "error": self.load_error,
            "geo_prior": geo if geo else None,
        }

    def classify_crop(
        self,
        crop: Image.Image,
        top_k: int = 50,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        country: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Classify a PIL image crop with SpeciesNet.

        SpeciesNet's public API is filepath-based, so we write the crop
        to a temp JPEG, classify it, then clean up immediately.

        Geographic parameters (lat/lng/country) activate SpeciesNet's
        built-in geographic prior, which biases predictions toward species
        known to occur in the target region. Per-call values override the
        instance-level defaults set at construction time.

        Returns list of (label, confidence) sorted descending, up to top_k.
        """
        if self.classifier is None:
            return []

        # Resolve geo params: per-call override → instance default → None
        _lat = lat if lat is not None else self.lat
        _lng = lng if lng is not None else self.lng
        _country = country if country is not None else self.country

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                tmp_path = tmp.name
                crop.save(tmp_path, format="JPEG", quality=92)

            preprocessed = self.classifier.preprocess(crop)

            geo_kwargs: Dict = {}
            if _lat is not None:
                geo_kwargs["lat"] = _lat
            if _lng is not None:
                geo_kwargs["lng"] = _lng
            if _country is not None:
                geo_kwargs["country"] = _country

            try:
                result = self.classifier.predict(tmp_path, img=preprocessed, **geo_kwargs)
            except TypeError:
                # Installed version doesn't accept all geo kwargs — retry without them
                result = self.classifier.predict(tmp_path, img=preprocessed)

            classifications = result.get("classifications", {})
            classes: List[str] = classifications.get("classes", [])
            scores: List[float] = classifications.get("scores", [])

            pairs = sorted(zip(classes, scores), key=lambda x: x[1], reverse=True)
            
            import json
            cleaned_pairs = []
            for label, score in pairs[:top_k]:
                if ";" in label:
                    parts = [p.strip() for p in label.split(";") if p.strip()]
                    if parts:
                        taxon_id = parts[0]
                        common_name = parts[-1]
                        hierarchy = parts[1:-1]
                        scientific_name = " ".join(parts[-3:-1]) if len(parts) >= 4 else ""
                        label = json.dumps({
                            "id": taxon_id,
                            "common_name": common_name,
                            "scientific_name": scientific_name,
                            "hierarchy": hierarchy,
                            "display": common_name
                        })
                elif "::::::" in label:
                    parts = label.split("::::::")
                    if len(parts) >= 2:
                        label = json.dumps({
                            "id": parts[0].strip(),
                            "common_name": parts[1].strip(),
                            "display": parts[1].strip()
                        })
                    else:
                        label = json.dumps({
                            "display": label.strip()
                        })
                else:
                    label = json.dumps({
                        "display": label.strip()
                    })
                cleaned_pairs.append((label, float(score)))
                
            return cleaned_pairs

        except Exception as exc:
            print(f"SpeciesNet classify_crop error: {exc}")
            return []

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
