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
        wrapper = SpeciesNetWrapper()
        results = wrapper.classify_crop(pil_image)
        # → [("panthera leo", 0.82), ("felidae", 0.09), ...]
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        low_spec: bool = False,
    ):
        self.model_name = model_name
        self.low_spec = low_spec
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
        return {
            "loaded": self.classifier is not None,
            "error": self.load_error,
        }

    def classify_crop(
        self,
        crop: Image.Image,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Classify a PIL image crop with SpeciesNet.

        SpeciesNet's public API is filepath-based, so we write the crop
        to a temp JPEG, classify it, then clean up immediately.

        Returns list of (label, confidence) sorted descending, up to top_k.
        """
        if self.classifier is None:
            return []

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                tmp_path = tmp.name
                crop.save(tmp_path, format="JPEG", quality=92)

            preprocessed = self.classifier.preprocess(crop)
            result = self.classifier.predict(tmp_path, img=preprocessed)

            classifications = result.get("classifications", {})
            classes: List[str] = classifications.get("classes", [])
            scores: List[float] = classifications.get("scores", [])

            pairs = sorted(zip(classes, scores), key=lambda x: x[1], reverse=True)
            
            cleaned_pairs = []
            for label, score in pairs[:top_k]:
                if "::::::" in label:
                    label = label.split("::::::")[-1].strip()
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
