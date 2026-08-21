"""
Wrapper around the SpeciesNet classifier (Google CameraTrapAI).

Install: pip install speciesnet
Model download: requires Kaggle credentials on first run.
  export KAGGLE_USERNAME=<your_username>
  export KAGGLE_KEY=<your_api_key>

SpeciesNet uses EfficientNetV2-M trained on 65M+ camera-trap images,
classifying into 2 000+ labels (species, higher taxa, blank, vehicle, human) —
drawn from camera traps worldwide, not just Africa.

Note: `SpeciesNetClassifier.predict()` (the low-level per-crop API used here)
does NOT accept lat/lng/country — those kwargs always trigger the TypeError
fallback below and are silently dropped. SpeciesNet's own geographic handling
only lives in a separate geofencing step (`speciesnet.geofence_utils`), which
this wrapper applies itself as a post-filter — see `_load_geofence` /
`_apply_region_filter`.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Tuple

from PIL import Image


# Default SpeciesNet model (PyTorch variant, v4.0.2a)
_DEFAULT_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"

# ISO 3166-1 alpha-3 codes for the East Africa + Horn of Africa region this
# deployment surveys. A SpeciesNet candidate is kept only if it's not
# geofenced out in at least one of these countries — i.e. it's plausible
# somewhere in the region, even if not in the project's specific country.
EAST_AFRICA_COUNTRIES = [
    "KEN", "TZA", "UGA", "RWA",  # Kenya, Tanzania, Uganda, Rwanda
    "BDI", "SSD",                # Burundi, South Sudan
    "ETH", "SOM", "DJI", "ERI",  # Ethiopia, Somalia, Djibouti, Eritrea
]

# Must match core/retrain_engine.py's TOP_K_FEATURES — how many ranked
# candidates the score-correction layer was trained on as features.
_CORRECTION_TOP_K = 10
_CORRECTION_MIN_CONFIDENCE = 0.5


class _EmbeddingExtractor:
    """
    Read-only instrumentation that captures SpeciesNet's penultimate-layer
    features via a forward-pre-hook on its final classification Linear layer
    (identified by matching out_features to the label vocabulary size, not by
    assuming a fixed architecture). Never modifies the model's weights — used
    both by core/retrain_engine.py (to build a training set from stored
    images) and by SpeciesNetWrapper itself (to score a live crop against an
    activated embedding-correction layer).
    """

    def __init__(self, classifier):
        self.ready = False
        self.error: Optional[str] = None
        self.device = getattr(classifier, "device", "cpu")
        self._captured = None
        self._hook = None
        self._classifier = classifier
        try:
            import torch

            model = getattr(classifier, "model", None)
            if model is None or not isinstance(model, torch.nn.Module):
                self.error = "Classifier has no underlying torch.nn.Module (`.model`)."
                return

            n_labels = len(getattr(classifier, "labels", {}) or {})
            head = None
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == n_labels:
                    head = module
            if head is None:
                self.error = (
                    "Could not identify the final classification layer "
                    f"(expected a Linear with out_features={n_labels})."
                )
                return

            def _hook_fn(_module, inputs):
                self._captured = inputs[0].detach()

            self._hook = head.register_forward_pre_hook(_hook_fn)
            self._model = model
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def extract(self, crop: Image.Image):
        """Return a 1-D numpy embedding for a PIL crop, or None on failure."""
        if not self.ready:
            return None
        try:
            import numpy as np
            import torch

            preprocessed = self._classifier.preprocess(crop)
            if preprocessed is None:
                return None
            arr = preprocessed.arr.astype(np.float32) / 255.0
            batch = torch.from_numpy(arr[None, ...]).to(self.device)
            self._captured = None
            with torch.no_grad():
                self._model(batch)
            if self._captured is None:
                return None
            vec = self._captured
            if vec.dim() > 2:
                vec = vec.flatten(start_dim=1)
            return vec[0].cpu().numpy()
        except Exception:
            return None

    def close(self) -> None:
        if self._hook is not None:
            try:
                self._hook.remove()
            except Exception:
                pass
            self._hook = None


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
        self._geofence_map: Optional[Dict] = None
        self.geofence_error: Optional[str] = None
        self._correction_layer: Optional[Dict] = None
        self._embedding_extractor: Optional[_EmbeddingExtractor] = None
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
            self._load_geofence()

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

    def _load_geofence(self) -> None:
        """
        Load SpeciesNet's own country-level geofence map (bundled alongside the
        classifier weights in the same downloaded model directory — no extra
        download needed once the classifier above has loaded).

        Used by `_apply_region_filter` to drop raw classifier candidates that
        aren't native to any EAST_AFRICA_COUNTRIES country. If this fails to
        load for any reason, region filtering is skipped (fail-open) rather
        than blocking classification entirely.
        """
        try:
            import json as _json
            from speciesnet.utils import ModelInfo

            model_info = ModelInfo(self.model_name)
            with open(model_info.geofence, encoding="utf-8") as fp:
                self._geofence_map = _json.load(fp)
            print("SpeciesNet geofence map loaded — East Africa region filtering active.")

        except Exception as exc:
            self.geofence_error = str(exc)
            print(
                f"Warning: SpeciesNet geofence map unavailable ({exc}). "
                "Region filtering is disabled; raw (global) SpeciesNet candidates will be used."
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
            "region_filter_active": self._geofence_map is not None,
            "region_filter_error": self.geofence_error,
            "correction_layer_active": self._correction_layer.get("method") if self._correction_layer else None,
        }

    # ------------------------------------------------------------------
    # HITL retraining hook (core/retrain_engine.py trains these artifacts)
    # ------------------------------------------------------------------

    def set_correction_layer(self, artifact: Optional[Dict]) -> None:
        """
        Activate (or, with `None`, clear) a HITL-trained correction layer.

        `artifact` is the dict saved by core/retrain_engine.py's
        `_save_artifact`: {"method": "correction_layer_scores" |
        "correction_layer_embeddings", "classifier": ..., "vectorizer": ...
        (scores method only)}. Applied in `classify_crop` after region
        filtering, on top of SpeciesNet's raw output — never touches the
        underlying pretrained model.
        """
        self._correction_layer = artifact

    def get_embedding_extractor(self) -> Optional[_EmbeddingExtractor]:
        """Lazily create and cache a penultimate-layer embedding extractor."""
        if self.classifier is None:
            return None
        if self._embedding_extractor is None:
            self._embedding_extractor = _EmbeddingExtractor(self.classifier)
        return self._embedding_extractor

    def _apply_correction_layer(
        self, cleaned_pairs: List[Tuple[str, float]], crop: Image.Image
    ) -> List[Tuple[str, float]]:
        """
        If a HITL correction layer is active and disagrees with SpeciesNet's
        own top pick with reasonable confidence, promote its prediction to
        the top of the list (tagged `corrected_by_review`) without discarding
        the rest of SpeciesNet's ranking — keeps the original signal visible
        for auditability instead of silently overwriting it.
        """
        layer = self._correction_layer
        if not layer or not cleaned_pairs:
            return cleaned_pairs

        import json as _json

        def _name(label: str) -> Optional[str]:
            if not label.startswith("{"):
                return None
            try:
                meta = _json.loads(label)
                return meta.get("common_name") or meta.get("display")
            except Exception:
                return None

        try:
            method = layer.get("method")
            clf = layer.get("classifier")

            if method == "correction_layer_scores":
                feat = {}
                for label, score in cleaned_pairs[:_CORRECTION_TOP_K]:
                    name = _name(label) or label
                    feat[f"score::{name}"] = float(score)
                vec = layer.get("vectorizer")
                if vec is None:
                    return cleaned_pairs
                X = vec.transform([feat])
                if X.nnz == 0:
                    # None of this crop's candidate species overlap with the
                    # correction layer's trained vocabulary at all — it has no
                    # real opinion here (a prediction off an all-zero input is
                    # just the model's class-imbalance prior, not signal).
                    return cleaned_pairs
                predicted = clf.predict(X)[0]
                confidence = float(max(clf.predict_proba(X)[0]))

            elif method == "correction_layer_embeddings":
                extractor = self.get_embedding_extractor()
                if extractor is None or not extractor.ready:
                    return cleaned_pairs
                emb = extractor.extract(crop)
                if emb is None:
                    return cleaned_pairs
                X = emb[None, :]
                predicted = clf.predict(X)[0]
                confidence = float(max(clf.predict_proba(X)[0]))

            else:
                return cleaned_pairs

            top_name = _name(cleaned_pairs[0][0])
            if predicted == top_name or confidence < _CORRECTION_MIN_CONFIDENCE:
                return cleaned_pairs

            corrected_label = _json.dumps({
                "common_name": predicted,
                "display": predicted,
                "corrected_by_review": True,
            })
            rest = [p for p in cleaned_pairs if _name(p[0]) != predicted]
            return [(corrected_label, confidence)] + rest

        except Exception as exc:
            print(f"Correction layer application error: {exc}")
            return cleaned_pairs

    # ------------------------------------------------------------------

    def _apply_region_filter(
        self, pairs: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Drop raw classifier candidates that SpeciesNet's geofence data says
        don't occur in any EAST_AFRICA_COUNTRIES country.

        `pairs` must be the untouched (label, score) tuples straight from
        `classifications["classes"]`/`["scores"]` — SpeciesNet's raw labels are
        7-part `uuid;class;order;family;genus;species;common_name` strings, and
        `should_geofence_animal_classification` parses that exact format.
        Non-species labels (blank/human/vehicle/higher-taxa) simply aren't in
        the geofence map and pass through unfiltered, same as today.
        """
        if self._geofence_map is None:
            return pairs

        from speciesnet.geofence_utils import should_geofence_animal_classification

        filtered: List[Tuple[str, float]] = []
        for label, score in pairs:
            try:
                allowed_in_region = any(
                    not should_geofence_animal_classification(
                        label, cc, None, self._geofence_map, enable_geofence=True
                    )
                    for cc in EAST_AFRICA_COUNTRIES
                )
            except Exception:
                allowed_in_region = True  # malformed/unexpected label — fail open
            if allowed_in_region:
                filtered.append((label, score))

        # If literally everything got filtered out (e.g. classifier was very
        # confident about a species with no East Africa range at all), keep
        # the original top candidate rather than returning nothing.
        return filtered if filtered else pairs[:1]

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
            pairs = self._apply_region_filter(pairs)

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

            cleaned_pairs = self._apply_correction_layer(cleaned_pairs, crop)
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
