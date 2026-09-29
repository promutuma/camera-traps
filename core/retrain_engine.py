"""
HITL retraining engine.

Turns accumulated human review corrections (core/review_engine.py ->
review_actions table) into a model that actually gets better at this
deployment's species mix, instead of corrections only ever fixing the
stored record and never the classifier.

SpeciesNet ships inference-only (no train()/fit() API, and its PyTorch
weights are an ONNX graph auto-converted by onnx2torch, not a normal
named-layer model) — so "retrain SpeciesNet" isn't a supported operation.
Two tiers are implemented instead, both of which train a small NEW
classifier on top of SpeciesNet rather than ever touching its pretrained
weights (always safe, always reversible, no risk of catastrophic
forgetting or corrupting the base model):

  Tier 1 — score-correction layer (always available)
      Trains on SpeciesNet's own already-computed output (the ranked
      candidate list stored per-detection in detections.top_candidates).
      No image I/O, runs on CPU in seconds. This is the reliable floor —
      it works as soon as there are a handful of corrections, on any
      machine.

  Tier 2 — embedding-correction layer (opportunistic upgrade)
      Re-crops the original images and extracts SpeciesNet's real
      penultimate-layer visual features (via a read-only forward-pre-hook
      on its final Linear classification layer — never modifies the
      model), then trains a classifier on those. More accurate than Tier 1
      because it sees actual pixels, not just SpeciesNet's opinion about
      them. Requires the SpeciesNet model to be loaded and the original
      images still present on disk; runs on whatever device SpeciesNet
      itself loaded onto (naturally GPU-accelerated when available, works
      on CPU otherwise — no hard GPU gate).

At runtime, `run()` always attempts Tier 1 first (the safe baseline), then
opportunistically attempts Tier 2, and activates whichever validates
better. Either failure mode degrades gracefully — Tier 2 failing never
blocks Tier 1, and a run with too little data yet fails cleanly with a
clear "not enough corrections" message rather than deploying a bad model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MIN_EXAMPLES = 5
MIN_CLASSES = 2
TOP_K_FEATURES = 10
MODELS_DIRNAME = "models/retrained"
DATASET_DIRNAME = "training_data"


def _crop_bbox(img, bbox: List[float], pad: float = 0.10):
    """Normalised [x, y, w, h] (top-left origin) -> padded PIL crop."""
    w, h = img.size
    x, y, bw, bh = bbox
    x_px, y_px = int(x * w), int(y * h)
    w_px, h_px = int(bw * w), int(bh * h)
    pad_w, pad_h = int(w_px * pad), int(h_px * pad)
    x1 = max(0, x_px - pad_w)
    y1 = max(0, y_px - pad_h)
    x2 = min(w, x_px + w_px + pad_w)
    y2 = min(h, y_px + h_px + pad_h)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _fit_classifier(X, y, min_examples: int, min_classes: int) -> Dict[str, Any]:
    """Shared fit/validate logic for both tiers, given feature matrix X and labels y."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    n_classes = len(set(y))
    if len(y) < min_examples or n_classes < min_classes:
        return {
            "success": False,
            "reason": (
                f"Need >= {min_examples} corrections across >= {min_classes} species "
                f"(have {len(y)} across {n_classes})."
            ),
        }

    class_counts: Dict[str, int] = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1
    can_stratify = all(c >= 2 for c in class_counts.values()) and len(y) >= 10

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    if can_stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        clf.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        clf.fit(X, y)  # refit on everything for the deployed model
        metrics = {"validation_accuracy": round(float(val_acc), 4), "validated": True}
    else:
        clf.fit(X, y)
        train_acc = accuracy_score(y, clf.predict(X))
        metrics = {
            "training_accuracy": round(float(train_acc), 4),
            "validated": False,
            "note": "Too little data for a held-out split; reporting training accuracy only.",
        }

    metrics["n_examples"] = len(y)
    metrics["n_classes"] = n_classes
    return {"success": True, "classifier": clf, "metrics": metrics}


def _score_key(score_score: float) -> float:
    return float(score_score)


class RetrainEngine:
    def __init__(self, db_manager, uploads_dir: Optional[str] = None, base_dir: Optional[str] = None):
        self.db_manager = db_manager
        self.uploads_dir = Path(uploads_dir) if uploads_dir else None
        base = Path(base_dir) if base_dir else Path(".")
        self.models_dir = base / MODELS_DIRNAME
        self.dataset_dir = base / DATASET_DIRNAME
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset export
    # ------------------------------------------------------------------

    def export_dataset(self, job_id: str, since_ts: Optional[str] = None) -> List[Dict]:
        """
        Pull corrections from the DB and write a versioned, reproducible
        snapshot to disk. Returns the rows (also usable directly in-memory).
        """
        rows = self.db_manager.get_correction_training_data(since_ts=since_ts)
        snapshot_path = self.dataset_dir / f"dataset_{job_id}.jsonl"
        with open(snapshot_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return rows

    # ------------------------------------------------------------------
    # Tier 1 — score-correction layer
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_score_features(rows: List[Dict]):
        X_dicts, y = [], []
        for row in rows:
            candidates = row.get("top_candidates") or []
            feat = {
                f"score::{c.get('common_name', '?')}": _score_key(c.get("confidence", 0.0))
                for c in candidates[:TOP_K_FEATURES]
            }
            if not feat:
                continue
            X_dicts.append(feat)
            y.append(row["corrected_species"])
        return X_dicts, y

    def train_score_correction_layer(self, rows: List[Dict]) -> Dict[str, Any]:
        from sklearn.feature_extraction import DictVectorizer

        X_dicts, y = self._rows_to_score_features(rows)
        if len(X_dicts) < MIN_EXAMPLES:
            return {
                "success": False,
                "reason": f"Only {len(X_dicts)} corrections have usable SpeciesNet output "
                          f"(need >= {MIN_EXAMPLES}).",
            }
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(X_dicts)
        result = _fit_classifier(X, y, MIN_EXAMPLES, MIN_CLASSES)
        if result["success"]:
            result["vectorizer"] = vectorizer
        return result

    # ------------------------------------------------------------------
    # Tier 2 — embedding-correction layer
    # ------------------------------------------------------------------

    def train_embedding_correction_layer(self, rows: List[Dict], speciesnet_wrapper) -> Dict[str, Any]:
        if speciesnet_wrapper is None or getattr(speciesnet_wrapper, "classifier", None) is None:
            return {"success": False, "reason": "SpeciesNet model is not loaded."}
        if self.uploads_dir is None:
            return {"success": False, "reason": "Uploads directory not configured."}

        # Reuse the wrapper's cached extractor (also used at inference time)
        # rather than attaching a second hook to the same model.
        extractor = speciesnet_wrapper.get_embedding_extractor()
        if extractor is None or not extractor.ready:
            reason = extractor.error if extractor else "Could not attach to model internals."
            return {"success": False, "reason": reason}

        import numpy as np
        from PIL import Image

        X_list, y = [], []
        for row in rows:
            filename = row.get("filename")
            if not filename:
                continue
            img_path = self.uploads_dir / filename
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                bbox = json.loads(row["bbox"]) if row.get("bbox") else None
                crop = _crop_bbox(img, bbox) if bbox else img
                emb = extractor.extract(crop)
                if emb is None:
                    continue
                X_list.append(emb)
                y.append(row["corrected_species"])
            except Exception:
                continue

        if len(X_list) < MIN_EXAMPLES:
            return {
                "success": False,
                "reason": f"Only {len(X_list)} corrections have source images still on "
                          f"disk (need >= {MIN_EXAMPLES}).",
            }

        X = np.stack(X_list)
        result = _fit_classifier(X, y, MIN_EXAMPLES, MIN_CLASSES)
        if result["success"]:
            result["metrics"]["embedding_dim"] = int(X.shape[1])
            result["metrics"]["device"] = extractor.device
        return result

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, job, speciesnet_wrapper, progress_cb=None) -> None:
        """
        Run a full retrain cycle for `job` (a RetrainJob), mutating its
        status/current_step/metrics/error in place. Never raises — all
        failure modes are captured on the job itself.
        """
        def step(msg: str) -> None:
            job.current_step = msg
            if progress_cb:
                try:
                    progress_cb(job)
                except Exception:
                    pass

        try:
            job.status = "running"
            step("Exporting corrections from review queue...")
            rows = self.export_dataset(job.job_id, since_ts=None)
            job.dataset_size = len(rows)

            if not rows:
                job.status = "error"
                job.error = "No review corrections found yet — accept/correct some detections in the Review Queue first."
                job.finished_at = time.time()
                return

            step(f"Training score-correction layer on {len(rows)} corrections...")
            tier1 = self.train_score_correction_layer(rows)

            step("Attempting embedding-correction layer (opportunistic upgrade)...")
            tier2 = self.train_embedding_correction_layer(rows, speciesnet_wrapper)

            candidates = []
            if tier1.get("success"):
                candidates.append(("correction_layer_scores", tier1))
            if tier2.get("success"):
                candidates.append(("correction_layer_embeddings", tier2))

            if not candidates:
                job.status = "error"
                job.error = tier1.get("reason") or "Retraining failed."
                job.metrics = {"tier1": tier1, "tier2": tier2}
                job.finished_at = time.time()
                return

            def _score(entry):
                m = entry[1]["metrics"]
                return m.get("validation_accuracy", m.get("training_accuracy", 0.0))

            best_method, best_result = max(candidates, key=_score)

            step(f"Saving best model ({best_method})...")
            artifact_path = self.models_dir / f"{job.job_id}.joblib"
            self._save_artifact(artifact_path, best_method, best_result)

            job.method = best_method
            job.model_version = job.job_id
            job.metrics = {
                "active": best_result["metrics"],
                "tier1": {k: v for k, v in tier1.items() if k != "classifier" and k != "vectorizer"},
                "tier2": {k: v for k, v in tier2.items() if k != "classifier"},
            }
            job.status = "done"
            job.activated = True
            job.finished_at = time.time()
            step("Done.")

        except Exception as exc:
            job.status = "error"
            job.error = f"Unexpected error: {exc}"
            job.finished_at = time.time()

    def _save_artifact(self, path: Path, method: str, result: Dict[str, Any]) -> None:
        import joblib

        payload = {"method": method, "classifier": result["classifier"]}
        if method == "correction_layer_scores":
            payload["vectorizer"] = result["vectorizer"]
        joblib.dump(payload, path)

    # ------------------------------------------------------------------
    # Loading a trained artifact for inference
    # ------------------------------------------------------------------

    def load_artifact(self, job_id: str) -> Optional[Dict[str, Any]]:
        import joblib

        path = self.models_dir / f"{job_id}.joblib"
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None
