"""
Privacy Scrubbing Module
Blurs human faces and vehicle bounding boxes in camera trap images.
Non-destructive: originals are never modified. Scrubbed copies are written
to a configurable output directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


# Labels that trigger scrubbing (MegaDetector category names)
_SCRUB_LABELS = {"person", "vehicle", "human"}

# Default Gaussian blur kernel — must be odd
_DEFAULT_BLUR = 51


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _label_needs_scrub(detection: dict) -> bool:
    label = str(
        detection.get("primary_label")
        or detection.get("detected_animal")
        or ""
    ).strip().lower()
    return label in _SCRUB_LABELS


class PrivacyScrubber:
    """
    Applies Gaussian blur to all Person and Vehicle bounding boxes
    in a camera trap image and saves the result as a separate scrubbed copy.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        blur_strength: int = _DEFAULT_BLUR,
        audit_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.blur_strength = _ensure_odd(max(11, blur_strength))
        self._audit_path = Path(audit_path) if audit_path else None
        self.audit_log: list[dict] = []
        self._load_audit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrub_image(
        self,
        image_path: str,
        detections: list,
    ) -> dict:
        """Blur all Person/Vehicle bounding boxes in one image."""
        result = {
            "original_path": image_path,
            "scrubbed_path": None,
            "boxes_blurred": 0,
            "scrubbed_at": datetime.now().isoformat(),
            "skipped": True,
            "reason": "No person/vehicle bounding boxes",
        }

        boxes_to_blur = [
            d["bbox"] for d in detections
            if _label_needs_scrub(d) and d.get("bbox") is not None
        ]

        if not boxes_to_blur:
            return result

        if not os.path.isfile(image_path):
            result["reason"] = "Source image file not found"
            return result

        img = cv2.imread(image_path)
        if img is None:
            result["reason"] = "Could not read source image"
            return result

        h, w = img.shape[:2]

        for bbox in boxes_to_blur:
            try:
                bx, by, bw, bh = bbox
                x1 = max(0, int(bx * w))
                y1 = max(0, int(by * h))
                x2 = min(w, int((bx + bw) * w))
                y2 = min(h, int((by + bh) * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = img[y1:y2, x1:x2]
                k = _ensure_odd(self.blur_strength)
                img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
                result["boxes_blurred"] += 1
            except Exception:
                continue

        if result["boxes_blurred"] == 0:
            result["reason"] = "Person/vehicle detected but no valid bounding boxes"
            return result

        scrubbed_path = self._scrubbed_path(image_path)
        scrubbed_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(scrubbed_path), img)

        result["scrubbed_path"] = str(scrubbed_path)
        result["skipped"] = False
        result["reason"] = f"Blurred {result['boxes_blurred']} region(s)"
        return result

    def scrub_batch(self, df, image_col: str = "filepath") -> list:
        """
        Scrub all images in a DataFrame that contain Person or Vehicle detections.
        Appends normalized entries to audit_log and persists to disk.
        """
        audit: list[dict] = []
        if df is None or df.empty or image_col not in df.columns:
            return audit

        for filepath, group in df.groupby(image_col):
            detections = group.to_dict("records")
            has_sensitive = any(_label_needs_scrub(d) for d in detections)
            if not has_sensitive:
                continue

            record = self.scrub_image(str(filepath), detections)
            record["image_id"] = (
                group["image_id"].iloc[0] if "image_id" in group.columns else ""
            )
            record["filename"] = (
                group["filename"].iloc[0]
                if "filename" in group.columns
                else os.path.basename(str(filepath))
            )
            entry = self._normalize_audit_entry(record)
            audit.append(entry)
            self._append_audit(entry)

        return audit

    def scrubbed_path_for_basename(self, filename: str) -> Optional[Path]:
        """Return scrubbed file path for a stored filename, if it exists."""
        if not self.output_dir:
            return None
        safe = Path(filename or "upload").name
        path = self.output_dir / safe
        return path if path.is_file() else None

    def resolve_display_file(
        self,
        original_path: Path,
        filename: str,
        scrubbing_enabled: bool = True,
    ) -> Path:
        """Pick scrubbed copy when scrubbing is enabled and the file exists."""
        if scrubbing_enabled:
            scrubbed = self.scrubbed_path_for_basename(filename)
            if scrubbed:
                return scrubbed
            scrubbed = self._scrubbed_path(str(original_path))
            if scrubbed.is_file():
                return scrubbed
        return original_path

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _normalize_audit_entry(self, record: dict) -> dict:
        skipped = bool(record.get("skipped", True))
        boxes = int(record.get("boxes_blurred") or 0)
        return {
            "filename": record.get("filename") or os.path.basename(record.get("original_path", "")),
            "image_id": record.get("image_id") or "",
            "scrubbed": "Yes" if (not skipped and boxes > 0) else "No",
            "skipped": "Yes" if skipped else "No",
            "boxes_blurred": boxes,
            "reason": record.get("reason") or ("Scrubbed" if boxes else "Skipped"),
            "scrubbed_at": record.get("scrubbed_at"),
            "scrubbed_path": record.get("scrubbed_path"),
            "original_path": record.get("original_path"),
        }

    def _append_audit(self, entry: dict) -> None:
        key = (str(entry.get("image_id", "")), str(entry.get("filename", "")))
        self.audit_log = [
            e for e in self.audit_log
            if (str(e.get("image_id", "")), str(e.get("filename", ""))) != key
        ]
        self.audit_log.append(entry)
        self._persist_audit()

    def _load_audit(self) -> None:
        if not self._audit_path or not self._audit_path.is_file():
            return
        try:
            data = json.loads(self._audit_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.audit_log = data
        except Exception:
            self.audit_log = []

    def _persist_audit(self) -> None:
        if not self._audit_path:
            return
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_path.write_text(
                json.dumps(self.audit_log, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrubbed_path(self, image_path: str) -> Path:
        src = Path(image_path)
        out_dir = self.output_dir if self.output_dir else src.parent / "scrubbed"
        return out_dir / src.name
