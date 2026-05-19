"""
Privacy Scrubbing Module
Blurs human faces and vehicle bounding boxes in camera trap images.
Non-destructive: originals are never modified. Scrubbed copies are written
to a configurable output directory.
"""

import os
import cv2
import json
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


# Labels that trigger scrubbing (MegaDetector category names)
_SCRUB_LABELS = {"person", "vehicle", "human"}

# Default Gaussian blur kernel — must be odd
_DEFAULT_BLUR = 51


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


class PrivacyScrubber:
    """
    Applies Gaussian blur to all Person and Vehicle bounding boxes
    in a camera trap image and saves the result as a separate scrubbed copy.

    Parameters
    ----------
    output_dir : str | Path
        Directory where scrubbed images are written.
        Defaults to a `scrubbed/` folder next to the source image.
    blur_strength : int
        Gaussian kernel size (odd integer). Larger = stronger blur.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        blur_strength: int = _DEFAULT_BLUR,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.blur_strength = _ensure_odd(max(11, blur_strength))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrub_image(
        self,
        image_path: str,
        detections: list,
    ) -> dict:
        """
        Blur all Person/Vehicle bounding boxes in one image.

        Parameters
        ----------
        image_path : str
            Absolute path to the original image.
        detections : list[dict]
            Detection records for this image. Each dict must contain:
                - primary_label  : str  ('Person', 'Vehicle', 'Animal', …)
                - bbox           : list [x, y, w, h] in relative coords (0–1),
                                   or None

        Returns
        -------
        dict with keys:
            original_path   : str
            scrubbed_path   : str | None   (None if no scrubbing was needed)
            boxes_blurred   : int
            scrubbed_at     : str (ISO timestamp)
            skipped         : bool         (True if no Person/Vehicle found)
        """
        result = {
            "original_path": image_path,
            "scrubbed_path": None,
            "boxes_blurred": 0,
            "scrubbed_at": datetime.now().isoformat(),
            "skipped": True,
        }

        # Identify bboxes that need blurring
        boxes_to_blur = [
            d["bbox"] for d in detections
            if d.get("primary_label", "").lower() in _SCRUB_LABELS
            and d.get("bbox") is not None
        ]

        if not boxes_to_blur:
            return result

        img = cv2.imread(image_path)
        if img is None:
            result["skipped"] = True
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
            return result

        # Write scrubbed copy
        scrubbed_path = self._scrubbed_path(image_path)
        scrubbed_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(scrubbed_path), img)

        result["scrubbed_path"] = str(scrubbed_path)
        result["skipped"] = False
        return result

    def scrub_batch(self, df, image_col: str = "filepath") -> list:
        """
        Scrub all images in a DataFrame that contain Person or Vehicle detections.

        Parameters
        ----------
        df : pd.DataFrame
            Processed detection data. Must contain `image_col` and
            `primary_label` / `bbox` columns.
        image_col : str
            Column holding the absolute image path.

        Returns
        -------
        list[dict]
            One audit record per image that was attempted.
        """
        import pandas as pd

        audit = []
        if df is None or df.empty or image_col not in df.columns:
            return audit

        for filepath, group in df.groupby(image_col):
            detections = group.to_dict("records")
            record = self.scrub_image(str(filepath), detections)
            record["image_id"] = group["image_id"].iloc[0] if "image_id" in group.columns else ""
            record["filename"] = group["filename"].iloc[0] if "filename" in group.columns else os.path.basename(str(filepath))
            audit.append(record)

        return audit

    def get_display_path(self, image_path: str) -> str:
        """
        Return the scrubbed path if it exists, otherwise the original.
        Used by the review UI to show privacy-safe images by default.
        """
        scrubbed = self._scrubbed_path(image_path)
        if scrubbed.exists():
            return str(scrubbed)
        return image_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrubbed_path(self, image_path: str) -> Path:
        src = Path(image_path)
        out_dir = self.output_dir if self.output_dir else src.parent / "scrubbed"
        return out_dir / src.name
