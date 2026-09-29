#!/usr/bin/env python3
"""
Pre-download MegaDetector v5a and SpeciesNet weights before first run.

Usage:
  python force_download.py

Requires KAGGLE_USERNAME and KAGGLE_KEY in the environment (or .env) for SpeciesNet.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("force_download")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.models.state import AppState
    from backend.services.model_loader import load_all_models, load_speciesnet

    state = AppState()
    logger.info("Downloading MegaDetector v5a weights...")
    load_all_models(state, ROOT)

    if state.config.enable_low_spec:
        logger.info("Low-Spec is enabled — skipping SpeciesNet download.")
        return 0

    logger.info("Downloading SpeciesNet weights (Kaggle)...")
    load_speciesnet(state)
    if state.speciesnet_model and state.speciesnet_model.classifier:
        logger.info("All models ready.")
        return 0

    logger.error("SpeciesNet failed: %s", state.speciesnet_error or "unknown error")
    logger.error("Set KAGGLE_USERNAME and KAGGLE_KEY in .env and retry.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
