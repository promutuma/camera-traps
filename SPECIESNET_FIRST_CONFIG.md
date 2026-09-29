# SpeciesNet Configuration (Deprecated Doc)

> **Update (2026):** The production pipeline uses **MegaDetector v5a + SpeciesNet only**. BioCLIP, MegaDetector v1000, and classifier fusion modes described in older versions of this file are **removed**. There is no `use_speciesnet_first` or fusion-weight configuration.

## Current behaviour

- **Detection:** MegaDetector v5a returns Animal / Person / Vehicle boxes.
- **Classification:** SpeciesNet classifies each animal crop; all candidates are returned ranked by confidence.
- **Agreement:** Derived from SpeciesNet confidence (High ≥ 0.7, Medium ≥ 0.4, Low otherwise).

## SpeciesNet geographic prior

Set in the sidebar or via `AppConfig`:

```python
speciesnet_lat: float = -1.0      # default Kenya
speciesnet_lng: float = 37.0
speciesnet_country: str = "KEN"
```

Coordinates sync from the active project when one is selected.

## Kaggle credentials

SpeciesNet requires Kaggle API credentials in `.env`:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Without credentials, SpeciesNet is skipped; MegaDetector detection still runs.

See [README.md](README.md) for full setup.
