# Camera Trap Pipeline — Model Context

**Production stack (2026):** MegaDetector **v5a** (detection) + **SpeciesNet** (classification). BioCLIP and MegaDetector v1000 are not loaded.

- SpeciesNet is the sole classifier; all ranked candidates are returned per detection.
- Agreement tiers come from SpeciesNet confidence (High ≥ 0.7, Medium ≥ 0.4, Low otherwise).
- Geographic prior: `speciesnet_lat/lng/country` in AppConfig, synced from active project.

See [README.md](../README.md) for setup and pipeline details.
