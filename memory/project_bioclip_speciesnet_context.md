---
name: project-bioclip-speciesnet-context
description: Technical context on how BioCLIP and SpeciesNet differ and how they're used in this camera-trap pipeline
metadata:
  type: project
---

BioCLIP is a broad vision-transformer foundation model (450K+ taxa, tree-of-life hierarchy) best for zero-shot and diverse organism classification. SpeciesNet is a CNN trained on 65M+ human-labeled camera trap images, specialized for mammals/birds/reptiles under field conditions (nocturnal IR, poor lighting, partial visibility, ~2500 categories).

**Why:** The two models have complementary strengths — SpeciesNet is the domain expert for camera traps; BioCLIP adds taxonomic breadth but is not camera-trap-optimized.

**How to apply:**
- SpeciesNet should carry higher ensemble weight than BioCLIP for standard camera trap scenarios (currently 0.55 vs 0.45 — directionally correct but could go higher)
- Day/night classification output should dynamically boost SpeciesNet weight at night (it was trained on nocturnal IR shots)
- BioCLIP's hierarchical taxonomy is unused — the app treats it as a flat classifier; fuse_species uses substring matching instead of taxonomy-aware comparison
- BioCLIP's species list is capped at 129 African species (WILDLIFE_CLASSES) — this prevents misclassification but loses BioCLIP's breadth advantage
- Agreement detection between the two models is unreliable because BioCLIP may return scientific names while SpeciesNet returns common names or different label formats
