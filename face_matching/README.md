# Face matching

`embedder.py` aligns each SCRFD detection from five landmarks and extracts a
normalized 512-value ArcFace R50 embedding. `matcher.py` builds quality-weighted
participant templates only from bib-confirmed detections, removes outliers, and
accepts unmatched faces only when both cosine similarity and the best-vs-second
margin pass `config.yaml`.

Face-derived matches are deliberately excluded from participant templates to
prevent one wrong match from contaminating later matches.

Bib search also retrieves non-mutating face suggestions. Its cosine threshold
is `search_similarity` in `config.yaml`; suggested photos are displayed
separately from bib-confirmed photos.

The included InsightFace pretrained weight is licensed for non-commercial
research unless separate authorization is obtained from InsightFace.

