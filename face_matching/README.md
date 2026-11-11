# Face matching

`embedder.py` aligns each SCRFD detection from five landmarks and extracts a
normalized 512-value ArcFace R50 embedding. `matcher.py` builds quality-weighted
participant templates only from bib-confirmed detections, removes outliers, and
accepts unmatched faces only when both cosine similarity and the best-vs-second
margin pass `config.yaml`.

Face-derived matches are deliberately excluded from participant templates to
prevent one wrong match from contaminating later matches.

Bib search also performs a non-mutating similarity pass using
`search_similarity` from `config.yaml`. Bib/manual associations are returned
first, followed by additional face-similar photos ordered by similarity; the UI
combines them in one gallery. This search-time pass does not write new
associations, while the stricter post-import matcher does persist accepted face
associations in the database.

The InsightFace pretrained weight (`weights/w600k_r50.onnx`) is not included in this
repository. Run `python setup_models.py` from the project root to download it
from InsightFace's official release. It is licensed for non-commercial research
unless separate authorization is obtained from InsightFace.

