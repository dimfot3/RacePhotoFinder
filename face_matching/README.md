# Face matching

`embedder.py` aligns each YuNet detection from five landmarks and extracts a
normalized 128-value SFace (MobileFaceNet) embedding. `matcher.py` builds quality-weighted
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

The weight `weights/face_recognition_sface_2021dec.onnx` is stored with Git LFS.
It is the official OpenCV Zoo release, licensed under the Apache License 2.0
(`../licenses/SFace-Apache-2.0.txt`).

It runs with ONNX Runtime, using the CUDA execution provider when available
and CPU otherwise. Faces are aligned to the standard 112x112 five-point
template with the same least-squares similarity transform as OpenCV's
`FaceRecognizerSF.alignCrop`, so embeddings match OpenCV's.

The similarity thresholds in `config.yaml` are set for SFace, which scores
differently from ArcFace. Re-tune them on real event photos. Embeddings from a
different face model cannot be compared, so re-import existing events with
`--rerun` after changing the model.
