# Person detection

`detector.py` runs Roboflow's COCO-pretrained **RF-DETR Medium** and keeps only
the `person` class. No training is needed. `PersonDetector.predict(images)`
takes a list of BGR images and returns one `(N, 5)` array per image with
`[x1, y1, x2, y2, confidence]` rows, highest confidence first.

`config.yaml` selects the model size (`nano`, `small`, `medium`, or `large`;
these are the Apache 2.0 sizes), confidence, device, and maximum detections.
The checkpoint downloads automatically to `weights/` on first use and is not
stored in Git.

## Coverage evaluation

Run from the project root:

```bash
python person_detection/evaluate_model.py
```

A bib is found if its **entire annotated number box** is inside any predicted
person box. Multiple predictions covering one bib count only once. This measures
coverage of bib wearers; it cannot measure person precision or standard person AP
because the dataset has no person bounding box annotations.

The script compares full images against exactly four corner-anchored crops in a
2×2 grid with 20% overlap; crop detections are mapped back and merged with IoU
NMS at 0.5. `evaluation_results.json` contains summary counts and per-image
person boxes, found bib IDs, and missed bib IDs. Use `--splits val` for
validation only, `--max-images 8` for a quick check, or `--output path.json` to
save a separate run.

Validation split, full images (1242 bibs in 524 images):

| Model | Confidence | Bib coverage | Person boxes |
|---|---|---|---|
| YOLO26s (previous) | 0.25 | 97.34% | 4210 |
| RF-DETR Medium | 0.25 | 97.99% | 6155 |
| RF-DETR Medium (default) | 0.35 | 97.58% | 4549 |
| RF-DETR Medium | 0.45 | 96.86% | 3686 |

Model reference: https://github.com/roboflow/rf-detr
