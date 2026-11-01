# YOLO26 person coverage evaluation

Run from the project root:

```bash
python person_detection/evaluate_model.py
```

Uses COCO pretrained `yolo26n.pt`, class `person` only. Weights download on first run.
Full images are resized with aspect ratio preserved and letterboxed to 640×640.
The second mode uses exactly four corner-anchored crops in a 2×2 grid, with
20% overlap measured relative to crop size. Each crop is letterboxed to 640×640.
Crop detections are mapped to original coordinates and merged with IoU NMS at 0.5.
Both modes use confidence 0.25 and the same images.

A bib is found if its **entire annotated number box** is inside any predicted
person box. Multiple predictions covering one bib count only once. This measures
coverage of bib wearers; it cannot measure person precision or standard person AP
because the dataset has no person bounding box annotations.

`config.yaml` controls dataset, weights, splits, input size, confidence and batch.
`evaluation_results.json` contains summary counts and per-image person boxes,
found bib IDs, and missed bib IDs. It is saved after each completed mode/split.
Use `--splits val` for validation only, `--max-images 8` for a quick check,
or `--output path.json` to save a separate run.

The local `.deps` directory contains an updated Ultralytics package for YOLO26;
the existing Python environment is left intact. If it is absent, install a current
Ultralytics version in your environment (`pip install -U ultralytics`).

Model reference: https://docs.ultralytics.com/tasks/detect/
