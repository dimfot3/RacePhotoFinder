# Bib-number detection on person crops

Starts from COCO-pretrained **RF-DETR Medium** (Apache 2.0) and fine-tunes it as
one `bib-number` class on `datasets/person_cropped_images`. Defaults: 50 epochs
with early stopping (patience 10), batch 4 × 4 gradient-accumulation steps
(effective batch 16), CUDA. Edit `config.yaml` to change.

## Train and evaluate

```bash
python -m pip install -r bib_detection/requirements.txt
python bib_detection/train_model.py
python bib_detection/evaluate_model.py
```

Training links the original crops into `.rfdetr_dataset/` (`val` becomes
`valid`, the layout RF-DETR expects); no images are copied or resized. Training
uses train and valid, leaving test for the evaluation script. Checkpoints are
written to `runs/bib_rfdetr_medium/`; the pipeline loads
`checkpoint_best_total.pth`, which is the only file from the run kept in Git.
Training again overwrites that directory.

Useful options:

- `--prepare-only` only builds `.rfdetr_dataset/`.
- `--limit 300 --epochs 4 --output-dir /tmp/bib_check` runs a quick check on a
  subset without touching the real run directory.
- `evaluate_model.py --weights path/to/checkpoint.pth` evaluates another run.

## Evaluation

`evaluate_model.py` reports COCO mAP50, mAP50–95, mAR, and RF-DETR's best-F1
precision/recall on val and test, and writes them to `evaluation_stats.json`.
It also sweeps confidence on **val** only (IoU 0.5) and records the best-F1
threshold as `recommended_confidence`. Copy that value to `confidence` in
`config.yaml`. These metrics measure box localization, not OCR reading accuracy.

Previous YOLO26s detector, for comparison:

| Split | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---|---|---|
| val | 0.972 | 0.873 | 0.935 | 0.664 |
| test | 0.961 | 0.896 | 0.945 | 0.730 |

## Inference

```bash
python infer.py path/to/photo.jpg
python infer.py path/to/photo.jpg --output result.jpg --no-show
```

`detector.py` provides `BibDetector.predict(crops, confidence=None, max_det=None)`,
which returns one `[x1, y1, x2, y2, confidence]` array per BGR person crop.

Folder imports use a bounded fallback when a person crop has no valid bib at
the normal threshold. The fallback detector threshold, stricter OCR threshold,
and maximum candidates are configured in `config.yaml`. At most one bib is
retained per person crop, selected by combined detection and OCR confidence.

Model reference: https://github.com/roboflow/rf-detr
