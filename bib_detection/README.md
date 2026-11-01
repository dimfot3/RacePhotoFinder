# Bib-number detection on person crops

Start with COCO-pretrained **YOLO26s**, fine-tuned as one `bib-number` class on
`datasets/person_cropped_images`. Images remain at original crop resolution on
disk; Ultralytics resizes and pads during training. Defaults: input 640, batch 8,
50 epochs, early stopping patience 10, CUDA device 0. Edit `config.yaml` to change.

```bash
python bib_detection/train_model.py
python bib_detection/evaluate_model.py
```

For a new environment: `pip install -r bib_detection/requirements.txt`.
A local `.deps` copy of the current Ultralytics package is provided for the
existing environment; it supports YOLO26 without upgrading the older installation.
The official `weights/yolo26s.pt` checkpoint downloads automatically if absent.
Training adapts the COCO detector to the bib class; it does not start from random weights.

COCO boxes are converted to normalized YOLO labels in `.yolo_dataset` with
symlinks to the original crop PNGs. No image copies or resizing are performed.
Truncated boxes are included as labeled in the crop dataset.
Use `python bib_detection/train_model.py --prepare-only` to only prepare labels.

Training uses train and val, leaving test for the evaluation script. Best and
last checkpoints are saved under `runs/bib_yolo26s/weights/` on the first run.
Ultralytics increments the run directory on subsequent runs to preserve weights;
use `--weights path/to/best.pt` when evaluating a later run or change `weights`
in the config.

Evaluation uses trained weights and standard bounding box IoU metrics: precision,
recall, mAP50, and mAP50–95 on val and test. Precision/recall in Ultralytics summaries
are selected from each split's F1 curve; mAP integrates the confidence curve.
`evaluation_stats.json` also recommends confidence from the **val** F1 curve only.
Per-split curves and plots are saved in `runs/evaluate_val*` and `runs/evaluate_test*`.
These metrics measure box localization, not OCR reading accuracy.

Single-photo person-to-bib inference and visualization:

```bash
python infer.py path/to/photo.jpg
python infer.py path/to/photo.jpg --output result.jpg --no-show
```

The root script reads both detection configs, uses YOLO26s for people and the
trained bib `best.pt` for person crops, and displays the final bib boxes on the
original image. Bib confidence is configured as 0.43 from validation.

Folder imports use a bounded fallback when a person crop has no valid bib at
the normal threshold. The fallback detector threshold, stricter OCR threshold,
and maximum candidates are configured in `config.yaml`. At most one bib is
retained per person crop, selected by combined detection and OCR confidence.

Model documentation: https://docs.ultralytics.com/models/yolo26/
