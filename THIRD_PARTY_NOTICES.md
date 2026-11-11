# Third-party software and model notices

The GNU Affero General Public License v3.0 in `LICENSE` applies to the original
Race Photo Finder source code. It does not replace or alter the licenses of the
third-party software, pretrained models, datasets, or other materials described
below. Those components remain subject to their respective upstream terms.

## Ultralytics YOLO26

Race Photo Finder uses Ultralytics YOLO26 models for person and bib detection.
The relevant files include:

- `weights/yolo26n.pt`
- `person_detection/weights/yolo26n.pt`
- `person_detection/weights/yolo26s.pt`
- `bib_detection/weights/yolo26s.pt`
- `bib_detection/runs/bib_yolo26s/weights/best.pt`

The final file is a fine-tuned derivative of the YOLO26s checkpoint. Ultralytics
offers its software and models under AGPL-3.0 and a separate Enterprise
license. This repository uses the AGPL-3.0 option. Users who cannot satisfy the
AGPL-3.0 requirements must obtain appropriate authorization from Ultralytics.
Rights in the dataset used to fine-tune the bib detector are separate and are
not granted by this repository.

- Project: https://github.com/ultralytics/ultralytics
- Model documentation: https://docs.ultralytics.com/models/yolo26/
- Licensing: https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed
- AGPL-3.0 text: https://www.gnu.org/licenses/agpl-3.0.html

## InsightFace buffalo_l models

The application expects these InsightFace `buffalo_l` model files locally:

- `face_detection/weights/det_10g.onnx`
  - SHA-256: `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`
- `face_matching/weights/w600k_r50.onnx`
  - SHA-256: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`

These pretrained weights are not covered by InsightFace's MIT source-code
license. InsightFace states that its public pretrained models are available for
non-commercial research only unless separate authorization is obtained. The
model binaries are therefore intentionally not redistributed by this
repository. Each user must review the upstream terms and obtain the official
`buffalo_l` package independently.

`setup_models.py` asks the user to accept these terms, downloads `buffalo_l.zip`
directly from InsightFace's official GitHub release
(https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip),
and installs only the two files above after verifying the SHA-256 checksums
listed here.

- Project and license policy: https://github.com/deepinsight/insightface
- Official model-zoo instructions: https://github.com/deepinsight/insightface/tree/master/model_zoo
- Commercial licensing: https://www.insightface.ai/

## PaddleOCR PP-OCRv6

Race Photo Finder includes the official `PP-OCRv6_small_rec` inference files in
`bib_ocr/PP-OCRv6_small_rec_infer/`. PaddleOCR is distributed under the Apache
License 2.0. Its copyright and license terms remain with the PaddlePaddle
authors.

- Project: https://github.com/PaddlePaddle/PaddleOCR
- Included license text: `licenses/PaddleOCR-Apache-2.0.txt`
- Upstream license: https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
- Model documentation: https://www.paddleocr.ai/

## Datasets and user-provided media

The `datasets/` directory and event photographs are excluded from this
repository. No license or permission for those materials is granted here. A
user who trains or redistributes a model is responsible for confirming that
the source dataset and media permissions allow that use.
