# Third-party software and model notices

The GNU Affero General Public License v3.0 in `LICENSE` applies to the original
Race Photo Finder source code. It does not replace or alter the licenses of the
third-party software, pretrained models, datasets, or other materials described
below. Those components remain subject to their respective upstream terms.

## Roboflow RF-DETR

Person and bib detection use the `rfdetr` package and its official
COCO-pretrained RF-DETR Medium checkpoint. The checkpoint is not stored in this
repository: `rfdetr` downloads it to `person_detection/weights/rf-detr-medium.pth`
on first use and verifies its MD5 hash.

The bib detector, `bib_detection/runs/bib_rfdetr_medium/checkpoint_best_total.pth`,
is fine-tuned from that checkpoint. Rights in the dataset used to fine-tune it
are separate and are not granted by this repository.

- `person_detection/weights/rf-detr-medium.pth`
  - SHA-256: `749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8`
- License: Apache License 2.0 for the `rfdetr` code and the Nano, Small,
  Medium, and Large detection checkpoints. Included text:
  `licenses/RF-DETR-Apache-2.0.txt`
- The Atto, Femto, Pico, XLarge, and 2XLarge checkpoints are under Roboflow's
  Platform Model License instead; this project's detector code refuses them.
- Project: https://github.com/roboflow/rf-detr

The COCO-pretrained checkpoint was trained on the COCO dataset, whose images
carry their own Flickr license terms.

## OpenCV Zoo YuNet and SFace

Race Photo Finder includes these official OpenCV Zoo face models:

- `face_detection/weights/face_detection_yunet_2026may.onnx` (YuNet; the
  official re-export of the 2023mar weights with dynamic input size)
  - SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
  - License: MIT License, Copyright (c) 2020 Shiqi Yu. Included text:
    `licenses/YuNet-MIT.txt`
  - Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  - Upstream training code: https://github.com/ShiqiYu/libfacedetection.train
- `face_matching/weights/face_recognition_sface_2021dec.onnx` (SFace)
  - SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
  - License: Apache License 2.0. Included text: `licenses/SFace-Apache-2.0.txt`
  - Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
  - Paper: https://arxiv.org/abs/2205.12010

YuNet's upstream project states that it was trained on WIDER FACE. The SFace
model card does not name its training dataset. Rights in third-party training
data are separate from the model licenses above, and anyone deploying this
software commercially should review them independently.

## PaddleOCR PP-OCRv6

Race Photo Finder includes the official `PP-OCRv6_small_rec` inference files in
`bib_ocr/PP-OCRv6_small_rec_infer/`. PaddleOCR is distributed under the Apache
License 2.0. Its copyright and license terms remain with the PaddlePaddle
authors.

- Project: https://github.com/PaddlePaddle/PaddleOCR
- Included license text: `licenses/PaddleOCR-Apache-2.0.txt`
- Upstream license: https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
- Model documentation: https://www.paddleocr.ai/

## Bib-number training dataset

The bib detector (`bib_detection/runs/bib_rfdetr_medium/checkpoint_best_total.pth`)
was fine-tuned on person crops derived from the following dataset:

- Title: "bib number labeling"
- Creator: Marco Cheung (Roboflow Universe workspace `marco-cheung`)
- Source: https://universe.roboflow.com/marco-cheung/bib-number-labeling
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/

Changes: the original images were cropped to individual people using an
automatic person detector, bib boxes were re-expressed relative to each crop,
and crops without a complete bib were excluded (see
`person_detection/create_person_dataset.py`). The original train/valid/test
split was preserved. The dataset itself is not redistributed with this
repository.

## Datasets and user-provided media

The `datasets/` directory and event photographs are excluded from this
repository. No license or permission for those materials is granted here. A
user who trains or redistributes a model is responsible for confirming that
the source dataset and media permissions allow that use.
