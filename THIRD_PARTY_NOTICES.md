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

## OpenCV Zoo YuNet and SFace

Race Photo Finder includes these official OpenCV Zoo face models:

- `face_detection/weights/face_detection_yunet_2023mar.onnx` (YuNet)
  - SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
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

## Datasets and user-provided media

The `datasets/` directory and event photographs are excluded from this
repository. No license or permission for those materials is granted here. A
user who trains or redistributes a model is responsible for confirming that
the source dataset and media permissions allow that use.
