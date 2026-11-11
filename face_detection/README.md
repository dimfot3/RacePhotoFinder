# Face detection

`inferer.py` runs the SCRFD-10G ONNX detector inside each detected person crop.
It returns the face box, confidence, and five facial landmarks used for
alignment. Settings live in `config.yaml`.

The InsightFace pretrained weight (`weights/det_10g.onnx`) is not included in this
repository. Run `python setup_models.py` from the project root to download it
from InsightFace's official release. It is licensed for non-commercial research
unless separate authorization is obtained from InsightFace.

