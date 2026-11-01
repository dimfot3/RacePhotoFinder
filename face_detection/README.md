# Face detection

`inferer.py` runs the SCRFD-10G ONNX detector inside each detected person crop.
It returns the face box, confidence, and five facial landmarks used for
alignment. Settings live in `config.yaml`.

The included InsightFace pretrained weight is licensed for non-commercial
research unless separate authorization is obtained from InsightFace.

