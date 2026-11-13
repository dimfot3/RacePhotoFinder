# Face detection

`inferer.py` runs the OpenCV Zoo YuNet ONNX detector inside each detected
person crop. Crops larger than `max_input_size` are downscaled for detection,
and results are mapped back to crop coordinates. It returns the face box,
confidence, and five facial landmarks used for alignment. Settings live in
`config.yaml`.

The weight `weights/face_detection_yunet_2026may.onnx` is stored with Git LFS.
It is the official OpenCV Zoo release, licensed under the MIT License
(`../licenses/YuNet-MIT.txt`).

Inference runs with ONNX Runtime. When `device` is not `cpu`, the CUDA
execution provider is tried first and CPU is the fallback; the startup log
prints which one is active. `onnxruntime-gpu` needs CUDA 12 and cuDNN 9, from a
system install or NVIDIA's pip packages. The box decoding and NMS reproduce
OpenCV's `FaceDetectorYN` exactly.
