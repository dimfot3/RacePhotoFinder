# Face detection

`inferer.py` runs the OpenCV Zoo YuNet ONNX detector inside each detected
person crop. Crops larger than `max_input_size` are downscaled for detection,
and results are mapped back to crop coordinates. It returns the face box,
confidence, and five facial landmarks used for alignment. Settings live in
`config.yaml`.

The weight `weights/face_detection_yunet_2023mar.onnx` is stored with Git LFS.
It is the official OpenCV Zoo release, licensed under the MIT License
(`../licenses/YuNet-MIT.txt`). Inference runs through OpenCV's DNN module; the
CUDA backend is used when `device` is not `cpu` and OpenCV was built with CUDA,
otherwise it runs on CPU.
