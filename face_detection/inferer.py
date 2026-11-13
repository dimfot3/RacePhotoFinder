"""YuNet face detector for person crops (OpenCV Zoo, MIT license)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class DetectedFace:
    """One face in coordinates relative to its input person crop."""

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    confidence: float

    @property
    def size(self):
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


class YuNetFaceDetector:
    def __init__(self, config_path=DEFAULT_CONFIG):
        self.config = yaml.safe_load(Path(config_path).read_text())
        model_path = ROOT / self.config["model"]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}. Run `git lfs pull`."
            )

        self.max_input_size = int(self.config["max_input_size"])
        self.min_face_size = float(self.config["min_face_size"])
        self.max_faces = int(self.config["max_faces_per_person"])
        backend, target, name = _dnn_backend(self.config.get("device"))
        self.model = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (self.max_input_size, self.max_input_size),
            score_threshold=float(self.config["confidence"]),
            nms_threshold=float(self.config["nms_threshold"]),
            top_k=5000,
            backend_id=backend,
            target_id=target,
        )
        print(f"YuNet face detection: {name}")

    def detect(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return []

        # Downscale large crops so the longest side fits max_input_size, then
        # map the results back to the original crop coordinates.
        height, width = person_crop.shape[:2]
        scale = min(1.0, self.max_input_size / max(height, width))
        image = person_crop
        if scale < 1.0:
            image = cv2.resize(
                person_crop,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self.model.setInputSize((image.shape[1], image.shape[0]))
        _, detections = self.model.detect(image)
        if detections is None:
            return []

        faces = []
        for row in detections:
            x, y, w, h = (float(value) / scale for value in row[:4])
            # YuNet landmark order (right eye, left eye, nose, right and left
            # mouth corners) matches the ArcFace alignment template order.
            points = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2) / scale
            face = DetectedFace(
                bbox=(x, y, x + w, y + h),
                landmarks=points,
                confidence=float(row[14]),
            )
            if face.size >= self.min_face_size:
                faces.append(face)
        faces.sort(
            key=lambda face: face.confidence * face.size * face.size,
            reverse=True,
        )
        return faces[: self.max_faces]


def _dnn_backend(device):
    """Use OpenCV's CUDA DNN backend when requested and available."""
    if str(device).lower() != "cpu":
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA"
        except (AttributeError, cv2.error):
            pass
    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "CPU"
