"""CUDA-first SCRFD face detector for person crops."""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from insightface.model_zoo import get_model


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


class SCRFDFaceDetector:
    def __init__(self, config_path=DEFAULT_CONFIG):
        self.config = yaml.safe_load(Path(config_path).read_text())
        model_path = ROOT / self.config["model"]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"SCRFD model not found: {model_path}. Run `python setup_models.py`."
            )

        self.providers, self.ctx_id = _execution_providers(self.config.get("device"))
        self.model = get_model(str(model_path), providers=self.providers)
        self.model.prepare(
            ctx_id=self.ctx_id,
            input_size=tuple(self.config["input_size"]),
            det_thresh=float(self.config["confidence"]),
        )
        self.min_face_size = float(self.config["min_face_size"])
        self.max_faces = int(self.config["max_faces_per_person"])
        active_provider = self.model.session.get_providers()[0]
        print(f"SCRFD face detection: {active_provider}")

    def detect(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return []
        boxes, landmarks = self.model.detect(
            person_crop,
            max_num=self.max_faces,
            metric="max",
        )
        if boxes is None or landmarks is None:
            return []

        faces = []
        for box, points in zip(boxes, landmarks):
            face = DetectedFace(
                bbox=tuple(float(value) for value in box[:4]),
                landmarks=np.asarray(points, dtype=np.float32),
                confidence=float(box[4]),
            )
            if face.size >= self.min_face_size:
                faces.append(face)
        return sorted(
            faces,
            key=lambda face: face.confidence * face.size * face.size,
            reverse=True,
        )


def _execution_providers(device):
    available = set(ort.get_available_providers())
    if str(device).lower() != "cpu" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1

