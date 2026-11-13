"""Five-point alignment and SFace embedding extraction (OpenCV Zoo, Apache 2.0)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from face_detection.inferer import _dnn_backend


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class EmbeddedFace:
    embedding: np.ndarray
    quality: float
    aligned_face: np.ndarray


class SFaceEmbedder:
    def __init__(self, config_path=DEFAULT_CONFIG):
        self.config = yaml.safe_load(Path(config_path).read_text())
        model_path = ROOT / self.config["model"]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"SFace model not found: {model_path}. Run `git lfs pull`."
            )

        backend, target, name = _dnn_backend(self.config.get("device"))
        self.model = cv2.FaceRecognizerSF.create(
            str(model_path), "", backend_id=backend, target_id=target
        )
        self.model_name = str(self.config["model_name"])
        self.dimension = int(self.config["embedding_dimension"])
        self.min_quality = float(self.config["min_embedding_quality"])
        print(f"SFace embedding: {name}")

    def embed(self, person_crop, detected_face):
        x1, y1, x2, y2 = detected_face.bbox
        face_row = np.array(
            [
                x1, y1, x2 - x1, y2 - y1,
                *np.asarray(detected_face.landmarks, dtype=np.float32).reshape(-1),
                detected_face.confidence,
            ],
            dtype=np.float32,
        )
        # alignCrop warps the five landmarks onto the standard 112x112 template.
        aligned = self.model.alignCrop(person_crop, face_row)
        quality = _face_quality(aligned, detected_face)
        if quality < self.min_quality:
            return None

        embedding = np.asarray(self.model.feature(aligned), dtype=np.float32).reshape(-1)
        if embedding.size != self.dimension:
            raise ValueError(
                f"SFace returned {embedding.size} values; expected {self.dimension}"
            )
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            return None
        embedding = np.ascontiguousarray(embedding / norm, dtype=np.float32)
        return EmbeddedFace(embedding=embedding, quality=quality, aligned_face=aligned)


def _face_quality(aligned_face, detected_face):
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, sharpness / 100.0)
    size_score = min(1.0, detected_face.size / 112.0)
    return float(
        detected_face.confidence
        * np.sqrt(max(0.0, sharpness_score * size_score))
    )
