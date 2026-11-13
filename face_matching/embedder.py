"""Five-point alignment and SFace embedding extraction (OpenCV Zoo, Apache 2.0)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from face_detection.inferer import onnx_session


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
ALIGNED_SIZE = 112
# Standard 112x112 five-point face template (eyes, nose, mouth corners).
TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


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

        self.session = onnx_session(model_path, self.config.get("device"))
        self.input_name = self.session.get_inputs()[0].name
        self.model_name = str(self.config["model_name"])
        self.dimension = int(self.config["embedding_dimension"])
        self.min_quality = float(self.config["min_embedding_quality"])
        print(f"SFace embedding: {self.session.get_providers()[0]}")

    def embed(self, person_crop, detected_face):
        aligned = align_face(person_crop, detected_face.landmarks)
        quality = _face_quality(aligned, detected_face)
        if quality < self.min_quality:
            return None

        # SFace expects raw 0-255 RGB values in NCHW layout.
        blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        embedding = self.session.run(None, {self.input_name: blob.astype(np.float32)})[0]
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.size != self.dimension:
            raise ValueError(
                f"SFace returned {embedding.size} values; expected {self.dimension}"
            )
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            return None
        embedding = np.ascontiguousarray(embedding / norm, dtype=np.float32)
        return EmbeddedFace(embedding=embedding, quality=quality, aligned_face=aligned)


def align_face(image, landmarks):
    """Warp five landmarks onto the standard 112x112 template."""
    matrix = _similarity_transform(np.asarray(landmarks, dtype=np.float64).reshape(5, 2), TEMPLATE)
    return cv2.warpAffine(image, matrix, (ALIGNED_SIZE, ALIGNED_SIZE), borderValue=0.0)


def _similarity_transform(source, target):
    """Least-squares similarity transform (Umeyama), as in OpenCV's alignCrop."""
    target = np.asarray(target, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(2)
    if np.linalg.det(covariance) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    scale = (singular * sign).sum() / source_centered.var(axis=0).sum()
    translation = target_mean - scale * rotation @ source_mean
    return np.hstack([scale * rotation, translation[:, None]])


def _face_quality(aligned_face, detected_face):
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, sharpness / 100.0)
    size_score = min(1.0, detected_face.size / 112.0)
    return float(
        detected_face.confidence
        * np.sqrt(max(0.0, sharpness_score * size_score))
    )
