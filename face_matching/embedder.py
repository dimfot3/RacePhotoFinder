"""Five-point alignment and ArcFace embedding extraction."""

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class EmbeddedFace:
    embedding: np.ndarray
    quality: float
    aligned_face: np.ndarray


class ArcFaceEmbedder:
    def __init__(self, config_path=DEFAULT_CONFIG):
        self.config = yaml.safe_load(Path(config_path).read_text())
        model_path = ROOT / self.config["model"]
        if not model_path.is_file():
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")

        providers, ctx_id = _execution_providers(self.config.get("device"))
        self.model = get_model(str(model_path), providers=providers)
        self.model.prepare(ctx_id=ctx_id)
        self.aligned_size = int(self.config["aligned_size"])
        self.dimension = int(self.config["embedding_dimension"])
        self.min_quality = float(self.config["min_embedding_quality"])
        active_provider = self.model.session.get_providers()[0]
        print(f"ArcFace embedding: {active_provider}")

    def embed(self, person_crop, detected_face):
        aligned = norm_crop(
            person_crop,
            landmark=detected_face.landmarks,
            image_size=self.aligned_size,
        )
        quality = _face_quality(aligned, detected_face)
        if quality < self.min_quality:
            return None

        embedding = np.asarray(self.model.get_feat(aligned), dtype=np.float32).reshape(-1)
        if embedding.size != self.dimension:
            raise ValueError(
                f"ArcFace returned {embedding.size} values; expected {self.dimension}"
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


def _execution_providers(device):
    available = set(ort.get_available_providers())
    if str(device).lower() != "cpu" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1

