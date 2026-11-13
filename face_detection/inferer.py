"""YuNet face detector for person crops (OpenCV Zoo, MIT license)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
STRIDES = (8, 16, 32)


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
        self.score_threshold = float(self.config["confidence"])
        self.nms_threshold = float(self.config["nms_threshold"])
        self.min_face_size = float(self.config["min_face_size"])
        self.max_faces = int(self.config["max_faces_per_person"])
        self.session = onnx_session(model_path, self.config.get("device"))
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"YuNet face detection: {self.session.get_providers()[0]}")

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
        faces = []
        for row in self._run(image):
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

    def _run(self, image):
        """Return YuNet rows [x, y, w, h, 10 landmark coords, score] after NMS.

        Mirrors OpenCV's FaceDetectorYN: pad bottom/right to a multiple of 32,
        feed raw BGR values, decode each stride's grid, then apply NMS.
        """
        height, width = image.shape[:2]
        pad_h = (height - 1) // 32 * 32 + 32
        pad_w = (width - 1) // 32 * 32 + 32
        padded = cv2.copyMakeBorder(
            image, 0, pad_h - height, 0, pad_w - width, cv2.BORDER_CONSTANT, value=0
        )
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = dict(zip(self.output_names, self.session.run(None, {self.input_name: blob})))

        rows = []
        for stride in STRIDES:
            cols, grid_rows = pad_w // stride, pad_h // stride
            cls = outputs[f"cls_{stride}"].reshape(-1)
            obj = outputs[f"obj_{stride}"].reshape(-1)
            bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
            kps = outputs[f"kps_{stride}"].reshape(-1, 10)
            scores = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            keep = np.flatnonzero(scores >= self.score_threshold)
            if keep.size == 0:
                continue
            grid_x = (keep % cols).astype(np.float32)
            grid_y = (keep // cols).astype(np.float32)
            cx = (grid_x + bbox[keep, 0]) * stride
            cy = (grid_y + bbox[keep, 1]) * stride
            w = np.exp(bbox[keep, 2]) * stride
            h = np.exp(bbox[keep, 3]) * stride
            points = kps[keep].reshape(-1, 5, 2).copy()
            points[:, :, 0] = (points[:, :, 0] + grid_x[:, None]) * stride
            points[:, :, 1] = (points[:, :, 1] + grid_y[:, None]) * stride
            rows.append(np.column_stack(
                [cx - w / 2, cy - h / 2, w, h, points.reshape(-1, 10), scores[keep]]
            ))
        if not rows:
            return np.empty((0, 15), dtype=np.float32)
        rows = np.concatenate(rows).astype(np.float32)
        kept = cv2.dnn.NMSBoxes(
            rows[:, :4].tolist(), rows[:, 14].tolist(),
            self.score_threshold, self.nms_threshold, top_k=5000,
        )
        return rows[np.asarray(kept, dtype=int).reshape(-1)]


def onnx_session(model_path, device):
    """ONNX Runtime session that prefers CUDA and falls back to CPU."""
    providers = ["CPUExecutionProvider"]
    if str(device).lower() != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
        # onnxruntime-gpu needs CUDA 12 / cuDNN 9; preload them from the system
        # or NVIDIA pip packages so the provider can initialise.
        try:
            ort.preload_dlls()
        except Exception:
            pass
        providers.insert(0, "CUDAExecutionProvider")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), options, providers=providers)
