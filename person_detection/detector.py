"""RF-DETR person detector (Roboflow, Apache 2.0 code and COCO weights)."""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
COCO_PERSON_CLASS_ID = 1

# Only these RF-DETR detection sizes ship under Apache 2.0. The smaller
# (Atto/Femto/Pico) and larger (XLarge/2XLarge) sizes use Roboflow's PML license.
APACHE_MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


class PersonDetector:
    """Detect people in BGR images and return [x1, y1, x2, y2, confidence] rows."""

    def __init__(self, config_path=DEFAULT_CONFIG):
        config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(config_path.read_text())
        size = str(self.config["model"]).lower()
        if size not in APACHE_MODELS:
            raise ValueError(
                f"Unsupported RF-DETR model {size!r}; choose one of {sorted(APACHE_MODELS)}"
            )
        self.confidence = float(self.config["confidence"])
        self.max_det = int(self.config["max_det"])
        self.device = _device(self.config.get("device"))
        weights = (config_path.parent / self.config["weights"]).resolve()
        weights.parent.mkdir(parents=True, exist_ok=True)

        import rfdetr

        # RF-DETR downloads the official checkpoint to `weights` when it is
        # missing and verifies its MD5 hash.
        self.model = getattr(rfdetr, APACHE_MODELS[size])(
            pretrain_weights=str(weights),
            device=self.device,
        )
        print(f"RF-DETR person detection: {size}, {weights}, device={self.device}, "
              f"confidence={self.confidence}")

    def predict(self, images):
        """Return one float32 (N, 5) array per BGR image, highest confidence first."""
        images = list(images)
        if not images:
            return []
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        detections = self.model.predict(
            rgb, threshold=self.confidence, include_source_image=False
        )
        if not isinstance(detections, list):
            detections = [detections]
        if len(detections) != len(images):
            raise RuntimeError(
                f"RF-DETR returned {len(detections)} results for {len(images)} images"
            )

        results = []
        for result in detections:
            keep = result.class_id == COCO_PERSON_CLASS_ID
            boxes = np.concatenate(
                [result.xyxy[keep], result.confidence[keep, None]], axis=1
            ).astype(np.float32)
            order = np.argsort(-boxes[:, 4], kind="stable")[: self.max_det]
            results.append(boxes[order])
        return results


def _device(device):
    if str(device).lower() == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return "cuda"
