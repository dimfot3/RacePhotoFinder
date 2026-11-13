"""Fine-tuned RF-DETR bib-number detector for person crops."""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# Only these RF-DETR detection sizes ship under Apache 2.0.
APACHE_MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


class BibDetector:
    """Detect bib numbers in BGR person crops."""

    def __init__(self, config_path=DEFAULT_CONFIG):
        config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(config_path.read_text())
        self.confidence = float(self.config["confidence"])
        self.batch_size = int(self.config["batch"])
        self.device = (
            "cpu"
            if str(self.config.get("device")) == "cpu" or not torch.cuda.is_available()
            else "cuda"
        )
        weights = (config_path.parent / self.config["weights"]).resolve()
        if not weights.is_file():
            raise FileNotFoundError(
                f"Bib detector weights not found: {weights}. "
                "Run `python bib_detection/train_model.py` first."
            )

        from rfdetr import RFDETR

        from person_detection.detector import _predictor

        self.model = RFDETR.from_checkpoint(str(weights), device=self.device)
        self.predictor = _predictor(self.model, self.device, self.config.get("precision"))
        self.predictor_dtype = self.predictor.dtype_name
        print(f"RF-DETR bib detection: {weights}, device={self.device}, "
              f"precision={self.predictor_dtype}, confidence={self.confidence}")

    def predict(self, crops, confidence=None, max_det=None):
        """Return one float32 (N, 5) [x1, y1, x2, y2, conf] array per crop."""
        crops = list(crops)
        threshold = self.confidence if confidence is None else float(confidence)
        results = []
        for start in range(0, len(crops), self.batch_size):
            batch = [
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                for crop in crops[start:start + self.batch_size]
            ]
            detections = self.predictor.predict(
                batch, threshold=threshold, include_source_image=False
            )
            if not isinstance(detections, list):
                detections = [detections]
            for result in detections:
                # Single-class model: every detection is a bib number.
                boxes = np.concatenate(
                    [result.xyxy, result.confidence[:, None]], axis=1
                ).astype(np.float32)
                order = np.argsort(-boxes[:, 4], kind="stable")[:max_det]
                results.append(boxes[order])
        if len(results) != len(crops):
            raise RuntimeError(f"RF-DETR returned {len(results)} results for {len(crops)} crops")
        return results
