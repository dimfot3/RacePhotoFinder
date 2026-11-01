"""Inference wrapper for the local PP-OCRv6_small_rec model."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import yaml


OCR_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OCR_DIR / "config.yaml"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(tempfile.gettempdir()) / "race-infer-paddlex"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "race-infer-matplotlib"))


@dataclass(frozen=True)
class OCRResult:
    """Text and confidence returned for one bib crop."""

    text: str
    confidence: float


class BibOCRInferer:
    """Recognize text in already-cropped bib images."""

    def __init__(
        self,
        config_path: Path | str = CONFIG_PATH,
    ):
        config_path = Path(config_path).resolve()
        config = yaml.safe_load(config_path.read_text())
        model_dir = Path(config["model_dir"])
        self.model_dir = (config_path.parent / model_dir).resolve()
        self.batch_size = int(config["batch_size"])
        self.min_confidence = float(config["confidence"])
        self.digits_only = bool(config["digits_only"])
        self.number_band = tuple(float(value) for value in config["number_band"])
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not 0 <= self.number_band[0] < self.number_band[1] <= 1:
            raise ValueError("number_band must be an increasing pair between 0 and 1")
        required = ("inference.json", "inference.pdiparams", "inference.yml")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete PP-OCRv6_small_rec model in {self.model_dir}; missing: {', '.join(missing)}"
            )
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required for bib OCR. Install bib_ocr/requirements.txt."
            ) from exc

        self._model = TextRecognition(
            model_name="PP-OCRv6_small_rec",
            model_dir=str(self.model_dir),
            device=str(config["device"]),
        )

    def infer(self, images: Iterable[np.ndarray]) -> list[OCRResult]:
        """Recognize a sequence of BGR bib crops, preserving input order."""
        crops = list(images)
        if not crops:
            return []
        for crop in crops:
            if not isinstance(crop, np.ndarray) or crop.size == 0:
                raise ValueError("Each bib crop must be a non-empty numpy array")

        # The recognition model expects one text line, while the detector returns
        # a whole bib. Remove the usual logo and sponsor strips above/below the
        # large race number before recognition.
        line_crops = []
        for crop in crops:
            height = crop.shape[0]
            top = int(height * self.number_band[0])
            bottom = max(top + 1, int(height * self.number_band[1]))
            line_crops.append(crop[top:bottom])

        raw_results = self._model.predict(input=line_crops, batch_size=self.batch_size)
        results = [self._convert_result(result) for result in raw_results]
        if len(results) != len(crops):
            raise RuntimeError(f"OCR returned {len(results)} results for {len(crops)} crops")
        return results

    def is_valid(self, result: OCRResult) -> bool:
        """Return whether a result is a confident, non-empty bib number."""
        return bool(result.text) and result.confidence >= self.min_confidence

    def _convert_result(self, result: Any) -> OCRResult:
        """Convert PaddleOCR/PaddleX result objects across supported 3.x releases."""
        payload = getattr(result, "json", result)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            try:
                payload = dict(payload)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Unexpected OCR result type: {type(result).__name__}") from exc
        payload = payload.get("res", payload)
        text = str(payload.get("rec_text", "")).strip()
        if self.digits_only:
            text = "".join(character for character in text if character in "0123456789")
        return OCRResult(
            text=text,
            confidence=float(payload.get("rec_score", 0.0)),
        )
