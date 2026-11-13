"""Adaptive batched person detection for the folder importer."""

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from bib_ocr import BibOCRInferer
from face_detection import YuNetFaceDetector
from face_matching.embedder import SFaceEmbedder
from infer import load_detector
from person_detection.detector import PersonDetector


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class FaceInference:
    box: tuple[float, float, float, float]
    confidence: float
    quality: float
    embedding: np.ndarray


@dataclass
class PersonInference:
    box: tuple[float, float, float, float, float]
    bibs: list = field(default_factory=list)
    face: FaceInference | None = None


@dataclass
class ImageInference:
    people: list[PersonInference] = field(default_factory=list)


class AdaptivePersonBatcher:
    """Keep person batches within a measured fraction of free CUDA memory."""

    def __init__(self, config, enabled):
        fraction = float(config["gpu_memory_fraction"])
        initial = int(config["initial_person_batch"])
        maximum = int(config["max_person_batch"])
        if not 0 < fraction <= 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        if not 1 <= initial <= maximum:
            raise ValueError("Person batch sizes must satisfy 1 <= initial <= max")

        self.enabled = enabled
        self.current = initial if enabled else 1
        self.maximum = maximum
        self.memory_fraction = fraction
        self.bytes_per_image = None
        self.budget_bytes = 0
        self._baseline_bytes = 0
        if enabled:
            free_bytes, _ = torch.cuda.mem_get_info()
            self.budget_bytes = int(free_bytes * fraction)
            print(
                "Adaptive person batching: "
                f"initial={self.current}, max={self.maximum}, "
                f"budget={self.budget_bytes / 2**30:.2f} GiB "
                f"({fraction:.0%} of free CUDA memory)"
            )
        else:
            print("Person detection batching: CUDA unavailable, using batch 1")

    def begin_measurement(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self._baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def finish_measurement(self, image_count):
        if not self.enabled or image_count < 1:
            return
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        additional_bytes = max(1, peak_bytes - self._baseline_bytes)
        observed_per_image = additional_bytes / image_count
        if self.bytes_per_image is None:
            self.bytes_per_image = observed_per_image
        else:
            # Retain the highest observation so crowded batches cannot make the
            # controller increasingly aggressive.
            self.bytes_per_image = max(self.bytes_per_image, observed_per_image)

        safe_batch = max(
            1,
            min(self.maximum, int(self.budget_bytes / self.bytes_per_image)),
        )
        previous = self.current
        if safe_batch > self.current:
            self.current = min(safe_batch, self.current * 2)
        elif safe_batch < self.current:
            self.current = safe_batch
        if self.current != previous:
            print(f"Person batch adjusted: {previous} -> {self.current}")

    def recover_from_oom(self):
        if not self.enabled or self.current <= 1:
            return False
        previous = self.current
        self.current = max(1, self.current // 2)
        self.bytes_per_image = None
        torch.cuda.empty_cache()
        print(f"CUDA memory limit reached; retrying person batch {previous} as {self.current}")
        return True


class BatchedInferencePipeline:
    """Batch originals for person detection; bound bib and OCR crop batches."""

    def __init__(self, config_path=CONFIG_PATH):
        config = yaml.safe_load(Path(config_path).read_text())["inference"]
        self.person_detector = PersonDetector()
        self.bib_model, self.bib_config = load_detector("bib_detection")
        self.ocr_inferer = BibOCRInferer()
        self.face_detector = YuNetFaceDetector()
        self.face_embedder = SFaceEmbedder()
        self.bib_fallback_confidence = float(
            self.bib_config["fallback_confidence"]
        )
        self.bib_fallback_ocr_confidence = float(
            self.bib_config["fallback_ocr_confidence"]
        )
        self.bib_max_candidates = int(
            self.bib_config["max_candidates_per_person"]
        )
        if not 0 <= self.bib_fallback_confidence < self.bib_config["confidence"]:
            raise ValueError("Bib fallback confidence must be below normal confidence")
        if not 0 <= self.bib_fallback_ocr_confidence <= 1:
            raise ValueError("Bib fallback OCR confidence must be between 0 and 1")
        if self.bib_max_candidates < 1:
            raise ValueError("Bib max candidates per person must be positive")
        self.cuda_enabled = (
            torch.cuda.is_available()
            and self.person_detector.device != "cpu"
            and self.bib_config.get("device") != "cpu"
        )
        if self.cuda_enabled:
            self._warm_cuda_models()
        self.batcher = AdaptivePersonBatcher(config, self.cuda_enabled)

    @property
    def batch_size(self):
        return self.batcher.current

    @staticmethod
    def is_cuda_oom(error):
        return isinstance(error, torch.cuda.OutOfMemoryError) or (
            isinstance(error, RuntimeError)
            and "out of memory" in str(error).lower()
        )

    def recover_from_oom(self):
        return self.batcher.recover_from_oom()

    def _warm_cuda_models(self):
        """Move both models to CUDA before calculating the free-memory budget."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.person_detector.predict([dummy])
        self.bib_model.predict(
            dummy,
            imgsz=self.bib_config["imgsz"],
            conf=self.bib_config["confidence"],
            device=self.bib_config["device"],
            max_det=1,
            verbose=False,
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def predict(self, images):
        """Detect people, bibs, faces, and embeddings for every image."""
        if not images:
            return []
        self.batcher.begin_measurement()
        people_results = self.person_detector.predict(images)
        self.batcher.finish_measurement(len(images))

        inference_by_image = [ImageInference() for _ in images]
        person_crops = []
        person_locations = []
        for image_index, (image, result) in enumerate(zip(images, people_results)):
            height, width = image.shape[:2]
            for x1, y1, x2, y2, confidence in result.tolist():
                x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
                x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
                if x2 > x1 and y2 > y1:
                    person_index = len(inference_by_image[image_index].people)
                    inference_by_image[image_index].people.append(
                        PersonInference(
                            box=(x1, y1, x2, y2, float(confidence)),
                        )
                    )
                    person_crops.append(image[y1:y2, x1:x2].copy())
                    person_locations.append((image_index, person_index, x1, y1))

        if person_crops:
            self._recognize_one_bib_per_person(
                person_crops,
                person_locations,
                inference_by_image,
                confidence=float(self.bib_config["confidence"]),
                min_ocr_confidence=self.ocr_inferer.min_confidence,
            )
            fallback_indexes = [
                index
                for index, (image_index, person_index, _, _) in enumerate(
                    person_locations
                )
                if not inference_by_image[image_index].people[person_index].bibs
            ]
            if fallback_indexes:
                self._recognize_one_bib_per_person(
                    [person_crops[index] for index in fallback_indexes],
                    [person_locations[index] for index in fallback_indexes],
                    inference_by_image,
                    confidence=self.bib_fallback_confidence,
                    min_ocr_confidence=self.bib_fallback_ocr_confidence,
                )

        for crop, (image_index, person_index, offset_x, offset_y) in zip(
            person_crops, person_locations
        ):
            faces = self.face_detector.detect(crop)
            if not faces:
                continue
            detected_face = faces[0]
            embedded_face = self.face_embedder.embed(crop, detected_face)
            if embedded_face is None:
                continue
            x1, y1, x2, y2 = detected_face.bbox
            inference_by_image[image_index].people[person_index].face = FaceInference(
                box=(
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ),
                confidence=detected_face.confidence,
                quality=embedded_face.quality,
                embedding=embedded_face.embedding,
            )

        return inference_by_image

    def _recognize_one_bib_per_person(
        self,
        person_crops,
        person_locations,
        inference_by_image,
        confidence,
        min_ocr_confidence,
    ):
        """OCR a bounded candidate set and retain at most one bib per person."""
        predictions = self.bib_model.predict(
            person_crops,
            imgsz=self.bib_config["imgsz"],
            conf=confidence,
            device=self.bib_config["device"],
            batch=self.bib_config["batch"],
            max_det=self.bib_max_candidates,
            rect=False,
            stream=True,
            verbose=False,
        )

        bib_crops = []
        candidate_locations = []
        for result, crop, location in zip(
            predictions, person_crops, person_locations
        ):
            image_index, person_index, offset_x, offset_y = location
            height, width = crop.shape[:2]
            detections = result.boxes.data[:, :5].cpu().tolist()
            for local_box in detections[: self.bib_max_candidates]:
                x1, y1, x2, y2, detection_confidence = local_box
                left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
                right = min(width, math.ceil(x2))
                bottom = min(height, math.ceil(y2))
                if right <= left or bottom <= top:
                    continue
                global_box = (
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                    detection_confidence,
                )
                bib_crops.append(crop[top:bottom, left:right].copy())
                candidate_locations.append(
                    (image_index, person_index, global_box)
                )

        if not bib_crops:
            return

        accepted = {}
        for location, ocr_result in zip(
            candidate_locations, self.ocr_inferer.infer(bib_crops)
        ):
            image_index, person_index, box = location
            if not ocr_result.text or ocr_result.confidence < min_ocr_confidence:
                continue
            score = float(box[4]) * float(ocr_result.confidence)
            key = (image_index, person_index)
            current = accepted.get(key)
            if current is None or score > current[0]:
                accepted[key] = (score, box, ocr_result)

        for (image_index, person_index), (_, box, ocr_result) in accepted.items():
            inference_by_image[image_index].people[person_index].bibs = [
                (box, ocr_result)
            ]
