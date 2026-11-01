"""Participant template construction and cosine-similarity matching."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class ParticipantTemplate:
    participant_id: int
    embedding: np.ndarray
    reference_count: int


@dataclass(frozen=True)
class Match:
    participant_id: int
    similarity: float
    second_similarity: float | None
    margin: float


def build_templates(reference_rows, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    grouped = {}
    for row in reference_rows:
        quality = float(row["face_quality"])
        if quality < float(config["min_template_quality"]):
            continue
        embedding = decode_embedding(row["embedding"], config)
        grouped.setdefault(int(row["participant_id"]), []).append((embedding, quality))

    templates = {}
    outlier_threshold = float(config["template_outlier_similarity"])
    for participant_id, samples in grouped.items():
        embeddings = np.stack([sample[0] for sample in samples])
        qualities = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        if len(embeddings) >= 3:
            provisional = _normalize(np.average(embeddings, axis=0, weights=qualities))
            keep = embeddings @ provisional >= outlier_threshold
            if np.any(keep):
                embeddings = embeddings[keep]
                qualities = qualities[keep]
        template = _normalize(np.average(embeddings, axis=0, weights=qualities))
        templates[participant_id] = ParticipantTemplate(
            participant_id=participant_id,
            embedding=template,
            reference_count=len(embeddings),
        )
    return templates


def find_match(embedding, face_quality, templates, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    if face_quality < float(config["min_match_quality"]) or not templates:
        return None

    candidate = _normalize(np.asarray(embedding, dtype=np.float32))
    scored = sorted(
        (
            (float(candidate @ template.embedding), template)
            for template in templates.values()
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_similarity, best = scored[0]
    second_similarity = scored[1][0] if len(scored) > 1 else None
    margin = (
        best_similarity - second_similarity
        if second_similarity is not None
        else float("inf")
    )
    threshold = float(config["minimum_similarity"])
    if best.reference_count == 1:
        threshold = max(threshold, float(config["single_reference_similarity"]))
    if best_similarity < threshold or margin < float(config["minimum_margin"]):
        return None
    return Match(
        participant_id=best.participant_id,
        similarity=best_similarity,
        second_similarity=second_similarity,
        margin=margin,
    )


def encode_embedding(embedding):
    return np.ascontiguousarray(embedding, dtype="<f4").tobytes()


def decode_embedding(blob, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    embedding = np.frombuffer(blob, dtype="<f4").copy()
    expected = int(config["embedding_dimension"])
    if embedding.size != expected:
        raise ValueError(f"Stored embedding has {embedding.size} values; expected {expected}")
    return _normalize(embedding)


def _normalize(embedding):
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize an empty or invalid face embedding")
    return np.ascontiguousarray(embedding / norm, dtype=np.float32)


def load_config(config_path=DEFAULT_CONFIG):
    if isinstance(config_path, dict):
        return config_path
    return yaml.safe_load(Path(config_path).read_text())

