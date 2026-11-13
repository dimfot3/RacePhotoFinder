import numpy as np

from face_matching.matcher import build_templates, encode_embedding, find_match


TEST_CONFIG = {
    "embedding_dimension": 128,
    "min_template_quality": 0.4,
    "min_match_quality": 0.4,
    "template_outlier_similarity": 0.25,
    "minimum_similarity": 0.5,
    "single_reference_similarity": 0.58,
    "minimum_margin": 0.08,
}


def unit_embedding(index):
    embedding = np.zeros(128, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


def test_matching_uses_bib_confirmed_participant_template():
    reference = unit_embedding(0)
    templates = build_templates(
        [
            {
                "participant_id": 12,
                "embedding": encode_embedding(reference),
                "face_quality": 0.9,
            }
        ],
        TEST_CONFIG,
    )

    match = find_match(reference, 0.9, templates, TEST_CONFIG)

    assert match.participant_id == 12
    assert match.similarity == 1.0


def test_ambiguous_face_is_not_associated():
    candidate = unit_embedding(0)
    almost_same = candidate.copy()
    almost_same[1] = 0.05
    almost_same /= np.linalg.norm(almost_same)
    templates = build_templates(
        [
            {
                "participant_id": 12,
                "embedding": encode_embedding(candidate),
                "face_quality": 0.9,
            },
            {
                "participant_id": 34,
                "embedding": encode_embedding(almost_same),
                "face_quality": 0.9,
            },
        ],
        TEST_CONFIG,
    )

    assert find_match(candidate, 0.9, templates, TEST_CONFIG) is None


def test_low_quality_face_is_not_associated():
    reference = unit_embedding(0)
    templates = build_templates(
        [
            {
                "participant_id": 12,
                "embedding": encode_embedding(reference),
                "face_quality": 0.9,
            }
        ],
        TEST_CONFIG,
    )

    assert find_match(reference, 0.2, templates, TEST_CONFIG) is None
