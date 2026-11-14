import numpy as np

from database_core.core import (
    connect_database,
    find_participant_search_results,
    initialize_database,
    rebuild_face_suggestions,
)
from database_core.events import Event
from face_matching.matcher import encode_embedding


def unit(*values):
    vector = np.zeros(128, dtype=np.float32)
    vector[: len(values)] = values
    return encode_embedding(vector / np.linalg.norm(vector))


def add_person(connection, detection_id, picture_id, embedding, participant_id=None):
    method = "bib" if participant_id else "unmatched"
    connection.execute(
        "INSERT INTO person_detections (id, picture_id, participant_id, person_x1, "
        "person_y1, person_x2, person_y2, detection_confidence, association_method) "
        "VALUES (?, ?, ?, ?, 0, ?, 10, 0.9, ?)",
        (
            detection_id, picture_id, participant_id,
            detection_id * 20, detection_id * 20 + 10, method,
        ),
    )
    connection.execute(
        "INSERT INTO face_embeddings (person_detection_id, embedding, dimension, "
        "model_name, face_x1, face_y1, face_x2, face_y2, "
        "face_detection_confidence, face_quality) "
        "VALUES (?, ?, 128, 'test', 0, 0, 5, 5, 0.9, 0.9)",
        (detection_id, embedding),
    )
    if participant_id:
        connection.execute(
            "INSERT INTO appearances (participant_id, picture_id, "
            "person_detection_id, association_method) VALUES (?, ?, ?, 'bib')",
            (participant_id, picture_id, detection_id),
        )


def test_search_reads_precomputed_face_suggestions(tmp_path, monkeypatch):
    monkeypatch.setenv("RPF_DATA_DIR", str(tmp_path))
    event = Event("test-event")
    initialize_database(event)

    with connect_database(event) as connection:
        connection.execute("INSERT INTO participants (id, bib) VALUES (1, '100')")
        connection.executemany(
            "INSERT INTO pictures (id, full_path) VALUES (?, ?)",
            [(i, f"/photos/{i}.jpg") for i in range(1, 5)],
        )
        add_person(connection, 1, 1, unit(1, 0), participant_id=1)  # bib-confirmed
        add_person(connection, 2, 1, unit(1, 0.05))  # same picture: not suggested
        add_person(connection, 3, 2, unit(1, 0.1))  # strong match
        add_person(connection, 4, 3, unit(1, 0.6))  # weaker match
        add_person(connection, 5, 4, unit(0, 1))  # someone else
        rebuild_face_suggestions(connection)

    participant, confirmed, possible = find_participant_search_results(event, "100")
    assert participant["bib"] == "100"
    assert [row["picture_id"] for row in confirmed] == [1]
    assert [row["picture_id"] for row in possible] == [2, 3]
    assert possible[0]["similarity"] > possible[1]["similarity"]

    _, _, without_faces = find_participant_search_results(
        event, "100", include_faces=False
    )
    assert without_faces == []
