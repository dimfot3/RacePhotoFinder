"""Database access and race-photo import services."""

import csv
import hashlib
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
THUMBNAIL_DIR = Path(__file__).resolve().parent / "thumbnails"
THUMBNAIL_MAX_SIZE = 640


def database_path():
    configured_path = os.environ.get("DATABASE_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parent / "race_photo_finder.db"


def connect_database():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with connect_database() as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        picture_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(pictures)").fetchall()
        }
        if "thumbnail_path" not in picture_columns:
            connection.execute("ALTER TABLE pictures ADD COLUMN thumbnail_path TEXT")
        appearance_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(appearances)").fetchall()
        }
        if "person_detection_id" not in appearance_columns:
            connection.execute(
                "ALTER TABLE appearances ADD COLUMN person_detection_id INTEGER "
                "REFERENCES person_detections(id) ON DELETE SET NULL"
            )
        if "association_method" not in appearance_columns:
            connection.execute(
                "ALTER TABLE appearances ADD COLUMN association_method TEXT "
                "NOT NULL DEFAULT 'bib'"
            )
        if "association_confidence" not in appearance_columns:
            connection.execute(
                "ALTER TABLE appearances ADD COLUMN association_confidence REAL"
            )


def find_participant_pictures(bib):
    """Return participant details and associated pictures for an exact bib."""
    query = """
        SELECT
            participant.id AS participant_id,
            participant.bib,
            participant.name,
            participant.email,
            participant.net_finish_time,
            picture.id AS picture_id,
            picture.full_path,
            picture.thumbnail_path
        FROM participants AS participant
        LEFT JOIN appearances AS appearance
          ON appearance.participant_id = participant.id
        LEFT JOIN pictures AS picture
          ON picture.id = appearance.picture_id
        WHERE participant.bib = ?
        ORDER BY picture.id
    """
    with connect_database() as connection:
        return connection.execute(query, (bib,)).fetchall()


def find_participant_search_results(bib):
    """Return confirmed bib photos and non-mutating face suggestions."""
    from face_matching.matcher import (
        build_templates,
        decode_embedding,
        load_config,
    )

    with connect_database() as connection:
        participant = connection.execute(
            """
            SELECT id, bib, name, email, net_finish_time
            FROM participants
            WHERE bib = ?
            """,
            (bib,),
        ).fetchone()
        if participant is None:
            return None, [], []

        confirmed_rows = connection.execute(
            """
            SELECT
                picture.id AS picture_id,
                picture.full_path,
                picture.thumbnail_path,
                appearance.association_method,
                appearance.association_confidence
            FROM appearances AS appearance
            JOIN pictures AS picture ON picture.id = appearance.picture_id
            WHERE appearance.participant_id = ?
              AND appearance.association_method IN ('bib', 'manual')
            ORDER BY picture.id
            """,
            (participant["id"],),
        ).fetchall()
        confirmed = [dict(row) for row in confirmed_rows]
        confirmed_ids = {row["picture_id"] for row in confirmed}

        reference_rows = connection.execute(
            """
            SELECT person.participant_id, face.embedding, face.face_quality
            FROM person_detections AS person
            JOIN face_embeddings AS face ON face.person_detection_id = person.id
            WHERE person.participant_id = ?
              AND person.association_method = 'bib'
            """,
            (participant["id"],),
        ).fetchall()
        config = load_config()
        template = build_templates(reference_rows, config).get(participant["id"])
        if template is None:
            return dict(participant), confirmed, []

        candidate_rows = connection.execute(
            """
            SELECT
                picture.id AS picture_id,
                picture.full_path,
                picture.thumbnail_path,
                face.embedding,
                face.face_quality
            FROM face_embeddings AS face
            JOIN person_detections AS person
              ON person.id = face.person_detection_id
            JOIN pictures AS picture ON picture.id = person.picture_id
            WHERE face.face_quality >= ?
            """,
            (float(config["min_match_quality"]),),
        ).fetchall()

    threshold = float(config["search_similarity"])
    possible_by_picture = {}
    for row in candidate_rows:
        if row["picture_id"] in confirmed_ids:
            continue
        embedding = decode_embedding(row["embedding"], config)
        similarity = float(embedding @ template.embedding)
        if similarity < threshold:
            continue
        current = possible_by_picture.get(row["picture_id"])
        if current is None or similarity > current["similarity"]:
            possible_by_picture[row["picture_id"]] = {
                "picture_id": row["picture_id"],
                "full_path": row["full_path"],
                "thumbnail_path": row["thumbnail_path"],
                "similarity": similarity,
            }

    possible = sorted(
        possible_by_picture.values(),
        key=lambda row: (-row["similarity"], row["picture_id"]),
    )
    return dict(participant), confirmed, possible


def find_picture(picture_id):
    with connect_database() as connection:
        row = connection.execute(
            "SELECT id, full_path, thumbnail_path FROM pictures WHERE id = ?",
            (picture_id,),
        ).fetchone()
    return row


def local_path(full_path):
    path = Path(full_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def create_thumbnail(image_path, image=None):
    """Create or refresh a small JPEG preview without changing the original."""
    import cv2

    image_path = Path(image_path).resolve()
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(image_path).encode()).hexdigest()[:24]
    thumbnail_path = THUMBNAIL_DIR / f"{digest}.jpg"
    if (
        thumbnail_path.is_file()
        and thumbnail_path.stat().st_mtime >= image_path.stat().st_mtime
    ):
        return thumbnail_path.resolve()

    if image is None:
        image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot create thumbnail for unreadable image: {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, THUMBNAIL_MAX_SIZE / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(str(thumbnail_path), image, [cv2.IMWRITE_JPEG_QUALITY, 82]):
        raise OSError(f"Could not write thumbnail: {thumbnail_path}")
    return thumbnail_path.resolve()


def read_participant_csv(csv_path):
    """Read optional participant details, accepting common header variants."""
    participants = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("Participant CSV must have a header row")
        for line_number, row in enumerate(reader, start=2):
            normalized = {
                key.strip().lower().replace(" ", "_"): value.strip() if value else None
                for key, value in row.items()
                if key
            }
            bib = normalized.get("bib") or normalized.get("bib_number")
            if not bib:
                raise ValueError(f"Participant CSV row {line_number} has no bib")
            participants.append(
                {
                    "bib": bib,
                    "name": normalized.get("name") or normalized.get("full_name"),
                    "email": normalized.get("email"),
                    "net_finish_time": (
                        normalized.get("net_finish_time")
                        or normalized.get("finish_time")
                        or normalized.get("time")
                    ),
                }
            )
    return participants


def import_participant_csv(connection, csv_path):
    participants = read_participant_csv(csv_path)
    query = """
        INSERT INTO participants (bib, name, email, net_finish_time)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (bib) DO UPDATE SET
            name = COALESCE(EXCLUDED.name, participants.name),
            email = COALESCE(EXCLUDED.email, participants.email),
            net_finish_time = COALESCE(
                EXCLUDED.net_finish_time,
                participants.net_finish_time
            )
    """
    with connection:
        for participant in participants:
            connection.execute(
                query,
                (
                    participant["bib"],
                    participant["name"],
                    participant["email"],
                    participant["net_finish_time"],
                ),
            )
    print(f"Imported {len(participants)} participant row(s) from {csv_path}")


def save_picture_results(connection, image_path, thumbnail_path, inference):
    """Persist every person, face embedding, and bib-confirmed association."""
    from face_matching.matcher import encode_embedding, load_config

    face_config = load_config()

    with connection:
        connection.execute(
            """
            INSERT INTO pictures (full_path, thumbnail_path)
            VALUES (?, ?)
            ON CONFLICT (full_path) DO UPDATE SET
                thumbnail_path = EXCLUDED.thumbnail_path
            """,
            (str(image_path.resolve()), str(thumbnail_path.resolve())),
        )
        picture_id = connection.execute(
            "SELECT id FROM pictures WHERE full_path = ?",
            (str(image_path.resolve()),),
        ).fetchone()[0]

        connection.execute("DELETE FROM appearances WHERE picture_id = ?", (picture_id,))
        connection.execute(
            "DELETE FROM person_detections WHERE picture_id = ?", (picture_id,)
        )

        bib_count = 0
        face_count = 0
        for person in inference.people:
            selected_bib = max(
                person.bibs,
                key=lambda item: float(item[0][4]) * float(item[1].confidence),
                default=None,
            )
            participant_id = None
            association_method = "unmatched"
            association_confidence = None
            if selected_bib:
                _, ocr_result = selected_bib
                connection.execute(
                    "INSERT INTO participants (bib) VALUES (?) "
                    "ON CONFLICT (bib) DO NOTHING",
                    (ocr_result.text,),
                )
                participant_id = connection.execute(
                    "SELECT id FROM participants WHERE bib = ?",
                    (ocr_result.text,),
                ).fetchone()[0]
                association_method = "bib"
                association_confidence = float(ocr_result.confidence)
                bib_count += 1

            x1, y1, x2, y2, person_confidence = person.box
            cursor = connection.execute(
                """
                INSERT INTO person_detections (
                    picture_id, participant_id,
                    person_x1, person_y1, person_x2, person_y2,
                    detection_confidence, association_method,
                    association_confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    picture_id,
                    participant_id,
                    round(x1), round(y1), round(x2), round(y2),
                    float(person_confidence),
                    association_method,
                    association_confidence,
                ),
            )
            person_detection_id = cursor.lastrowid

            if person.face is not None:
                fx1, fy1, fx2, fy2 = person.face.box
                connection.execute(
                    """
                    INSERT INTO face_embeddings (
                        person_detection_id, embedding, dimension, model_name,
                        face_x1, face_y1, face_x2, face_y2,
                        face_detection_confidence, face_quality
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person_detection_id,
                        sqlite3.Binary(encode_embedding(person.face.embedding)),
                        int(face_config["embedding_dimension"]),
                        str(face_config["model_name"]),
                        round(fx1), round(fy1), round(fx2), round(fy2),
                        float(person.face.confidence),
                        float(person.face.quality),
                    ),
                )
                face_count += 1

            if selected_bib:
                box, ocr_result = selected_bib
                bx1, by1, bx2, by2, bib_confidence = box
                connection.execute(
                    """
                    INSERT INTO appearances (
                        participant_id, picture_id, person_detection_id,
                        association_method, association_confidence,
                        bib_x1, bib_y1, bib_x2, bib_y2,
                        bib_detection_confidence, ocr_confidence
                    )
                    VALUES (?, ?, ?, 'bib', ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (participant_id, picture_id) DO UPDATE SET
                        person_detection_id = EXCLUDED.person_detection_id,
                        association_method = 'bib',
                        association_confidence = EXCLUDED.association_confidence,
                        bib_x1 = EXCLUDED.bib_x1,
                        bib_y1 = EXCLUDED.bib_y1,
                        bib_x2 = EXCLUDED.bib_x2,
                        bib_y2 = EXCLUDED.bib_y2,
                        bib_detection_confidence = EXCLUDED.bib_detection_confidence,
                        ocr_confidence = EXCLUDED.ocr_confidence
                    """,
                    (
                        participant_id,
                        picture_id,
                        person_detection_id,
                        float(ocr_result.confidence),
                        round(bx1), round(by1), round(bx2), round(by2),
                        float(bib_confidence),
                        float(ocr_result.confidence),
                    ),
                )

    return len(inference.people), face_count, bib_count


def run_face_matching(connection):
    """Associate reliable unmatched faces using bib-confirmed templates only."""
    from face_matching.matcher import (
        build_templates,
        decode_embedding,
        find_match,
        load_config,
    )

    matching_config = load_config()

    with connection:
        connection.execute("DELETE FROM appearances WHERE association_method = 'face'")
        connection.execute(
            """
            UPDATE person_detections
            SET participant_id = NULL,
                association_method = 'unmatched',
                association_confidence = NULL,
                match_margin = NULL
            WHERE association_method = 'face'
            """
        )

    reference_rows = connection.execute(
        """
        SELECT person.participant_id, face.embedding, face.face_quality
        FROM person_detections AS person
        JOIN face_embeddings AS face ON face.person_detection_id = person.id
        WHERE person.association_method = 'bib'
          AND person.participant_id IS NOT NULL
        """
    ).fetchall()
    templates = build_templates(reference_rows, matching_config)
    if not templates:
        print("Face matching skipped: no suitable bib-confirmed face templates.")
        return 0

    unmatched_rows = connection.execute(
        """
        SELECT person.id AS person_detection_id, person.picture_id,
               face.embedding, face.face_quality
        FROM person_detections AS person
        JOIN face_embeddings AS face ON face.person_detection_id = person.id
        WHERE person.association_method = 'unmatched'
          AND person.participant_id IS NULL
        ORDER BY person.picture_id, person.id
        """
    ).fetchall()
    participants_by_picture = {
        (row["picture_id"], row["participant_id"])
        for row in connection.execute(
            "SELECT picture_id, participant_id FROM appearances"
        ).fetchall()
    }

    matched = 0
    with connection:
        for row in unmatched_rows:
            match = find_match(
                decode_embedding(row["embedding"], matching_config),
                float(row["face_quality"]),
                templates,
                matching_config,
            )
            if match is None:
                continue
            picture_participant = (row["picture_id"], match.participant_id)
            if picture_participant in participants_by_picture:
                continue
            connection.execute(
                """
                UPDATE person_detections
                SET participant_id = ?, association_method = 'face',
                    association_confidence = ?, match_margin = ?
                WHERE id = ?
                """,
                (
                    match.participant_id,
                    match.similarity,
                    match.margin,
                    row["person_detection_id"],
                ),
            )
            connection.execute(
                """
                INSERT INTO appearances (
                    participant_id, picture_id, person_detection_id,
                    association_method, association_confidence
                )
                VALUES (?, ?, ?, 'face', ?)
                ON CONFLICT (participant_id, picture_id) DO NOTHING
                """,
                (
                    match.participant_id,
                    row["picture_id"],
                    row["person_detection_id"],
                    match.similarity,
                ),
            )
            participants_by_picture.add(picture_participant)
            matched += 1

    print(
        f"Face matching complete: {len(templates)} participant template(s), "
        f"{matched}/{len(unmatched_rows)} unmatched face(s) associated."
    )
    return matched


def reset_database():
    """Delete the current event records and their generated thumbnails."""
    initialize_database()
    thumbnail_root = THUMBNAIL_DIR.resolve()
    with connect_database() as connection:
        thumbnail_rows = connection.execute(
            "SELECT thumbnail_path FROM pictures WHERE thumbnail_path IS NOT NULL"
        ).fetchall()
        connection.execute("DELETE FROM appearances")
        connection.execute("DELETE FROM pictures")
        connection.execute("DELETE FROM participants")
        connection.execute(
            "DELETE FROM sqlite_sequence WHERE name IN "
            "('appearances', 'face_embeddings', 'person_detections', "
            "'pictures', 'participants')"
        )

    removed = 0
    for row in thumbnail_rows:
        thumbnail_path = Path(row["thumbnail_path"]).resolve()
        if thumbnail_path.is_relative_to(thumbnail_root) and thumbnail_path.is_file():
            thumbnail_path.unlink()
            removed += 1
    print(f"Previous event cleared: {removed} thumbnail(s) removed.")


def import_photo_folder(folder, csv_path=None, rerun=False):
    """Run the existing detection pipeline and persist results recursively."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Photo folder does not exist: {folder}")
    image_paths = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {folder}")

    # Import lazily so serving the search UI does not load the ML models.
    import cv2
    from database_core.batch_inference import BatchedInferencePipeline

    pipeline = BatchedInferencePipeline()
    if rerun:
        reset_database()
    processed = 0
    unreadable = 0
    accepted_bibs = 0
    detected_people = 0
    embedded_faces = 0
    face_matches = 0

    initialize_database()
    with connect_database() as connection:
        if csv_path:
            import_participant_csv(connection, csv_path)

        cursor = 0
        while cursor < len(image_paths):
            chunk_size = min(pipeline.batch_size, len(image_paths) - cursor)
            chunk_paths = image_paths[cursor:cursor + chunk_size]
            valid_paths = []
            valid_numbers = []
            images = []
            for offset, image_path in enumerate(chunk_paths):
                image = cv2.imread(str(image_path))
                if image is None:
                    unreadable += 1
                    image_number = cursor + offset + 1
                    print(
                        f"[{image_number}/{len(image_paths)}] "
                        f"Skipped unreadable image: {image_path}"
                    )
                    continue
                valid_paths.append(image_path)
                valid_numbers.append(cursor + offset + 1)
                images.append(image)

            try:
                batch_results = pipeline.predict(images)
            except RuntimeError as error:
                if pipeline.is_cuda_oom(error) and pipeline.recover_from_oom():
                    del images
                    continue
                raise

            for image_number, image_path, image, inference in zip(
                valid_numbers, valid_paths, images, batch_results
            ):
                thumbnail_path = create_thumbnail(image_path, image)
                people_count, face_count, bib_count = save_picture_results(
                    connection,
                    image_path,
                    thumbnail_path,
                    inference,
                )
                processed += 1
                detected_people += people_count
                embedded_faces += face_count
                accepted_bibs += bib_count
                bibs = [
                    max(
                        person.bibs,
                        key=lambda item: float(item[1].confidence),
                    )[1].text
                    for person in inference.people
                    if person.bibs
                ]
                print(
                    f"[{image_number}/{len(image_paths)}] {image_path.name}: "
                    f"people={people_count}, faces={face_count}, "
                    f"bibs={', '.join(bibs) or 'none'}"
                )
            cursor += chunk_size

        face_matches = run_face_matching(connection)

    print(
        f"Import complete: {processed} picture(s), {detected_people} person "
        f"detection(s), {embedded_faces} face embedding(s), {accepted_bibs} "
        f"bib appearance(s), {face_matches} face-matched appearance(s), "
        f"{unreadable} unreadable file(s)."
    )


def import_csv_only(csv_path):
    initialize_database()
    with connect_database() as connection:
        import_participant_csv(connection, csv_path)


def rebuild_thumbnails():
    """Create missing previews for pictures already stored in SQLite."""
    initialize_database()
    with connect_database() as connection:
        pictures = connection.execute(
            "SELECT id, full_path, thumbnail_path FROM pictures ORDER BY id"
        ).fetchall()

        created = 0
        skipped = 0
        for index, picture_row in enumerate(pictures, start=1):
            parsed = urlparse(picture_row["full_path"])
            if parsed.scheme:
                skipped += 1
                continue
            original_path = local_path(picture_row["full_path"])
            if not original_path.is_file():
                skipped += 1
                continue
            try:
                thumbnail_path = create_thumbnail(original_path)
            except (OSError, ValueError):
                skipped += 1
                continue
            connection.execute(
                "UPDATE pictures SET thumbnail_path = ? WHERE id = ?",
                (str(thumbnail_path), picture_row["id"]),
            )
            created += 1
            if index % 25 == 0 or index == len(pictures):
                connection.commit()
                print(f"Thumbnails: {index}/{len(pictures)}")
        connection.commit()
    print(f"Thumbnail rebuild complete: {created} ready, {skipped} skipped.")


