PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bib TEXT UNIQUE,
    net_finish_time TEXT,
    name TEXT,
    email TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT participants_bib_not_blank
        CHECK (bib IS NULL OR TRIM(bib) <> ''),
    CONSTRAINT participants_name_not_blank
        CHECK (name IS NULL OR TRIM(name) <> ''),
    CONSTRAINT participants_email_not_blank
        CHECK (email IS NULL OR TRIM(email) <> '')
);

CREATE TABLE IF NOT EXISTS pictures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_path TEXT NOT NULL UNIQUE,
    thumbnail_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pictures_full_path_not_blank
        CHECK (TRIM(full_path) <> '')
);

CREATE TABLE IF NOT EXISTS person_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    picture_id INTEGER NOT NULL,
    participant_id INTEGER,
    person_x1 INTEGER NOT NULL,
    person_y1 INTEGER NOT NULL,
    person_x2 INTEGER NOT NULL,
    person_y2 INTEGER NOT NULL,
    detection_confidence REAL NOT NULL,
    association_method TEXT NOT NULL DEFAULT 'unmatched',
    association_confidence REAL,
    match_margin REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT person_detections_picture_fk
        FOREIGN KEY (picture_id)
        REFERENCES pictures (id)
        ON DELETE CASCADE,
    CONSTRAINT person_detections_participant_fk
        FOREIGN KEY (participant_id)
        REFERENCES participants (id)
        ON DELETE SET NULL,
    CONSTRAINT person_detections_method_valid
        CHECK (association_method IN ('unmatched', 'bib', 'face', 'manual')),
    CONSTRAINT person_detections_box_valid
        CHECK (person_x2 > person_x1 AND person_y2 > person_y1),
    CONSTRAINT person_detections_picture_box_unique
        UNIQUE (picture_id, person_x1, person_y1, person_x2, person_y2)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_detection_id INTEGER NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    face_x1 INTEGER NOT NULL,
    face_y1 INTEGER NOT NULL,
    face_x2 INTEGER NOT NULL,
    face_y2 INTEGER NOT NULL,
    face_detection_confidence REAL NOT NULL,
    face_quality REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT face_embeddings_person_detection_fk
        FOREIGN KEY (person_detection_id)
        REFERENCES person_detections (id)
        ON DELETE CASCADE,
    CONSTRAINT face_embeddings_dimension_valid
        CHECK (dimension > 0),
    CONSTRAINT face_embeddings_box_valid
        CHECK (face_x2 > face_x1 AND face_y2 > face_y1)
);

CREATE TABLE IF NOT EXISTS appearances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id INTEGER NOT NULL,
    picture_id INTEGER NOT NULL,
    person_detection_id INTEGER,
    association_method TEXT NOT NULL DEFAULT 'bib',
    association_confidence REAL,
    bib_x1 INTEGER,
    bib_y1 INTEGER,
    bib_x2 INTEGER,
    bib_y2 INTEGER,
    bib_detection_confidence REAL,
    ocr_confidence REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT appearances_participant_fk
        FOREIGN KEY (participant_id)
        REFERENCES participants (id)
        ON DELETE CASCADE,
    CONSTRAINT appearances_picture_fk
        FOREIGN KEY (picture_id)
        REFERENCES pictures (id)
        ON DELETE CASCADE,
    CONSTRAINT appearances_person_detection_fk
        FOREIGN KEY (person_detection_id)
        REFERENCES person_detections (id)
        ON DELETE SET NULL,
    CONSTRAINT appearances_method_valid
        CHECK (association_method IN ('bib', 'face', 'manual')),
    CONSTRAINT appearances_participant_picture_unique
        UNIQUE (participant_id, picture_id)
);

CREATE INDEX IF NOT EXISTS appearances_picture_id_idx
    ON appearances (picture_id);

CREATE INDEX IF NOT EXISTS person_detections_picture_id_idx
    ON person_detections (picture_id);

CREATE INDEX IF NOT EXISTS person_detections_participant_id_idx
    ON person_detections (participant_id);

CREATE INDEX IF NOT EXISTS face_embeddings_quality_idx
    ON face_embeddings (face_quality);

-- Face suggestions precomputed after each import, so a search is a simple
-- lookup instead of comparing against every face in the event.
CREATE TABLE IF NOT EXISTS face_suggestions (
    participant_id INTEGER NOT NULL,
    picture_id INTEGER NOT NULL,
    similarity REAL NOT NULL,

    PRIMARY KEY (participant_id, picture_id),
    CONSTRAINT face_suggestions_participant_fk
        FOREIGN KEY (participant_id)
        REFERENCES participants (id)
        ON DELETE CASCADE,
    CONSTRAINT face_suggestions_picture_fk
        FOREIGN KEY (picture_id)
        REFERENCES pictures (id)
        ON DELETE CASCADE
);
