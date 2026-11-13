# Database core

The application uses SQLite and stores all records in one local file:

```text
database_core/race_photo_finder.db
```

The database is created automatically from `schema.sql`. No database server,
Docker container, or separate database client is required.

The schema contains:

- `participants`: one row per person. `bib`, `net_finish_time`, `name`, and
  `email` are optional.
- `pictures`: one row per local picture, identified by `full_path`.
- `appearances`: connects participants and pictures and stores bib detection
  bounds, the association method (`bib`, `face`, or `manual`), and confidence.
- `person_detections`: one auditable row for every person found in a picture;
  anonymous people remain here without creating fake participants.
- `face_embeddings`: one normalized SFace embedding for each suitable face,
  connected to its person detection.

Imports also create cached JPEG thumbnails under `database_core/thumbnails/`.
The search grid loads only these small previews. Selecting a photo opens the
full-size image, while its Download control downloads the untouched original.

## Install

```bash
python -m pip install -r requirements.txt
```

## Import a photo folder

Pass a folder to `app.py` to recursively run person detection, bib detection,
and bib OCR. Accepted bibs are created automatically and connected to their
pictures:

```bash
python app.py "datasets/tsr_test/Fox Trail 2026"
```

Original photos are grouped for person detection using adaptive CUDA batches.
The importer starts at 4 photos, measures actual peak GPU memory, grows up to
16 only while staying within 60% of the free CUDA memory measured after both
models are loaded, and automatically halves/retries a batch after CUDA OOM.
Bib detection and OCR keep their existing bounded crop batches. These settings
are configurable in `database_core/config.yaml`.

During the first pass, YuNet detects a face inside every person crop and
SFace creates a five-point-aligned, normalized 128-value embedding. After all
photos finish, the importer builds participant templates only from
bib-confirmed faces and runs conservative cosine matching over suitable
anonymous faces. Face-derived matches never feed back into the templates.

The import is safe to rerun: existing paths and bibs are reused, and existing
appearance metadata is updated.

To replace the previous event completely, use `--rerun`. The new folder is
validated first; then old participants, pictures, appearances, and generated
thumbnails are removed before inference starts:

```bash
python app.py "datasets/tsr_test/Fox Trail 2026" --rerun
```

Without `--rerun`, a folder import keeps and updates existing records.

## Optional participant CSV

A CSV may contain `bib`, `name`, `email`, and either `net_finish_time` or
`finish_time`. Only `bib` is required. Times can use values such as `01:32:45`.

```csv
bib,name,email,finish_time
1389,Alex Runner,alex@example.com,01:32:45
2041,,,02:04:18
```

See `participants.example.csv` for a ready-to-copy template.

Import the roster together with the photos:

```bash
python app.py "datasets/tsr_test/Fox Trail 2026" \
  --participants-csv database_core/participants.example.csv
```

Import only the CSV:

```bash
python app.py \
  --participants-csv database_core/participants.example.csv
```

## Start the bib-search UI

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

Add `--serve` to start the search UI immediately after a photo or CSV import.

For pictures imported before thumbnail support was added, create previews
without rerunning detection or OCR:

```bash
python app.py --rebuild-thumbnails
```

To place the SQLite file somewhere else, set `DATABASE_PATH`:

```bash
export DATABASE_PATH='/data/race/race_photo_finder.db'
python app.py
```
