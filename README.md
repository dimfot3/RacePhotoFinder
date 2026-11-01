# Race Photo Finder

Race Photo Finder organizes event photographs by participant and lets runners
find their photos using a bib number.

It is more than a bib detector. The first pass reads visible bibs and uses them
to build reliable participant identities. It also extracts face embeddings.
After all photos are processed, conservative face matching can recover
additional photos where a bib is hidden, turned away, blurred, or outside the
frame.

```text
photo → person detection → bib recognition + face embedding → participant match
```

Confirmed bib photos and possible face matches are shown separately. Face
suggestions do not change participant records or become training references.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Process an event folder and replace the previous event:

```bash
python app.py "/path/to/event/photos" --rerun
```

Start the search interface:

```bash
python app.py
```

Open `http://127.0.0.1:5000` and search by bib number.

Participant details can optionally be imported from CSV with
`--participants-csv`. Original photos remain unchanged; the application stores
its index in SQLite and creates smaller previews for browsing.

> The included InsightFace model weights are intended for non-commercial
> research unless separately licensed.
