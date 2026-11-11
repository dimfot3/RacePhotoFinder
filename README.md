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

<p align="center">
  <img src="docs/pipeline_example.jpg" alt="Runner annotated with person, face, and bib 1389 detections" width="220">
  <br>
  <sub>Real pipeline output: <b>person</b> detection (blue), <b>face</b> detection (red), and <b>bib</b> detection with OCR reading "1389" (green).</sub>
</p>

The importer builds each participant's face template only from photos where the
bib was recognized. It then associates sufficiently confident unmatched faces
with that participant. Search combines bib-confirmed and face-matched photos in
one gallery. Face-derived matches never feed back into the templates, which
prevents one uncertain match from influencing later matches.

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

## Model setup

Model binaries that may be redistributed are stored with Git LFS. After
cloning, install Git LFS and retrieve them:

```bash
git lfs install
git lfs pull
```

The InsightFace `buffalo_l` weights are not redistributed. Review the
[official InsightFace model terms](https://github.com/deepinsight/insightface#license),
obtain the official package, and place these files at:

```text
face_detection/weights/det_10g.onnx
face_matching/weights/w600k_r50.onnx
```

Public InsightFace pretrained models are limited to non-commercial research
unless separately licensed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for model provenance, checksums, and license details.

## License

Original Race Photo Finder source code is licensed under the
[GNU Affero General Public License v3.0](LICENSE). Third-party software and
model artifacts retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
