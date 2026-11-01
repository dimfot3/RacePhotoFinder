# Bib OCR

`BibOCRInferer` recognizes text in bib crops with the local
`PP-OCRv6_small_rec` PaddleOCR inference model. It focuses each detected bib on
the central number band before running the text-line recognizer. OCR settings
are in `config.yaml`; by default, output is restricted to digits and results
below `0.85` confidence are rejected.

Install its runtime dependencies if they are not already available:

```bash
python -m pip install -r bib_ocr/requirements.txt
```

Install them with the same Python environment used to run `infer.py`. Compiled
packages such as PaddlePaddle and NumPy cannot be shared between Python minor
versions.

The official model is stored in `PP-OCRv6_small_rec_infer/` and is loaded
locally; inference does not download model weights.
