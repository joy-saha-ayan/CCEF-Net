# OCR baseline usage

## Scope

Tesseract Bengali+English extraction with word data.
This is not a misinformation classifier or an OCR accuracy claim.

## Fresh Ubuntu/Colab runtime

Mount Google Drive first if manifests/images are stored there.
Run these commands from the code repository root:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng
python -m pip install -r environments/requirements-ocr.txt
python -m pip install --no-deps -e .
```

Colab normally runs with sufficient privileges to omit `sudo`.
Use the Python interpreter associated with the active notebook/runtime.
The observed engine version was Tesseract 5.3.4.
The apt commands do not guarantee that exact version on another OS.

Verify:

```bash
tesseract --version
tesseract --list-langs
python -c "import pytesseract; print(pytesseract.__version__)"
```

Required languages: ben and eng.

## Tests

```bash
python -m unittest discover -s tests -p 'test_ocr_utils.py' -v
```

The three utility tests cover line reconstruction, empty output and
unknown confidence. They do not validate transcription accuracy.

## Input manifest

CSV columns required:
- candidate_id: unique, non-empty ASCII letters/digits/underscore/hyphen
- local_image_path: image path accessible to the process

Absolute paths are supported. Relative image paths are resolved from
the process working directory, not from the manifest directory.

## Run

Replace the example input/output paths before execution:

```bash
python scripts/run_ocr.py \
  --manifest /path/to/candidate_manifest.csv \
  --config configs/ocr/tesseract_baseline.json \
  --output /path/to/new_ocr_run
```

The output directory must not already exist.
CLI arguments select paths; config paths fields are not used by this CLI.

## Outputs

- ocr_results.csv
- word_data/<candidate_id>.csv
- run_metadata.json

Bounding boxes refer to EXIF-oriented original-resolution image pixels.
The baseline uses ben+eng, OEM 1, PSM 11.
No resize or thresholding is applied.

## Verification

On the existing 20-candidate development set:
- 20 non-empty OCR outputs
- 408 recognized tokens
- 20/20 exact-text matches against the notebook implementation
- 20/20 word-count matches

This is a refactor-equivalence check, not a representative benchmark.
Reference transcription and CER/WER evaluation remain pending.

## Storage and privacy

Keep data, OCR outputs and private annotations outside the code repository
or in ignored directories. Do not commit credentials or tokens.
