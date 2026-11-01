# CCEF-Net

**Cultural Context-Aware Evidence Fusion Network for Bengali Meme Misinformation Detection**

CCEF-Net is a multimodal, evidence-grounded, and explainable framework for detecting misleading Bengali and English-Bengali code-mixed memes.

## Signals Used

- Bengali OCR text
- Meme image
- Bengali cultural context
- External evidence
- Explainability layer

## Output Labels

- Authentic
- Misleading / Out-of-Context
- Satire / Humor
- Insufficient Evidence — Human Review

## Main Components

- Bengali OCR preprocessing
- BanglaBERT / XLM-R text encoder
- ViT / CLIP image encoder
- Cultural-context-aware fusion
- Evidence retrieval and verification
- Explainability module
- FastAPI / Gradio demo

## Status

Active development: reusable OCR baseline verified; full CCEF-Net implementation and evaluation pending.

<!-- BEGIN CCEF VERIFIED DEVELOPMENT STATUS -->

## Verified development status

The full CCEF-Net architecture is under development.

Implemented and checked:
- Reusable Bengali+English Tesseract OCR module and CLI.
- Three OCR utility tests passed.
- On 20 development images, modular OCR matched notebook text and
  word counts for all 20 samples.

Not yet established:
- OCR transcription accuracy.
- Adjudicated Bengali misinformation benchmark.
- Effectiveness of cultural context or evidence fusion.
- Final CCEF-Net classification performance.

## Development documentation

- [OCR setup and usage](docs/ocr_usage.md)
- [Target architecture](docs/architecture.md)
- [Experiment status and limitations](docs/experiment_status.md)
- [Data sources and attribution](docs/data_sources.md)

Canonical source layout: `src/ccef_net/`.
Root-level baseline scripts and legacy requirements are retained for
historical continuity; they are not the current OCR setup instructions.

<!-- END CCEF VERIFIED DEVELOPMENT STATUS -->
