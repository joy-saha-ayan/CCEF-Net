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

Project setup phase.