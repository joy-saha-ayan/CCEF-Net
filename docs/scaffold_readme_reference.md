# CCEF-Net

Cultural Context-Aware Evidence Fusion Network for Bengali
meme misinformation detection.

## Status
Capstone research in progress. Full CCEF-Net is not yet evaluated.
See docs/experiment_status.md.

## Planned outputs
- Authentic
- Misleading / Out-of-Context
- Satire / Humor
- Insufficient Evidence — Human Review

Annotation mapping and the review policy require pilot validation.

## Layout
- src/ccef_net/: reusable package
- configs/: experiment settings
- scripts/: command-line entry points
- notebooks/: reviewed exploration and demonstrations
- tests/: automated checks
- docs/: design, provenance and research decisions
- reports/: reviewed publishable summaries
- data/, models/, outputs/: local artifacts excluded from Git

## Setup
A complete training environment specification is pending.
For package development:
    pip install -e .

This command installs the package scaffold, not all ML/OCR dependencies.
See environments/ for component-specific environment notes.

## Data and licensing
See docs/data_sources.md.
Do not commit downloaded datasets, private annotations or credentials.
Project source-code licensing must be agreed before publication.
