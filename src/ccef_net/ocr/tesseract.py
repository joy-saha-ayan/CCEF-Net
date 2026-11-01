"""Tesseract OCR baseline. Confidence is not measured accuracy."""

from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
import pytesseract


def summarize_words(data):
    """Reconstruct text in Tesseract's emitted line traversal order."""
    words = data.copy()
    words["text"] = words["text"].fillna("").astype(str).str.strip()
    words["conf"] = pd.to_numeric(words["conf"], errors="coerce")

    words = words[
        words["level"].eq(5) & words["text"].ne("")
    ].copy()

    lines = [
        " ".join(line["text"].tolist())
        for _, line in words.groupby(
            ["page_num", "block_num", "par_num", "line_num"],
            sort=False,
        )
    ]

    valid_conf = words.loc[words["conf"].ge(0), "conf"]

    return {
        "status": "ok" if len(words) else "no_text",
        "ocr_text": "\n".join(lines),
        "recognized_words": len(words),
        "mean_word_confidence": (
            float(valid_conf.mean()) if len(valid_conf) else None
        ),
    }


def check_environment(language):
    available = set(pytesseract.get_languages(config=""))
    required = set(language.split("+"))
    missing = required - available
    if missing:
        raise RuntimeError(f"Missing OCR languages: {sorted(missing)}")
    return {
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "available_languages": sorted(available),
    }


def recognize_image(image_path, config):
    """Return summary and full word data in oriented-image coordinates."""
    preprocessing = config["preprocessing"]
    if preprocessing != {
        "exif_orientation": True,
        "color_mode": "RGB",
        "resize": False,
    }:
        raise ValueError("Unsupported preprocessing configuration.")

    with Image.open(Path(image_path)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")

    flags = f"--oem {int(config['oem'])} --psm {int(config['psm'])}"

    data = pytesseract.image_to_data(
        rgb,
        lang=config["language"],
        config=flags,
        output_type=pytesseract.Output.DATAFRAME,
        timeout=float(config["timeout_seconds"]),
    )

    result = summarize_words(data)
    result["oriented_width"], result["oriented_height"] = rgb.size
    return result, data
