"""Run the OCR baseline on a candidate manifest."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import pandas as pd
from ccef_net.ocr.tesseract import check_environment, recognize_image


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.manifest, keep_default_na=False)

    required = {"candidate_id", "local_image_path"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest requires columns: {sorted(required)}")
    if frame.empty:
        raise ValueError("Manifest is empty.")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs.")

    for value in frame["candidate_id"]:
        if not value or any(c not in
                            "abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                            for c in value):
            raise ValueError(f"Unsafe candidate ID: {value}")

    environment = check_environment(config["language"])

    # Refuse to reuse an output folder.
    args.output.mkdir(parents=True, exist_ok=False)
    word_dir = args.output / "word_data"
    word_dir.mkdir()

    records = []

    for row in frame.to_dict("records"):
        record = {
            "candidate_id": row["candidate_id"],
            "image_path": row["local_image_path"],
            "status": "failed",
            "ocr_text": "",
            "recognized_words": 0,
            "mean_word_confidence": None,
            "oriented_width": None,
            "oriented_height": None,
            "error": "",
        }

        try:
            result, data = recognize_image(
                row["local_image_path"], config
            )
            record.update(result)
            data.to_csv(
                word_dir / f"{row['candidate_id']}.csv",
                index=False,
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )[:500]

        records.append(record)
        pd.DataFrame(records).to_csv(
            args.output / "ocr_results.csv", index=False
        )
        print(
            f"{len(records)}/{len(frame)} "
            f"{record['candidate_id']} {record['status']}",
            flush=True,
        )

    results = pd.DataFrame(records)
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": checksum(args.manifest),
        "config_sha256": checksum(args.config),
        "config": config,
        "environment": environment,
        "versions": {
            name: metadata.version(name)
            for name in ["pandas", "Pillow", "pytesseract"]
        },
        "status_counts": results["status"].value_counts().to_dict(),
        "recognized_words": int(results["recognized_words"].sum()),
        "box_coordinates": "EXIF-oriented original image pixels",
        "scope": "OCR extraction; not accuracy evaluation",
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(json.dumps(run["status_counts"], indent=2))
    print("Words:", run["recognized_words"])
    print("Output:", args.output)

    if results["status"].eq("failed").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
