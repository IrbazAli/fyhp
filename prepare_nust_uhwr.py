"""Prepare locally available NUST-repository handwriting for Qwen training."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROMPT = "<image>\nاس تصویر میں لکھا ہوا اردو ہاتھ کا لکھا لفظ یا عدد پڑھ کر بتائیں۔"


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    records, fingerprints = {}, defaultdict(list)
    for split in ("train", "val", "test"):
        with (source / "data_uhwr" / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            records[split] = list(csv.DictReader(handle))
        for row in records[split]:
            path = (source / row["file_name"]).resolve()
            if not path.is_relative_to(source) or not path.is_file() or not row["text"].strip():
                raise ValueError(f"Invalid record: {row}")
            with Image.open(path) as image:
                image.verify()
            row["image"] = str(path)
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprints[row["sha256"]].append((split, row))

    # Quarantine ambiguous images and any image duplicated across supplied splits.
    excluded = set()
    quarantine = []
    for digest, group in fingerprints.items():
        if len({s for s, _ in group}) > 1 or len({r["text"] for _, r in group}) > 1:
            excluded.add(digest)
            quarantine.extend({"split": s, **r} for s, r in group)

    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, rows in records.items():
        samples = [
            {"image": row["image"], "conversations": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": row["text"]},
            ]}
            for row in rows if row["sha256"] not in excluded
        ]
        counts[split] = len(samples)
        (destination / f"{split}.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))

    unavailable = {}
    for name in ("data_upti", "data_urti"):
        for csv_path in sorted((source / name).glob("*.csv")):
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            unavailable[str(csv_path.relative_to(source))] = {
                "labels": len(rows),
                "available_images": sum((source / row["file_name"]).is_file() for row in rows),
            }
    report = {"source": str(source), "splits": counts, "quarantined": quarantine,
              "other_datasets": unavailable,
              "split_policy": "Supplied splits retained; ambiguous/cross-split identical files quarantined. Writer independence not verified."}
    (destination / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/home/irbazali/urdu-ocr-models/nust-unified-urdu-ocr")
    parser.add_argument("--output", default="dataset/nust_uhwr")
    args = parser.parse_args()
    prepare(args.source, args.output)
