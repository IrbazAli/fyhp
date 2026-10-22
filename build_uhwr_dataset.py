"""
build_uhwr_dataset.py
=============================================================================
Compiles the 8,483 handwritten Urdu lines from the NUST UHWR dataset and
blends them with the 627 Aasaan Khata handwritten ledger word crops (oversampled 4x)
to create a balanced, robust handwriting dataset for Qwen2.5-VL-7B fine-tuning.
=============================================================================
"""

import csv
import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UHWR_DIR = "/home/irbazali/urdu-ocr-models/nust-unified-urdu-ocr"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_CSV = os.path.join(UHWR_DIR, "data_uhwr/train.csv")
VAL_CSV = os.path.join(UHWR_DIR, "data_uhwr/val.csv")
LEDGER_JSON = os.path.join(DATASET_DIR, "train_words.json")

OUTPUT_TRAIN_JSON = os.path.join(DATASET_DIR, "train_combined_uhwr.json")
OUTPUT_VAL_JSON = os.path.join(DATASET_DIR, "val_uhwr.json")


def build():
    print("=" * 65)
    print("Building Combined UHWR + Aasaan Khata Ledger Dataset")
    print("=" * 65)

    # 1. Load UHWR Training lines
    train_records = []
    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fpath = os.path.join(UHWR_DIR, r["file_name"])
            text = r["text"].strip()
            if os.path.exists(fpath) and text:
                train_records.append({
                    "image": fpath,
                    "conversations": [
                        {
                            "role": "user",
                            "content": "<image>\nاس تصویر میں لکھا ہوا اردو ہاتھ کا لکھا متن پڑھ کر بتائیں۔"
                        },
                        {
                            "role": "assistant",
                            "content": text
                        }
                    ]
                })

    print(f"Loaded {len(train_records)} handwritten Urdu line samples from UHWR train set.")

    # 2. Load UHWR Validation lines
    val_records = []
    with open(VAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fpath = os.path.join(UHWR_DIR, r["file_name"])
            text = r["text"].strip()
            if os.path.exists(fpath) and text:
                val_records.append({
                    "image": fpath,
                    "conversations": [
                        {
                            "role": "user",
                            "content": "<image>\nاس تصویر میں لکھا ہوا اردو ہاتھ کا لکھا متن پڑھ کر بتائیں۔"
                        },
                        {
                            "role": "assistant",
                            "content": text
                        }
                    ]
                })

    print(f"Loaded {len(val_records)} validation samples from UHWR val set.")

    # 3. Load Existing Ledger Word/Number Crops
    with open(LEDGER_JSON, "r", encoding="utf-8") as f:
        ledger_records = json.load(f)
    print(f"Loaded {len(ledger_records)} existing ledger word crops from Aasaan Khata.")

    # 4. Anti-Forgetting Oversampling: 4x ledger crops blended with general handwriting
    combined_records = train_records + (ledger_records * 4)
    random.seed(42)
    random.shuffle(combined_records)

    print(f"Total blended training records: {len(combined_records)}")

    # 5. Save manifests
    with open(OUTPUT_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(combined_records, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_TRAIN_JSON} ({len(combined_records)} samples)")

    with open(OUTPUT_VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_VAL_JSON} ({len(val_records)} samples)")
    print("=" * 65)


if __name__ == "__main__":
    build()
