"""
build_training_data.py
=============================================================================
Combines word-level cropped image patches and full-page ledger annotations
into a unified Qwen2.5-VL training dataset: dataset/train.json
"""

import os
import json
import re
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WORD_ANN_FILE = os.path.join(DATASET_DIR, "word_annotations.json")
PAGE_ANN_FILE = os.path.join(DATASET_DIR, "annotations.json")
CROPS_DIR = os.path.join(DATASET_DIR, "word_crops")
TRAIN_FILE = os.path.join(DATASET_DIR, "train.json")

os.makedirs(CROPS_DIR, exist_ok=True)


def safe_name(text: str) -> str:
    cleaned = re.sub(r'[\/\\:\*\?"<>\|\s]+', '_', text.strip())
    return cleaned[:30] if cleaned else "word"


def build_dataset(padding: int = 4):
    train_records = []
    
    # 1. Process Word Bounding Boxes into Cropped Images
    if os.path.exists(WORD_ANN_FILE):
        with open(WORD_ANN_FILE, 'r', encoding='utf-8') as f:
            word_annotations = json.load(f)

        for img_name, boxes in word_annotations.items():
            img_path = os.path.join(DATASET_DIR, img_name)
            if not os.path.exists(img_path):
                continue

            master_img = Image.open(img_path).convert("RGB")
            w, h = master_img.size

            for i, b in enumerate(boxes):
                text = b.get("text", "").strip()
                if not text:
                    continue

                x1 = max(0, int(b.get("x1", 0)) - padding)
                y1 = max(0, int(b.get("y1", 0)) - padding)
                x2 = min(w, int(b.get("x2", 0)) + padding)
                y2 = min(h, int(b.get("y2", 0)) + padding)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop_img = master_img.crop((x1, y1, x2, y2))
                crop_fname = f"{os.path.splitext(img_name)[0]}_crop_{i+1:03d}_{safe_name(text)}.png"
                crop_path = os.path.join(CROPS_DIR, crop_fname)
                crop_img.save(crop_path)

                rel_crop_path = os.path.relpath(crop_path, BASE_DIR)
                train_records.append({
                    "image": rel_crop_path,
                    "conversations": [
                        {
                            "role": "user",
                            "content": "<image>\nاس تصویر میں لکھا ہوا اردو ہاتھ کا لکھا لفظ یا عدد پڑھ کر بتائیں۔"
                        },
                        {
                            "role": "assistant",
                            "content": text
                        }
                    ]
                })

    # 2. Add Full-Page Ledger Annotations
    if os.path.exists(PAGE_ANN_FILE):
        with open(PAGE_ANN_FILE, 'r', encoding='utf-8') as f:
            page_annotations = json.load(f)

        for item in page_annotations:
            if "image" in item and os.path.exists(item["image"]):
                train_records.append({
                    "image": item["image"],
                    "conversations": item["conversations"]
                })

    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(train_records)} total training samples in '{TRAIN_FILE}'")
    print(f"   - Word image crops: {len(train_records) - (len(page_annotations) if os.path.exists(PAGE_ANN_FILE) else 0)}")
    print(f"   - Full-page samples: {len(page_annotations) if os.path.exists(PAGE_ANN_FILE) else 0}")


if __name__ == "__main__":
    build_dataset()
