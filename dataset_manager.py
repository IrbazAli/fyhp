"""
dataset_manager.py
=============================================================================
Aasaan Khata - Dataset & Annotation Manager
=============================================================================
Manages ground truth annotations, serialization, validation, and export
into the Hugging Face / Qwen conversational fine-tuning format.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from math_engine import MathValidator


DEFAULT_ANNOTATIONS_PATH = "dataset/annotations.json"
DEFAULT_IMAGES_DIR = "dataset"

URDU_SYSTEM_PROMPT = "<image>\nExtract all ledger transactions into structured JSON. Capture Nastaliq product words, rates, line totals, and grand total."


class DatasetManager:
    def __init__(self, annotations_path: str = DEFAULT_ANNOTATIONS_PATH, images_dir: str = DEFAULT_IMAGES_DIR):
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        self.math_validator = MathValidator()
        self.data: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        """Loads existing annotations from disk."""
        if os.path.exists(self.annotations_path):
            try:
                with open(self.annotations_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading {self.annotations_path}: {e}")
                self.data = []
        else:
            self.data = []

    def save(self):
        """Persists annotations to disk."""
        os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)
        with open(self.annotations_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_available_images(self) -> List[str]:
        """Lists all image filenames in dataset folder."""
        if not os.path.exists(self.images_dir):
            return []
        valid_exts = (".jpeg", ".jpg", ".png", ".webp")
        files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_exts)]
        return sorted(files)

    def get_annotation_for_image(self, image_filename: str) -> Optional[Dict[str, Any]]:
        """Finds existing annotation for given image filename."""
        rel_path = os.path.join(self.images_dir, image_filename)
        for record in self.data:
            if record.get("image") == rel_path or os.path.basename(record.get("image", "")) == image_filename:
                return record
        return None

    def upsert_annotation(
        self,
        image_filename: str,
        structured_ledger: Dict[str, Any],
        prompt_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adds or updates an annotation record in Qwen fine-tuning conversational schema.
        """
        rel_path = os.path.join(self.images_dir, image_filename)
        prompt_text = prompt_override or URDU_SYSTEM_PROMPT

        # Serialized target JSON string for assistant response
        assistant_json_str = json.dumps(structured_ledger, ensure_ascii=False, indent=2)

        record = {
            "image": rel_path,
            "conversations": [
                {
                    "role": "user",
                    "content": prompt_text
                },
                {
                    "role": "assistant",
                    "content": assistant_json_str
                }
            ],
            "metadata": {
                "validated": self.math_validator.validate_document(structured_ledger).is_valid,
                "item_count": len(structured_ledger.get("items", [])),
                "grand_total": structured_ledger.get("grand_total")
            }
        }

        # Check if updating existing
        found_idx = None
        for i, existing in enumerate(self.data):
            if existing.get("image") == rel_path or os.path.basename(existing.get("image", "")) == image_filename:
                found_idx = i
                break

        if found_idx is not None:
            self.data[found_idx] = record
        else:
            self.data.append(record)

        self.save()
        return record

    def export_splits(
        self,
        train_path: str = "dataset/train.json",
        val_path: str = "dataset/val.json",
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[int, int]:
        """
        Splits annotated dataset into train and validation sets for fine-tuning.
        """
        if not self.data:
            return 0, 0

        # Remove internal metadata for clean Hugging Face ingestion
        clean_records = []
        for r in self.data:
            clean_records.append({
                "image": r["image"],
                "conversations": r["conversations"]
            })

        random.seed(seed)
        shuffled = list(clean_records)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        if split_idx == 0 and len(shuffled) > 0:
            split_idx = 1

        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:] if len(shuffled) > split_idx else shuffled[:1]

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        return len(train_data), len(val_data)
