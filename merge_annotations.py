"""
merge_annotations.py
=============================================================================
Merges downloaded JSON annotation files from teammates into:
  - dataset/word_annotations.json
  - frontend/dataset/word_annotations.json

Usage:
  1. Drop received JSON files into the `incoming_labels/` folder, or pass them as arguments:
       python merge_annotations.py
       python merge_annotations.py path/to/annotations.json

  2. The script merges all bounding boxes & Urdu transcriptions cleanly,
     and automatically triggers build_training_data.py so you are ready to train!
=============================================================================
"""

import os
import sys
import glob
import json
import subprocess

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FRONTEND_DATASET_DIR = os.path.join(BASE_DIR, "frontend", "dataset")
MASTER_ANN_FILE = os.path.join(DATASET_DIR, "word_annotations.json")
FRONTEND_ANN_FILE = os.path.join(FRONTEND_DATASET_DIR, "word_annotations.json")
INCOMING_DIR = os.path.join(BASE_DIR, "incoming_labels")

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(FRONTEND_DATASET_DIR, exist_ok=True)


def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not read {filepath}: {e}")
        return {}


def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_same_box(b1, b2, tolerance=6):
    # Check if box coordinates match within tolerance and text matches
    close_x1 = abs(b1.get("x1", 0) - b2.get("x1", 0)) <= tolerance
    close_y1 = abs(b1.get("y1", 0) - b2.get("y1", 0)) <= tolerance
    close_x2 = abs(b1.get("x2", 0) - b2.get("x2", 0)) <= tolerance
    close_y2 = abs(b1.get("y2", 0) - b2.get("y2", 0)) <= tolerance
    return close_x1 and close_y1 and close_x2 and close_y2


def merge_boxes(existing_list, incoming_list):
    merged = list(existing_list)
    added_count = 0

    for new_box in incoming_list:
        text = str(new_box.get("text", "")).strip()
        if not text:
            continue

        # Check if already present
        found = False
        for ex in merged:
            if is_same_box(ex, new_box):
                # Update text if existing was empty
                if not ex.get("text") and text:
                    ex["text"] = text
                found = True
                break

        if not found:
            merged.append(new_box)
            added_count += 1

    return merged, added_count


def main():
    print("=" * 60)
    print("  Aasaan Khata - Annotations Merger")
    print("=" * 60)

    # 1. Determine files to merge
    input_files = []
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        # Look in incoming_labels/
        incoming_patterns = [
            os.path.join(INCOMING_DIR, "*.json"),
            os.path.join(BASE_DIR, "word_annotations_*.json")
        ]
        for p in incoming_patterns:
            input_files.extend(glob.glob(p))

    if not input_files:
        print(f"\nℹ️  No incoming JSON files found in '{os.path.relpath(INCOMING_DIR)}/'.")
        print("   Place teammates' downloaded JSON files into 'incoming_labels/' or run:")
        print("   python merge_annotations.py <path_to_file.json>\n")
        return

    # 2. Load master annotations
    master_data = load_json(MASTER_ANN_FILE)
    total_added = 0
    files_processed = 0

    print(f"\n📂 Processing {len(input_files)} incoming file(s)...")

    for fpath in input_files:
        data = load_json(fpath)
        if not data:
            continue

        files_processed += 1
        # Data can be { "image_name": [boxes...] }
        for img_name, boxes in data.items():
            if not isinstance(boxes, list):
                continue

            existing_boxes = master_data.get(img_name, [])
            merged, added = merge_boxes(existing_boxes, boxes)
            master_data[img_name] = merged
            total_added += added
            print(f"  ✓ {os.path.basename(fpath)} -> [{img_name}]: {len(merged)} total boxes (+{added} new)")

    # 3. Save merged results
    save_json(MASTER_ANN_FILE, master_data)
    save_json(FRONTEND_ANN_FILE, master_data)

    print("\n" + "=" * 60)
    print(f"🎉 Success! Merged {files_processed} file(s). Added {total_added} new boxes.")
    print(f"💾 Updated: {os.path.relpath(MASTER_ANN_FILE)}")
    print("=" * 60)

    # 4. Automatically run build_training_data.py
    build_script = os.path.join(BASE_DIR, "build_training_data.py")
    if os.path.exists(build_script):
        print("\n⚙️  Re-building word crops and Qwen training dataset...")
        res = subprocess.run([sys.executable, build_script], capture_output=True, text=True)
        if res.returncode == 0:
            print("✓ build_training_data.py finished successfully!")
        else:
            print(f"⚠️ Warning during build_training_data: {res.stderr}")

    print("\n🚀 Ready for local training: run `python train_local.py` anytime!\n")


if __name__ == '__main__':
    main()
