"""
server.py
=============================================================================
Aasaan Khata - Production Flask Server (Qwen2.5-VL-7B Offline Backend)
=============================================================================
Connects the dark-mode glassmorphism web frontend directly to the local
4-bit Qwen2.5-VL-7B Vision-Language Model running on NVIDIA RTX 3080.
"""

import torch_patch
import os
import sys
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

app = Flask(__name__, static_folder='frontend', static_url_path='')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "qwen2.5-vl-7b")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global model state
vlm_model = None
vlm_processor = None


def get_vlm():
    """Lazy loader for Qwen2.5-VL-7B 4-bit model."""
    global vlm_model, vlm_processor
    if vlm_model is None or vlm_processor is None:
        print(f"Loading Qwen2.5-VL-7B into 4-bit memory from {MODEL_PATH}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        vlm_processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("Model loaded successfully.")
    return vlm_model, vlm_processor


def run_vlm_on_image(image_path: str):
    """
    Runs Qwen2.5-VL forward pass and parses table rows & columns.
    """
    model, processor = get_vlm()

    prompt = (
        "Extract all information from this handwritten ledger image into structured tabular JSON format.\n\n"
        "Instructions:\n"
        "1. Dynamically identify all columns/fields that exist in the image (e.g., date, description, folio/page, amount, particulars, etc.).\n"
        "2. Extract every line/row from top to bottom into 'entries'.\n"
        "3. Output format:\n"
        "{\n"
        "  \"detected_fields\": [\"field1\", \"field2\", ...],\n"
        "  \"entries\": [\n"
        "    {\"field1\": \"...\", \"field2\": \"...\"}\n"
        "  ]\n"
        "}\n"
        "Output valid JSON only."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 602112  # Capped to prevent CUDA OOM
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=1280)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse JSON
    cleaned = raw_output.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
    if match:
        cleaned = match.group(1).strip()
    
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1:
        cleaned = cleaned[first_brace:last_brace+1]

    try:
        parsed_json = json.loads(cleaned)
    except Exception:
        # Fallback regex trailing commas
        cleaned_fix = re.sub(r',\s*([}\]])', r'\1', cleaned)
        try:
            parsed_json = json.loads(cleaned_fix)
        except Exception:
            parsed_json = {"detected_fields": ["Raw Output"], "entries": [{"Raw Output": raw_output}]}

    # Convert parsed JSON into dynamic columns & rows
    columns = parsed_json.get("detected_fields", [])
    entries = parsed_json.get("entries", [])

    # If detected_fields wasn't explicitly populated, infer from entry keys
    if not columns and entries and isinstance(entries[0], dict):
        keys_set = []
        for entry in entries:
            for k in entry.keys():
                if k not in keys_set:
                    keys_set.append(k)
        columns = keys_set

    # Build rows matrix
    table_rows = []
    for entry in entries:
        if isinstance(entry, dict):
            row_cells = [str(entry.get(col, "")) for col in columns]
            table_rows.append(row_cells)
        elif isinstance(entry, list):
            table_rows.append([str(c) for c in entry])

    return columns, table_rows, parsed_json


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/temp_uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image attached"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename) or "uploaded_ledger.jpg"
    dest_path = os.path.join(UPLOADS_DIR, filename)
    file.save(dest_path)

    try:
        columns, table_rows, raw_json = run_vlm_on_image(dest_path)
        return jsonify({
            "columns": columns,
            "table": table_rows,
            "raw_json": raw_json,
            "image_url": f"/temp_uploads/{filename}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/sample', methods=['GET'])
def process_sample():
    sample_name = request.args.get('name', '2.jpeg')
    safe_name = secure_filename(sample_name)
    sample_path = os.path.join(DATASET_DIR, safe_name)

    if not os.path.exists(sample_path):
        return jsonify({"error": f"Sample not found: {sample_name}"}), 404

    try:
        columns, table_rows, raw_json = run_vlm_on_image(sample_path)
        return jsonify({
            "columns": columns,
            "table": table_rows,
            "raw_json": raw_json,
            "image_url": f"/dataset/{safe_name}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 65)
    print("Aasaan Khata Web App starting on http://localhost:5000")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=False)
