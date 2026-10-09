"""
server.py
=============================================================================
Aasaan Khata - Production Flask Server (Qwen2.5-VL-7B Offline Backend)
=============================================================================
Connects the dark-mode web frontend directly to the local 4-bit Qwen2.5-VL-7B
model running on NVIDIA RTX 3080.
Outputs dynamic tabular data with Urdu headings and original data languages.
"""

import torch_patch
import os
import sys
import json
import re
import uuid
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "qwen2.5-vl-7b")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
UPLOADS_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Global model state
vlm_model = None
vlm_processor = None


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(getattr(e, 'description', e))}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found (404)"}), 404


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File size exceeds 50MB limit"}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error occurred during processing"}), 500


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
        if os.path.exists("./local_adapter/adapter_model.safetensors"):
            try:
                from peft import PeftModel
                print("Applying fine-tuned LoRA adapter from ./local_adapter...")
                vlm_model = PeftModel.from_pretrained(vlm_model, "./local_adapter")
                print("LoRA adapter loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load local adapter: {e}")
        print("Model loaded successfully.")
    return vlm_model, vlm_processor


def run_vlm_on_image(image_path: str):
    """
    Runs Qwen2.5-VL forward pass and extracts ledger in native Urdu/numeric format.
    """
    model, processor = get_vlm()

    prompt = (
        "You are an expert handwritten Urdu ledger transcription system.\n"
        "Extract EVERY transaction row from this ledger page into structured JSON format matching the exact printed columns and rows.\n\n"
        "CRITICAL RULES FOR ROW EXTRACTION:\n"
        "1. Every individual transaction row written under 'رقم روپیہ' (Amount) MUST be an INDEPENDENT object in the 'entries' list. DO NOT combine or concatenate multiple rows into one.\n"
        "2. Even if the date (e.g., 11-03-018) is written once for several items, repeat that date for each row.\n"
        "3. Transcribe each item description in authentic handwritten Urdu Nastaliq (e.g., کیش, مونگ پھلی, کمیشن, عبد المتین ڈھلی).\n"
        "4. Output valid JSON only with this structure:\n"
        "{\n"
        "  \"entries\": [\n"
        "    {\n"
        "      \"تاریخ (Date)\": \"7-03-018\",\n"
        "      \"تفصیل آمدن (Description / Particulars)\": \"کیش\",\n"
        "      \"صفحہ (Folio)\": \"\",\n"
        "      \"رقم روپیہ (Amount Rs)\": \"6344/-\"\n"
        "    },\n"
        "    {\n"
        "      \"تاریخ (Date)\": \"7-03-018\",\n"
        "      \"تفصیل آمدن (Description / Particulars)\": \"ملک سلطان امیر اینڈ کو بل ڈنڈل\",\n"
        "      \"صفحہ (Folio)\": \"11\",\n"
        "      \"رقم روپیہ (Amount Rs)\": \"22377/-\"\n"
        "    },\n"
        "    {\n"
        "      \"تاریخ (Date)\": \"11-03-018\",\n"
        "      \"تفصیل آمدن (Description / Particulars)\": \"کیش\",\n"
        "      \"صفحہ (Folio)\": \"\",\n"
        "      \"رقم روپیہ (Amount Rs)\": \"4024/-\"\n"
        "    },\n"
        "    {\n"
        "      \"تاریخ (Date)\": \"11-03-018\",\n"
        "      \"تفصیل آمدن (Description / Particulars)\": \"مونگ پھلی\",\n"
        "      \"صفحہ (Folio)\": \"40\",\n"
        "      \"رقم روپیہ (Amount Rs)\": \"11745/-\"\n"
        "    },\n"
        "    {\n"
        "      \"تاریخ (Date)\": \"11-03-018\",\n"
        "      \"تفصیل آمدن (Description / Particulars)\": \"کمیشن\",\n"
        "      \"صفحہ (Folio)\": \"20\",\n"
        "      \"رقم روپیہ (Amount Rs)\": \"479/-\"\n"
        "    },\n"
        "    {\n"
        "      \"تاریخ (Date)\": \"12-03-018\",\n"
        "      \"تفصیل آمدن (Description / Particulars)\": \"عبد المتین ڈھلی\",\n"
        "      \"صفحہ (Folio)\": \"200\",\n"
        "      \"رقم روپیہ (Amount Rs)\": \"48000/-\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Output valid JSON only without markdown explanation."
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
        generated_ids = model.generate(**inputs, max_new_tokens=1536)

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

    parsed_json = None
    try:
        parsed_json = json.loads(cleaned)
    except Exception:
        cleaned_fix = re.sub(r',\s*([}\]])', r'\1', cleaned)
        try:
            parsed_json = json.loads(cleaned_fix)
        except Exception:
            pass

    if not parsed_json or "entries" not in parsed_json:
        parsed_json = {
            "detected_fields": ["تاریخ (Date)", "تفصیل آمدن (Description / Particulars)", "صفحہ (Folio)", "رقم روپیہ (Amount Rs)"],
            "entries": []
        }

    raw_fields = parsed_json.get("detected_fields", [])
    raw_entries = parsed_json.get("entries", [])

    # Standardize column headers to exact Urdu + English format
    standard_columns = ["تاریخ (Date)", "تفصیل آمدن (Description / Particulars)", "صفحہ (Folio)", "رقم روپیہ (Amount Rs)"]

    def map_key_to_standard(k):
        k_lower = k.lower()
        if "تاریخ" in k or "date" in k_lower:
            return "تاریخ (Date)"
        if "آمدن" in k or "تفصیل" in k or "توضیحات" in k or "تخصیص" in k or "desc" in k_lower or "particular" in k_lower or "name" in k_lower:
            return "تفصیل آمدن (Description / Particulars)"
        if "صفحہ" in k or "صفحه" in k or "folio" in k_lower or "page" in k_lower:
            return "صفحہ (Folio)"
        if "رقم" in k or "amount" in k_lower or "price" in k_lower or "total" in k_lower:
            return "رقم روپیہ (Amount Rs)"
        return k

    table_rows = []
    for entry in raw_entries:
        if isinstance(entry, dict):
            mapped_row = {}
            for k, v in entry.items():
                std_k = map_key_to_standard(k)
                mapped_row[std_k] = str(v).strip()

            row_cells = [
                mapped_row.get("تاریخ (Date)", ""),
                mapped_row.get("تفصیل آمدن (Description / Particulars)", ""),
                mapped_row.get("صفحہ (Folio)", ""),
                mapped_row.get("رقم روپیہ (Amount Rs)", "")
            ]
            table_rows.append(row_cells)
        elif isinstance(entry, list):
            table_rows.append([str(c).strip() for c in entry])

    return standard_columns, table_rows, parsed_json


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

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
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


@app.route('/api/images', methods=['GET'])
def get_dataset_images():
    valid_exts = ('.jpeg', '.jpg', '.png', '.webp')
    files = [f for f in sorted(os.listdir(DATASET_DIR)) if f.lower().endswith(valid_exts)]
    return jsonify({"images": files})


WORD_ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "word_annotations.json")


@app.route('/api/annotations/words', methods=['GET'])
def get_word_annotations():
    image_name = request.args.get('image', '')
    safe_name = secure_filename(image_name) if image_name else ''
    if not os.path.exists(WORD_ANNOTATIONS_PATH):
        return jsonify({"image": safe_name, "boxes": []})
    try:
        with open(WORD_ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
            all_ann = json.load(f)
        return jsonify({
            "image": safe_name,
            "boxes": all_ann.get(safe_name, [])
        })
    except Exception as e:
        return jsonify({"error": str(e), "boxes": []}), 500


@app.route('/api/annotations/words', methods=['POST'])
def save_word_annotations():
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image filename"}), 400
    safe_name = secure_filename(data['image'])
    boxes = data.get('boxes', [])

    all_ann = {}
    if os.path.exists(WORD_ANNOTATIONS_PATH):
        try:
            with open(WORD_ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
                all_ann = json.load(f)
        except Exception:
            all_ann = {}

    all_ann[safe_name] = boxes
    with open(WORD_ANNOTATIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_ann, f, ensure_ascii=False, indent=2)

    return jsonify({"status": "success", "saved_count": len(boxes), "image": safe_name})


if __name__ == '__main__':
    print("=" * 65)
    print("Aasaan Khata Web App running at http://localhost:5000")
    print("=" * 65)
    # Pre-warm model in memory so first user request is instant
    get_vlm()
    app.run(host='0.0.0.0', port=5000, debug=False)
