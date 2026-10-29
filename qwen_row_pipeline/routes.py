"""
qwen_row_pipeline/routes.py
=============================================================================
Flask Blueprint for Qwen Row Pipeline endpoints:
  - Job creation & Stage A segmentation
  - Crop review & boundary edits
  - Two-row Stage B refinement
  - Stage D & E transcription and field assignment
  - Artifact serving & Excel export
=============================================================================
"""

import os
import uuid
from flask import Blueprint, request, jsonify, send_file, send_from_directory, current_app
from werkzeug.utils import secure_filename

from qwen_row_pipeline.engine import QwenRowPipelineRunner
from qwen_row_pipeline.kraken_engine import KrakenBLLAPreprocessor
from qwen_row_pipeline.storage import JOBS_BASE_DIR, load_manifest
from segmentation_benchmark import (
    run_single_engine,
    run_comparative_benchmark,
    ENGINE_META
)

pipeline_bp = Blueprint("qwen_pipeline", __name__, url_prefix="/api/qwen_pipeline")
runner = QwenRowPipelineRunner()
kraken_preprocessor = KrakenBLLAPreprocessor()


@pipeline_bp.route("/benchmark_meta", methods=["GET"])
def benchmark_meta():
    """Returns metadata for all 5 document line/row segmentation benchmark engines."""
    return jsonify({
        "status": "success",
        "engines": ENGINE_META,
        "default_engine": "kraken"
    })


@pipeline_bp.route("/benchmark_segment", methods=["POST"])
def benchmark_segment():
    """
    Executes a chosen segmentation candidate or all 5 candidates in a comparative benchmark.
    Engine options: 'kraken', 'opencv_rulings', 'table_transformer', 'yolo', 'paddleocr', or 'all'.
    """
    engine_key = request.form.get("engine", "kraken").strip().lower()
    sample_name = request.form.get("sample_name", "").strip()
    image_file = request.files.get("image")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    if sample_name:
        safe_name = secure_filename(sample_name)
        sample_path = os.path.join(base_dir, "dataset", safe_name)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(base_dir, "dataset", "word_crops", safe_name)
        if not os.path.exists(sample_path):
            return jsonify({"error": f"Sample image not found: {sample_name}"}), 404
        target_path = sample_path
        orig_filename = safe_name
    elif image_file and image_file.filename:
        ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        temp_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        target_path = os.path.join(temp_dir, temp_name)
        image_file.save(target_path)
        orig_filename = secure_filename(image_file.filename)
    else:
        return jsonify({"error": "No image or sample provided"}), 400

    try:
        if engine_key == "all" or engine_key == "benchmark":
            result = run_comparative_benchmark(target_path, source_filename=orig_filename)
        else:
            result = run_single_engine(engine_key, target_path, source_filename=orig_filename)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/kraken_segment", methods=["POST"])
def kraken_segment():
    """
    Runs Kraken BLLA neural baseline and layout analysis segmentation (backward compatible).
    Accepts either an uploaded file ('image') or a sample name ('sample_name').
    """
    sample_name = request.form.get("sample_name", "").strip()
    image_file = request.files.get("image")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    if sample_name:
        safe_name = secure_filename(sample_name)
        sample_path = os.path.join(base_dir, "dataset", safe_name)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(base_dir, "dataset", "word_crops", safe_name)
        if not os.path.exists(sample_path):
            return jsonify({"error": f"Sample image not found: {sample_name}"}), 404
        target_path = sample_path
        orig_filename = safe_name
    elif image_file and image_file.filename:
        ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        temp_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        target_path = os.path.join(temp_dir, temp_name)
        image_file.save(target_path)
        orig_filename = secure_filename(image_file.filename)
    else:
        return jsonify({"error": "No image or sample provided"}), 400

    try:
        direction = request.form.get("direction", "horizontal-rl")
        result = kraken_preprocessor.segment_image(target_path, text_direction=direction, source_filename=orig_filename)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/upload", methods=["POST"])
def pipeline_upload():
    """
    Creates a new job and runs Stage A (whole-page layout proposal).
    Accepts either an uploaded file ('image') or a sample name ('sample_name').
    """
    sample_name = request.form.get("sample_name", "").strip()
    image_file = request.files.get("image")
    engine = request.form.get("engine", "").strip().lower()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    if sample_name:
        safe_name = secure_filename(sample_name)
        sample_path = os.path.join(base_dir, "dataset", safe_name)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(base_dir, "dataset", "word_crops", safe_name)
        if not os.path.exists(sample_path):
            return jsonify({"error": f"Sample image not found: {sample_name}"}), 404
        target_path = sample_path
        orig_filename = safe_name
    elif image_file and image_file.filename:
        ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        temp_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        target_path = os.path.join(temp_dir, temp_name)
        image_file.save(target_path)
        orig_filename = secure_filename(image_file.filename)
    else:
        return jsonify({"error": "No image or sample provided"}), 400

    try:
        if engine == "kraken" or engine == "kraken_blla":
            direction = request.form.get("direction", "horizontal-rl")
            result = kraken_preprocessor.segment_image(target_path, text_direction=direction, source_filename=orig_filename)
            return jsonify(result)

        # Default vision patch preprocessor
        result = runner.extract_vision_patches(target_path, orig_filename)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    Returns current manifest and stage details for an existing job.
    """
    safe_job_id = secure_filename(job_id)
    manifest_path = os.path.join(JOBS_BASE_DIR, safe_job_id, "manifest.json")
    if not os.path.exists(manifest_path):
        return jsonify({"error": f"Job not found: {job_id}"}), 404

    try:
        manifest = load_manifest(manifest_path)
        return jsonify({
            "job_id": safe_job_id,
            "manifest": manifest
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/job/<job_id>/crop_edit", methods=["POST"])
def edit_crop_boundary(job_id: str):
    """
    Stage C: Saves user-adjusted top/bottom boundary or row type.
    Regenerates crop and marks downstream results stale.
    """
    safe_job_id = secure_filename(job_id)
    data = request.get_json(force=True)
    if not data or "row_id" not in data or "new_top" not in data or "new_bottom" not in data:
        return jsonify({"error": "Missing required fields (row_id, new_top, new_bottom)"}), 400

    row_id = data["row_id"]
    new_top = int(data["new_top"])
    new_bottom = int(data["new_bottom"])
    new_row_type = data.get("new_row_type")

    try:
        res = runner.update_crop_boundary(safe_job_id, row_id, new_top, new_bottom, new_row_type)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/job/<job_id>/restore_proposal", methods=["POST"])
def restore_proposal(job_id: str):
    """
    Stage C: Restores original proposed bounds for a row.
    """
    safe_job_id = secure_filename(job_id)
    data = request.get_json(force=True)
    if not data or "row_id" not in data:
        return jsonify({"error": "Missing row_id"}), 400

    row_id = data["row_id"]
    try:
        res = runner.restore_original_row_proposal(safe_job_id, row_id)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/job/<job_id>/refine_pair", methods=["POST"])
def refine_pair(job_id: str):
    """
    Stage B: Refines adjacent rows using two-row context crop.
    """
    safe_job_id = secure_filename(job_id)
    data = request.get_json(force=True)
    if not data or "row_a_id" not in data or "row_b_id" not in data:
        return jsonify({"error": "Missing row_a_id or row_b_id"}), 400

    row_a_id = data["row_a_id"]
    row_b_id = data["row_b_id"]
    padding_y = int(data.get("padding_y", 24))

    try:
        res = runner.refine_row_pair_stage_b(safe_job_id, row_a_id, row_b_id, padding_y=padding_y)
        return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/job/<job_id>/transcribe", methods=["POST"])
def transcribe_crops(job_id: str):
    """
    Stages D & E: Runs faithful transcription and field assignment on reviewed crops.
    """
    safe_job_id = secure_filename(job_id)
    data = request.get_json(silent=True) or {}
    pixel_budget = int(data.get("pixel_budget", 100352))

    try:
        res = runner.transcribe_and_assign_reviewed_crops(safe_job_id, pixel_budget=pixel_budget)
        return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/job/<job_id>/artifact/<path:subpath>", methods=["GET"])
def serve_artifact(job_id: str, subpath: str):
    """
    Securely serves job artifacts (e.g., canonical_page.png, crops/row_001.png, context_crops/...).
    """
    safe_job_id = secure_filename(job_id)
    job_dir = os.path.join(JOBS_BASE_DIR, safe_job_id)
    if not os.path.exists(job_dir):
        return jsonify({"error": "Job not found"}), 404

    # Security check: resolve canonical target path
    full_target = os.path.abspath(os.path.join(job_dir, subpath))
    if not full_target.startswith(os.path.abspath(job_dir)):
        return jsonify({"error": "Access denied: invalid artifact path"}), 403

    if not os.path.exists(full_target):
        return jsonify({"error": "Artifact file not found"}), 404

    dir_name = os.path.dirname(full_target)
    file_name = os.path.basename(full_target)
    return send_from_directory(dir_name, file_name)


@pipeline_bp.route("/job/<job_id>/export/xlsx", methods=["GET"])
def download_xlsx(job_id: str):
    """
    Downloads the genuine multi-sheet XLSX export file.
    """
    safe_job_id = secure_filename(job_id)
    xlsx_path = os.path.join(JOBS_BASE_DIR, safe_job_id, "exports", "ledger_export.xlsx")
    if not os.path.exists(xlsx_path):
        return jsonify({"error": "XLSX export not found"}), 404

    return send_file(
        xlsx_path,
        as_attachment=True,
        download_name=f"{safe_job_id}_ledger.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@pipeline_bp.route("/job/<job_id>/export/json", methods=["GET"])
def download_json(job_id: str):
    """
    Downloads the complete result.json export.
    """
    safe_job_id = secure_filename(job_id)
    json_path = os.path.join(JOBS_BASE_DIR, safe_job_id, "exports", "result.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "JSON export not found"}), 404

    return send_file(
        json_path,
        as_attachment=True,
        download_name=f"{safe_job_id}_result.json",
        mimetype="application/json"
    )
