"""
qwen_row_pipeline/engine.py
=============================================================================
Modular execution engine for Qwen Row Pipeline:
  - Global GPU Lock & Single Model Instance Management
  - Deterministic Decoding & Truncation Detection
  - Stages A, B, C, D, E, F orchestration
=============================================================================
"""

import torch_patch
import io
import base64
import os
import re
import json
import time
import threading
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageOps

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from qwen_row_pipeline.geometry import (
    make_canonical_image,
    prepare_localization_image,
    map_stage_a_coordinates,
    validate_row_bounds,
    create_two_row_context_crop,
    map_stage_b_coordinates,
    extract_row_crop
)
from qwen_row_pipeline.prompts import (
    build_stage_a_prompt,
    build_stage_b_prompt,
    build_stage_d_prompt,
    build_stage_e_prompt
)
from qwen_row_pipeline.storage import (
    JOBS_BASE_DIR,
    create_job_directory,
    load_manifest,
    save_manifest,
    save_stage_artifact,
    export_genuine_xlsx
)

# Global lock to serialize ALL GPU generation calls across all threads and endpoints
GPU_GENERATION_LOCK = threading.Lock()

# Global shared model reference
_SHARED_MODEL = None
_SHARED_PROCESSOR = None
_ADAPTER_STATUS = {
    "intended_adapter": "./combined_adapter (627 ledger words + NUST UHWR)",
    "actual_adapter": "none",
    "status": "uninitialized",
    "details": ""
}


def get_shared_vlm(model_path: str = "./qwen2.5-vl-7b") -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Lazy loader for local 4-bit Qwen2.5-VL model.
    Guarantees exactly one model copy in GPU memory.
    Explicitly reports adapter status.
    """
    global _SHARED_MODEL, _SHARED_PROCESSOR, _ADAPTER_STATUS

    if _SHARED_MODEL is not None and _SHARED_PROCESSOR is not None:
        return _SHARED_MODEL, _SHARED_PROCESSOR, _ADAPTER_STATUS

    with GPU_GENERATION_LOCK:
        if _SHARED_MODEL is not None:
            return _SHARED_MODEL, _SHARED_PROCESSOR, _ADAPTER_STATUS

        print(f"[QwenEngine] Loading Qwen2.5-VL-7B from {model_path} with 4-bit NF4...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )

        _SHARED_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        _SHARED_PROCESSOR = AutoProcessor.from_pretrained(model_path)

        adapter_path = "./combined_adapter"
        if not os.path.exists(adapter_path):
            adapter_path = "./local_adapter"
        if not os.path.exists(adapter_path):
            adapter_path = "./local_adapter_10991"

        _ADAPTER_STATUS["intended_adapter"] = "./combined_adapter (627 ledger words + NUST UHWR)"

        safetensors_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(safetensors_file):
            try:
                from peft import PeftModel
                print(f"[QwenEngine] Applying LoRA adapter from {adapter_path}...")
                _SHARED_MODEL = PeftModel.from_pretrained(_SHARED_MODEL, adapter_path)
                _ADAPTER_STATUS["actual_adapter"] = adapter_path
                _ADAPTER_STATUS["status"] = "loaded"
                _ADAPTER_STATUS["details"] = f"LoRA weights loaded successfully from {adapter_path}."
                print(f"[QwenEngine] {_ADAPTER_STATUS['details']}")
            except Exception as e:
                _ADAPTER_STATUS["actual_adapter"] = "none (base model fallback)"
                _ADAPTER_STATUS["status"] = "failed"
                _ADAPTER_STATUS["details"] = f"Failed to attach adapter from {adapter_path}: {e}"
                print(f"[QwenEngine] Warning: {_ADAPTER_STATUS['details']}")
        else:
            _ADAPTER_STATUS["actual_adapter"] = "none (base model fallback)"
            _ADAPTER_STATUS["status"] = "not_found"
            _ADAPTER_STATUS["details"] = f"Adapter safetensors not found at {adapter_path}."
            print(f"[QwenEngine] {_ADAPTER_STATUS['details']}")

        return _SHARED_MODEL, _SHARED_PROCESSOR, _ADAPTER_STATUS


def run_deterministic_vlm_generation(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 1024,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    repetition_penalty: Optional[float] = 1.15
) -> Tuple[str, bool, float]:
    """
    Executes a deterministic forward pass (do_sample=False).
    Uses GPU_GENERATION_LOCK to ensure thread-safety.
    Detects token truncation and records runtime duration.
    Returns (raw_output_text, is_truncated, duration_seconds).
    """
    img_dict: Dict[str, Any] = {"type": "image", "image": image_path}
    if min_pixels is not None:
        img_dict["min_pixels"] = min_pixels
    if max_pixels is not None:
        img_dict["max_pixels"] = max_pixels

    messages = [
        {
            "role": "user",
            "content": [
                img_dict,
                {"type": "text", "text": prompt}
            ],
        }
    ]

    t0 = time.time()
    with GPU_GENERATION_LOCK:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None
        }
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                **gen_kwargs
            )

        in_ids = inputs.input_ids
        out_ids = generated_ids[0]
        gen_tokens = out_ids[len(in_ids[0]):]
        is_truncated = (len(gen_tokens) >= max_new_tokens)

        raw_output = processor.batch_decode(
            [gen_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    duration = time.time() - t0
    return raw_output, is_truncated, duration


def parse_json_from_model_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses JSON from raw model output.
    Supports JSON objects {...} and arrays [...].
    Recovers valid partial structures if truncated at token limit.
    """
    cleaned = raw_text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)(?:```|$)', cleaned)
    if match:
        cleaned = match.group(1).strip()

    first_brace = cleaned.find('{')
    first_bracket = cleaned.find('[')

    is_array = False
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        is_array = True
        start_idx = first_bracket
    elif first_brace != -1:
        start_idx = first_brace
    else:
        return None

    cleaned = cleaned[start_idx:]

    def try_parse(s: str) -> Optional[Any]:
        try:
            return json.loads(s)
        except Exception:
            s_fix = re.sub(r',\s*([}\]])', r'\1', s)
            try:
                return json.loads(s_fix)
            except Exception:
                return None

    def normalize_parsed_result(obj: Any, is_partial: bool = False) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        res = None
        if isinstance(obj, list):
            # Model wrapped root dict in a list: [{"coordinate_space": ..., "rows": [...]}]
            if len(obj) == 1 and isinstance(obj[0], dict) and ("rows" in obj[0] or "sections" in obj[0] or "coordinate_space" in obj[0] or "section_space" in obj[0]):
                res = obj[0]
            # Model output a list of row objects directly: [{"top": ..., "bottom": ...}, ...]
            else:
                res = {"rows": obj}
        elif isinstance(obj, dict):
            # If model nested the dict under rows: {"rows": [{"coordinate_space": ..., "rows": [...]}]}
            if "rows" in obj and isinstance(obj["rows"], list) and len(obj["rows"]) == 1:
                inner = obj["rows"][0]
                if isinstance(inner, dict) and "rows" in inner and isinstance(inner["rows"], list):
                    obj["sections"] = inner.get("sections", obj.get("sections", []))
                    obj["columns"] = inner.get("columns", obj.get("columns", []))
                    obj["coordinate_space"] = inner.get("coordinate_space", inner.get("section_space", obj.get("coordinate_space")))
                    obj["rows"] = inner["rows"]
            res = obj

        if res is not None and is_partial:
            res["is_partial_recovery"] = True
        return res

    parsed = try_parse(cleaned)
    if parsed is not None:
        normalized = normalize_parsed_result(parsed, is_partial=False)
        if normalized is not None:
            return normalized

    # Robust stack-based truncation recovery if cut off at token limit
    def compute_closing_suffix(text: str) -> Tuple[str, str]:
        last_delimiter = max(text.rfind('}'), text.rfind(']'))
        if last_delimiter == -1:
            return text, ""
        sub = text[:last_delimiter+1]

        stack = []
        in_string = False
        escape = False
        for ch in sub:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    stack.append('}')
                elif ch == '[':
                    stack.append(']')
                elif ch == '}' and stack and stack[-1] == '}':
                    stack.pop()
                elif ch == ']' and stack and stack[-1] == ']':
                    stack.pop()

        suffix = ''.join(reversed(stack))
        return sub, suffix

    sub, suffix = compute_closing_suffix(cleaned)
    if suffix:
        recovered = try_parse(sub + suffix)
        if recovered is not None:
            normalized = normalize_parsed_result(recovered, is_partial=True)
            if normalized is not None:
                return normalized

    # Fallback: scan candidate delimiter positions backwards
    idx = len(cleaned)
    while True:
        idx = max(cleaned.rfind('}', 0, idx), cleaned.rfind(']', 0, idx))
        if idx == -1:
            break
        sub_candidate = cleaned[:idx+1]
        _, cand_suffix = compute_closing_suffix(sub_candidate)
        recovered = try_parse(sub_candidate + cand_suffix)
        if recovered is not None:
            normalized = normalize_parsed_result(recovered, is_partial=True)
            if normalized is not None:
                return normalized

    return None



class QwenRowPipelineRunner:
    """
    Orchestrates the entire row segmentation, review, transcription,
    and field assignment workflow for an inspectable job.
    """

    def __init__(self, model_path: str = "./qwen2.5-vl-7b"):
        self.model_path = model_path

    def extract_vision_patches(
        self,
        image_path: str,
        source_filename: str
    ) -> Dict[str, Any]:
        """
        Vision Preprocessor (Vision Transformer) Patch Extraction:
        Decomposes input image into ViT patches (base 14x14) and merged visual tokens (28x28)
        using Qwen2.5-VL's processor.image_processor.
        
        Zero patch files are written to disk / PC.
        Downstream translation and post-processing are commented down for testing.
        """
        global _SHARED_PROCESSOR, _ADAPTER_STATUS
        if _SHARED_PROCESSOR is not None:
            processor = _SHARED_PROCESSOR
            adapter_info = _ADAPTER_STATUS
        else:
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_path)
                adapter_info = _ADAPTER_STATUS
            except Exception:
                _, processor, adapter_info = get_shared_vlm(self.model_path)

        with Image.open(image_path) as raw_img:
            canonical = ImageOps.exif_transpose(raw_img)
            if canonical is None:
                canonical = raw_img
            canonical = canonical.convert("RGB")
            orig_w, orig_h = canonical.size

        ip = processor.image_processor
        # Run vision processor
        features = ip(images=[canonical], return_tensors="pt")
        thw = features["image_grid_thw"][0].tolist()  # [1, grid_h, grid_w]
        grid_h = int(thw[1])
        grid_w = int(thw[2])

        patch_size = int(getattr(ip, "patch_size", 14))
        merge_size = int(getattr(ip, "merge_size", 2))
        token_size = patch_size * merge_size
        temporal_patch_size = int(getattr(ip, "temporal_patch_size", 2))

        resized_w = grid_w * patch_size
        resized_h = grid_h * patch_size

        resized_img = canonical.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

        # In-memory JPEG encoding (NO files written to PC)
        buf = io.BytesIO()
        resized_img.save(buf, format="JPEG", quality=90)
        b64_image = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        total_base_patches = grid_h * grid_w
        token_grid_h = grid_h // merge_size
        token_grid_w = grid_w // merge_size
        total_tokens = token_grid_h * token_grid_w

        pixel_values_shape = list(features["pixel_values"].shape)

        # Step 4: Multimodal Fusion Token Sequence Analysis
        prompt_text = "اس تصویر میں لکھا ہوا اردو ہاتھ کا لکھا لفظ یا عدد پڑھ کر بتائیں۔"
        multimodal_info: Dict[str, Any] = {}
        try:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None and hasattr(processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": canonical},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                token_inputs = processor(text=[text], images=[canonical], return_tensors="pt")
                input_ids = token_inputs["input_ids"][0].tolist()

                v_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
                v_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
                pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

                v_start_idx = input_ids.index(v_start_id) if v_start_id in input_ids else -1
                v_end_idx = input_ids.index(v_end_id) if v_end_id in input_ids else -1

                prefix_tokens = [
                    {"seq_idx": i, "id": input_ids[i], "text": tokenizer.decode([input_ids[i]])}
                    for i in range(v_start_idx + 1)
                ] if v_start_idx >= 0 else []

                postfix_tokens = [
                    {"seq_idx": i, "id": input_ids[i], "text": tokenizer.decode([input_ids[i]])}
                    for i in range(v_end_idx, len(input_ids))
                ] if v_end_idx >= 0 else []

                multimodal_info = {
                    "total_sequence_length": len(input_ids),
                    "prefix_length": len(prefix_tokens),
                    "postfix_length": len(postfix_tokens),
                    "visual_span_start": v_start_idx + 1 if v_start_idx >= 0 else 0,
                    "visual_span_end": v_end_idx - 1 if v_end_idx >= 0 else 0,
                    "visual_tokens_count": total_tokens,
                    "image_pad_id": pad_id,
                    "vision_start_id": v_start_id,
                    "vision_end_id": v_end_id,
                    "embedding_dim": 3584,
                    "vit_dim": 1280,
                    "prompt_text": prompt_text,
                    "prefix_tokens": prefix_tokens,
                    "postfix_tokens": postfix_tokens
                }
        except Exception as e:
            multimodal_info = {
                "error": str(e),
                "total_sequence_length": 15 + total_tokens + 50,
                "prefix_length": 15,
                "postfix_length": 50,
                "visual_span_start": 15,
                "visual_span_end": 15 + total_tokens - 1,
                "visual_tokens_count": total_tokens,
                "image_pad_id": 151655,
                "embedding_dim": 3584,
                "vit_dim": 1280,
                "prompt_text": prompt_text
            }

        return {
            "status": "success",
            "mode": "vision_patches",
            "source_filename": source_filename,
            "model_info": adapter_info,
            "patch_dimensions": {
                "base_patch_size": [patch_size, patch_size],
                "merge_size": merge_size,
                "merged_token_size": [token_size, token_size],
                "temporal_patch_size": temporal_patch_size
            },
            "grid": {
                "base_grid": {
                    "rows": grid_h,
                    "cols": grid_w,
                    "total_patches": total_base_patches
                },
                "token_grid": {
                    "rows": token_grid_h,
                    "cols": token_grid_w,
                    "total_tokens": total_tokens
                }
            },
            "dimensions": {
                "original": {"width": orig_w, "height": orig_h},
                "resized_input": {"width": resized_w, "height": resized_h}
            },
            "pixel_values_shape": pixel_values_shape,
            "image_data_url": b64_image,
            "multimodal_fusion": multimodal_info,
            "notes": "Vision Transformer patches extracted in-memory. Zero patch files saved on PC. Downstream translation and post-processing commented down."
        }

    def start_job(self, uploaded_image_path: str, source_filename: str) -> Dict[str, Any]:
        """
        Stage A:
        1. Creates isolated directory results/qwen_row_pipeline/<job_id>
        2. Applies EXIF orientation transpose -> canonical_page.png
        3. Prepares factor-28 resolution localization image -> localization_image.png
        4. Queries Qwen for layout proposals with strict JSON schema
        5. Inverts coordinates, validates bounds, generates stable IDs
        6. Extracts initial crop images
        7. Persists job state and manifest
        """
        job_paths = create_job_directory(source_filename)
        job_id = job_paths["job_id"]
        manifest = load_manifest(job_paths["manifest_path"])

        # Load shared model
        model, processor, adapter_info = get_shared_vlm(self.model_path)
        manifest["model_info"] = adapter_info

        # 1. Canonicalize EXIF orientation
        canonical_img, can_w, can_h = make_canonical_image(
            uploaded_image_path, job_paths["canonical_image_path"]
        )

        # 2. Prepare factor-28 localization image
        transform = prepare_localization_image(
            canonical_img, job_paths["localization_image_path"], max_pixels=602112
        )
        with open(job_paths["transforms_path"], "w", encoding="utf-8") as f:
            json.dump(transform, f, indent=2)

        input_w = transform["input_width"]
        input_h = transform["input_height"]

        # 3. Stage A prompt
        stage_a_prompt = build_stage_a_prompt(input_w, input_h)
        gen_settings = {
            "max_new_tokens": 3072,
            "do_sample": False,
            "min_pixels": input_w * input_h,
            "max_pixels": input_w * input_h
        }

        # 4. Deterministic Qwen generation
        raw_output, is_truncated, duration = run_deterministic_vlm_generation(
            model,
            processor,
            job_paths["localization_image_path"],
            stage_a_prompt,
            max_new_tokens=gen_settings["max_new_tokens"],
            min_pixels=gen_settings["min_pixels"],
            max_pixels=gen_settings["max_pixels"]
        )

        # 5. Parse JSON
        parsed_json = parse_json_from_model_output(raw_output)
        save_stage_artifact(
            job_paths["stages_dir"],
            "stage_a",
            raw_prompt=stage_a_prompt,
            raw_response=raw_output,
            generation_settings=gen_settings,
            parsed_json=parsed_json,
            duration_s=duration,
            is_truncated=is_truncated
        )

        manifest["stage_timings"]["stage_a_seconds"] = duration
        manifest["stages"]["stage_a"] = {
            "status": "completed" if parsed_json else "failed",
            "is_truncated": is_truncated,
            "duration_seconds": duration,
            "error": None if parsed_json else "Failed to parse valid layout JSON from model response"
        }

        if not parsed_json or not isinstance(parsed_json.get("rows"), list):
            manifest["overall_flags"].append("stage_a_json_malformed")
            manifest["current_stage"] = "stage_a_failed"
            save_manifest(job_paths["manifest_path"], manifest)
            raise ValueError(f"Stage A proposal failed: Model output was malformed or empty JSON. Raw output saved in job {job_id}")

        # 6. Map coordinates and validate
        mapped = map_stage_a_coordinates(parsed_json, transform)
        validated_rows, issues = validate_row_bounds(mapped["rows"], can_w, can_h)

        manifest["sections"] = mapped["sections"]
        manifest["columns"] = mapped["columns"]
        manifest["validation_issues"] = issues

        # 7. Assign stable IDs and extract crops
        row_manifests = []
        for idx, r in enumerate(validated_rows, start=1):
            row_id = f"row_{idx:03d}"
            can_coords = r["canonical_coords"]
            crop_filename = f"{row_id}.png"
            crop_path = os.path.join(job_paths["crops_dir"], crop_filename)

            # Crop from canonical page
            crop_meta = extract_row_crop(
                canonical_img,
                (can_coords["top"], can_coords["bottom"]),
                crop_path
            )

            row_entry = {
                "row_id": row_id,
                "proposed_index": idx,
                "row_type": r.get("row_type", "transaction"),
                "original_proposal": {
                    "top": can_coords["top"],
                    "bottom": can_coords["bottom"],
                    "left": can_coords["left"],
                    "right": can_coords["right"],
                    "ref_x": can_coords.get("ref_x"),
                    "ref_y": can_coords.get("ref_y")
                },
                "current_bounds": {
                    "top": can_coords["top"],
                    "bottom": can_coords["bottom"],
                    "left": can_coords["left"],
                    "right": can_coords["right"]
                },
                "crop_filename": crop_filename,
                "crop_width": crop_meta["width"],
                "crop_height": crop_meta["height"],
                "status": "proposed" if not r.get("validation_flags") else "flagged",
                "is_manually_edited": False,
                "is_refined_stage_b": False,
                "validation_flags": r.get("validation_flags", []),
                "stage_d_transcription": None,
                "stage_e_fields": None
            }
            row_manifests.append(row_entry)

        manifest["rows"] = row_manifests
        manifest["current_stage"] = "ready_for_review"
        manifest["stages"]["stage_c"] = {"status": "in_review"}
        save_manifest(job_paths["manifest_path"], manifest)

        return {
            "job_id": job_id,
            "status": "ready_for_review",
            "model_info": adapter_info,
            "canonical_dimensions": {"width": can_w, "height": can_h},
            "rows_count": len(row_manifests),
            "columns": manifest["columns"],
            "manifest": manifest
        }

    def refine_row_pair_stage_b(
        self,
        job_id: str,
        row_a_id: str,
        row_b_id: str,
        padding_y: int = 24
    ) -> Dict[str, Any]:
        """
        Stage B:
        Refines boundaries between adjacent rows using two-row context crop.
        Derives from unchanged canonical page.
        Inverts coordinates: model input -> context crop -> canonical full page.
        """
        job_dir = os.path.join(JOBS_BASE_DIR, job_id)
        manifest_path = os.path.join(job_dir, "manifest.json")
        canonical_path = os.path.join(job_dir, "canonical_page.png")
        stages_dir = os.path.join(job_dir, "stages")
        context_dir = os.path.join(job_dir, "context_crops")

        manifest = load_manifest(manifest_path)
        canonical_img = Image.open(canonical_path)

        # Locate Row A and Row B
        rows_by_id = {r["row_id"]: r for r in manifest["rows"]}
        if row_a_id not in rows_by_id or row_b_id not in rows_by_id:
            raise ValueError(f"Invalid row IDs: {row_a_id}, {row_b_id}")

        row_a = rows_by_id[row_a_id]
        row_b = rows_by_id[row_b_id]

        bounds_a = (row_a["current_bounds"]["top"], row_a["current_bounds"]["bottom"])
        bounds_b = (row_b["current_bounds"]["top"], row_b["current_bounds"]["bottom"])

        context_filename = f"context_{row_a_id}_{row_b_id}.png"
        context_output_path = os.path.join(context_dir, context_filename)

        # 1. Create context crop and transform
        ctx_transform = create_two_row_context_crop(
            canonical_img, bounds_a, bounds_b, context_output_path, padding_y=padding_y
        )

        in_dim = ctx_transform["model_input_dimensions"]
        approx = ctx_transform["approx_input_bounds"]

        # 2. Build Stage B prompt
        prompt = build_stage_b_prompt(
            in_dim["width"],
            in_dim["height"],
            row_a_id,
            row_b_id,
            approx["row_a"],
            approx["row_b"]
        )

        # 3. Deterministic generation
        model, processor, _ = get_shared_vlm(self.model_path)
        gen_settings = {
            "max_new_tokens": 768,
            "do_sample": False,
            "min_pixels": in_dim["width"] * in_dim["height"],
            "max_pixels": in_dim["width"] * in_dim["height"]
        }

        raw_output, is_truncated, duration = run_deterministic_vlm_generation(
            model,
            processor,
            context_output_path,
            prompt,
            max_new_tokens=gen_settings["max_new_tokens"],
            min_pixels=gen_settings["min_pixels"],
            max_pixels=gen_settings["max_pixels"]
        )

        parsed_json = parse_json_from_model_output(raw_output)
        save_stage_artifact(
            stages_dir,
            f"stage_b_{row_a_id}_{row_b_id}",
            raw_prompt=prompt,
            raw_response=raw_output,
            generation_settings=gen_settings,
            parsed_json=parsed_json,
            duration_s=duration,
            is_truncated=is_truncated
        )

        if not parsed_json or "row_a" not in parsed_json or "row_b" not in parsed_json:
            raise ValueError(f"Stage B refinement failed: malformed JSON response for {row_a_id} and {row_b_id}")

        # 4. Map back: model input -> context crop -> canonical page
        mapped_refinement = map_stage_b_coordinates(parsed_json, ctx_transform)
        ref_rows = mapped_refinement["refined_rows"]

        # 5. Apply refined bounds and regenerate crops
        crops_dir = os.path.join(job_dir, "crops")
        for r_key, r_target in [("row_a", row_a), ("row_b", row_b)]:
            refined = ref_rows[r_key]
            can_coords = refined["canonical_coords"]
            r_target["current_bounds"]["top"] = can_coords["top"]
            r_target["current_bounds"]["bottom"] = can_coords["bottom"]
            r_target["is_refined_stage_b"] = True
            if refined["ambiguity_flags"]:
                r_target["validation_flags"].extend(refined["ambiguity_flags"])

            # Re-crop row
            crop_path = os.path.join(crops_dir, r_target["crop_filename"])
            extract_row_crop(canonical_img, (can_coords["top"], can_coords["bottom"]), crop_path)

            # Invalidate downstream if existed
            if r_target.get("stage_d_transcription"):
                r_target["stage_d_transcription"]["stale"] = True
            if r_target.get("stage_e_fields"):
                r_target["stage_e_fields"]["stale"] = True

        # Record refinement history
        ref_record = {
            "timestamp": time.time(),
            "pair": [row_a_id, row_b_id],
            "context_image": context_filename,
            "duration_seconds": duration,
            "separation_notes": mapped_refinement["separation_notes"],
            "conflict_detected": mapped_refinement["conflict_detected"]
        }
        manifest["stages"]["stage_b"]["refinements"].append(ref_record)
        save_manifest(manifest_path, manifest)

        return {
            "status": "success",
            "pair": [row_a_id, row_b_id],
            "refined": mapped_refinement,
            "manifest": manifest
        }

    def update_crop_boundary(
        self,
        job_id: str,
        row_id: str,
        new_top: int,
        new_bottom: int,
        new_row_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage C Edit:
        Updates bounds for a row, marks it as manually edited,
        regenerates the full-resolution crop, and invalidates downstream stages.
        """
        job_dir = os.path.join(JOBS_BASE_DIR, job_id)
        manifest_path = os.path.join(job_dir, "manifest.json")
        canonical_path = os.path.join(job_dir, "canonical_page.png")
        crops_dir = os.path.join(job_dir, "crops")

        manifest = load_manifest(manifest_path)
        canonical_img = Image.open(canonical_path)

        target_row = None
        for r in manifest["rows"]:
            if r["row_id"] == row_id:
                target_row = r
                break

        if not target_row:
            raise ValueError(f"Row {row_id} not found in job {job_id}")

        if new_top >= new_bottom:
            raise ValueError(f"Invalid boundaries: top ({new_top}) must be less than bottom ({new_bottom})")

        # Update bounds
        target_row["current_bounds"]["top"] = int(new_top)
        target_row["current_bounds"]["bottom"] = int(new_bottom)
        target_row["is_manually_edited"] = True
        target_row["status"] = "accepted"
        if new_row_type:
            target_row["row_type"] = new_row_type

        # Re-crop row
        crop_path = os.path.join(crops_dir, target_row["crop_filename"])
        crop_meta = extract_row_crop(canonical_img, (new_top, new_bottom), crop_path)
        target_row["crop_width"] = crop_meta["width"]
        target_row["crop_height"] = crop_meta["height"]

        # Invalidate downstream
        if target_row.get("stage_d_transcription"):
            target_row["stage_d_transcription"]["stale"] = True
        if target_row.get("stage_e_fields"):
            target_row["stage_e_fields"]["stale"] = True

        save_manifest(manifest_path, manifest)
        return {"status": "success", "row": target_row, "manifest": manifest}

    def restore_original_row_proposal(self, job_id: str, row_id: str) -> Dict[str, Any]:
        """
        Stage C: Restores the original Qwen proposed bounds for a row.
        """
        job_dir = os.path.join(JOBS_BASE_DIR, job_id)
        manifest_path = os.path.join(job_dir, "manifest.json")
        canonical_path = os.path.join(job_dir, "canonical_page.png")
        crops_dir = os.path.join(job_dir, "crops")

        manifest = load_manifest(manifest_path)
        canonical_img = Image.open(canonical_path)

        target_row = None
        for r in manifest["rows"]:
            if r["row_id"] == row_id:
                target_row = r
                break

        if not target_row:
            raise ValueError(f"Row {row_id} not found in job {job_id}")

        orig = target_row["original_proposal"]
        target_row["current_bounds"]["top"] = orig["top"]
        target_row["current_bounds"]["bottom"] = orig["bottom"]
        target_row["is_manually_edited"] = False

        crop_path = os.path.join(crops_dir, target_row["crop_filename"])
        crop_meta = extract_row_crop(canonical_img, (orig["top"], orig["bottom"]), crop_path)
        target_row["crop_width"] = crop_meta["width"]
        target_row["crop_height"] = crop_meta["height"]

        if target_row.get("stage_d_transcription"):
            target_row["stage_d_transcription"]["stale"] = True
        if target_row.get("stage_e_fields"):
            target_row["stage_e_fields"]["stale"] = True

        save_manifest(manifest_path, manifest)
        return {"status": "success", "row": target_row, "manifest": manifest}

    def transcribe_and_assign_reviewed_crops(
        self,
        job_id: str,
        pixel_budget: int = 100352
    ) -> Dict[str, Any]:
        """
        Stages D & E: (Currently commented down for testing as requested by user).
        Downstream transcription and post-processing paused to focus on Vision Transformer patch testing.
        """
        # --- TEST MODE: TRANSLATION AND POST-PROCESSING COMMENTED DOWN ---
        return {
            "status": "commented_down",
            "message": "Stages D, E, and F (translating/transcribing and structured post-processing) are commented down for testing as requested."
        }

    def _dormant_transcribe_and_assign_reviewed_crops(
        self,
        job_id: str,
        pixel_budget: int = 100352
    ) -> Dict[str, Any]:
        """Historical full pipeline Stages D & E preserved for later re-activation."""
        job_dir = os.path.join(JOBS_BASE_DIR, job_id)
        manifest_path = os.path.join(job_dir, "manifest.json")
        stages_dir = os.path.join(job_dir, "stages")
        crops_dir = os.path.join(job_dir, "crops")

        manifest = load_manifest(manifest_path)
        model, processor, _ = get_shared_vlm(self.model_path)

        rows = manifest.get("rows", [])
        columns = manifest.get("columns", [])
        total_rows = len(rows)

        stage_d_prompt = build_stage_d_prompt()

        total_d_time = 0.0
        total_e_time = 0.0

        for idx, row in enumerate(rows, start=1):
            if row.get("status") == "deleted":
                continue

            row_id = row["row_id"]
            row_type = row.get("row_type", "transaction")
            crop_path = os.path.join(crops_dir, row["crop_filename"])

            if not os.path.exists(crop_path):
                continue

            print(f"[QwenEngine] Running Stage D & E on {row_id} ({idx}/{total_rows})...")

            # --- STAGE D: FAITHFUL ROW TRANSCRIPTION ---
            raw_d_output, is_d_trunc, d_duration = run_deterministic_vlm_generation(
                model,
                processor,
                crop_path,
                stage_d_prompt,
                max_new_tokens=256,
                max_pixels=pixel_budget
            )
            total_d_time += d_duration

            cleaned_d = raw_d_output.strip()
            # Clean outer fences if present
            cleaned_d = re.sub(r'^```[\w]*\s*', '', cleaned_d)
            cleaned_d = re.sub(r'\s*```$', '', cleaned_d).strip()

            row["stage_d_transcription"] = {
                "raw_text": raw_d_output,
                "cleaned_text": cleaned_d,
                "is_truncated": is_d_trunc,
                "duration_seconds": d_duration,
                "stale": False,
                "pixel_budget": pixel_budget
            }

            # --- STAGE E: FIELD ASSIGNMENT ---
            stage_e_prompt = build_stage_e_prompt(
                row_id=row_id,
                row_type=row_type,
                raw_transcription=cleaned_d,
                columns=columns,
                crop_width=row.get("crop_width", 1000),
                crop_height=row.get("crop_height", 60)
            )

            raw_e_output, is_e_trunc, e_duration = run_deterministic_vlm_generation(
                model,
                processor,
                crop_path,
                stage_e_prompt,
                max_new_tokens=512,
                max_pixels=pixel_budget
            )
            total_e_time += e_duration

            parsed_e = parse_json_from_model_output(raw_e_output)
            cells = {}
            unassigned = ""
            ambiguous_spans = []
            review_flags = []

            if parsed_e and isinstance(parsed_e, dict):
                raw_cells = parsed_e.get("cells", {})
                if isinstance(raw_cells, dict):
                    cells = {k: str(v) for k, v in raw_cells.items()}
                unassigned = str(parsed_e.get("unassigned_text", ""))
                ambiguous_spans = parsed_e.get("ambiguous_spans", [])
                review_flags = parsed_e.get("review_flags", [])

                # Validation: check if structured output hallucinated words unsupported by raw transcription
                if cleaned_d:
                    # Token-level overlap heuristic
                    raw_tokens = set(re.findall(r'\w+', cleaned_d))
                    struct_tokens = set()
                    for cell_val in cells.values():
                        struct_tokens.update(re.findall(r'\w+', cell_val))
                    new_tokens = struct_tokens - raw_tokens
                    # Ignore pure numbers or single punctuation marks
                    unsupported_words = [t for t in new_tokens if not t.isdigit() and len(t) > 2]
                    if unsupported_words:
                        review_flags.append(f"unsupported_words_introduced: {','.join(unsupported_words[:3])}")
            else:
                review_flags.append("stage_e_parsing_failed")
                cells = {col.get("col_id", f"col_{i+1:02d}"): "" for i, col in enumerate(columns)}
                if columns:
                    cells[columns[0].get("col_id", "col_01")] = cleaned_d

            row["stage_e_fields"] = {
                "raw_response": raw_e_output,
                "cells": cells,
                "unassigned_text": unassigned,
                "ambiguous_spans": ambiguous_spans,
                "review_flags": review_flags,
                "is_truncated": is_e_trunc,
                "duration_seconds": e_duration,
                "stale": False
            }

            # Save per-row artifacts for auditability
            save_stage_artifact(
                stages_dir,
                f"stage_d_{row_id}",
                raw_prompt=stage_d_prompt,
                raw_response=raw_d_output,
                generation_settings={"max_new_tokens": 256, "do_sample": False},
                parsed_json=None,
                duration_s=d_duration,
                is_truncated=is_d_trunc
            )
            save_stage_artifact(
                stages_dir,
                f"stage_e_{row_id}",
                raw_prompt=stage_e_prompt,
                raw_response=raw_e_output,
                generation_settings={"max_new_tokens": 512, "do_sample": False},
                parsed_json=parsed_e,
                duration_s=e_duration,
                is_truncated=is_e_trunc
            )

        manifest["stage_timings"]["stage_d_total_seconds"] = total_d_time
        manifest["stage_timings"]["stage_e_total_seconds"] = total_e_time
        manifest["stages"]["stage_d"]["status"] = "completed"
        manifest["stages"]["stage_e"]["status"] = "completed"
        manifest["current_stage"] = "recognized"

        # --- STAGE F: PYTHON DETERMINISTIC ASSEMBLY & EXPORTS ---
        exports_dir = os.path.join(job_dir, "exports")
        json_export_path = os.path.join(exports_dir, "result.json")
        xlsx_export_path = os.path.join(exports_dir, "ledger_export.xlsx")

        with open(json_export_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        export_genuine_xlsx(manifest, xlsx_export_path)

        manifest["stages"]["stage_f"] = {
            "status": "completed",
            "json_path": json_export_path,
            "xlsx_path": xlsx_export_path
        }
        manifest["current_stage"] = "completed"
        save_manifest(manifest_path, manifest)

        return {
            "status": "completed",
            "job_id": job_id,
            "manifest": manifest,
            "xlsx_url": f"/api/qwen_pipeline/job/{job_id}/export/xlsx",
            "json_url": f"/api/qwen_pipeline/job/{job_id}/export/json"
        }
