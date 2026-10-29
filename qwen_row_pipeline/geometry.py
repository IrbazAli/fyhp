"""
qwen_row_pipeline/geometry.py
=============================================================================
Geometry, EXIF canonicalization, factor-28 resolution scaling,
coordinate transformations, validation, and crop extraction.
=============================================================================
"""

import re
import math
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageOps


def parse_coordinate_float(val: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """
    Safely converts a coordinate value to float.
    Handles numeric types, strings with trailing hyphens/slashes ('281-', '50000/-'),
    ranges ('281-300'), currency/non-digit artifacts, and invalid tokens ('N/', 'N/A').
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(val) else default
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return default
        try:
            f = float(s)
            return f if math.isfinite(f) else default
        except ValueError:
            pass

        # Clean trailing slashes, dashes, spaces, currency symbols
        cleaned = re.sub(r'[/,\s]+', '', s).strip()
        cleaned = cleaned.rstrip('-_')
        try:
            f = float(cleaned)
            return f if math.isfinite(f) else default
        except ValueError:
            pass

        # Extract first valid signed/unsigned number
        m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
        if m:
            try:
                f = float(m.group(0))
                return f if math.isfinite(f) else default
            except ValueError:
                pass
    return default



def make_canonical_image(image_path: str, output_path: str) -> Tuple[Image.Image, int, int]:
    """
    Loads an image, applies EXIF orientation transpose, converts to RGB,
    saves the canonical image to output_path, and returns (image, width, height).
    """
    with Image.open(image_path) as raw_img:
        # ImageOps.exif_transpose orients the image based on EXIF tag
        transposed = ImageOps.exif_transpose(raw_img)
        if transposed is None:
            transposed = raw_img
        canonical = transposed.convert("RGB")
        canonical.save(output_path, format="PNG")
        w, h = canonical.size
        return canonical, w, h


def compute_factor28_dimensions(
    orig_w: int,
    orig_h: int,
    max_pixels: int = 602112,
    min_pixels: int = 28 * 28 * 4
) -> Tuple[int, int]:
    """
    Computes (width, height) that are both strictly divisible by 28,
    preserve the aspect ratio as closely as possible, and fit within [min_pixels, max_pixels].
    Qwen2.5-VL operates on 28x28 spatial token grids (factor=28).
    """
    orig_pixels = orig_w * orig_h
    scale = 1.0
    if orig_pixels > max_pixels:
        scale = math.sqrt(max_pixels / orig_pixels)
    elif orig_pixels < min_pixels:
        scale = math.sqrt(min_pixels / orig_pixels)

    target_w = max(28, int(round((orig_w * scale) / 28.0)) * 28)
    target_h = max(28, int(round((orig_h * scale) / 28.0)) * 28)

    # If rounding pushed it slightly over max_pixels, step down by 28 on the longer edge
    while (target_w * target_h) > max_pixels and (target_w > 28 or target_h > 28):
        if target_w >= target_h and target_w > 28:
            target_w -= 28
        elif target_h > 28:
            target_h -= 28
        else:
            break

    return target_w, target_h


def prepare_localization_image(
    canonical_img: Image.Image,
    output_path: str,
    max_pixels: int = 602112
) -> Dict[str, Any]:
    """
    Resizes canonical image to explicit factor-28 dimensions.
    Saves the localized image and returns transform metadata.
    """
    can_w, can_h = canonical_img.size
    input_w, input_h = compute_factor28_dimensions(can_w, can_h, max_pixels=max_pixels)

    resized_img = canonical_img.resize((input_w, input_h), Image.Resampling.LANCZOS)
    resized_img.save(output_path, format="PNG")

    scale_x = can_w / float(input_w)
    scale_y = can_h / float(input_h)

    return {
        "canonical_width": can_w,
        "canonical_height": can_h,
        "input_width": input_w,
        "input_height": input_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "image_path": output_path
    }


def map_stage_a_coordinates(
    raw_coords: Dict[str, Any],
    transform: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Maps Stage A whole-page coordinates from model input space to canonical page space.
    """
    scale_x = transform["scale_x"]
    scale_y = transform["scale_y"]
    can_w = transform["canonical_width"]
    can_h = transform["canonical_height"]

    mapped_rows = []
    raw_rows = raw_coords.get("rows", [])

    # Defensive unnesting if model nested full root dict inside rows list
    if isinstance(raw_rows, list) and len(raw_rows) == 1 and isinstance(raw_rows[0], dict) and "rows" in raw_rows[0] and isinstance(raw_rows[0]["rows"], list):
        inner = raw_rows[0]
        if "sections" in inner and not raw_coords.get("sections"):
            raw_coords["sections"] = inner["sections"]
        if "columns" in inner and not raw_coords.get("columns"):
            raw_coords["columns"] = inner["columns"]
        raw_rows = inner["rows"]

    input_h = transform.get("input_height", 1000)

    # Detect if model generated coordinates in 0-100 percentage space
    is_percentage_space = False
    if raw_rows and input_h > 200:
        valid_bottoms = [
            parse_coordinate_float(r.get("bottom"), default=None)
            for r in raw_rows
        ]
        valid_bottoms = [b for b in valid_bottoms if b is not None]
        max_bottom = max(valid_bottoms) if valid_bottoms else 0.0
        if 0 < max_bottom <= 100.0:
            is_percentage_space = True

    pct_scale = (input_h / 100.0) if is_percentage_space else 1.0

    for idx, r in enumerate(raw_rows):
        raw_top = parse_coordinate_float(r.get("top"), default=0.0)
        raw_bot = parse_coordinate_float(r.get("bottom"), default=0.0)

        top_in = raw_top * pct_scale
        bottom_in = raw_bot * pct_scale

        # Model input reference point
        ref_x_in = parse_coordinate_float(r.get("ref_x"), default=None)
        ref_y_in = parse_coordinate_float(r.get("ref_y"), default=None)
        if is_percentage_space and ref_y_in is not None:
            ref_y_in = ref_y_in * pct_scale

        top_can = int(round(top_in * scale_y))
        bottom_can = int(round(bottom_in * scale_y))

        ref_x_can = int(round(ref_x_in * scale_x)) if ref_x_in is not None else None
        ref_y_can = int(round(ref_y_in * scale_y)) if ref_y_in is not None else None

        row_flags = list(r.get("review_flags", []))
        if is_percentage_space:
            row_flags.append("detected_0_100_percent_coordinates")
        if isinstance(r.get("top"), str) and '-' in str(r.get("top")):
            row_flags.append("sanitized_hyphenated_coordinate")
        if isinstance(r.get("bottom"), str) and '-' in str(r.get("bottom")):
            row_flags.append("sanitized_hyphenated_coordinate")

        mapped_rows.append({
            "proposed_index": idx + 1,
            "row_type": r.get("row_type", "transaction"),
            "proposed_label": r.get("label", ""),
            "model_coords": {
                "top": top_in,
                "bottom": bottom_in,
                "ref_x": ref_x_in,
                "ref_y": ref_y_in
            },
            "canonical_coords": {
                "top": top_can,
                "bottom": bottom_can,
                "left": 0,
                "right": can_w,
                "ref_x": ref_x_can,
                "ref_y": ref_y_can
            },
            "raw_review_flags": row_flags
        })

    # Columns mapping
    mapped_columns = []
    for c_idx, col in enumerate(raw_coords.get("columns", [])):
        guide_x_in = parse_coordinate_float(col.get("guide_x"), default=None)
        guide_x_can = int(round(guide_x_in * scale_x)) if guide_x_in is not None else None
        left_in = parse_coordinate_float(col.get("left"), default=0.0)
        right_in = parse_coordinate_float(col.get("right"), default=float(can_w / scale_x))
        left_can = int(round(left_in * scale_x)) if left_in is not None else 0
        right_can = int(round(right_in * scale_x)) if right_in is not None else can_w

        mapped_columns.append({
            "col_id": f"col_{c_idx+1:02d}",
            "label": str(col.get("label", f"Column {c_idx+1}")).strip(),
            "guide_x_model": guide_x_in,
            "guide_x_canonical": guide_x_can,
            "left_canonical": left_can,
            "right_canonical": right_can
        })

    # Sections mapping
    mapped_sections = []
    for s_idx, sec in enumerate(raw_coords.get("sections", [])):
        sec_top_in = parse_coordinate_float(sec.get("top"), default=0.0)
        sec_bottom_in = parse_coordinate_float(sec.get("bottom"), default=0.0)
        mapped_sections.append({
            "section_id": f"sec_{s_idx+1:02d}",
            "label": sec.get("label", ""),
            "canonical_top": int(round(sec_top_in * scale_y)),
            "canonical_bottom": int(round(sec_bottom_in * scale_y))
        })

    return {
        "rows": mapped_rows,
        "columns": mapped_columns,
        "sections": mapped_sections
    }


def validate_row_bounds(
    rows: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    boundary_tolerance_px: float = 8.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validates row candidates.
    Flags material out-of-bounds, inverted bounds, tiny/huge rows, overlaps, and duplicates.
    Does NOT silently clamp material out-of-bounds!
    Returns (validated_rows_with_flags, overall_validation_issues).
    """
    issues = []
    annotated_rows = []

    for i, row in enumerate(rows):
        can = dict(row.get("canonical_coords", {}))
        top = can.get("top", 0)
        bottom = can.get("bottom", 0)
        flags = list(row.get("raw_review_flags", []))

        # Check finite numeric
        if not (isinstance(top, (int, float)) and isinstance(bottom, (int, float)) and
                math.isfinite(top) and math.isfinite(bottom)):
            flags.append("non_finite_coordinates")
            issues.append({"row_index": i, "issue": "Non-finite coordinates detected", "severity": "error"})

        height = bottom - top

        # Inverted or zero height
        if height <= 0:
            flags.append("inverted_or_empty_height")
            issues.append({"row_index": i, "issue": f"Row has invalid height {height}px (top={top}, bottom={bottom})", "severity": "error"})

        # Out-of-bounds checks
        # Documented small rounding at image edges is acceptable (within tolerance)
        if top < -boundary_tolerance_px:
            flags.append("material_out_of_bounds_top")
            issues.append({"row_index": i, "issue": f"Row top ({top}px) severely exceeds top edge", "severity": "warning"})
        elif top < 0:
            # Minor edge rounding tolerance
            can["top"] = 0

        if bottom > (image_height + boundary_tolerance_px):
            flags.append("material_out_of_bounds_bottom")
            issues.append({"row_index": i, "issue": f"Row bottom ({bottom}px) severely exceeds bottom edge ({image_height}px)", "severity": "warning"})
        elif bottom > image_height:
            # Minor edge rounding tolerance
            can["bottom"] = image_height

        # Suspiciously small or huge
        if 0 < height < 12:
            flags.append("suspiciously_small_height")
            issues.append({"row_index": i, "issue": f"Row height is suspiciously small ({height}px)", "severity": "warning"})
        elif height > (image_height * 0.5):
            flags.append("suspiciously_large_height")
            issues.append({"row_index": i, "issue": f"Row height spans over 50% of the entire page ({height}px)", "severity": "warning"})

        row_copy = dict(row)
        row_copy["canonical_coords"] = can
        row_copy["validation_flags"] = flags
        annotated_rows.append(row_copy)

    # Order, overlap, and duplicate check
    for i in range(len(annotated_rows) - 1):
        curr_row = annotated_rows[i]
        next_row = annotated_rows[i+1]
        curr_can = curr_row["canonical_coords"]
        next_can = next_row["canonical_coords"]

        # Check top-to-bottom ordering
        if curr_can["top"] > next_can["top"]:
            curr_row["validation_flags"].append("out_of_order")
            next_row["validation_flags"].append("out_of_order")
            issues.append({"rows": [i, i+1], "issue": f"Row {i+1} starts lower than Row {i+2}", "severity": "warning"})

        # Overlap check
        overlap_top = max(curr_can["top"], next_can["top"])
        overlap_bottom = min(curr_can["bottom"], next_can["bottom"])
        if overlap_bottom > overlap_top:
            overlap_h = overlap_bottom - overlap_top
            curr_h = max(1, curr_can["bottom"] - curr_can["top"])
            overlap_ratio = overlap_h / float(curr_h)
            if overlap_ratio > 0.85:
                curr_row["validation_flags"].append("duplicate_candidate")
                next_row["validation_flags"].append("duplicate_candidate")
                issues.append({"rows": [i, i+1], "issue": f"Rows {i+1} and {i+2} overlap by {int(overlap_ratio*100)}% (potential duplicate)", "severity": "warning"})
            elif overlap_h >= 6:
                curr_row["validation_flags"].append(f"overlap_with_row_{i+2}")
                next_row["validation_flags"].append(f"overlap_with_row_{i+1}")
                issues.append({"rows": [i, i+1], "issue": f"Vertical overlap of {overlap_h}px between rows {i+1} and {i+2}", "severity": "info"})

    return annotated_rows, issues


def create_two_row_context_crop(
    canonical_img: Image.Image,
    row_a_bounds: Tuple[int, int],
    row_b_bounds: Tuple[int, int],
    context_output_path: str,
    padding_y: int = 24,
    max_pixels: int = 250880
) -> Dict[str, Any]:
    """
    Creates a full-width context crop containing two adjacent candidate rows with vertical padding.
    Saves both the clean canonical-resolution context crop and the model input image.
    Returns full transform metadata for bidirectional coordinate mapping:
      model input -> context crop -> canonical full page.
    """
    can_w, can_h = canonical_img.size
    top_a, bot_a = row_a_bounds
    top_b, bot_b = row_b_bounds

    min_y = min(top_a, top_b)
    max_y = max(bot_a, bot_b)

    ctx_top = max(0, min_y - padding_y)
    ctx_bottom = min(can_h, max_y + padding_y)
    ctx_h = ctx_bottom - ctx_top
    ctx_w = can_w

    # Extract crop from canonical page
    raw_ctx_crop = canonical_img.crop((0, ctx_top, ctx_w, ctx_bottom))

    # Compute factor-28 dimensions for model input
    in_w, in_h = compute_factor28_dimensions(ctx_w, ctx_h, max_pixels=max_pixels)
    scaled_ctx = raw_ctx_crop.resize((in_w, in_h), Image.Resampling.LANCZOS)
    scaled_ctx.save(context_output_path, format="PNG")

    scale_x = ctx_w / float(in_w)
    scale_y = ctx_h / float(in_h)

    # Reference points in model input coordinates
    rel_top_a_in = ((top_a - ctx_top) / float(ctx_h)) * in_h
    rel_bot_a_in = ((bot_a - ctx_top) / float(ctx_h)) * in_h
    rel_top_b_in = ((top_b - ctx_top) / float(ctx_h)) * in_h
    rel_bot_b_in = ((bot_b - ctx_top) / float(ctx_h)) * in_h

    return {
        "context_image_path": context_output_path,
        "context_box_in_canonical": {
            "top": ctx_top,
            "bottom": ctx_bottom,
            "left": 0,
            "right": can_w,
            "width": ctx_w,
            "height": ctx_h
        },
        "model_input_dimensions": {
            "width": in_w,
            "height": in_h
        },
        "scale_x": scale_x,
        "scale_y": scale_y,
        "approx_input_bounds": {
            "row_a": {"top": rel_top_a_in, "bottom": rel_bot_a_in},
            "row_b": {"top": rel_top_b_in, "bottom": rel_bot_b_in}
        }
    }


def map_stage_b_coordinates(
    refinement_result: Dict[str, Any],
    context_transform: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Inverts coordinate transforms from Stage B model input space back to canonical page space:
      model input -> context crop -> canonical full page.
    """
    ctx_box = context_transform["context_box_in_canonical"]
    ctx_top = ctx_box["top"]
    scale_y = context_transform["scale_y"]

    refined_rows = {}
    for r_key in ["row_a", "row_b"]:
        r_data = refinement_result.get(r_key, {})
        pred_top_in = parse_coordinate_float(r_data.get("top"), default=0.0)
        pred_bot_in = parse_coordinate_float(r_data.get("bottom"), default=0.0)

        # Model input -> context crop (unscaled)
        ctx_crop_top = pred_top_in * scale_y
        ctx_crop_bot = pred_bot_in * scale_y

        # Context crop -> Canonical page
        canonical_top = int(round(ctx_top + ctx_crop_top))
        canonical_bottom = int(round(ctx_top + ctx_crop_bot))

        refined_rows[r_key] = {
            "row_id": r_data.get("row_id"),
            "model_coords": {"top": pred_top_in, "bottom": pred_bot_in},
            "canonical_coords": {
                "top": canonical_top,
                "bottom": canonical_bottom,
                "left": 0,
                "right": ctx_box["right"]
            },
            "ambiguity_flags": r_data.get("ambiguity_flags", [])
        }

    return {
        "refined_rows": refined_rows,
        "separation_notes": refinement_result.get("separation_notes", ""),
        "conflict_detected": refinement_result.get("conflict_detected", False)
    }


def extract_row_crop(
    canonical_img: Image.Image,
    bounds: Tuple[int, int],
    crop_output_path: str
) -> Dict[str, Any]:
    """
    Extracts a clean full-resolution rectangular crop from the canonical page.
    bounds = (top, bottom).
    Returns crop dimensions and saved path.
    """
    can_w, can_h = canonical_img.size
    top, bottom = bounds

    # Defensive boundary protection for crop slicing
    safe_top = max(0, min(top, can_h - 1))
    safe_bottom = max(safe_top + 1, min(bottom, can_h))

    crop_img = canonical_img.crop((0, safe_top, can_w, safe_bottom))
    crop_img.save(crop_output_path, format="PNG")

    crop_w, crop_h = crop_img.size
    return {
        "crop_path": crop_output_path,
        "width": crop_w,
        "height": crop_h,
        "canonical_bounds": {
            "top": top,
            "bottom": bottom,
            "left": 0,
            "right": can_w
        }
    }
