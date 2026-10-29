"""
segmentation_benchmark/opencv_ruling_engine.py
=============================================================================
Classical OpenCV Morphological Ruling Detector
Extracts physical notebook horizontal and vertical rulings using morphological
kernels and slices ledger pages into ruled row and column bands.
0 files saved to disk (in-memory Base64 data URLs).
=============================================================================
"""

import io
import time
import base64
from typing import Dict, Any, List
import cv2
import numpy as np
from PIL import Image, ImageOps


class OpenCVRulingPreprocessor:
    """
    Classical computer vision ruling line and row extraction using OpenCV morphology.
    Detects physical notebook rulings and extracts inter-ruling horizontal row crops.
    """

    def __init__(self):
        self.device = "CPU (OpenCV Morphology)"

    def segment_image(self, image_input: Any, source_filename: str = "") -> Dict[str, Any]:
        t0 = time.time()

        if isinstance(image_input, str):
            with Image.open(image_input) as raw_img:
                canonical = ImageOps.exif_transpose(raw_img)
                if canonical is None:
                    canonical = raw_img
                canonical = canonical.convert("RGB")
        else:
            canonical = ImageOps.exif_transpose(image_input)
            if canonical is None:
                canonical = image_input
            canonical = canonical.convert("RGB")

        orig_w, orig_h = canonical.size
        img_np = np.array(canonical)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Inverted binary threshold: ink and rulings become white (255), background is black (0)
        bw = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        # 1. Horizontal ruling extraction
        horiz_scale = 30
        horiz_size = max(10, orig_w // horiz_scale)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        horiz_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)

        # Dilate slightly horizontally to connect broken ruled lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horiz_connected = cv2.dilate(horiz_morph, dilate_kernel, iterations=1)

        horiz_contours, _ = cv2.findContours(
            horiz_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter horizontal ruling lines spanning significant page width (> 20% page width)
        min_line_width = int(orig_w * 0.20)
        detected_rulings = []
        for c in horiz_contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= min_line_width:
                # Find line centroid y
                line_y = y + h // 2
                detected_rulings.append({
                    "x1": x,
                    "y1": line_y,
                    "x2": x + w,
                    "y2": line_y,
                    "width": w,
                    "height": h,
                    "y": line_y
                })

        # Sort rulings top to bottom
        detected_rulings.sort(key=lambda item: item["y"])

        # Merge close duplicate ruling detections (within 10 pixels of each other)
        merged_rulings = []
        for r in detected_rulings:
            if not merged_rulings:
                merged_rulings.append(r)
            else:
                prev = merged_rulings[-1]
                if abs(r["y"] - prev["y"]) < 12:
                    # Merge by taking wider coordinates
                    prev["x1"] = min(prev["x1"], r["x1"])
                    prev["x2"] = max(prev["x2"], r["x2"])
                    prev["width"] = prev["x2"] - prev["x1"]
                    prev["y"] = (prev["y"] + r["y"]) // 2
                    prev["y1"] = prev["y"]
                    prev["y2"] = prev["y"]
                else:
                    merged_rulings.append(r)

        # 2. Extract physical row bands between adjacent ruling lines
        row_bands: List[Dict[str, Any]] = []
        for idx in range(len(merged_rulings) - 1):
            top_rule = merged_rulings[idx]
            bot_rule = merged_rulings[idx + 1]
            row_y1 = max(0, top_rule["y"])
            row_y2 = min(orig_h, bot_rule["y"])
            row_h = row_y2 - row_y1

            # Ignore unreasonably thin slices (< 12 px)
            if row_h < 12:
                continue

            row_x1 = min(top_rule["x1"], bot_rule["x1"])
            row_x2 = max(top_rule["x2"], bot_rule["x2"])
            row_w = row_x2 - row_x1
            if row_w < min_line_width:
                row_x1 = 0
                row_x2 = orig_w
                row_w = orig_w

            # Extract in-memory row crop
            crop_b64 = ""
            try:
                crop_img = canonical.crop((row_x1, row_y1, row_x2, row_y2))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=90)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf.getvalue()).decode("ascii")
            except Exception:
                crop_b64 = ""

            row_bands.append({
                "line_idx": len(row_bands),
                "id": f"ruled_row_{len(row_bands)}",
                "label": f"Ruled Row #{len(row_bands) + 1}",
                "bbox": [row_x1, row_y1, row_x2, row_y2],
                "x": row_x1,
                "y": row_y1,
                "width": row_w,
                "height": row_h,
                "top_rule_y": row_y1,
                "bottom_rule_y": row_y2,
                "baseline": [[row_x1, row_y2], [row_x2, row_y2]],
                "baseline_points_count": 2,
                "boundary": [
                    [row_x1, row_y1],
                    [row_x2, row_y1],
                    [row_x2, row_y2],
                    [row_x1, row_y2]
                ],
                "boundary_points_count": 4,
                "tags": ["ruled_row", f"height_{row_h}px"],
                "crop_data_url": crop_b64
            })

        # 3. Vertical ruling extraction (column separators)
        vert_scale = 30
        vert_size = max(10, orig_h // vert_scale)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
        vert_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel)
        vert_contours, _ = cv2.findContours(
            vert_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_col_height = int(orig_h * 0.25)
        detected_columns = []
        for c in vert_contours:
            x, y, w, h = cv2.boundingRect(c)
            if h >= min_col_height:
                detected_columns.append({
                    "x1": x + w // 2,
                    "y1": y,
                    "x2": x + w // 2,
                    "y2": y + h,
                    "height": h
                })

        elapsed = time.time() - t0

        # Full image base64
        full_buf = io.BytesIO()
        canonical.save(full_buf, format="JPEG", quality=90)
        full_b64 = "data:image/jpeg;base64," + base64.b64encode(full_buf.getvalue()).decode("ascii")

        return {
            "status": "success",
            "mode": "opencv_rulings",
            "engine": "OpenCV Morphological Ruling & Row Detector",
            "model_name": "OpenCV Morphology (cv2.MORPH_OPEN)",
            "source_filename": source_filename,
            "dimensions": {"width": orig_w, "height": orig_h},
            "device": self.device,
            "latency_seconds": round(elapsed, 3),
            "total_lines": len(row_bands),
            "line_count": len(row_bands),
            "rule_count": len(merged_rulings),
            "detected_rulings_count": len(merged_rulings),
            "detected_vertical_rulings_count": len(detected_columns),
            "lines": row_bands,
            "rulings": merged_rulings,
            "column_rulings": detected_columns,
            "image_data_url": full_b64,
            "full_image_data_url": full_b64,
            "objective": "Cut between the notebook's existing horizontal rulings",
            "notes": f"Detected {len(merged_rulings)} horizontal ruling lines and extracted {len(row_bands)} row bands in {elapsed*1000:.1f}ms (0 files on disk)."
        }
