"""
segmentation_benchmark/paddle_det_engine.py
=============================================================================
PaddleOCR DBNet Text Detection Preprocessor Engine
Deep learning Differentiable Binarization (DBNet) text region detector.
Extracts irregular polygon text boundaries and word/line bounding boxes.
0 files saved to disk (in-memory Base64 data URLs).
=============================================================================
"""

import os
import io
import time
import base64
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps
import paddle.inference as pdi

_CACHED_PADDLE_PREDICTOR: Optional[pdi.Predictor] = None


def get_cached_paddle_det_predictor():
    global _CACHED_PADDLE_PREDICTOR
    if _CACHED_PADDLE_PREDICTOR is None:
        model_dir = os.path.expanduser("~/.paddlex/official_models/PP-OCRv4_server_det")
        model_json = os.path.join(model_dir, "inference.json")
        model_params = os.path.join(model_dir, "inference.pdiparams")

        if not os.path.exists(model_json) or not os.path.exists(model_params):
            raise FileNotFoundError(f"PaddleOCR detection model not found in {model_dir}")

        config = pdi.Config(model_json, model_params)
        # Disable MKLDNN/OneDNN on Linux Python 3.12 to prevent PIR array attribute crash
        config.disable_mkldnn()
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(4)
        _CACHED_PADDLE_PREDICTOR = pdi.create_predictor(config)
    return _CACHED_PADDLE_PREDICTOR


class PaddleDBNetPreprocessor:
    """
    PaddleOCR PP-OCR DBNet neural text detector.
    Detects irregular text polygon regions and bounding boxes across ledger pages.
    """

    def __init__(self):
        self.device = "CPU (Paddle Static Inference)"

    def segment_image(self, image_input: Any, thresh: float = 0.3, max_side: int = 960, source_filename: str = "") -> Dict[str, Any]:
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

        predictor = get_cached_paddle_det_predictor()

        # Compute aspect-ratio preserved dimensions divisible by 32
        ratio = max_side / max(orig_h, orig_w)
        target_h = max(32, int(round(orig_h * ratio / 32) * 32))
        target_w = max(32, int(round(orig_w * ratio / 32) * 32))

        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        # Preprocess for DBNet
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (target_w, target_h))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = ((resized.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)
        norm_img = norm_img.transpose((2, 0, 1))[np.newaxis, ...]

        # Run inference
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(norm_img.shape)
        input_tensor.copy_from_cpu(norm_img)

        predictor.run()

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        out_data = output_tensor.copy_to_cpu()

        pred_map = out_data[0, 0, :, :]
        bin_map = (pred_map > thresh).astype(np.uint8) * 255

        contours, _ = cv2.findContours(bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        detected_lines: List[Dict[str, Any]] = []
        min_box_area = 16

        for c_idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < min_box_area:
                continue

            # Scale contour points back to original image
            scaled_poly = []
            xs, ys = [], []
            for pt in cnt:
                px = int(round(pt[0][0] * scale_x))
                py = int(round(pt[0][1] * scale_y))
                scaled_poly.append([px, py])
                xs.append(px)
                ys.append(py)

            if len(xs) < 3:
                continue

            min_x = max(0, min(xs))
            min_y = max(0, min(ys))
            max_x = min(orig_w, max(xs))
            max_y = min(orig_h, max(ys))
            bw = max(1, max_x - min_x)
            bh = max(1, max_y - min_y)

            # Extract in-memory crop
            crop_b64 = ""
            try:
                crop_img = canonical.crop((min_x, min_y, max_x, max_y))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=90)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf.getvalue()).decode("ascii")
            except Exception:
                crop_b64 = ""

            detected_lines.append({
                "line_idx": len(detected_lines),
                "id": f"paddle_poly_{len(detected_lines)}",
                "label": f"Text Region #{len(detected_lines) + 1}",
                "bbox": [min_x, min_y, max_x, max_y],
                "x": min_x,
                "y": min_y,
                "width": bw,
                "height": bh,
                "baseline": [[min_x, max_y], [max_x, max_y]],
                "baseline_points_count": 2,
                "boundary": scaled_poly,
                "polygon": scaled_poly,
                "boundary_points_count": len(scaled_poly),
                "tags": ["paddle_dbnet", "text_region"],
                "crop_data_url": crop_b64
            })

        # Sort top to bottom
        detected_lines.sort(key=lambda item: item["y"])
        for idx, line in enumerate(detected_lines):
            line["line_idx"] = idx

        elapsed = time.time() - t0

        # Encode full image
        full_buf = io.BytesIO()
        canonical.save(full_buf, format="JPEG", quality=90)
        full_b64 = "data:image/jpeg;base64," + base64.b64encode(full_buf.getvalue()).decode("ascii")

        return {
            "status": "success",
            "mode": "paddleocr",
            "engine": "PaddleOCR DBNet Text Region Detector",
            "model_name": "PP-OCRv4_server_det (Differentiable Binarization)",
            "source_filename": source_filename,
            "dimensions": {"width": orig_w, "height": orig_h},
            "device": self.device,
            "latency_seconds": round(elapsed, 3),
            "total_lines": len(detected_lines),
            "line_count": len(detected_lines),
            "lines": detected_lines,
            "image_data_url": full_b64,
            "full_image_data_url": full_b64,
            "objective": "Detect text-region polygon coordinates across the page",
            "notes": f"Detected {len(detected_lines)} text region polygons in {elapsed*1000:.1f}ms on CPU (0 files saved to disk)."
        }
