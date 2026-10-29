"""
segmentation_benchmark/yolo_engine.py
=============================================================================
Ultralytics YOLO Document Layout Preprocessor Engine
Detects document layout elements and rectangular row/table bounding boxes.
0 files saved to disk (in-memory Base64 data URLs).
=============================================================================
"""

import os
import io
import time
import base64
from typing import Dict, Any, List, Optional
from PIL import Image, ImageOps
import torch
from ultralytics import YOLO

_CACHED_YOLO_MODEL: Optional[YOLO] = None


def get_cached_yolo_model(weights_path: Optional[str] = None) -> YOLO:
    global _CACHED_YOLO_MODEL
    if _CACHED_YOLO_MODEL is None:
        if weights_path and os.path.exists(weights_path):
            _CACHED_YOLO_MODEL = YOLO(weights_path)
        else:
            # Check for cached doc-layout-yolov8n weights
            cache_base = os.path.expanduser("~/.cache/huggingface/hub")
            doclayout_found = False
            for root, dirs, files in os.walk(cache_base):
                if "best.pt" in files and "doc-layout" in root:
                    candidate = os.path.join(root, "best.pt")
                    try:
                        _CACHED_YOLO_MODEL = YOLO(candidate)
                        doclayout_found = True
                        break
                    except Exception:
                        pass
            if not doclayout_found:
                # Fallback to standard yolov8n.pt
                _CACHED_YOLO_MODEL = YOLO("yolov8n.pt")
    return _CACHED_YOLO_MODEL


class YOLOLayoutPreprocessor:
    """
    Ultralytics YOLO document layout and rectangular bounding box extractor.
    Identifies document structures, tables, text regions, and rows.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path

    def segment_image(self, image_input: Any, conf_threshold: float = 0.15, source_filename: str = "") -> Dict[str, Any]:
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

        model = get_cached_yolo_model(self.weights_path)

        # Run inference
        results = model(canonical, conf=conf_threshold, device=self.device, verbose=False)

        detected_boxes: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                x1 = max(0, int(xyxy[0]))
                y1 = max(0, int(xyxy[1]))
                x2 = min(orig_w, int(xyxy[2]))
                y2 = min(orig_h, int(xyxy[3]))
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                crop_b64 = ""
                try:
                    crop_img = canonical.crop((x1, y1, x2, y2))
                    crop_buf = io.BytesIO()
                    crop_img.save(crop_buf, format="JPEG", quality=90)
                    crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf.getvalue()).decode("ascii")
                except Exception:
                    crop_b64 = ""

                detected_boxes.append({
                    "line_idx": i,
                    "id": f"yolo_box_{i}",
                    "label": f"{cls_name} ({conf:.2f})",
                    "class_name": cls_name,
                    "score": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "x": x1,
                    "y": y1,
                    "width": w,
                    "height": h,
                    "baseline": [[x1, y2], [x2, y2]],
                    "baseline_points_count": 2,
                    "boundary": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "boundary_points_count": 4,
                    "tags": [cls_name, f"conf_{conf:.2f}"],
                    "crop_data_url": crop_b64
                })

        # Sort top to bottom
        detected_boxes.sort(key=lambda b: b["y"])
        for idx, b in enumerate(detected_boxes):
            b["line_idx"] = idx

        elapsed = time.time() - t0

        # Encode full image
        full_buf = io.BytesIO()
        canonical.save(full_buf, format="JPEG", quality=90)
        full_b64 = "data:image/jpeg;base64," + base64.b64encode(full_buf.getvalue()).decode("ascii")

        return {
            "status": "success",
            "mode": "yolo",
            "engine": "Ultralytics YOLO Document Layout Detector",
            "model_name": getattr(model, "model_name", "YOLOv8 Document Layout"),
            "source_filename": source_filename,
            "dimensions": {"width": orig_w, "height": orig_h},
            "device": self.device,
            "latency_seconds": round(elapsed, 3),
            "total_lines": len(detected_boxes),
            "line_count": len(detected_boxes),
            "lines": detected_boxes,
            "image_data_url": full_b64,
            "full_image_data_url": full_b64,
            "objective": "Rectangular crops covering entire ledger rows / layout structures",
            "notes": f"Detected {len(detected_boxes)} bounding box regions in {elapsed*1000:.1f}ms on {self.device} (0 files saved to disk)."
        }
