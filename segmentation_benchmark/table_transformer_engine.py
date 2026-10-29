"""
segmentation_benchmark/table_transformer_engine.py
=============================================================================
Microsoft Table Transformer (TATR) Preprocessor Engine
Deep learning transformer model for Table Structure Recognition.
Detects table rows, columns, headers, and full table boundaries.
0 files saved to disk (in-memory Base64 data URLs).
=============================================================================
"""

import io
import time
import base64
from typing import Dict, Any, List, Optional
from PIL import Image, ImageOps

import torch_patch
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

_CACHED_TATR_PROCESSOR: Optional[AutoImageProcessor] = None
_CACHED_TATR_MODEL: Optional[TableTransformerForObjectDetection] = None


def get_cached_table_transformer(device: str = "cuda:0"):
    global _CACHED_TATR_PROCESSOR, _CACHED_TATR_MODEL
    if _CACHED_TATR_PROCESSOR is None or _CACHED_TATR_MODEL is None:
        model_id = "microsoft/table-transformer-structure-recognition"
        _CACHED_TATR_PROCESSOR = AutoImageProcessor.from_pretrained(model_id)
        _CACHED_TATR_MODEL = TableTransformerForObjectDetection.from_pretrained(model_id).to(device)
        _CACHED_TATR_MODEL.eval()
    return _CACHED_TATR_PROCESSOR, _CACHED_TATR_MODEL


class TableTransformerPreprocessor:
    """
    Microsoft Table Transformer (TATR) structure recognition preprocessor.
    Identifies full table boundaries, rows, columns, and headers using DETR-based transformer.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def segment_image(self, image_input: Any, confidence_threshold: float = 0.45, source_filename: str = "") -> Dict[str, Any]:
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

        processor, model = get_cached_table_transformer(self.device)

        inputs = processor(images=canonical, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
        results = processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()

        detected_rows: List[Dict[str, Any]] = []
        detected_columns: List[Dict[str, Any]] = []
        table_boxes: List[Dict[str, Any]] = []
        headers: List[Dict[str, Any]] = []
        all_elements: List[Dict[str, Any]] = []

        for score, label_idx, box in zip(scores, labels, boxes):
            label_name = model.config.id2label.get(label_idx, f"class_{label_idx}")
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(orig_w, int(box[2]))
            y2 = min(orig_h, int(box[3]))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            elem = {
                "label": label_name,
                "score": round(score, 3),
                "bbox": [x1, y1, x2, y2],
                "x": x1,
                "y": y1,
                "width": w,
                "height": h
            }
            all_elements.append(elem)

            if label_name in ["table row", "table projected row header"]:
                detected_rows.append(elem)
            elif label_name == "table column":
                detected_columns.append(elem)
            elif label_name == "table":
                table_boxes.append(elem)
            elif label_name == "table column header":
                headers.append(elem)

        # Sort rows top to bottom
        detected_rows.sort(key=lambda r: r["y"])
        detected_columns.sort(key=lambda c: c["x"])

        # Format rows into standard line structure with in-memory crops
        extracted_lines: List[Dict[str, Any]] = []
        for idx, row in enumerate(detected_rows):
            x1, y1, x2, y2 = row["bbox"]
            crop_b64 = ""
            try:
                crop_img = canonical.crop((x1, y1, x2, y2))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=90)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf.getvalue()).decode("ascii")
            except Exception:
                crop_b64 = ""

            extracted_lines.append({
                "line_idx": idx,
                "id": f"tatr_row_{idx}",
                "label": f"TATR Row #{idx + 1} ({row['label']})",
                "category": row["label"],
                "score": row["score"],
                "bbox": [x1, y1, x2, y2],
                "x": x1,
                "y": y1,
                "width": row["width"],
                "height": row["height"],
                "baseline": [[x1, y2], [x2, y2]],
                "baseline_points_count": 2,
                "boundary": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "boundary_points_count": 4,
                "tags": [row["label"], f"conf_{row['score']}"],
                "crop_data_url": crop_b64
            })

        elapsed = time.time() - t0

        # Encode full image
        full_buf = io.BytesIO()
        canonical.save(full_buf, format="JPEG", quality=90)
        full_b64 = "data:image/jpeg;base64," + base64.b64encode(full_buf.getvalue()).decode("ascii")

        return {
            "status": "success",
            "mode": "table_transformer",
            "engine": "Microsoft Table Transformer (TATR)",
            "model_name": "microsoft/table-transformer-structure-recognition",
            "source_filename": source_filename,
            "dimensions": {"width": orig_w, "height": orig_h},
            "device": self.device,
            "latency_seconds": round(elapsed, 3),
            "total_lines": len(extracted_lines),
            "line_count": len(extracted_lines),
            "detected_rows_count": len(detected_rows),
            "detected_columns_count": len(detected_columns),
            "detected_tables_count": len(table_boxes),
            "detected_headers_count": len(headers),
            "lines": extracted_lines,
            "columns": detected_columns,
            "tables": table_boxes,
            "headers": headers,
            "image_data_url": full_b64,
            "full_image_data_url": full_b64,
            "objective": "Locate ledger rows and columns together",
            "notes": f"Detected {len(detected_rows)} rows, {len(detected_columns)} columns, and {len(table_boxes)} table regions on GPU in {elapsed:.2f}s (0 files saved to disk)."
        }
