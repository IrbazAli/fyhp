"""
qwen_row_pipeline/kraken_engine.py
=============================================================================
Kraken BLLA (Baseline and Layout Analysis) Preprocessor Engine
Performs neural baseline detection and polygon segmentation for handwritten
and printed document layouts.
=============================================================================
"""

import io
import time
import base64
from typing import Dict, Any, List, Optional
from PIL import Image, ImageOps

import torch
# Shim torch.accelerator for transformers / torch compatibility if missing
if not hasattr(torch, "accelerator"):
    class _DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.accelerator = _DummyAccelerator()

from kraken import blla
from kraken.lib import vgsl
from importlib import resources

_CACHED_BLLA_MODEL: Optional[vgsl.TorchVGSLModel] = None


def get_cached_blla_model(device: str = "cuda:0") -> vgsl.TorchVGSLModel:
    """Loads and caches the default kraken blla.mlmodel in memory."""
    global _CACHED_BLLA_MODEL
    if _CACHED_BLLA_MODEL is None:
        model_path = resources.files("kraken").joinpath("blla.mlmodel")
        _CACHED_BLLA_MODEL = vgsl.TorchVGSLModel.load_model(model_path)
    return _CACHED_BLLA_MODEL


class KrakenBLLAPreprocessor:
    """
    Executes Kraken BLLA neural baseline and boundary polygon segmentation.
    Produces in-memory image crops and coordinate geometry with 0 files saved on disk.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def segment_image(self, image_input: Any, text_direction: str = "horizontal-rl", source_filename: str = "") -> Dict[str, Any]:
        """
        Runs Kraken BLLA segmentation on an image path or PIL Image.

        Args:
            image_input: File path (str) or PIL Image.
            text_direction: Text reading direction ('horizontal-rl' for Urdu).
            source_filename: Display filename.

        Returns:
            Dictionary containing segmentation lines, baselines, polygons, bboxes,
            line crop data URLs, and full image data URL.
        """
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

        # Load / get cached model and run segmentation
        try:
            model = get_cached_blla_model(device=self.device)
            res = blla.segment(
                canonical,
                text_direction=text_direction,
                model=model,
                device=self.device,
                raise_on_error=False
            )
            used_device = self.device
        except Exception as err:
            if "out of memory" in str(err).lower() and self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                model_cpu = get_cached_blla_model(device="cpu")
                res = blla.segment(
                    canonical,
                    text_direction=text_direction,
                    model=model_cpu,
                    device="cpu",
                    raise_on_error=False
                )
                used_device = "cpu (GPU OOM fallback)"
            else:
                raise

        elapsed = time.time() - t0

        # Encode full canonical image to Base64 (0 files on disk)
        full_buf = io.BytesIO()
        canonical.save(full_buf, format="JPEG", quality=90)
        full_b64 = "data:image/jpeg;base64," + base64.b64encode(full_buf.getvalue()).decode("ascii")

        # Process each detected line
        extracted_lines: List[Dict[str, Any]] = []
        for idx, line in enumerate(res.lines):
            line_id = line.id or f"line_{idx}"
            baseline = line.baseline or []
            boundary = line.boundary or []

            # Compute bounding box
            try:
                raw_bbox = line.to_bbox().bbox  # (min_x, min_y, max_x, max_y)
                min_x = max(0, int(raw_bbox[0]))
                min_y = max(0, int(raw_bbox[1]))
                max_x = min(orig_w, int(raw_bbox[2]))
                max_y = min(orig_h, int(raw_bbox[3]))
            except Exception:
                # Fallback to computing from boundary / baseline points
                all_pts = (boundary or []) + (baseline or [])
                if all_pts:
                    xs = [p[0] for p in all_pts]
                    ys = [p[1] for p in all_pts]
                    min_x = max(0, int(min(xs)))
                    min_y = max(0, int(min(ys)))
                    max_x = min(orig_w, int(max(xs)))
                    max_y = min(orig_h, int(max(ys)))
                else:
                    min_x, min_y, max_x, max_y = 0, 0, 10, 10

            box_w = max(1, max_x - min_x)
            box_h = max(1, max_y - min_y)

            # Extract in-memory line crop
            crop_b64 = ""
            try:
                crop_img = canonical.crop((min_x, min_y, max_x, max_y))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=90)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf.getvalue()).decode("ascii")
            except Exception:
                crop_b64 = ""

            tags_list = []
            if line.tags and isinstance(line.tags, dict):
                for k, v in line.tags.items():
                    tags_list.append(f"{k}:{v}")

            extracted_lines.append({
                "line_idx": idx,
                "id": line_id,
                "bbox": [min_x, min_y, max_x, max_y],
                "x": min_x,
                "y": min_y,
                "width": box_w,
                "height": box_h,
                "baseline": baseline,
                "baseline_points_count": len(baseline),
                "boundary": boundary,
                "boundary_points_count": len(boundary),
                "tags": tags_list,
                "crop_data_url": crop_b64
            })

        # Process regions if any
        extracted_regions: List[Dict[str, Any]] = []
        if res.regions and isinstance(res.regions, dict):
            for reg_type, reg_list in res.regions.items():
                for r_idx, reg in enumerate(reg_list):
                    extracted_regions.append({
                        "id": getattr(reg, "id", f"region_{reg_type}_{r_idx}"),
                        "type": reg_type,
                        "boundary": getattr(reg, "boundary", [])
                    })

        return {
            "status": "success",
            "mode": "kraken_blla",
            "engine": "Kraken BLLA (Neural Baseline & Layout Analysis)",
            "model_name": "blla.mlmodel",
            "source_filename": source_filename,
            "dimensions": {
                "width": orig_w,
                "height": orig_h
            },
            "text_direction": text_direction,
            "device": used_device,
            "latency_seconds": round(elapsed, 3),
            "total_lines": len(extracted_lines),
            "line_count": len(extracted_lines),
            "total_regions": len(extracted_regions),
            "lines": extracted_lines,
            "regions": extracted_regions,
            "image_data_url": full_b64,
            "full_image_data_url": full_b64,
            "notes": "Neural baseline detection and layout analysis via Kraken BLLA. Zero files saved to disk."
        }
