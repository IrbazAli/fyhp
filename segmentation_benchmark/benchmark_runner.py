"""
segmentation_benchmark/benchmark_runner.py
=============================================================================
Unified Benchmark Runner for Document Line & Row Segmentation
Coordinates execution and side-by-side evaluation across all 5 candidates:
1. Kraken BLLA (Handwriting-aware baseline & polygon segmentation)
2. OpenCV Rulings (Classical morphology horizontal ruled line & row slicing)
3. Microsoft Table Transformer TATR (Deep learning table structure, rows & columns)
4. Ultralytics YOLO (Rectangular layout / row object detection)
5. PaddleOCR DBNet (Differentiable binarization text region polygons)
=============================================================================
"""

import time
from typing import Dict, Any, Optional

from qwen_row_pipeline.kraken_engine import KrakenBLLAPreprocessor
from segmentation_benchmark.opencv_ruling_engine import OpenCVRulingPreprocessor
from segmentation_benchmark.table_transformer_engine import TableTransformerPreprocessor
from segmentation_benchmark.yolo_engine import YOLOLayoutPreprocessor
from segmentation_benchmark.paddle_det_engine import PaddleDBNetPreprocessor

_ENGINES = {
    "kraken": KrakenBLLAPreprocessor(),
    "opencv_rulings": OpenCVRulingPreprocessor(),
    "table_transformer": TableTransformerPreprocessor(),
    "yolo": YOLOLayoutPreprocessor(),
    "paddleocr": PaddleDBNetPreprocessor(),
}

ENGINE_META = {
    "kraken": {
        "name": "Kraken BLLA",
        "title": "Kraken BLLA (Neural Baseline & Layout Analysis)",
        "objective": "Extract complete handwritten lines, including descending Nastaliq strokes",
        "output_type": "Neural Baseline Polylines + Tight Boundary Envelopes",
        "device": "GPU (cuda:0)",
        "strengths": "Follows irregular handwriting baselines; preserves descending strokes without rigid rectangle clipping.",
        "limitations": "Does not automatically associate amounts with descriptions across columns.",
        "fine_tunable": "Yes, documented training workflow with PAGE XML / ALTO."
    },
    "opencv_rulings": {
        "name": "OpenCV Rulings",
        "title": "OpenCV Morphological Ruling & Row Detector",
        "objective": "Cut between the notebook's existing horizontal rulings",
        "output_type": "Horizontal Ruled Lines & Inter-Rule Row Bounding Slices",
        "device": "CPU (Classical Morphology)",
        "strengths": "Ultra-fast (<50ms), deterministic, 0 VRAM footprint, no training data required.",
        "limitations": "Struggles with curved notebook pages, shadows, and descenders crossing rule lines.",
        "fine_tunable": "Parameter tuning (structuring element scale, adaptive threshold constants)."
    },
    "table_transformer": {
        "name": "Table Transformer (TATR)",
        "title": "Microsoft Table Transformer (Structure Recognition)",
        "objective": "Locate ledger rows and columns together",
        "output_type": "Table BBoxes, Row Strips, Column Bands & Headers",
        "device": "GPU (cuda:0)",
        "strengths": "Simultaneously predicts rows AND columns, reconstructing table cell grid geometry.",
        "limitations": "Pretrained on printed English tables; needs fine-tuning for handwritten ledgers.",
        "fine_tunable": "Yes, fine-tunable via HuggingFace transformers DETR pipeline."
    },
    "yolo": {
        "name": "YOLO Layout",
        "title": "Ultralytics YOLO Document Layout Detector",
        "objective": "Rectangular crops covering entire ledger rows / table blocks",
        "output_type": "Rectangular Bounding Boxes with Class Labels",
        "device": "GPU (cuda:0)",
        "strengths": "Very fast, high frame rate, straightforward bounding box annotation format.",
        "limitations": "Rigid rectangular boxes cannot isolate interwoven strokes from adjacent rows.",
        "fine_tunable": "Yes, standard YOLOv8/v11 detection fine-tuning with YAML datasets."
    },
    "paddleocr": {
        "name": "PaddleOCR DBNet",
        "title": "PaddleOCR DBNet Text Region Detector",
        "objective": "Detect text-region polygon coordinates across the page",
        "output_type": "Arbitrary-Shape Text Boundary Polygons",
        "device": "CPU (Paddle Static Inference)",
        "strengths": "Separates individual text clusters into irregular bounding polygons.",
        "limitations": "Splits entries into scattered word fragments rather than cohesive ledger rows.",
        "fine_tunable": "Yes, documented PP-OCR detection training workflow."
    }
}


def run_single_engine(engine_key: str, image_input: Any, source_filename: str = "") -> Dict[str, Any]:
    """Runs a single selected segmentation engine."""
    engine_key = engine_key.lower().strip()
    if engine_key not in _ENGINES:
        raise ValueError(f"Unknown segmentation engine '{engine_key}'. Valid engines: {list(_ENGINES.keys())}")

    engine = _ENGINES[engine_key]
    result = engine.segment_image(image_input, source_filename=source_filename)
    result["engine_key"] = engine_key
    result["metadata"] = ENGINE_META.get(engine_key, {})
    return result


def run_comparative_benchmark(image_input: Any, source_filename: str = "") -> Dict[str, Any]:
    """
    Executes all 5 candidate engines on the same input image and generates
    a comparative scorecard and benchmark dataset.
    """
    t_start = time.time()
    benchmark_results = {}
    scorecard = []

    for key, engine in _ENGINES.items():
        try:
            res = engine.segment_image(image_input, source_filename=source_filename)
            res["metadata"] = ENGINE_META.get(key, {})
            benchmark_results[key] = res

            scorecard.append({
                "engine_key": key,
                "name": ENGINE_META[key]["name"],
                "objective": ENGINE_META[key]["objective"],
                "output_type": ENGINE_META[key]["output_type"],
                "items_detected": res.get("line_count", len(res.get("lines", []))),
                "latency_seconds": res.get("latency_seconds", 0.0),
                "device": res.get("device", "N/A"),
                "status": "success",
                "notes": res.get("notes", "")
            })
        except Exception as e:
            scorecard.append({
                "engine_key": key,
                "name": ENGINE_META[key]["name"],
                "objective": ENGINE_META[key]["objective"],
                "output_type": ENGINE_META[key]["output_type"],
                "items_detected": 0,
                "latency_seconds": 0.0,
                "device": ENGINE_META[key]["device"],
                "status": f"failed: {str(e)}",
                "notes": str(e)
            })

    total_time = round(time.time() - t_start, 3)

    return {
        "status": "success",
        "mode": "comparative_benchmark",
        "source_filename": source_filename,
        "total_benchmark_time_seconds": total_time,
        "engines_tested": len(_ENGINES),
        "scorecard": scorecard,
        "results": benchmark_results,
        "comparison_summary": {
            "fastest": min(scorecard, key=lambda s: s["latency_seconds"] if s["status"] == "success" else 999)["name"],
            "most_detailed_lines": max(scorecard, key=lambda s: s["items_detected"] if s["status"] == "success" else -1)["name"],
            "handwriting_preservation_choice": "Kraken BLLA",
            "table_structure_choice": "Microsoft Table Transformer (TATR)",
            "speed_baseline_choice": "OpenCV Rulings"
        }
    }
