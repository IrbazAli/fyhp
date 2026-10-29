"""
segmentation_benchmark package
=============================================================================
Suite of 5 distinct document layout & row segmentation engines:
1. Kraken BLLA (Neural Baseline and Layout Analysis)
2. OpenCV Rulings (Classical morphology horizontal ruled lines & rows)
3. Table Transformer (Microsoft TATR Table Structure Recognition)
4. YOLO Layout (Ultralytics document layout bounding boxes)
5. PaddleOCR DBNet (Differentiable binarization text region polygons)
=============================================================================
"""

from segmentation_benchmark.benchmark_runner import (
    run_single_engine,
    run_comparative_benchmark,
    ENGINE_META
)
from segmentation_benchmark.opencv_ruling_engine import OpenCVRulingPreprocessor
from segmentation_benchmark.table_transformer_engine import TableTransformerPreprocessor
from segmentation_benchmark.yolo_engine import YOLOLayoutPreprocessor
from segmentation_benchmark.paddle_det_engine import PaddleDBNetPreprocessor

__all__ = [
    "run_single_engine",
    "run_comparative_benchmark",
    "ENGINE_META",
    "OpenCVRulingPreprocessor",
    "TableTransformerPreprocessor",
    "YOLOLayoutPreprocessor",
    "PaddleDBNetPreprocessor"
]
