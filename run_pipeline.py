"""
run_pipeline.py
=============================================================================
Aasaan Khata - End-to-End CLI Pipeline (Offline VLM + Math Validation)
=============================================================================
Run batch or single document transcription and math auditing from the terminal.
Usage:
    python run_pipeline.py --image dataset/1.jpeg
    python run_pipeline.py --dir dataset --output results
"""

import torch_patch
import os
import json
import argparse
from typing import List

from inference_engine import LedgerVLMEngine
from math_engine import MathValidator


def process_single(engine: LedgerVLMEngine, image_path: str, output_dir: str = "results"):
    print("\n" + "=" * 70)
    print(f"📄 Processing: {image_path}")
    print("=" * 70)

    raw_output, parsed_json, val_result = engine.process_image(image_path)

    if not parsed_json:
        print("❌ Error: Failed to extract valid JSON from VLM output.")
        print("Raw Output:")
        print(raw_output)
        return

    print(f"📅 Date: {parsed_json.get('date', 'N/A')}")
    print(f"👤 Khata Holder: {parsed_json.get('khata_holder', parsed_json.get('customer_name', 'N/A'))}")
    print("\n--- Line Item Audit ---")

    for line in val_result.line_results:
        symbol = "✅" if line.is_valid else "❌"
        print(f" {symbol} [{line.item_name}] {line.quantity} × {line.rate} = {line.extracted_total} (Calc: {line.calculated_total}) | Status: {line.status}")
        if line.warning_message:
            print(f"    ⚠️  {line.warning_message}")

    print("-" * 70)
    print(f"Sum of Line Items:      Rs. {val_result.sum_of_lines:,.2f}")
    ext_gt = val_result.extracted_grand_total or 0.0
    print(f"Extracted Grand Total:  Rs. {ext_gt:,.2f}")
    if val_result.grand_total_valid:
        print("✅ Grand Total: Matched")
    else:
        print(f"❌ Grand Total Mismatch! Difference: Rs. {val_result.grand_total_diff:,.2f}")

    print("-" * 70)
    overall_status = "✅ PASS (VERIFIED)" if val_result.is_valid else "⚠️ HITL REQUIRED (DISCREPANCY)"
    print(f"Overall Status: {overall_status}")
    print(f"Audit Summary:  {val_result.audit_summary}")

    # Save to output file
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_result.json")

    final_payload = {
        "image": image_path,
        "extracted_ledger": parsed_json,
        "validation_report": val_result.to_dict()
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print(f"💾 Results saved to {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Aasaan Khata CLI Pipeline")
    parser.add_argument("--image", type=str, help="Path to single ledger image")
    parser.add_argument("--dir", type=str, help="Directory containing ledger images")
    parser.add_argument("--output", type=str, default="results", help="Directory to save output JSON files")
    parser.add_argument("--model_path", type=str, default="./qwen2.5-vl-7b", help="Path to local Qwen2.5-VL-7B checkpoint")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("Please provide either --image or --dir. Example: python run_pipeline.py --image dataset/1.jpeg")
        return

    print("Initializing Aasaan Khata Inference Engine...")
    engine = LedgerVLMEngine(model_path=args.model_path)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
        process_single(engine, args.image, args.output)

    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return
        valid_exts = (".jpeg", ".jpg", ".png", ".webp")
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(valid_exts)]
        files.sort()
        print(f"Found {len(files)} images to process in {args.dir}")

        for img_path in files:
            process_single(engine, img_path, args.output)


if __name__ == "__main__":
    main()
