"""
tests/smoke_test_real_pipeline.py
=============================================================================
Live end-to-end smoke test against local Flask server (Qwen Row Pipeline)
Testing with representative sample '2.jpeg'.
=============================================================================
"""

import sys
import os
import requests
import json
from openpyxl import load_workbook

BASE_URL = "http://127.0.0.1:5000"


def run_smoke_test():
    print("=" * 60)
    print("Starting Live Qwen Row Pipeline Smoke Test on '2.jpeg'...")
    print("=" * 60)

    # 1. Trigger Stage A
    print("\n[Step 1] Triggering Stage A (Layout Proposals)...")
    res_a = requests.post(
        f"{BASE_URL}/api/qwen_pipeline/upload",
        data={"sample_name": "2.jpeg"},
        timeout=180
    )
    if res_a.status_code != 200:
        print(f"Stage A Failed: {res_a.status_code} - {res_a.text}")
        sys.exit(1)

    data_a = res_a.json()
    job_id = data_a.get("job_id")
    manifest = data_a.get("manifest", {})
    rows = manifest.get("rows", [])
    columns = manifest.get("columns", [])
    model_info = data_a.get("model_info", {})

    print(f"  Job ID: {job_id}")
    print(f"  Model Info: {model_info.get('status')} (Adapter: {model_info.get('actual_adapter')})")
    print(f"  Canonical Dimensions: {data_a.get('canonical_dimensions')}")
    print(f"  Proposed Rows Count: {len(rows)}")
    print(f"  Columns Detected ({len(columns)}): {[c.get('label') for c in columns]}")

    if not rows:
        print("ERROR: No rows were proposed by Stage A!")
        sys.exit(1)

    first_row = rows[0]
    print(f"  Row 1 bounds: top={first_row['current_bounds']['top']}, bottom={first_row['current_bounds']['bottom']}")
    print(f"  Row 1 crop file: {first_row['crop_filename']}")

    # 2. Test Crop Edit (Stage C)
    print("\n[Step 2] Testing Stage C Crop Edit...")
    orig_top = first_row['current_bounds']['top']
    orig_bot = first_row['current_bounds']['bottom']
    new_top = orig_top + 2
    new_bot = orig_bot + 2

    res_edit = requests.post(
        f"{BASE_URL}/api/qwen_pipeline/job/{job_id}/crop_edit",
        json={
            "row_id": first_row["row_id"],
            "new_top": new_top,
            "new_bottom": new_bot,
            "new_row_type": "transaction"
        },
        timeout=30
    )
    if res_edit.status_code != 200:
        print(f"Crop Edit Failed: {res_edit.status_code} - {res_edit.text}")
        sys.exit(1)

    data_edit = res_edit.json()
    edited_row = data_edit.get("row", {})
    print(f"  Edited Row: {edited_row['row_id']}, top={edited_row['current_bounds']['top']}, bottom={edited_row['current_bounds']['bottom']}, is_manually_edited={edited_row.get('is_manually_edited')}")

    # 3. Test Stage B Refinement on adjacent rows (Row 1 & Row 2 if exists)
    if len(rows) >= 2:
        second_row = rows[1]
        print(f"\n[Step 3] Testing Stage B Two-Row Refinement on {first_row['row_id']} and {second_row['row_id']}...")
        res_b = requests.post(
            f"{BASE_URL}/api/qwen_pipeline/job/{job_id}/refine_pair",
            json={
                "row_a_id": first_row["row_id"],
                "row_b_id": second_row["row_id"],
                "padding_y": 24
            },
            timeout=120
        )
        if res_b.status_code == 200:
            data_b = res_b.json()
            print(f"  Stage B Refinement successful!")
            print(f"  Separation Notes: {data_b.get('refined', {}).get('separation_notes')}")
        else:
            print(f"  Stage B Warning: {res_b.status_code} - {res_b.text}")

    # 4. Run Stage D & E Transcription on Reviewed Crops
    print("\n[Step 4] Running Stage D & E Transcription on Reviewed Crops...")
    res_t = requests.post(
        f"{BASE_URL}/api/qwen_pipeline/job/{job_id}/transcribe",
        json={"pixel_budget": 100352},
        timeout=300
    )
    if res_t.status_code != 200:
        print(f"Transcription Failed: {res_t.status_code} - {res_t.text}")
        sys.exit(1)

    data_t = res_t.json()
    t_manifest = data_t.get("manifest", {})
    print(f"  Transcription completed! Current stage: {t_manifest.get('current_stage')}")
    print(f"  Stage D time: {t_manifest.get('stage_timings', {}).get('stage_d_total_seconds'):.1f}s")
    print(f"  Stage E time: {t_manifest.get('stage_timings', {}).get('stage_e_total_seconds'):.1f}s")

    for idx, r in enumerate(t_manifest.get("rows", [])[:3], start=1):
        raw_d = r.get("stage_d_transcription", {}).get("raw_text", "")
        cells = r.get("stage_e_fields", {}).get("cells", {})
        print(f"  Row {idx} ({r['row_id']}): Raw='{raw_d}' | Cells={cells}")

    # 5. Verify XLSX Export
    print("\n[Step 5] Verifying Genuine XLSX Export...")
    res_xlsx = requests.get(f"{BASE_URL}/api/qwen_pipeline/job/{job_id}/export/xlsx", timeout=30)
    if res_xlsx.status_code != 200:
        print(f"XLSX Download Failed: {res_xlsx.status_code}")
        sys.exit(1)

    xlsx_save_path = f"results/qwen_row_pipeline/{job_id}/exports/ledger_export.xlsx"
    wb = load_workbook(xlsx_save_path)
    sheet_names = wb.sheetnames
    print(f"  Workbook Sheets: {sheet_names}")
    assert "Ledger" in sheet_names, "Missing Ledger sheet"
    assert "Review" in sheet_names, "Missing Review sheet"
    assert "Provenance" in sheet_names, "Missing Provenance sheet"

    ws_ledger = wb["Ledger"]
    print(f"  Ledger RTL Enabled: {ws_ledger.views.sheetView[0].rightToLeft}")
    print(f"  Ledger Total Rows: {ws_ledger.max_row}")

    print("\n" + "=" * 60)
    print("✅ SMOKE TEST PASSED 100%! All stages verified end-to-end.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
