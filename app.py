"""
app.py
=============================================================================
Aasaan Khata - Handwritten Urdu (Nastaliq) Wholesale Ledger AI Platform
=============================================================================
Features:
- Injected Google Font (Noto Nastaliq Urdu) & Right-to-Left (RTL) input styling.
- Recurring wholesale commodity chips (چاول, چینی, گندم, etc.).
- Live deterministic math validation comparing entered vs calculated line & grand totals.
- Offline Qwen2.5-VL-7B 4-bit VLM inference on physical RTX 3080 GPU.
- Human-in-the-Loop (HITL) review and one-click discrepancy resolution.
- Direct export into Qwen conversational fine-tuning schema (Hugging Face format).
"""

import torch_patch
import os
import json
import streamlit as st
from PIL import Image
import pandas as pd

from math_engine import MathValidator, clean_numeric_string
from dataset_manager import DatasetManager

# Page Configuration
st.set_page_config(
    page_title="Aasaan Khata (آسان کھاتہ) - VLM Pipeline",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Urdu Typography and Modern Glassmorphism Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif !important;
        direction: rtl !important;
        text-align: right !important;
        line-height: 2.2 !important;
        font-size: 1.25rem !important;
    }
    
    .urdu-header {
        font-family: 'Noto Nastaliq Urdu', serif !important;
        direction: rtl !important;
        text-align: right !important;
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #10B981;
    }
    
    .rtl-input input {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Nastaliq Urdu', serif !important;
        font-size: 1.15rem !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .badge-verified {
        background-color: #065F46;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-mismatch {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Common wholesale commodities & units
WHOLESALE_COMMODITIES = [
    "چاول", "چاول باسمتی", "چاول کائنات", "چینی", "گندم", 
    "دال چنا", "دال ماش", "دال مسور", "دال مونگ",
    "گھی", "تیل", "آٹا", "شکر", "گڑ", 
    "سرخ مرچ", "ہلدی", "دھنیا", "نمک", "پیاز", "آلو"
]

WHOLESALE_UNITS = ["من", "بوری", "تھیلا", "کلو", "پیٹی", "کارٹن", "نگ"]

# Initialize Session State & Managers
if "dataset_mgr" not in st.session_state:
    st.session_state.dataset_mgr = DatasetManager()

if "math_validator" not in st.session_state:
    st.session_state.math_validator = MathValidator()

if "vlm_engine" not in st.session_state:
    st.session_state.vlm_engine = None

if "current_items" not in st.session_state:
    st.session_state.current_items = [
        {"item": "چاول باسمتی", "quantity": 10.0, "unit": "من", "rate": 3500.0, "line_total": 35000.0}
    ]

# Sidebar
st.sidebar.markdown("<div class='urdu-header'>آسان کھاتہ</div>", unsafe_allow_html=True)
st.sidebar.markdown("**Aasaan Khata** | FYP")
st.sidebar.caption("Offline VLM & Deterministic Math Engine for Nastaliq Ledgers")

gpu_device = "NVIDIA RTX 3080 (10 GB)"
st.sidebar.info(f"🖥️ **Hardware Engine:** {gpu_device}\n\n🤖 **Model:** Qwen2.5-VL-7B (4-bit NF4)")

app_mode = st.sidebar.radio(
    "Select Workflow Mode:",
    [
        "🤖 AI VLM Transcribe & HITL Audit",
        "✍️ Manual Ground Truth Studio",
        "📊 Dataset Explorer & Split Export"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Quick Urdu Terms")
for term in WHOLESALE_COMMODITIES[:6]:
    st.sidebar.code(term)


# =============================================================================
# MODE 1: AI VLM Transcribe & HITL Audit
# =============================================================================
if app_mode == "🤖 AI VLM Transcribe & HITL Audit":
    st.title("🤖 Offline VLM Transcription & HITL Audit")
    st.markdown("Run local 4-bit `Qwen2.5-VL-7B` inference and pass outputs through the **Deterministic Math Validation Layer**.")

    available_images = st.session_state.dataset_mgr.get_available_images()
    
    col_img_pick, col_upload = st.columns([1, 1])
    with col_img_pick:
        selected_file = st.selectbox(
            "Select sample from dataset directory:",
            available_images if available_images else ["No images found"]
        )
    with col_upload:
        uploaded_file = st.file_uploader("Or upload new ledger photo:", type=["jpg", "jpeg", "png"])

    # Determine active image
    active_img_path = None
    display_image = None

    if uploaded_file is not None:
        display_image = Image.open(uploaded_file).convert("RGB")
        # Save into dataset dir for tracking
        save_name = f"uploaded_{uploaded_file.name}"
        active_img_path = os.path.join("dataset", save_name)
        display_image.save(active_img_path)
    elif selected_file and selected_file != "No images found":
        active_img_path = os.path.join("dataset", selected_file)
        if os.path.exists(active_img_path):
            display_image = Image.open(active_img_path)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("📄 Ledger Image")
        if display_image:
            st.image(display_image, use_container_width=True, caption=os.path.basename(active_img_path))
        else:
            st.warning("Please select or upload an image.")

    with col_right:
        st.subheader("⚙️ VLM Processing & Validation")

        run_infer_btn = st.button("🚀 Run Qwen2.5-VL-7B Inference", type="primary", use_container_width=True)

        if run_infer_btn and active_img_path:
            with st.spinner("Executing 4-bit VLM forward pass on RTX 3080..."):
                # Lazy-load VLM engine
                if st.session_state.vlm_engine is None:
                    from inference_engine import LedgerVLMEngine
                    st.session_state.vlm_engine = LedgerVLMEngine()

                raw_output, parsed_json, val_result = st.session_state.vlm_engine.process_image(active_img_path)
                st.session_state["last_raw_output"] = raw_output
                st.session_state["last_parsed_json"] = parsed_json
                st.session_state["last_val_result"] = val_result

        # Display inference and validation results if available
        if "last_parsed_json" in st.session_state and st.session_state["last_parsed_json"]:
            parsed = st.session_state["last_parsed_json"]
            val_res = st.session_state["last_val_result"]

            # Math Audit Status Banner
            if val_res.is_valid:
                st.success(f"✅ **MATH VERIFIED**: {val_res.audit_summary}")
            else:
                st.error(f"⚠️ **HITL INTERVENTION REQUIRED**: {val_res.audit_summary}")

            # General Header details
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.markdown(f"**Date (تاریخ):** `{parsed.get('date', 'N/A')}`")
            with h_col2:
                holder = parsed.get('khata_holder', parsed.get('customer_name', 'N/A'))
                st.markdown(f"**Khata Holder (کھاتہ دار):** <span class='urdu-text'>{holder}</span>", unsafe_allow_html=True)

            # Table of Line Items with Verification Badges
            st.markdown("### 📋 Line Items & Arithmetic Audit")
            table_rows = []
            items_list = parsed.get("items", parsed.get("transactions", parsed.get("entries", [])))

            for i, line in enumerate(val_res.line_results):
                status_badge = "✅ PASS" if line.is_valid else f"❌ MISMATCH (Diff: {line.diff:.1f})"
                table_rows.append({
                    "Index": i + 1,
                    "Item (جنس)": line.item_name,
                    "Qty (تعداد)": line.quantity,
                    "Unit (اکائی)": line.unit,
                    "Rate (نرخ)": line.rate,
                    "Extracted Total": line.extracted_total,
                    "Calculated Total": line.calculated_total,
                    "Audit Status": status_badge
                })

            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True)

            # Grand Total Math Check
            gt_col1, gt_col2 = st.columns(2)
            with gt_col1:
                st.metric("Sum of Line Items", f"Rs. {val_res.sum_of_lines:,.2f}")
            with gt_col2:
                ext_gt = val_res.extracted_grand_total or 0.0
                st.metric(
                    "Extracted Grand Total", 
                    f"Rs. {ext_gt:,.2f}", 
                    delta=f"{val_res.grand_total_diff:,.2f}" if not val_res.grand_total_valid else "Matched",
                    delta_color="inverse"
                )

            # Human-In-The-Loop Correction & Save
            st.markdown("### ✍️ Human-In-The-Loop (HITL) Actions")
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🔧 Auto-Fix Line Totals with Calculated Math", use_container_width=True):
                    # Update line totals to calculated values
                    for item in items_list:
                        q = clean_numeric_string(item.get("quantity"))
                        r = clean_numeric_string(item.get("rate"))
                        if q is not None and r is not None:
                            item["line_total"] = round(q * r, 2)
                    parsed["grand_total"] = sum(clean_numeric_string(it.get("line_total", 0)) for it in items_list)
                    st.session_state["last_val_result"] = st.session_state.math_validator.validate_document(parsed)
                    st.rerun()

            with c_btn2:
                if st.button("💾 Approve & Save to Ground Truth Dataset", type="primary", use_container_width=True):
                    img_name = os.path.basename(active_img_path)
                    st.session_state.dataset_mgr.upsert_annotation(img_name, parsed)
                    st.success(f"Successfully saved validated ground truth for `{img_name}` into dataset!")

            with st.expander("🔍 View Raw Model JSON Output"):
                st.code(json.dumps(parsed, ensure_ascii=False, indent=2), language="json")


# =============================================================================
# MODE 2: Manual Ground Truth Studio
# =============================================================================
elif app_mode == "✍️ Manual Ground Truth Studio":
    st.title("✍️ Nastaliq Ground Truth Annotation Studio")
    st.markdown("Create high-precision training annotations with RTL Urdu typography, commodity chips, and live math.")

    available_images = st.session_state.dataset_mgr.get_available_images()
    selected_img = st.selectbox("Select Image to Annotate:", available_images)

    if selected_img:
        img_path = os.path.join("dataset", selected_img)
        img = Image.open(img_path)

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(img, use_container_width=True, caption=selected_img)

        with col2:
            st.markdown("#### Document Header")
            existing_ann = st.session_state.dataset_mgr.get_annotation_for_image(selected_img)
            
            init_date = "2024-03-15"
            init_holder = "حاجی محمد اکرم گلہ منڈی"
            if existing_ann:
                try:
                    loaded_json = json.loads(existing_ann["conversations"][1]["content"])
                    init_date = loaded_json.get("date", init_date)
                    init_holder = loaded_json.get("khata_holder", init_holder)
                except Exception:
                    pass

            doc_date = st.text_input("Date (تاریخ):", value=init_date)
            doc_holder = st.text_input("Customer / Merchant Name (کھاتہ دار / دکان):", value=init_holder)

            st.markdown("#### Wholesale Commodities Quick Select")
            chip_cols = st.columns(5)
            for idx, commodity in enumerate(WHOLESALE_COMMODITIES[:10]):
                with chip_cols[idx % 5]:
                    if st.button(commodity, key=f"chip_{commodity}_{idx}"):
                        st.session_state.current_items.append({
                            "item": commodity,
                            "quantity": 10.0,
                            "unit": "من",
                            "rate": 3000.0,
                            "line_total": 30000.0
                        })
                        st.rerun()

            st.markdown("#### Line Items (Transcribing Entries)")
            items_to_save = []
            running_grand_total = 0.0

            for i, item_data in enumerate(st.session_state.current_items):
                with st.container():
                    c_del, c_name, c_qty, c_unit, c_rate, c_total = st.columns([0.5, 2, 1, 1, 1.2, 1.5])
                    with c_del:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.current_items.pop(i)
                            st.rerun()
                    with c_name:
                        name_val = st.text_input("Item Name (جنس)", value=item_data["item"], key=f"item_{i}")
                    with c_qty:
                        qty_val = st.number_input("Qty", value=float(item_data["quantity"]), min_value=0.0, step=1.0, key=f"qty_{i}")
                    with c_unit:
                        unit_val = st.selectbox("Unit", WHOLESALE_UNITS, index=WHOLESALE_UNITS.index(item_data.get("unit", "من")) if item_data.get("unit") in WHOLESALE_UNITS else 0, key=f"unit_{i}")
                    with c_rate:
                        rate_val = st.number_input("Rate (نرخ)", value=float(item_data["rate"]), min_value=0.0, step=50.0, key=f"rate_{i}")
                    with c_total:
                        # Live deterministic calculation
                        calc_tot = round(qty_val * rate_val, 2)
                        st.markdown(f"**Calculated Total:**\n\n`Rs. {calc_tot:,.1f}`")

                    items_to_save.append({
                        "item": name_val,
                        "quantity": qty_val,
                        "unit": unit_val,
                        "rate": rate_val,
                        "line_total": calc_tot
                    })
                    running_grand_total += calc_tot

            if st.button("➕ Add Transaction Line", use_container_width=True):
                st.session_state.current_items.append({
                    "item": "چینی",
                    "quantity": 5.0,
                    "unit": "بوری",
                    "rate": 140.0,
                    "line_total": 700.0
                })
                st.rerun()

            st.divider()
            st.metric("Deterministic Grand Total (میزان)", f"Rs. {running_grand_total:,.2f}")

            if st.button("💾 Save Annotation to Training Set", type="primary", use_container_width=True):
                structured_doc = {
                    "date": doc_date,
                    "khata_holder": doc_holder,
                    "items": items_to_save,
                    "grand_total": running_grand_total
                }
                st.session_state.dataset_mgr.upsert_annotation(selected_img, structured_doc)
                st.success(f"Saved verified ground truth for {selected_img}!")


# =============================================================================
# MODE 3: Dataset Explorer & Split Export
# =============================================================================
elif app_mode == "📊 Dataset Explorer & Split Export":
    st.title("📊 Dataset Explorer & Cloud Training Export")
    st.markdown("Manage dataset splits for **Kaggle GPU fine-tuning (90 hrs/week pooled)** and Hugging Face Hub synchronization.")

    mgr = st.session_state.dataset_mgr
    total_records = len(mgr.data)
    available_imgs = len(mgr.get_available_images())

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Ledger Images", available_imgs)
    with col_m2:
        st.metric("Annotated Documents", total_records)
    with col_m3:
        progress_pct = (total_records / available_imgs * 100) if available_imgs > 0 else 0
        st.metric("Dataset Labeling Progress", f"{progress_pct:.1f}%")

    st.divider()

    st.subheader("📁 Export Splits for Kaggle QLoRA Training")
    split_col1, split_col2 = st.columns([2, 1])
    with split_col1:
        train_ratio = st.slider("Train Split Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
    with split_col2:
        if st.button("⚡ Generate train.json & val.json", type="primary", use_container_width=True):
            n_train, n_val = mgr.export_splits(train_ratio=train_ratio)
            st.success(f"Generated `dataset/train.json` ({n_train} samples) and `dataset/val.json` ({n_val} samples)!")

    if total_records > 0:
        st.subheader("📑 Current Annotations")
        for i, item in enumerate(mgr.data):
            with st.expander(f"Sample #{i+1}: {item['image']}"):
                try:
                    parsed_content = json.loads(item['conversations'][1]['content'])
                    st.json(parsed_content)
                except Exception:
                    st.text(item['conversations'][1]['content'])
    else:
        st.info("No annotations created yet. Use Mode 1 or Mode 2 to annotate images.")
