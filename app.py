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

st.sidebar.markdown(
    """<a href="http://localhost:5000/label.html" target="_blank" style="text-decoration:none;">
    <div style="background:rgba(0,255,163,0.12); border:1px solid #00FFA3; border-radius:8px; padding:10px; text-align:center; color:#00FFA3; font-weight:700; margin-bottom:12px;">
    ✏️ Fullscreen Visual Labeler ↗
    </div></a>""",
    unsafe_allow_html=True
)

app_mode = st.sidebar.radio(
    "Select Workflow Mode:",
    [
        "✏️ Visual Mouse Box Labeler (Canvas Studio)",
        "✍️ Manual Ground Truth Studio",
        "🤖 AI VLM Transcribe & HITL Audit",
        "📊 Dataset Explorer & Split Export"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Quick Urdu Terms")
for term in WHOLESALE_COMMODITIES[:6]:
    st.sidebar.code(term)


# =============================================================================
# MODE 0: Interactive Visual Mouse Box Labeler (Canvas Studio)
# =============================================================================
if app_mode == "✏️ Visual Mouse Box Labeler (Canvas Studio)":
    st.title("✏️ Interactive Mouse Word Labeling Studio")
    st.markdown("Draw bounding boxes directly around words with your mouse, view the cropped word, and transcribe Nastaliq labels.")
    import streamlit.components.v1 as components
    components.iframe("http://localhost:5000/label.html", height=840, scrolling=True)


# =============================================================================
# MODE 1: AI VLM Transcribe & HITL Audit
# =============================================================================
elif app_mode == "🤖 AI VLM Transcribe & HITL Audit":
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
    st.markdown("Create high-precision training annotations with RTL Urdu typography, interactive table editing, and live math validation.")

    mgr = st.session_state.dataset_mgr
    available_images = mgr.get_available_images()
    
    # Build list of images with labeling status
    image_options = []
    for img_name in available_images:
        existing = mgr.get_annotation_for_image(img_name)
        if existing:
            try:
                content = json.loads(existing["conversations"][1]["content"])
                entries_count = len(content.get("entries", content.get("items", [])))
                image_options.append(f"{img_name}  [✅ Labeled - {entries_count} rows]")
            except Exception:
                image_options.append(f"{img_name}  [✅ Labeled]")
        else:
            image_options.append(f"{img_name}  [⚪ Unlabeled]")

    col_pick, col_up = st.columns([1.5, 1])
    with col_pick:
        selected_display = st.selectbox("Select Image to Annotate:", image_options if image_options else ["No images found"])
        selected_img = selected_display.split(" ")[0] if selected_display != "No images found" else None
    with col_up:
        uploaded_ann_img = st.file_uploader("Or add new ledger image to dataset:", type=["jpg", "jpeg", "png"])
        if uploaded_ann_img is not None:
            save_name = uploaded_ann_img.name
            target_path = os.path.join("dataset", save_name)
            if not os.path.exists(target_path):
                img_obj = Image.open(uploaded_ann_img).convert("RGB")
                img_obj.save(target_path)
                st.success(f"Added `{save_name}` to dataset directory! Refreshing...")
                st.rerun()

    if selected_img and selected_img != "No images found":
        img_path = os.path.join("dataset", selected_img)
        img = Image.open(img_path)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(img, use_container_width=True, caption=f"Selected: {selected_img}")
            st.caption("💡 Zoom or view in fullscreen to inspect faint handwritten Nastaliq cursive lines.")

        with col2:
            st.markdown("### 📝 Annotation Metadata")
            existing_ann = mgr.get_annotation_for_image(selected_img)
            
            init_date = "11-03-2018"
            init_holder = "روزنامچہ دکان"
            init_doc_type = "روزنامچہ (Daily Ledger / Cash Book)"
            existing_content = None

            if existing_ann:
                try:
                    existing_content = json.loads(existing_ann["conversations"][1]["content"])
                    init_date = existing_content.get("date", init_date)
                    init_holder = existing_content.get("khata_holder", existing_content.get("customer_name", init_holder))
                    init_doc_type = existing_content.get("document_type", init_doc_type)
                except Exception:
                    pass

            meta_c1, meta_c2, meta_c3 = st.columns([1.2, 1.2, 1.2])
            with meta_c1:
                doc_type = st.selectbox(
                    "Document Type:", 
                    ["روزنامچہ (Daily Ledger / Cash Book)", "بل کھاتہ (Invoice / Bill)", "کھاتہ دار حساب (Account Ledger)"],
                    index=0 if "روزنامچہ" in init_doc_type else (1 if "بل" in init_doc_type else 2)
                )
            with meta_c2:
                doc_date = st.text_input("Date (تاریخ):", value=init_date)
            with meta_c3:
                doc_holder = st.text_input("Account / Shop (کھاتہ دار / دکان):", value=init_holder)

            # Choose schema template
            schema_type = st.radio(
                "Table Schema Format:",
                ["📜 Ledger Grid (تاریخ، تفصیل آمدن، صفحہ، رقم روپیہ)", "📦 Commodity Bill (جنس، تعداد، اکائی، نرخ، رقم)"],
                horizontal=True
            )

            st.markdown("#### ⚡ Quick Urdu Nastaliq Keywords (Click to copy / reference)")
            quick_terms = [
                "کیش", "مونگ پھلی", "کمیشن", "عبد المتین ڈھلی", 
                "مال سلطان", "عوامی کمیشن", "میزان", "نام بنام خرچ", 
                "بقیہ", "77-5", "82", "شریف بلخی", "کشت"
            ]
            chip_cols = st.columns(6)
            for idx, term in enumerate(quick_terms):
                with chip_cols[idx % 6]:
                    st.code(term)

            st.markdown("### 📋 Interactive Transaction Grid Editor")
            st.caption("Edit directly in the table below. Use Tab to move cells, press Enter to confirm, or click '➕ Add row'.")

            # Prepare dataframe based on existing content or schema
            if "Ledger Grid" in schema_type:
                initial_rows = []
                if existing_content and "entries" in existing_content:
                    for e in existing_content["entries"]:
                        initial_rows.append({
                            "تاریخ (Date)": str(e.get("date", "")),
                            "تفصیل آمدن (Description / Particulars)": str(e.get("description", "")),
                            "صفحہ (Folio)": str(e.get("folio", "")),
                            "رقم روپیہ (Amount Rs)": float(e.get("amount", 0.0) or 0.0),
                            "نوعیت (Type)": str(e.get("type", "credit"))
                        })
                elif existing_content and "items" in existing_content:
                    for item in existing_content["items"]:
                        initial_rows.append({
                            "تاریخ (Date)": doc_date,
                            "تفصیل آمدن (Description / Particulars)": str(item.get("item", "")),
                            "صفحہ (Folio)": "",
                            "رقم روپیہ (Amount Rs)": float(item.get("line_total", 0.0) or 0.0),
                            "نوعیت (Type)": "credit"
                        })
                else:
                    initial_rows = [
                        {"تاریخ (Date)": doc_date, "تفصیل آمدن (Description / Particulars)": "کیش", "صفحہ (Folio)": "", "رقم روپیہ (Amount Rs)": 0.0, "نوعیت (Type)": "credit"},
                        {"تاریخ (Date)": doc_date, "تفصیل آمدن (Description / Particulars)": "", "صفحہ (Folio)": "", "رقم روپیہ (Amount Rs)": 0.0, "نوعیت (Type)": "credit"}
                    ]

                df_template = pd.DataFrame(initial_rows)
                edited_df = st.data_editor(
                    df_template,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "تاریخ (Date)": st.column_config.TextColumn("تاریخ (Date)", required=False),
                        "تفصیل آمدن (Description / Particulars)": st.column_config.TextColumn("تفصیل آمدن (Description / Particulars)", required=True),
                        "صفحہ (Folio)": st.column_config.TextColumn("صفحہ (Folio)", required=False),
                        "رقم روپیہ (Amount Rs)": st.column_config.NumberColumn("رقم روپیہ (Amount Rs)", format="Rs. %.1f", required=True),
                        "نوعیت (Type)": st.column_config.SelectboxColumn("نوعیت (Type)", options=["credit", "subtotal", "deduction", "balance"])
                    },
                    key=f"editor_ledger_{selected_img}"
                )

                # Compute live totals
                calc_grand_total = 0.0
                if not edited_df.empty and "رقم روپیہ (Amount Rs)" in edited_df.columns:
                    calc_grand_total = float(edited_df["رقم روپیہ (Amount Rs)"].sum())

                tot_c1, tot_c2 = st.columns(2)
                with tot_c1:
                    st.metric("Total Rows", len(edited_df))
                with tot_c2:
                    st.metric("Sum of Amounts (میزان)", f"Rs. {calc_grand_total:,.2f}")

                # Save button
                if st.button("💾 Save Ground Truth to dataset/annotations.json", type="primary", use_container_width=True):
                    entries_to_save = []
                    for _, row in edited_df.iterrows():
                        desc = str(row.get("تفصیل آمدن (Description / Particulars)", "")).strip()
                        amt = float(row.get("رقم روپیہ (Amount Rs)", 0.0) or 0.0)
                        if desc or amt > 0:
                            entries_to_save.append({
                                "date": str(row.get("تاریخ (Date)", "")).strip(),
                                "description": desc,
                                "folio": str(row.get("صفحہ (Folio)", "")).strip(),
                                "amount": amt,
                                "type": str(row.get("نوعیت (Type)", "credit")).strip()
                            })

                    structured_doc = {
                        "document_type": doc_type,
                        "date": doc_date,
                        "khata_holder": doc_holder,
                        "entries": entries_to_save,
                        "grand_total": calc_grand_total
                    }
                    mgr.upsert_annotation(selected_img, structured_doc)
                    st.success(f"✅ Successfully saved {len(entries_to_save)} rows for `{selected_img}` to `dataset/annotations.json`!")
                    st.rerun()

            else:
                # Commodity bill format
                initial_items = []
                if existing_content and "items" in existing_content:
                    for it in existing_content["items"]:
                        initial_items.append({
                            "جنس (Item)": str(it.get("item", "")),
                            "تعداد (Qty)": float(it.get("quantity", 1.0)),
                            "اکائی (Unit)": str(it.get("unit", "من")),
                            "نرخ (Rate)": float(it.get("rate", 0.0)),
                            "کل رقم (Total)": float(it.get("line_total", 0.0))
                        })
                else:
                    initial_items = [
                        {"جنس (Item)": "چاول باسمتی", "تعداد (Qty)": 10.0, "اکائی (Unit)": "من", "نرخ (Rate)": 3500.0, "کل رقم (Total)": 35000.0}
                    ]

                df_items = pd.DataFrame(initial_items)
                edited_items = st.data_editor(
                    df_items,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "جنس (Item)": st.column_config.TextColumn("جنس (Item)", required=True),
                        "تعداد (Qty)": st.column_config.NumberColumn("تعداد (Qty)", min_value=0.0, step=1.0),
                        "اکائی (Unit)": st.column_config.SelectboxColumn("اکائی (Unit)", options=WHOLESALE_UNITS),
                        "نرخ (Rate)": st.column_config.NumberColumn("نرخ (Rate)", min_value=0.0, step=50.0),
                        "کل رقم (Total)": st.column_config.NumberColumn("کل رقم (Total)", format="Rs. %.1f")
                    },
                    key=f"editor_items_{selected_img}"
                )

                # Compute live line totals
                calc_items_total = 0.0
                if not edited_items.empty and "کل رقم (Total)" in edited_items.columns:
                    calc_items_total = float(edited_items["کل رقم (Total)"].sum())

                it_c1, it_c2 = st.columns(2)
                with it_c1:
                    st.metric("Total Items", len(edited_items))
                with it_c2:
                    st.metric("Grand Total (میزان)", f"Rs. {calc_items_total:,.2f}")

                if st.button("💾 Save Commodity Ground Truth", type="primary", use_container_width=True):
                    items_to_save = []
                    for _, row in edited_items.iterrows():
                        item_name = str(row.get("جنس (Item)", "")).strip()
                        if item_name:
                            q = float(row.get("تعداد (Qty)", 0.0) or 0.0)
                            r = float(row.get("نرخ (Rate)", 0.0) or 0.0)
                            tot = float(row.get("کل رقم (Total)", q * r) or (q * r))
                            items_to_save.append({
                                "item": item_name,
                                "quantity": q,
                                "unit": str(row.get("اکائی (Unit)", "من")),
                                "rate": r,
                                "line_total": tot
                            })

                    structured_doc = {
                        "document_type": doc_type,
                        "date": doc_date,
                        "khata_holder": doc_holder,
                        "items": items_to_save,
                        "grand_total": calc_items_total
                    }
                    mgr.upsert_annotation(selected_img, structured_doc)
                    st.success(f"✅ Successfully saved {len(items_to_save)} commodities for `{selected_img}`!")
                    st.rerun()

            if existing_content:
                with st.expander("🔍 View Active Saved Ground Truth JSON"):
                    st.json(existing_content)


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
