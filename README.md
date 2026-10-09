# Aasaan Khata (آسان کھاتہ) - FYP

**Aasaan Khata** is an offline Vision-Language Model (VLM) pipeline designed to transcribe raw, uncropped images of handwritten Urdu (Nastaliq) wholesale ledgers directly into mathematically audited JSON.

---

## 🏛️ System Architecture

1. **Vision-Language Core:**
   - **Model:** `Qwen2.5-VL-7B-Instruct`
   - **Quantization:** 4-bit NF4 (`BitsAndBytesConfig`) running locally within 6.1 GB VRAM on an **NVIDIA GeForce RTX 3080 (10 GB)**.
   - **Spatial Reasoning:** Direct whole-document instruction tuning (no OCR bounding boxes or crop heuristics).
   - **Resolution Guard:** Capped at `max_pixels: 602112` (~768×784) with `expandable_segments:True` to eliminate CUDA Out-Of-Memory spikes.

2. **Deterministic Math Validation Layer:**
   - The model is **not** relied upon for arithmetic calculations.
   - A dedicated Python layer (`math_engine.py`) strips visual artifacts (such as Pakistani currency slashes `/-`, commas, Urdu numerals `۰-۹`), verifies that:
     $$\text{Quantity} \times \text{Rate} = \text{Line Total}$$
     $$\sum \text{Line Totals} = \text{Grand Total}$$
   - Flags discrepancies and routes failed rows to the **Human-in-the-Loop (HITL)** interface.

3. **Human-in-the-Loop & Annotation UI:**
   - Built with Streamlit (`app.py`), injecting Google Font **Noto Nastaliq Urdu** with Right-to-Left (RTL) input styling.
   - Live arithmetic calculation comparing user or model inputs against dynamic totals.
   - Direct export to the Hugging Face / Qwen conversational fine-tuning schema.

4. **Cloud Fine-Tuning Pipeline:**
   - QLoRA fine-tuning scripts and Kaggle Notebook for training on pooled Kaggle GPUs (90 hrs/week).

---

## 📂 Project Structure

```text
├── 1.py                           # Original zero-shot testing script
├── app.py                         # Streamlit UI (Annotation Studio + HITL Review + Dataset Explorer)
├── dataset/                       # Ledger photos (1.jpeg - 6.jpeg) & exported annotations
│   ├── annotations.json           # Ground-truth verified annotations
│   ├── train.json                 # Training split for fine-tuning
│   └── val.json                   # Validation split
├── dataset_manager.py             # Annotation schema management & train/val splitting
├── inference_engine.py            # Production offline 4-bit Qwen2.5-VL inference wrapper
├── math_engine.py                 # Deterministic Math Validation Layer
├── qwen2.5-vl-7b/                 # Local model weights (~16.6 GB)
├── requirements.txt               # Locked Python dependencies
├── run_pipeline.py                # Command-line batch runner
├── torch_patch.py                 # PyTorch 2.5 accelerator monkey-patch
├── train_kaggle_qwen_vl.py        # Cloud / Kaggle QLoRA fine-tuning script
└── train_qwen_vl_lora_kaggle.ipynb # Ready-to-upload Kaggle fine-tuning notebook
```

---

## 🚀 Quickstart Guide

### 1. Launch the Streamlit Platform (HITL & Annotation)
```bash
./venv/bin/streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Tab 1: AI VLM Transcribe & HITL Audit:** Run offline inference on any image, view mathematical discrepancy alerts, auto-fix math, and approve.
- **Tab 2: Manual Ground Truth Studio:** Annotate ledgers with RTL Urdu typography, commodity chips, and live math.
- **Tab 3: Dataset Explorer & Split Export:** Export `train.json` and `val.json` for fine-tuning.

### 2. Run CLI Inference on a Single Image
```bash
./venv/bin/python3 run_pipeline.py --image dataset/1.jpeg
```
Outputs parsed transactions, audits line items, checks grand totals, and saves the verified record to `results/1_result.json`.

### 3. Run CLI Batch Inference on the Entire Dataset
```bash
./venv/bin/python3 run_pipeline.py --dir dataset --output results
```

### 4. Fine-Tune on Kaggle GPUs
Upload `train_qwen_vl_lora_kaggle.ipynb` along with `dataset/train.json` to Kaggle, or execute:
```bash
python3 train_kaggle_qwen_vl.py \
    --train_json dataset/train.json \
    --val_json dataset/val.json \
    --output_dir ./aasaan_khata_adapter \
    --epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```
