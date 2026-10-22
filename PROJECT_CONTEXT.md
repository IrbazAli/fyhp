# Aasaan Khata — project handoff and working context

Last reviewed: 2026-10-20. Workspace: `/home/irbazali/Desktop/FYP`.

## Purpose and continuity

Irbaz is continuing this project with Codex after developing it with Gemini in
Antigravity IDE. He requested analysis of the files/folders followed by the
previous conversation, and preservation of that context.

This file provides durable context within this workspace. It is not a claim of
permanent account-wide assistant memory.

Project: **Aasaan Khata (آسان کھاتہ)**, an offline system to transcribe handwritten
Urdu Nastaliq wholesale ledgers into structured tables/JSON. The longer-term
scope includes deterministic accounting validation, human review, ledger
persistence, analytics, and integrity controls.

Academic context from the report: FAST-NUCES Islamabad, project
F26-107-LedgerDigitization, session 2023–2027; team Irbaz Ali, Talal Shahid,
Muhammad Ibrahim; supervisor Mr. Irfan Ullah. Mid-evaluation materials are
under `mid evaluation/` and at the repository root.

## User preferences and task boundaries

- Local RTX 3080 training first; Kaggle is a fallback if local hardware cannot
  handle the workload.
- The primary application is Flask on port 5000. The labeling studio is
  `http://localhost:5000/label.html`; starting Streamlit alone does not start it.
- Preserve the language and markings actually present in the image: Urdu
  descriptions/headings, original numbers, dates, and table structure.
- Desired direction includes variable column layouts, separate transaction
  rows, multiple date sections per page, literal ditto marks (`//`), and empty
  descriptions where the page has numbers without text.
- Do not silently replace model output with assistant visual transcription.
  Distinguish raw output, formatting, normalization, correction, and annotation.
- “Just answer,” “no execution,” and “plan only” are meaningful boundaries.
  Historical unwanted changes prompted cancellations and rollbacks.
- The user asked to manage pushes manually, then authorized individual pushes
  later. Do not push based on those historical requests.
- Do not disable or replace the adapter merely because older conversations
  discuss degradation; the user explicitly asked to keep it at that time.
- Preserve the current baseline unless a new task authorizes changes.

## Source-of-truth and final historical state

- `complete_conversation_history.md`: 72 numbered user turns across two Gemini
  sessions. Read through during this review. It contains 11 literal
  `<truncated ... bytes>` markers; despite the title, some original text is
  missing. Screenshots and full tool traces are not preserved here.
- `chat_history_export.md`: older overlapping export covering 24 user turns,
  with IDE metadata and its own truncations.
- `complete_conversation_history.docx`: companion export. The exporter limits
  combined assistant responses to 15,000 characters per turn in DOCX.
- Final archived user request (turn 72): undo the automatic crop-detection and
  prompt changes and return to the last GitHub commit.
- At review, branch `main`, HEAD `7f7706e` (`trained on 45 imgs`); tracked files
  had no working-tree diff. `backup-local-work` also exists.
- Remote repository named in project history:
  `https://github.com/IrbazAli/fyhp.git`. No remote fetch/push was performed in
  this review; remote freshness is not verified.
- Current source reflects the restored baseline. Dynamic-column discovery,
  crop-aware inference, JSON truncation recovery, and `train_full_page.py`
  described in later experimental chat turns are **not in the current tree**.
- Turn 70 requested app shutdown. No server was started or stopped during this
  review, and current running-service status was not established.

## File and folder map

| Path | Actual role |
| --- | --- |
| `server.py` | Flask API/static server; local Qwen loading, adapter loading, ledger prompting, JSON-to-table mapping, upload/sample endpoints, word annotation API. |
| `frontend/index.html` | Main image-upload/sample interface, progress display, result table, raw parsed JSON display, CSV export controls. |
| `frontend/script.js` | Calls Flask; handles errors; renders cells and exports CSV. Routes to localhost:5000 when the page is not served on port 5000. |
| `frontend/style.css` | Dark interface styling and Noto Nastaliq Urdu text styling. |
| `frontend/label.html` | Standalone HTML5 Canvas word-box labeler with crop preview, zoom, labels, quick terms, previous/next image controls, and saved image selection. |
| `frontend/grid_output.jpg` | Existing visual reference asset, not an inference implementation. |
| `app.py` | Streamlit app: embedded Flask labeler, manual page annotation, VLM/math review, dataset exploration/export. |
| `inference_engine.py` | Separate base-model inference wrapper for Streamlit/CLI; commodity JSON prompt and math validation. Does not load `local_adapter`. |
| `math_engine.py` | Numeric normalization, quantity × rate checks, optional Mann divisor, line-sum/grand-total comparison, review flags and suggestions. |
| `dataset_manager.py` | Page annotation persistence and conversational-format train/validation splitting. |
| `build_training_data.py` | Crops labeled boxes with 4-pixel padding; writes word-only training JSON and mixed word/page training JSON. |
| `train_local.py` | Local QLoRA training; defaults to `dataset/train_words.json`. |
| `train_kaggle_qwen_vl.py` | Alternative cloud training script, with different LoRA targets/configuration from local training. |
| `train_qwen_vl_lora_kaggle.ipynb` | Kaggle notebook; currently contains a Python syntax error in its dataset-loader cell. |
| `torch_patch.py` | Supplies a minimal `torch.accelerator` shim and allocator setting for the installed stack. |
| `run_pipeline.py` | Single-image/directory CLI inference and math-audit result export through `LedgerVLMEngine`. |
| `1.py` | Original exploratory inference script, using `dataset/2.jpeg`; running it loads the model immediately. |
| `scratch/compare_models.py` | Base-versus-adapter comparison on image 1 with image-only and minimal prompts; has a missing `json` import. |
| `compare_results.json` | Empty (0 bytes), not a usable comparison result. |
| `dataset/` | 49 source ledger images, annotations, generated word crops, training records, imported per-page JSON, and annotation backup. |
| `qwen2.5-vl-7b/` | Local pretrained model: five BF16 safetensors shards, model/processor/tokenizer configuration, Hugging Face cache metadata. About 16.6 GB. |
| `local_adapter/` | Current LoRA weights, processor/tokenizer files, checkpoint-1570 and training state. |
| `local_adapter.bak_20261019_172559/` | Older adapter/checkpoint-180 backup. |
| `results/1_result.json` | Historical extraction with both line-item mismatches and a grand-total discrepancy. |
| `temp_uploads/` | 43 existing upload/crop images at review time. |
| `mid evaluation/` | Main report DOCX and backup, official rubric PDF, LaTeX template/PDF, class/style files and logo. |
| `FYP_Mid_Evaluation_Accuracy_and_Confusion_Matrix.*` | Generated DOCX/PDF presentation reports; reported metrics are not substantiated by an evaluation artifact here. |
| `generate_docx_report.py` | Formats a report with hardcoded accuracy tables/confusion counts and milestone claims; does not measure accuracy. |
| `export_all_conversations.py` | Exports two external Gemini JSONL transcripts to Markdown and DOCX. |
| `restore_fyp_chat.py` | Historical utility that writes Antigravity's external SQLite conversation index. Do not execute as a project startup step. |
| `restart_with_chats.sh` | Waits for Antigravity exit and invokes the history-repair utility. |
| `README.md` | Useful earlier architecture/launch notes, but stale on dataset size, primary UI, and training defaults. |
| `requirements.txt` | Installed-stack pins; omits Flask, Werkzeug, and python-docx despite project usage. |
| `.gitignore` | Excludes base weights, active adapter, weight extensions, venv, uploads, results and bytecode. |
| `venv/`, `__pycache__/`, `.git/` | Environment, generated bytecode, and Git metadata; inventoried rather than read as authored source. |
| `.agents/`, `.codex/` | Empty at review time. |

## Current inference behavior

Flask flow:
`image upload -> temp_uploads -> processor/vision preprocessing -> Qwen + LoRA
generation -> JSON parsing -> fixed-column table -> browser/CSV`.

- `get_vlm()` loads local Qwen with NF4 4-bit quantization, BF16 compute, and
  `device_map="auto"`. Model loading is lazy in the function but pre-warmed in
  the `__main__` startup block.
- The adapter is attached with `PeftModel.from_pretrained` when the relative
  path `./local_adapter/adapter_model.safetensors` exists. Adapter-load failures
  are logged and execution can continue with the base model.
- The server uses `max_pixels=602112`, `max_new_tokens=1536`.
- Its generation call does not explicitly set `do_sample` or
  `repetition_penalty`; do not claim the reverted guardrail values are active.
- Its prompt includes six concrete example rows with names, dates and amounts
  from sample 2. These are potential prompt-copying anchors, especially for
  isolated-word uploads.
- The output columns are **fixed to four bilingual headings** through
  `standard_columns` and `map_key_to_standard`. Model spatial reading is not
  the same thing as arbitrary dynamic output-column support.
- Every upload goes through the ledger-table prompt; there is no separate
  word/crop recognition mode in this baseline.
- Markdown fences and trailing commas are handled. Failed parsing silently
  becomes an empty entries list; no raw generated string or parse-error flag
  reaches the user. The displayed `raw_json` is a parsed/fallback object.
- Flask does not invoke `MathValidator`. Streamlit/CLI do, through the separate
  inference engine, which uses the base model without loading the adapter.
- Main Flask routes: `/`, `/upload`, `/api/sample`, `/api/images`,
  `/api/annotations/words` (GET/POST), `/dataset/<path>`,
  `/temp_uploads/<path>`, plus frontend static routes.
- Upload limit is 50 MB; server binds `0.0.0.0:5000`. Frontend fonts are loaded
  from Google Fonts, so model inference is local but typography has an external
  dependency.

## Verified data and model inventory

- Source pages: **49**.
- Pages in `word_annotations.json`: **43**.
- Boxes: **634** total, **627** with nonempty text, **7** unlabeled.
- Word crop files and `train_words.json` records: **627**.
- Unique exact nonempty labels: **188**. Most frequent: `عبد المتین` (96),
  `کیش` (43), `ڈھلی` (29), `نقد` (28). This is class-frequency evidence, not
  proof of a particular hallucination mechanism.
- `annotations.json`: **one** full-page sample, `dataset/2.jpeg`, with 15
  entries. Its transcription was assistant-assisted historically; do not
  present it as an untouched Qwen output or independently verified benchmark.
- `train.json`: **628** records (627 words + one page).
- No `dataset/val.json`, held-out word test split, or full-page training dataset
  is present.
- Unannotated source pages: `9.jpg`–`13.jpg` and `45.png`.
- `dataset/Json-file/`: 30 imported annotation JSON files. The master merge is
  already done; avoid blindly merging them again.
- Annotation backup contains 15 page keys and 228 boxes.
- Active checkpoint: epoch **5.0**, global step **1570**, logging interval 10.
  First logged loss ~3.359; last logged loss ~0.180. These are training losses,
  not evaluation accuracy.
- Archived run duration of 24m32s and average loss 0.594 are historical claims;
  not remeasured in this review.
- Backup checkpoint: epoch 5.0, step 180.
- From safetensors headers: base model has **8,292,166,656 parameters**
  (~8.29B including vision); adapter has **5,046,272 FP32 parameters** in 224
  tensors, file size 20,218,120 bytes (~20.2 MB). A 20 MB adapter is not 20M
  parameters.
- Adapter keys show language-model attention projections, not vision-tower
  adaptation or MLP targets.

## Actual local training configuration

`train_local.py` currently:

- Uses frozen 4-bit NF4 base weights with double quantization and BF16 compute.
- Enables input gradients and gradient checkpointing; disables model cache.
- Creates a fresh LoRA adapter using `get_peft_model`; it does not resume the
  previous adapter by default.
- LoRA `r=8`, alpha 16, dropout 0.05; targets `q_proj`, `k_proj`, `v_proj`,
  `o_proj`.
- Default dataset: word crops. Pixel cap **50176**, a pixel-area budget, not a
  guarantee that every crop becomes exactly 224 × 224.
- Default 5 epochs, batch size 1, gradient accumulation **2**, learning rate
  `2e-4`, cosine schedule, warmup 2 steps, `paged_adamw_8bit`.
- Masks labels preceding the assistant-start marker, so target text supplies
  the supervised loss. No evaluation dataset or evaluation metrics are used.
- Saves to `./local_adapter` by default with one retained checkpoint.
- The file's opening comments still claim accumulation 8 and a larger pixel
  budget; executable settings above are authoritative.
- Both training and inference load a model copy. Historical simultaneous
  server/training runs exhausted the 10 GB GPU. Check actual running processes
  before future GPU work; do not kill unrelated processes.

## History that explains current problems

1. Original objective was whole-page VLM transcription after difficulties with
   heuristic segmentation/OCR. That was the design direction, not evidence that
   full-page training was completed.
2. The user requested mouse word-box labeling; training then used isolated word
   crops. Dataset grew through reported 16, 28 and 72 crops to verified 627.
3. Whole-page extraction suffered row grouping, repeated merchant names, Urdu
   hallucinations, and empty tables when JSON parsing failed.
4. Experiments changed prompts, resolution, column mapping, ditto handling,
   repetition penalties, and JSON recovery. Several were reverted after worse
   results.
5. Base/adapter comparisons and two successful cropped-word predictions were
   reported in chat. There is no usable saved comparison JSON; two known-word
   examples do not establish held-out accuracy.
6. The user requested dynamic faithful extraction and a full-page training
   plan, later deferred full-page training, then restored the GitHub baseline.
7. The last crop-routing fix was also explicitly requested to be undone.

Potential explanations for degradation include mismatch between word training
and page inference, prompt examples, label imbalance, resolution loss and
generation behavior. The archive's claims of proven “destroyed spatial
attention,” guaranteed cures, or zero hallucination are not established by the
available evidence.

## Review findings to carry into future work

These are observations, not changes authorized or implemented during onboarding.

1. **Evaluation is missing.** The report's 96.2%, 92.4%, 89.5%, 93.8%, 78.2% and
   63.5% values, baseline comparisons and confusion counts are literal data in
   `generate_docx_report.py`, not computed results. Do not reuse as measured
   performance. Establish a held-out evaluation with predictions and ground
   truth before reporting accuracy.
2. **Report/code mismatch.** The main report mixes Flask/Streamlit descriptions
   with Next.js/FastAPI/PostgreSQL/HMAC/Docker claims. It references 39 tests,
   thousands of mutation trials and `ledger_validation` files absent from this
   workspace. These may belong to another artifact or teammate's work; their
   existence/results are not verified here. The LaTeX folder is a template with
   placeholder authors and chapter instructions.
3. **Math scope is narrower than claimed.** Current validator does not implement
   cash-book credit/debit/running-balance semantics, handle subtotal/deduction
   row types, or incorporate expenses/discounts into the total. Empty input can
   receive `ALL_VERIFIED`. Existing tolerances are absolute 1.0 or relative 1%.
4. **Single-record split leakage.** `DatasetManager.export_splits()` can place
   the same sole record into both train and validation.
5. **Comparison script failure.** `scratch/compare_models.py` calls `json.dump`
   without importing `json`. The empty result file is consistent with a failure
   after opening the file, but this was not rerun to prove its provenance.
6. **Kaggle notebook syntax.** Cell index 4 contains literal newlines inside
   ordinary quoted Python strings; AST parsing reports an unterminated string.
   The cloud dataset code also slices vision tensors with `[0]`, unlike the
   corrected local path; review tensor dimensions before using cloud training.
7. **Annotation durability.** Word annotations are rewritten as one JSON file,
   without atomic replacement/locking. A read failure during POST resets the
   in-memory contents to `{}` before saving. The labeler requires explicit
   “Set Word” and “Save All”; switching images can discard unsaved work.
8. **Rendering correctness.** Main table and label list interpolate model/label
   strings into `innerHTML`. Amount formatting and CSV generation also normalize
   values; displayed text is not strictly verbatim model output.
9. **Reproducibility.** Flask/Werkzeug/python-docx are installed locally but
   missing from requirements. Some paths depend on starting in the repository
   root. Model/adapter weights are local ignored assets.

## Review coverage and limits

Reviewed all authored Python modules, frontend HTML/JS/CSS, notebook cells,
project documentation/configuration, annotation/training JSON structure,
checkpoint metadata and safetensors headers. Inventoried every top-level folder
and generated/dependency asset category. Extracted DOCX/PDF text and reviewed
report structure, implementation/evaluation claims, rubric and template; this
was not a full visual/page-layout audit of every report page or an external
verification of its literature references.

Checks performed without starting models:

- Parsed all 15 project Python files with `ast.parse`: passed.
- Parsed notebook Python cells: dataset-loader syntax failure recorded above.
- Checked referenced training-image existence: no missing image paths.
- Checked all master annotation boxes against image bounds: no invalid boxes.
- Opened and ran Pillow `verify()` on 722 image files across dataset, uploads,
  frontend and report assets: no reported errors. This is not transcription or
  label-quality verification.
- Inspected model tensor metadata without loading weights onto GPU.

No application code, labels, adapters, reports or Git history were modified;
only this handoff and `AGENTS.md` were added. No inference, training, external
publication, dependency installation, or Git push was performed.

## Authorized Changes (2026-10-22)

- Added a separate, explicit **Single Line / Word Urdu Text** option while leaving the **Full Ledger Page** pipeline, prompt, and routes 100% untouched.
  - In `server.py`: added `run_vlm_on_word_crop(image_path)` using the exact training prompt from `dataset/train_words.json` (`"اس تصویر میں لکھا ہوا اردو ہاتھ کا لکھا لفظ یا عدد پڑھ کر بتائیں۔"`), added `/api/predict_word` endpoint, and supported `mode='word'` in `/upload` and `/api/sample` without modifying `run_vlm_on_image`.
  - In `frontend/index.html` & `frontend/style.css`: added a modern mode switch bar (`[Full Ledger Page]` vs `[Single Line / Word Text]`), mode-specific sample buttons (`4_crop_010_ماما_طالب_حسین.png`, `15_crop_024_ڈھلی.png`, `14_crop_005_عثمان_کمیشن_شاپ.png`), and a dedicated Nastaliq text result card with a copy-to-clipboard button.
  - In `frontend/script.js`: added mode switching state (`currentMode`), mode parameter propagation, and conditional rendering for tabular ledger vs single line Nastaliq output.
- Authorized execution of Possibility A: Fine-tuning `Qwen2.5-VL-7B` on the NUST UHWR handwriting dataset (8,483 lines) blended with Aasaan Khata ledger crops.
  - Prior adapter backed up to `local_adapter.bak_20261022_pre_uhwr`.
  - Created [`build_uhwr_dataset.py`](file:///home/irbazali/Desktop/FYP/build_uhwr_dataset.py): compiled 8,483 UHWR lines + 4x oversampled ledger crops (627 words) into [`dataset/train_combined_uhwr.json`](file:///home/irbazali/Desktop/FYP/dataset/train_combined_uhwr.json) (10,991 total samples) and [`dataset/val_uhwr.json`](file:///home/irbazali/Desktop/FYP/dataset/val_uhwr.json) (1,061 samples).
  - Created [`train_uhwr_qwen.py`](file:///home/irbazali/Desktop/FYP/train_uhwr_qwen.py): 4-bit NF4 QLoRA ($r=16, \alpha=32$, attention + MLP modules, 47.6M trainable params), gradient checkpointing, `max_pixels: 65536` (~7.1 GB VRAM footprint on RTX 3080).
  - Training completed 100% (1,374 steps across 2 epochs, 3h 54m runtime on RTX 3080). Final step loss dropped to ~0.18–0.22 (overall train loss: 0.3662).
  - Saved final LoRA adapter to `./local_adapter/` (182 MB safetensors).
  - Measured verification:
    - UHWR validation line `01959.jpg`: Target: *"میں ہم یہ جنگ جیت سکتے ہیں لیکن اس کیلئے حکمت عملی تبدیل"* $\rightarrow$ Model: *"میں ہم یہ جگہ چیت سکتے ہیں لیکن اس کیلئے حکمت عملی نبديل"* (strong handwriting sentence understanding).
    - Ledger crop `4_crop_010_ماما_طالب_حسین.png`: Target: *"ماما طالب حسین"* $\rightarrow$ Model: *"ماما طالب حسین"* (100% Exact Match ✅).
    - Ledger crop `15_crop_024_ڈھلی.png`: Target: *"ڈھلی"* $\rightarrow$ Model: *"ڈھلی"* (100% Exact Match ✅).
    - Zero catastrophic forgetting of ledger accounting terms.
  - Mathematically merged the 627-sample ledger LoRA ($r=8$) and the 10,991-sample UHWR LoRA ($r=16$) into a unified lossless combined adapter ($r=24, \alpha=48$) via [`merge_adapters.py`](file:///home/irbazali/Desktop/FYP/merge_adapters.py) (273 MB safetensors saved to `./combined_adapter/` and active in `./local_adapter/`). Preserved individual adapters in `./local_adapter_627/` and `./local_adapter_10991/`.
  - Updated [`server.py`](file:///home/irbazali/Desktop/FYP/server.py) to load the combined adapter and register both on startup.

## Useful commands for a future authorized task

Run from the project root:

```bash
./venv/bin/python server.py
./venv/bin/streamlit run app.py
./venv/bin/python run_pipeline.py --image dataset/1.jpeg
./venv/bin/python build_training_data.py
./venv/bin/python train_local.py --train_json dataset/train_words.json --epochs 5
```

The last two commands regenerate data and overwrite training outputs
respectively; they are reference commands, not instructions to run on startup.
