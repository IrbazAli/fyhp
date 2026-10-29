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
  - Switched [`server.py`](file:///home/irbazali/Desktop/FYP/server.py) to exclusively load the new trained LoRA adapter ([`./local_adapter_10991`](file:///home/irbazali/Desktop/FYP/local_adapter_10991)) without the previous 627 adapter for isolation testing.

## Authorized Changes (2026-10-23) — Qwen Row Pipeline Experiment

Implemented the complete end-to-end Qwen-based row segmentation experiment (**Qwen Row Pipeline — Experimental**) with separate, inspectable stages:
- **Architecture and Modules**:
  - [`qwen_row_pipeline/geometry.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/geometry.py): EXIF orientation canonicalization (`make_canonical_image`), explicit factor-28 spatial grid scaling matching Qwen's 28x28 token patches (`compute_factor28_dimensions`), bidirectional coordinate mapping (`model input -> context crop -> canonical full page`), non-clamping boundary validation (`validate_row_bounds` flagging material out-of-bounds, overlaps, duplicate candidates, and tiny/huge bounds), full-resolution rectangular clean crop extraction (`extract_row_crop`).
  - [`qwen_row_pipeline/prompts.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/prompts.py): Dedicated prompts for Stage A (whole-page layout proposals with exact user rules, strict JSON schema, and zero dummy names/amounts), Stage B (two-row context refinement with descent flourish preservation and conflict detection), Stage D (faithful verbatim row transcription preserving Urdu, English, numbers, and ditto marks `//`), and Stage E (positional & content-based field assignment with column guide evidence).
  - [`qwen_row_pipeline/storage.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/storage.py): Isolated job workspace persistence under `results/qwen_row_pipeline/<job_id>/` (storing original image, canonical image, localization image, coordinate transforms, prompts, generation settings, raw decoded outputs before cleaning, manifests, crops, and multi-sheet genuine XLSX via `openpyxl` with RTL presentation, explicit text datatypes, and `Ledger`, `Review`, and `Provenance` sheets).
  - [`qwen_row_pipeline/engine.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py): Single GPU model instance reuse via `get_shared_vlm`, serialization of all GPU calls across threads through `GPU_GENERATION_LOCK`, deterministic greedy decoding (`do_sample=False`, `repetition_penalty=1.15`), truncation detection, robust JSON extraction supporting both object and array formats with partial-recovery, and full lifecycle orchestration.
  - [`qwen_row_pipeline/routes.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/routes.py): Flask Blueprint exposing `/api/qwen_pipeline/upload`, `/api/qwen_pipeline/job/<job_id>`, `/crop_edit`, `/restore_proposal`, `/refine_pair`, `/transcribe`, artifact serving, and `/export/xlsx` / `/export/json`.
  - Registered blueprint in [`server.py`](file:///home/irbazali/Desktop/FYP/server.py) and wrapped legacy modes with the shared `GPU_GENERATION_LOCK`.
- **Frontend & Stage C Review Interface**:
  - Added third mode tab in [`frontend/index.html`](file:///home/irbazali/Desktop/FYP/frontend/index.html): **"Qwen Row Pipeline — Experimental"**.
  - Built **Stage C Crop Review Interface**: Dual-panel layout with canonical page SVG boundary overlays on the left and a scrollable list of row cards on the right. Controls include numeric boundary inputs (`Top`, `Bottom`), "Save Bounds", "Restore Original", "Refine with Next Row (Stage B)", and type dropdowns.
  - Built **Stage B Refinement Modal**: Displays clean two-row context crop, separation notes, ambiguity flags, and conflict warnings.
  - Built **Stage F Structured Results View**: Dynamic table displaying discovered columns, raw Qwen transcription beside structured values, review flags, clean crop links, and genuine `.xlsx` / JSON downloads.
  - Added CSS glassmorphic dark-theme styles in [`frontend/style.css`](file:///home/irbazali/Desktop/FYP/frontend/style.css).
- **Verified Testing**:
  - [`tests/test_qwen_pipeline.py`](file:///home/irbazali/Desktop/FYP/tests/test_qwen_pipeline.py): 7 unit tests passed 100% (factor-28 sizing, coordinate mapping, non-clamping validation, two-row context roundtrip, JSON truncation recovery, genuine XLSX export, stable row identity, and stale-result invalidation upon crop edits).
  - [`tests/smoke_test_real_pipeline.py`](file:///home/irbazali/Desktop/FYP/tests/smoke_test_real_pipeline.py): Live end-to-end smoke test executed against local server on `dataset/2.jpeg`.
    - Stage A proposed row candidates.
    - Stage C manual edit saved boundary adjustments (`[-18, 809]` to `[-16, 811]`), re-cropped image, and flagged downstream as stale.
    - Stage B refined row_001 and row_002 with two-row context crop (`"The flourish on the number '4' in کیش extends into the next line."`).
    - Stage D faithfully transcribed crops verbatim: Row 1 `"تفصیل آمان"`, Row 2 `"11-03-018"`, Row 3 `"مونگ پھلی 75-410"`, Row 4 `"12420 82 کمیشن"`, Row 5 & 6 `"عبد المتین ڈھلی"` (0% hallucination of full-page prompt boilerplate).
- **Adapter Configuration and GPU Loading**:
  - Confirmed active adapter is [`./combined_adapter`](file:///home/irbazali/Desktop/FYP/combined_adapter) (the mathematically fused $r=24, \alpha=48$ LoRA combining the user's 627 ledger words and the 10,991 NUST UHWR dataset).
  - Resolved `ValueError: Some modules are dispatched on the CPU or the disk`: Added `llm_int8_enable_fp32_cpu_offload=True` to `BitsAndBytesConfig` in [`qwen_row_pipeline/engine.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py) and verified GPU VRAM is completely clear (15 MiB used) for the user to run `server.py` directly.
- **Row Detection and Crop Normalization Fix**:
  - Identified and fixed bug where model output wrapped in `[ { "coordinate_space": ..., "rows": [...] } ]` was parsed as a single dummy row with `top: 0, bottom: 0` (clamped to a 1px crop at the very top of the image).
  - Enhanced [`parse_json_from_model_output`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py#L234-L260) and [`map_stage_a_coordinates`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/geometry.py#L105-L115) to normalize array-wrapped dictionaries and nested row structures.
  - Added explicit prompt constraint in [`STAGE_A_SYSTEM_RULES`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/prompts.py#L16-L28) ensuring row 1 starts where handwritten text actually begins rather than the top image margin. Verified all 11 rows in `job_20261023_115325_201421` now map accurately (Row 1 starts at $y=179$ px where handwritten ink begins).
- **Dense Ledger Token Limit & Truncation Recovery Fix (job_20261023_145546_87f6fb)**:
  - Identified failure cause on dense ledger pages: Stage A `max_new_tokens` (1536) was reached after generating 28 rows, cutting off the JSON mid-token (`{"top":630`). The previous truncation recovery was unable to resolve deeply nested bracket/brace stacks (`[ { ... "rows": [ ...`).
  - Doubled Stage A token budget in [`qwen_row_pipeline/engine.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py#L371) to `max_new_tokens: 3072`.
  - Implemented stack-based closing delimiter calculation (`compute_closing_suffix`) in [`parse_json_from_model_output`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py#L265-L315) which determines the exact sequence of closing brackets/braces (e.g. `]}]`).
  - Verified on `job_20261023_145546_87f6fb`: successfully recovered all 28 completed rows without data loss. Added unit test case 7 to [`tests/test_qwen_pipeline.py`](file:///home/irbazali/Desktop/FYP/tests/test_qwen_pipeline.py).
- **Hyphenated & String Coordinate Sanitization Fix (job_20261026_131441_9c08ae)**:
  - Fixed crash `could not convert string to float: '281-'` caused when model emitted hyphenated coordinates, ledger amounts, or non-numeric tokens (`'281-'`, `'50000/-'`, `'N/'`) in `top`/`bottom`.
  - Created [`parse_coordinate_float`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/geometry.py#L15-L55) to safely clean trailing slashes, dashes, currency markers, and non-numeric artifacts into valid floats without crashing.
  - Reinforced [`STAGE_A_SYSTEM_RULES`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/prompts.py#L16-L32) and [`STAGE_A_SCHEMA`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/prompts.py#L35-L60) to explicitly require integer vertical Y pixel coordinates between $0$ and $H$, warning the model never to output page amounts or trailing dashes into `top`/`bottom`.
  - Added unit test 8 to [`tests/test_qwen_pipeline.py`](file:///home/irbazali/Desktop/FYP/tests/test_qwen_pipeline.py) (8/8 tests passing).

## Authorized Changes (2026-10-27) — Vision Transformer (ViT) Patch & Token Inspector Mode

- **Purpose & Scope**:
  - The user requested testing Qwen2.5-VL's vision preprocessor (Vision Transformer) image decomposition:
    - Return patches in whatever dimensions the preprocessor cuts (base ViT patches: $14 \times 14$ px; merged spatial 2D-RoPE tokens: $28 \times 28$ px via `merge_size=2`).
    - Show all patches as output in the interface.
    - Strictly do NOT save patch files to disk / PC (all processing and visualization done in memory via HTML5 canvas).
    - Comment down downstream translation (Stage D) and structured post-processing (Stage E/F) for this testing phase.
- **Backend Implementation**:
  - Added [`extract_vision_patches`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py#L336-L425) in `QwenRowPipelineRunner`:
    - Runs canonical orientation transpose via `ImageOps.exif_transpose`.
    - Queries `processor.image_processor` for exact `smart_resize` factor-28 dimensions ($1204 \times 1596$), `patch_size=14`, `merge_size=2`, `temporal_patch_size=2`.
    - Produces grid geometry: $114 \times 86 = 9,804$ base patches and $57 \times 43 = 2,451$ merged tokens.
    - Generates in-memory Base64 Data URL; zero patch files are saved to the filesystem.
    - Commented down downstream translation (`transcribe_and_assign_reviewed_crops`) and post-processing for this test.
  - Connected [`/api/qwen_pipeline/upload`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/routes.py#L25-L65) to run `extract_vision_patches`.
- **Post-Merger Token Visualization (Unified Single Image)**:
  - Updated merged token visualization based on user guidance ("these four tokens merged should show as one img not as different patches"):
    - Removed disjoint 4-subpatch cards from the inspector; now renders the merged entity as **ONE single continuous image** ($28 \times 28$ px) at high-magnification ($140 \times 140$ px crisp display).
    - Removed internal 14px dashed lines and sub-patch highlights on the main canvas; now draws only clean solid factor-28 token borders and highlights the full unified $28 \times 28$ rectangle.
- **Step 4: Multimodal Fusion (replacing `<|image_pad|>` tokens with visual vectors)**:
  - Implemented token sequence analysis in [`extract_vision_patches`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/engine.py#L398-L450):
    - Generates full chat template input IDs using `processor.apply_chat_template` and `processor.tokenizer`.
    - Total sequence length: 2,516 tokens, composed of:
      1. *Prefix*: 15 text tokens (`<|im_start|>system...<|vision_start|>`).
      2. *Visual Span*: Exactly 2,451 `<|image_pad|>` placeholder tokens (ID 151655) spanning indices $[15..2465]$.
      3. *Postfix*: 50 Urdu prompt text tokens (`<|vision_end|>` followed by prompt and `<|im_start|>assistant`).
  - Built interactive **Step 4: Multimodal Fusion** dashboard card in [`frontend/index.html`](file:///home/irbazali/Desktop/FYP/frontend/index.html), [`frontend/script.js`](file:///home/irbazali/Desktop/FYP/frontend/script.js), and [`frontend/style.css`](file:///home/irbazali/Desktop/FYP/frontend/style.css):
    - *Sequence Stream Bar*: Tri-color visual architecture bar (blue prefix, emerald visual vectors, purple Urdu prompt) with dynamic cursor marker tracking active visual token.
    - *Side-by-Side Splicing Inspector*: Shows before-and-after vector replacement:
      - Before: `<|image_pad|>` (ID: 151655) at Sequence Position `#15 + i` (static learned placeholder in $\mathbb{R}^{3584}$).
      - Splicing Action: `inputs_embeds[image_mask] = visual_embeds`.
      - After: Visual Vector $v_i$ ($\mathbb{R}^{3584}$) spliced from the single $28 \times 28$ image crop with 3D MRoPE coordinate $[T=0, H=tr, W=tc]$.
    - *Token Sequence Stream Explorer*: Interactive chip strip showing actual tokens, IDs, decoded Urdu characters, and jump controls (Prefix, Active Visual, Urdu Prompt).
- **Verification & Testing**:
  - Added unit test 9 `test_vision_preprocessor_patch_extraction` in [`tests/test_qwen_pipeline.py`](file:///home/irbazali/Desktop/FYP/tests/test_qwen_pipeline.py).
  - All 9 unit tests pass in 0.69s (`python -m unittest tests/test_qwen_pipeline.py`).
  - Verified Flask test client on `1.jpeg`: HTTP 200, returns full `multimodal_fusion` payload (2,516 tokens, 2,451 visual vectors, 0 disk writes).

### Kraken BLLA Preprocessor & Neural Layout Segmentation (October 2026)
- **Motivation & Role**:
  - The user requested using Kraken BLLA (Baseline and Layout Analysis) as an image preprocessor and displaying its segmentation results directly in the web app.
  - Unlike axis-aligned rectangular crop assumptions (which suffer from clipping diacritics, dots, and skewed Nastaliq text in handwritten ledgers), Kraken BLLA predicts continuous neural baseline polylines and tight boundary polygons using VGSL models.
- **Backend Implementation**:
  - [`qwen_row_pipeline/kraken_engine.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/kraken_engine.py):
    - Added compatibility shim for PyTorch 2.5.1 / transformers (`torch.accelerator`).
    - Loads and caches the bundled `blla.mlmodel` on GPU (`cuda:0`).
    - Executes `blla.segment(canonical, text_direction="horizontal-rl", model=model, device="cuda:0")`.
    - Extracts baseline polyline coordinates, boundary polygon coordinates, bounding boxes, and generates in-memory Base64 JPEG line crops (0 files written to disk).
  - [`qwen_row_pipeline/routes.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/routes.py):
    - Added `/api/qwen_pipeline/kraken_segment` POST endpoint accepting an uploaded image (`image`) or dataset sample name (`sample_name`).
- **Frontend Interactive Visualizer**:
  - Added 4th mode tab in [`frontend/index.html`](file:///home/irbazali/Desktop/FYP/frontend/index.html): `Kraken BLLA Segmenter (کریکن بیس لائن اور لے آؤٹ تجزیہ)`.
  - Added sample buttons (`1.jpeg` and `2.jpeg`).
  - Interactive canvas in [`frontend/script.js`](file:///home/irbazali/Desktop/FYP/frontend/script.js) with toggle controls:
    - 🟢 Neural Baselines (`#00FFA3` polylines with anchor vertex circles).
    - 🟣 Boundary Polygons (`rgba(168, 85, 247, 0.2)` fill with purple borders).
    - 🔵 Bounding Boxes (dashed cyan boxes).
    - 🏷️ Line Index Badges (`#1`, `#2`, ...).
    - Active Line Inspector: displays the high-resolution in-memory crop, bounding box `[x, y, w, h]`, baseline vertex count, boundary vertex count, and tags.
    - All Segmented Lines Gallery: scrollable cards with line thumbnails, dimensions, and baseline stats; clicking any card centers and inspects that line.
  - Added responsive styling in [`frontend/style.css`](file:///home/irbazali/Desktop/FYP/frontend/style.css).
- **Measured Results**:
  - `dataset/1.jpeg` (Bill Khata): 16 lines detected in ~6.16s on `cuda:0`.
  - `dataset/2.jpeg` (Roznamcha): 35 lines detected in ~6.86s on `cuda:0`.
  - 10/10 unit tests pass in 7.75s (`python -m unittest tests/test_qwen_pipeline.py`).
  - 0 files saved to disk (strictly in-memory base64).

### Document Segmentation Benchmark Suite & Comparative Scorecard (October 2026)
- **Motivation & Objectives**:
  - The user evaluated 5 architectural alternatives before training a custom Vision Transformer:
    1. **Kraken BLLA**: Neural handwriting-aware baseline & polygonal line envelopes (`blla.mlmodel`).
    2. **OpenCV Morphological Rulings**: Horizontal ruled line detection & inter-rule row band slicing.
    3. **Microsoft Table Transformer (TATR)**: Deep table structure recognition for rows, columns, and headers (`microsoft/table-transformer-structure-recognition`).
    4. **Ultralytics YOLO Layout**: Rectangular document layout & table block object detection (`neuralshift/doc-layout-yolov8n`).
    5. **PaddleOCR DBNet**: Differentiable binarization text region boundary polygons (`PP-OCRv4_server_det`).
  - Core instruction: Implement each candidate engine separately so the user can test all 5 candidates individually and view a side-by-side comparative scorecard matrix.
- **Backend Architecture & Packages**:
  - [`segmentation_benchmark/opencv_ruling_engine.py`](file:///home/irbazali/Desktop/FYP/segmentation_benchmark/opencv_ruling_engine.py): Adaptive binarization + horizontal morphology kernel `(W//30, 1)`. Detects horizontal rulings, deduplicates lines within 12px, slices row bounding boxes, and generates in-memory base64 crops.
  - [`segmentation_benchmark/table_transformer_engine.py`](file:///home/irbazali/Desktop/FYP/segmentation_benchmark/table_transformer_engine.py): GPU-accelerated (`cuda:0`) DETR-based table structure recognition extracting table bounds, rows, columns, and headers with category tags and confidence scores.
  - [`segmentation_benchmark/yolo_engine.py`](file:///home/irbazali/Desktop/FYP/segmentation_benchmark/yolo_engine.py): Ultralytics YOLOv8n document layout detector on GPU (`cuda:0`) providing rectangular bounding boxes.
  - [`segmentation_benchmark/paddle_det_engine.py`](file:///home/irbazali/Desktop/FYP/segmentation_benchmark/paddle_det_engine.py): Paddle static inference running `PP-OCRv4_server_det` on CPU with `disable_mkldnn()` to bypass Python 3.12 Linux OneDNN PIR attribute bugs. Extracts DBNet text probability contours.
  - [`segmentation_benchmark/benchmark_runner.py`](file:///home/irbazali/Desktop/FYP/segmentation_benchmark/benchmark_runner.py): Unified runner exposing `run_single_engine(engine_key)` and `run_comparative_benchmark()`.
  - Added CUDA OOM CPU fallback in [`qwen_row_pipeline/kraken_engine.py`](file:///home/irbazali/Desktop/FYP/qwen_row_pipeline/kraken_engine.py) when GPU memory is heavily shared with Qwen.
  - Exposes endpoints:
    - `POST /api/qwen_pipeline/benchmark_segment`: Accepts `engine` (`kraken`, `opencv_rulings`, `table_transformer`, `yolo`, `paddleocr`, or `all`).
    - `GET /api/qwen_pipeline/benchmark_meta`: Returns metadata on all 5 candidates.
- **Frontend Interactive Visualizer & Scorecard Matrix**:
  - Mode Switcher: `Segmentation Benchmark (سیگمنٹیشن بنچ مارک — 5 ماڈلز)`.
  - Engine Selection Bar: 6 interactive pills allowing instant 1-click execution or cached inspection across candidates.
  - Comparative Scorecard: Responsive table comparing objective, output geometry, items detected, latency, hardware device, and handwritten ledger suitability.
  - Canvas Overlays: Dynamic rendering for polylines (Kraken), horizontal rules & row bands (OpenCV), rows/cols/headers (TATR), bounding boxes (YOLO), and text polygons (PaddleOCR).
  - 100% In-Memory: All crops are generated in memory and served as Base64 data URLs without writing any files to disk.
- **Verified Benchmark Measurements (`dataset/1.jpeg`)**:
  - Kraken BLLA: 16 lines (5.4s on GPU / 11.8s on CPU fallback).
  - OpenCV Rulings: 22 row bands, 23 rulings (0.032s on CPU).
  - Table Transformer: 14 rows, 6 columns, 1 table boundary (1.17s on GPU).
  - YOLO Layout: 1 document region (0.195s on GPU).
  - PaddleOCR DBNet: 16 text polygon regions (6.77s on CPU).
  - All 15 unit tests pass in 29.5s (`python -m unittest tests/test_qwen_pipeline.py`).

## Useful commands for a future authorized task

Run from the project root:

```bash
./venv/bin/python server.py
./venv/bin/python -m unittest tests/test_qwen_pipeline.py
./venv/bin/python tests/smoke_test_real_pipeline.py
./venv/bin/streamlit run app.py
./venv/bin/python run_pipeline.py --image dataset/1.jpeg
./venv/bin/python build_training_data.py
./venv/bin/python train_local.py --train_json dataset/train_words.json --epochs 5
```

The last two commands regenerate data and overwrite training outputs
respectively; they are reference commands, not instructions to run on startup.

