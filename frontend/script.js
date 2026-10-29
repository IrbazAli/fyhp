document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');
    const errorBanner = document.getElementById('error-banner');
    const errorBannerText = document.getElementById('error-banner-text');
    const imageDisplay = document.getElementById('image-display');
    const thead = document.getElementById('results-thead');
    const tbody = document.getElementById('results-body');
    const rawJsonDisplay = document.getElementById('raw-json-display');
    const statsSpan = document.getElementById('detected-stats');

    // Mode-specific elements
    const btnModeLedger = document.getElementById('btn-mode-ledger');
    const btnModeWord = document.getElementById('btn-mode-word');
    const btnModePipeline = document.getElementById('btn-mode-pipeline');
    const btnModeKraken = document.getElementById('btn-mode-kraken');
    const uploadTitle = document.getElementById('upload-title');
    const uploadSubtitle = document.getElementById('upload-subtitle');
    const samplesLedger = document.getElementById('samples-ledger');
    const samplesWord = document.getElementById('samples-word');
    const samplesPipeline = document.getElementById('samples-pipeline');
    const samplesKraken = document.getElementById('samples-kraken');

    // Segmentation Benchmark UI elements
    const krakenSection = document.getElementById('kraken-section');
    const benchmarkEngineSelectorBox = document.getElementById('benchmark-engine-selector-box');
    const benchmarkScorecardSection = document.getElementById('benchmark-scorecard-section');
    const benchmarkScorecardTbody = document.getElementById('benchmark-scorecard-tbody');
    const scorecardTotalTime = document.getElementById('scorecard-total-time');
    const benchmarkSingleHeader = document.getElementById('benchmark-single-header');
    const benchmarkEngineTitle = document.getElementById('benchmark-engine-title');
    const benchmarkReviewGrid = document.getElementById('benchmark-review-grid');
    const benchmarkEngineNote = document.getElementById('benchmark-engine-note');
    const krakenMainCanvas = document.getElementById('kraken-main-canvas');
    const krakenActiveCrop = document.getElementById('kraken-active-crop');
    const krakenActiveLineTitle = document.getElementById('kraken-active-line-title');
    const krakenLinesList = document.getElementById('kraken-lines-list');
    const krakenModelBadge = document.getElementById('kraken-model-badge');
    const krakenLinesBadge = document.getElementById('kraken-lines-badge');
    const krakenDeviceBadge = document.getElementById('kraken-device-badge');
    const krakenLatencyBadge = document.getElementById('kraken-latency-badge');
    const krakenDirBadge = document.getElementById('kraken-dir-badge');
    const krakenHoverCoords = document.getElementById('kraken-hover-coords');
    const krakenMetaLineId = document.getElementById('kraken-meta-line-id');
    const krakenMetaBbox = document.getElementById('kraken-meta-bbox');
    const krakenMetaBaselinePts = document.getElementById('kraken-meta-baseline-pts');
    const krakenMetaBoundaryPts = document.getElementById('kraken-meta-boundary-pts');
    const krakenMetaTags = document.getElementById('kraken-meta-tags');
    const krakenGalleryCount = document.getElementById('kraken-gallery-count');
    const btnToggleBaselines = document.getElementById('btn-toggle-baselines');
    const btnTogglePolygons = document.getElementById('btn-toggle-polygons');
    const btnToggleBBoxes = document.getElementById('btn-toggle-bboxes');
    const btnToggleLabels = document.getElementById('btn-toggle-labels');

    // Pipeline UI elements
    const pipelineReviewSection = document.getElementById('pipeline-review-section');
    const pipelineResultsSection = document.getElementById('pipeline-results-section');
    const stageBModal = document.getElementById('stage-b-modal');
    const reviewCanonicalImg = document.getElementById('review-canonical-img');
    const reviewOverlaySvg = document.getElementById('review-overlay-svg');
    const rowsScrollContainer = document.getElementById('rows-scroll-container');
    const pipelineJobIdBadge = document.getElementById('pipeline-job-id');
    const pipelineModelBadge = document.getElementById('pipeline-model-badge');
    const pipelineRowsCountBadge = document.getElementById('pipeline-rows-count');
    const reviewActiveCount = document.getElementById('review-active-count');
    const btnTranscribeCrops = document.getElementById('btn-transcribe-crops');
    const pipelineThead = document.getElementById('pipeline-thead');
    const pipelineTbody = document.getElementById('pipeline-tbody');
    const btnDownloadXlsx = document.getElementById('btn-download-xlsx');
    const btnDownloadJson = document.getElementById('btn-download-json');
    const stageBContextImg = document.getElementById('stage-b-context-img');
    const stageBNotes = document.getElementById('stage-b-notes');
    const stageBModalTitle = document.getElementById('stage-b-modal-title');

    // Progress step descriptions
    const step1Title = document.getElementById('step-1-title');
    const step1Desc = document.getElementById('step-1-desc');
    const step2Title = document.getElementById('step-2-title');
    const step2Desc = document.getElementById('step-2-desc');
    const step3Title = document.getElementById('step-3-title');
    const step3Desc = document.getElementById('step-3-desc');

    // Results layout elements
    const resultsTitle = document.getElementById('results-title');
    const previewTitle = document.getElementById('preview-title');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const tableCardContainer = document.getElementById('table-card-container');
    const wordCardContainer = document.getElementById('word-card-container');
    const wordDisplayText = document.getElementById('word-display-text');
    const wordMetaInfo = document.getElementById('word-meta-info');

    // Mode state: 'pipeline' (default for patch/token merger testing), 'ledger', or 'word'
    let currentMode = 'pipeline';
    let activeJobId = null;
    let activeManifest = null;

    // Auto-detect backend: if opened from file:// or external port, route to local GPU port 5000
    const isPort5000 = (window.location.protocol.startsWith('http') && window.location.port === '5000');
    const BACKEND_BASE = isPort5000 ? '' : 'http://localhost:5000';

    let currentTableData = { columns: [], rows: [] };

    // Mode switching function
    window.switchMode = function(mode) {
        currentMode = (mode === 'word') ? 'word' : (mode === 'pipeline' ? 'pipeline' : (mode === 'kraken' ? 'kraken' : 'ledger'));

        if (btnModeLedger) btnModeLedger.classList.remove('active');
        if (btnModeWord) btnModeWord.classList.remove('active');
        if (btnModePipeline) btnModePipeline.classList.remove('active');
        if (btnModeKraken) btnModeKraken.classList.remove('active');

        if (samplesLedger) samplesLedger.classList.add('hidden');
        if (samplesWord) samplesWord.classList.add('hidden');
        if (samplesPipeline) samplesPipeline.classList.add('hidden');
        if (samplesKraken) samplesKraken.classList.add('hidden');
        if (benchmarkEngineSelectorBox) benchmarkEngineSelectorBox.classList.add('hidden');

        if (currentMode === 'word') {
            if (btnModeWord) btnModeWord.classList.add('active');
            if (uploadTitle) uploadTitle.textContent = "Upload Single Line / Word Crop";
            if (uploadSubtitle) uploadSubtitle.textContent = "Drag & drop cropped handwritten Urdu image or browse local files";
            if (samplesWord) samplesWord.classList.remove('hidden');
        } else if (currentMode === 'pipeline') {
            if (btnModePipeline) btnModePipeline.classList.add('active');
            if (uploadTitle) uploadTitle.textContent = "Qwen2.5-VL Vision Transformer & Patch Merger Output";
            if (uploadSubtitle) uploadSubtitle.textContent = "Extract in-memory 14×14 ViT patches and 28×28 merged visual tokens (0 disk files)";
            if (samplesPipeline) samplesPipeline.classList.remove('hidden');
        } else if (currentMode === 'kraken') {
            if (btnModeKraken) btnModeKraken.classList.add('active');
            if (uploadTitle) uploadTitle.textContent = "Document Segmentation Benchmark (5 Candidate Engines)";
            if (uploadSubtitle) uploadSubtitle.textContent = "Benchmark Kraken BLLA, OpenCV Rulings, Table Transformer, YOLO, and PaddleOCR on your ledger";
            if (benchmarkEngineSelectorBox) benchmarkEngineSelectorBox.classList.remove('hidden');
            if (samplesKraken) samplesKraken.classList.remove('hidden');
        } else {
            if (btnModeLedger) btnModeLedger.classList.add('active');
            if (uploadTitle) uploadTitle.textContent = "Upload Ledger Image";
            if (uploadSubtitle) uploadSubtitle.textContent = "Drag & drop raw photograph or browse local files";
            if (samplesLedger) samplesLedger.classList.remove('hidden');
        }
        hideError();
    };


    // Drag and Drop listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            startUpload(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            startUpload(e.target.files[0]);
        }
    }

    function showError(msg) {
        if (errorBanner && errorBannerText) {
            errorBannerText.textContent = msg;
            errorBanner.classList.remove('hidden');
        } else {
            alert(msg);
        }
    }

    function hideError() {
        if (errorBanner) {
            errorBanner.classList.add('hidden');
        }
    }

    async function parseJsonResponse(response) {
        const rawText = await response.text();
        let parsed;
        try {
            parsed = JSON.parse(rawText);
        } catch (e) {
            let snippet = rawText.trim().replace(/<[^>]*>?/gm, ' ').slice(0, 120);
            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status} (${response.statusText}): ${snippet}`);
            }
            throw new Error(`Invalid non-JSON response from ${response.url}: ${snippet}`);
        }
        if (!response.ok) {
            throw new Error(parsed.error || `HTTP ${response.status} Error`);
        }
        return parsed;
    }

    window.loadSample = async function(sampleName) {
        hideError();
        if (currentMode === 'pipeline') {
            await startPipelineJob(null, sampleName);
            return;
        }
        if (currentMode === 'kraken') {
            await startBenchmarkSegmentation(null, sampleName, currentBenchmarkEngine);
            return;
        }

        startPipelineUI();
        updateStep(1, 'active');

        try {
            updateStep(1, 'completed');
            updateStep(2, 'active');

            const url = `${BACKEND_BASE}/api/sample?name=${encodeURIComponent(sampleName)}&mode=${currentMode}`;
            const response = await fetch(url);
            const data = await parseJsonResponse(response);

            updateStep(2, 'completed');
            updateStep(3, 'active');
            await sleep(400);
            updateStep(3, 'completed');

            showResults(data);
        } catch (err) {
            console.error(err);
            showError("Failed to process sample: " + err.message);
            resetApp();
        }
    };

    async function startUpload(file) {
        hideError();
        if (currentMode === 'pipeline') {
            await startPipelineJob(file, null);
            return;
        }
        if (currentMode === 'kraken') {
            await startBenchmarkSegmentation(file, null, currentBenchmarkEngine);
            return;
        }

        startPipelineUI();
        updateStep(1, 'active');

        const formData = new FormData();
        formData.append('image', file);
        formData.append('mode', currentMode);

        try {
            updateStep(1, 'completed');
            updateStep(2, 'active');

            const url = `${BACKEND_BASE}/upload`;
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            const data = await parseJsonResponse(response);

            updateStep(2, 'completed');
            updateStep(3, 'active');
            await sleep(400);
            updateStep(3, 'completed');

            showResults(data);
        } catch (error) {
            console.error('Pipeline Error:', error);
            showError("Failed to transcribe uploaded image: " + error.message);
            resetApp();
        }
    }

    function startPipelineUI() {
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        if (pipelineReviewSection) pipelineReviewSection.classList.add('hidden');
        if (pipelineResultsSection) pipelineResultsSection.classList.add('hidden');
        if (krakenSection) krakenSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        progressSection.classList.add('fade-in');

        // Dynamically adjust step labels for current mode
        if (currentMode === 'word') {
            if (step1Title) step1Title.textContent = "Word Crop Optimization";
            if (step1Desc) step1Desc.textContent = "Scaling visual crop tokens for fine-tuned Nastaliq recognition";
            if (step2Title) step2Title.textContent = "Fine-Tuned Nastaliq LoRA Pass";
            if (step2Desc) step2Desc.textContent = "Recognizing handwritten Urdu characters, dots, and ligatures";
            if (step3Title) step3Title.textContent = "Urdu Text Decoding";
            if (step3Desc) step3Desc.textContent = "Synthesizing authentic Nastaliq text representation";
        } else if (currentMode === 'pipeline') {
            if (step1Title) step1Title.textContent = "Resolution Scaling & Token Geometry";
            if (step1Desc) step1Desc.textContent = "Computing factor-28 resolution & spatial grid boundaries";
            if (step2Title) step2Title.textContent = "Vision Transformer Preprocessor";
            if (step2Desc) step2Desc.textContent = "Extracting base 14×14 patches & 28×28 spatial tokens in-memory";
            if (step3Title) step3Title.textContent = "Patch Coordinate Mapping";
            if (step3Desc) step3Desc.textContent = "Organizing grid tiles without saving any files to disk";
        } else if (currentMode === 'kraken') {
            if (step1Title) step1Title.textContent = "Document Preprocessing & Canonical Scaling";
            if (step1Desc) step1Desc.textContent = "Validating in-memory channels & layout coordinates";
            if (step2Title) step2Title.textContent = (currentBenchmarkEngine === 'all')
                ? "Running 5-Engine Comparative Benchmark"
                : `Executing ${currentBenchmarkEngine.replace('_', ' ').toUpperCase()} Engine`;
            if (step2Desc) step2Desc.textContent = "Evaluating line boundaries, rulings, and layout geometry";
            if (step3Title) step3Title.textContent = "In-Memory Item Crops & Scorecard";
            if (step3Desc) step3Desc.textContent = "Generating segmentation geometry and crops (0 disk writes)";
        } else {
            if (step1Title) step1Title.textContent = "Resolution Capping & Token Allocation";
            if (step1Desc) step1Desc.textContent = "Capping visual grid tokens to prevent CUDA VRAM spikes";
            if (step2Title) step2Title.textContent = "Qwen2.5-VL-7B Forward Pass";
            if (step2Desc) step2Desc.textContent = "Offline 4-bit multi-row spatial layout reasoning on RTX 3080";
            if (step3Title) step3Title.textContent = "Tabular Ledger Reconstruction";
            if (step3Desc) step3Desc.textContent = "Extracting dynamic columns and cell records directly";
        }

        resetSteps();
    }


    function updateStep(stepNum, status) {
        const step = document.getElementById(`step-${stepNum}`);
        if (!step) return;
        const statusText = step.querySelector('.step-status');
        step.classList.remove('active', 'completed');

        if (status === 'active') {
            step.classList.add('active');
            statusText.textContent = 'Running...';
        } else if (status === 'completed') {
            step.classList.add('completed');
            statusText.textContent = 'Completed';
            step.querySelector('.step-indicator').innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00FFA3" stroke-width="2.5">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;
        }
    }

    function resetSteps() {
        [1, 2, 3].forEach(i => {
            const step = document.getElementById(`step-${i}`);
            if (!step) return;
            step.classList.remove('active', 'completed');
            step.querySelector('.step-status').textContent = 'Pending';
            step.querySelector('.step-indicator').textContent = i;
        });
    }

    function formatAmount(raw) {
        if (!raw) return '<span class="empty-dash">—</span>';
        let s = String(raw).trim();
        if (s === '' || s === '-' || s === '—') return '<span class="empty-dash">—</span>';
        let numStr = s.replace(/[\/\-=]/g, '').replace(/,/g, '').trim();
        let num = parseFloat(numStr);
        if (!isNaN(num)) {
            let formatted = num.toLocaleString('en-US');
            return `<span class="amount-val">${formatted}/-</span>`;
        }
        return `<span class="amount-val">${s}</span>`;
    }

    function showResults(data) {
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('fade-in');

        // Display image
        if (data.image_url) {
            const imgUrl = (data.image_url.startsWith('http')) 
                ? data.image_url 
                : (BACKEND_BASE + data.image_url);
            imageDisplay.src = imgUrl + "?t=" + new Date().getTime();
        }

        const isWordMode = (data.mode === 'word' || (currentMode === 'word' && data.text !== undefined));

        if (isWordMode) {
            // Render Single Line / Word Result
            if (resultsTitle) resultsTitle.textContent = "Recognized Urdu Text (ہاتھ کا لکھا اردو متن)";
            if (previewTitle) previewTitle.textContent = "Cropped Word / Line Patch";
            if (btnExportCsv) btnExportCsv.classList.add('hidden');
            if (statsSpan) statsSpan.textContent = "(Single Line / Word Mode)";

            if (tableCardContainer) tableCardContainer.classList.add('hidden');
            if (wordCardContainer) wordCardContainer.classList.remove('hidden');

            const textOutput = (data.text !== undefined && data.text !== null) ? String(data.text).trim() : '';
            if (wordDisplayText) {
                wordDisplayText.textContent = textOutput || "—";
            }
            if (wordMetaInfo) {
                wordMetaInfo.textContent = `Length: ${textOutput.length} characters | Model: Qwen2.5-VL-7B + Local Nastaliq Adapter`;
            }
        } else {
            // Render Full Ledger Page Table
            if (resultsTitle) resultsTitle.textContent = "Extracted Ledger Table";
            if (previewTitle) previewTitle.textContent = "Original Photograph";
            if (btnExportCsv) btnExportCsv.classList.remove('hidden');

            if (tableCardContainer) tableCardContainer.classList.remove('hidden');
            if (wordCardContainer) wordCardContainer.classList.add('hidden');

            const columns = data.columns || [];
            const rows = data.table || [];
            currentTableData = { columns, rows };

            statsSpan.textContent = `(${columns.length} Columns, ${rows.length} Rows)`;

            // Build Table Header
            thead.innerHTML = '';
            const headerTr = document.createElement('tr');
            columns.forEach(colName => {
                const th = document.createElement('th');
                th.textContent = colName;
                headerTr.appendChild(th);
            });
            thead.appendChild(headerTr);

            // Build Table Body
            tbody.innerHTML = '';
            rows.forEach(rowCells => {
                const tr = document.createElement('tr');
                rowCells.forEach((cellText, colIdx) => {
                    const td = document.createElement('td');
                    const colName = columns[colIdx] || '';
                    const val = (cellText !== null && cellText !== undefined) ? String(cellText).trim() : '';

                    if (colName.includes('تاریخ') || colName.includes('Date')) {
                        td.className = 'col-date';
                        if (val && val !== '—' && val !== '-') {
                            td.innerHTML = `<span class="cell-badge date-badge">${val}</span>`;
                        } else {
                            td.innerHTML = `<span class="empty-dash">—</span>`;
                        }
                    } else if (colName.includes('آمدن') || colName.includes('تفصیل') || colName.includes('Description') || colName.includes('Particulars')) {
                        td.className = 'col-desc';
                        if (val && val !== '—') {
                            td.innerHTML = `<span class="urdu-desc">${val}</span>`;
                        } else {
                            td.innerHTML = `<span class="empty-dash">—</span>`;
                        }
                    } else if (colName.includes('صفحہ') || colName.includes('صفحه') || colName.includes('Folio') || colName.includes('Page')) {
                        td.className = 'col-folio';
                        if (val && val !== '—' && val !== '-') {
                            td.innerHTML = `<span class="cell-badge folio-badge">${val}</span>`;
                        } else {
                            td.innerHTML = `<span class="empty-dash">—</span>`;
                        }
                    } else if (colName.includes('رقم') || colName.includes('Amount')) {
                        td.className = 'col-amount';
                        td.innerHTML = formatAmount(val);
                    } else {
                        td.textContent = (val !== '') ? val : '—';
                    }

                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });

            // Store raw JSON
            if (data.raw_json) {
                rawJsonDisplay.textContent = JSON.stringify(data.raw_json, null, 2);
            } else {
                rawJsonDisplay.textContent = JSON.stringify(data, null, 2);
            }
        }
    }

    window.copyRecognizedText = function() {
        if (!wordDisplayText) return;
        const text = wordDisplayText.textContent.trim();
        if (!text || text === '—') return;

        navigator.clipboard.writeText(text).then(() => {
            const copyBtn = document.getElementById('btn-copy-word');
            if (copyBtn) {
                const orig = copyBtn.innerHTML;
                copyBtn.innerHTML = '✅ Copied! (کاپی ہو گیا)';
                setTimeout(() => { copyBtn.innerHTML = orig; }, 2000);
            }
        }).catch(err => {
            alert("Failed to copy text: " + err);
        });
    };

    window.toggleRawJson = function() {
        rawJsonDisplay.classList.toggle('hidden');
    };

    window.exportCSV = function() {
        if (!currentTableData.rows || currentTableData.rows.length === 0) {
            alert("No table data to export.");
            return;
        }

        let csvContent = "data:text/csv;charset=utf-8,\uFEFF";
        csvContent += currentTableData.columns.map(c => `"${c.replace(/"/g, '""')}"`).join(",") + "\r\n";

        currentTableData.rows.forEach(row => {
            csvContent += row.map(cell => `"${String(cell || '').replace(/"/g, '""')}"`).join(",") + "\r\n";
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "ledger_extracted.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // =========================================================================
    // Qwen Row Pipeline — Experimental Workflow & Stage C/B/F Logic
    // =========================================================================

    async function startPipelineJob(file, sampleName) {
        startPipelineUI();
        updateStep(1, 'active');

        const formData = new FormData();
        if (file) {
            formData.append('image', file);
        } else if (sampleName) {
            formData.append('sample_name', sampleName);
        }

        try {
            updateStep(1, 'completed');
            updateStep(2, 'active');

            const response = await fetch(`${BACKEND_BASE}/api/qwen_pipeline/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await parseJsonResponse(response);

            updateStep(2, 'completed');
            updateStep(3, 'active');
            await sleep(350);
            updateStep(3, 'completed');

            if (data.mode === 'vision_patches') {
                renderPatchInspector(data);
            } else {
                activeJobId = data.job_id;
                activeManifest = data.manifest;
                renderStageCReview(data.manifest);
            }
        } catch (err) {
            console.error(err);
            showError("Pipeline Vision Patch extraction failed: " + err.message);
            resetApp();
        }
    }

    let patchData = null;
    let patchSourceImg = null;
    let currentPatchMode = 'token'; // Default to 'token' (28x28 post-merger output)
    let gridLineColor = '#00E5FF';
    let showGridLines = true;
    let activePatch = { r: 0, c: 0, tr: 0, tc: 0, patchId: 0, tokenId: 0 };
    let currentViewRow = 0;
    let currentBatchPage = 0;
    const BATCH_SIZE = 40;

    function renderPatchInspector(data) {
        patchData = data;
        progressSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.add('hidden');
        if (pipelineResultsSection) pipelineResultsSection.classList.add('hidden');
        pipelineReviewSection.classList.remove('hidden');
        pipelineReviewSection.classList.add('fade-in');

        const vitDimsBadge = document.getElementById('patch-vit-dims');
        const tokenDimsBadge = document.getElementById('patch-token-dims');
        const gridBadge = document.getElementById('patch-grid-badge');
        const memoryBadge = document.getElementById('patch-memory-badge');

        const basePs = data.patch_dimensions?.base_patch_size?.[0] || 14;
        const tokenPs = data.patch_dimensions?.merged_token_size?.[0] || 28;
        const baseGrid = data.grid?.base_grid || { rows: 0, cols: 0, total_patches: 0 };
        const tokenGrid = data.grid?.token_grid || { rows: 0, cols: 0, total_tokens: 0 };

        if (vitDimsBadge) vitDimsBadge.textContent = `ViT Base: ${basePs}×${basePs} px`;
        if (tokenDimsBadge) tokenDimsBadge.textContent = `Merged Token: ${tokenPs}×${tokenPs} px (4-to-1)`;
        if (gridBadge) gridBadge.textContent = `Tokens: ${tokenGrid.rows}×${tokenGrid.cols} (${tokenGrid.total_tokens.toLocaleString()}) | Patches: ${baseGrid.total_patches.toLocaleString()}`;
        if (memoryBadge) memoryBadge.textContent = `⚡ In-Memory (0 files on disk)`;

        setupMultimodalFusion(data);
        updateRowDropdown();

        // Load in-memory Base64 image
        patchSourceImg = new Image();
        patchSourceImg.onload = () => {
            setupPatchMainCanvas();
            selectPatch(0, 0, false);
            renderRowPatches(0);
            renderBatchPatches(0);
        };
        patchSourceImg.src = data.image_data_url;
    }

    function updateRowDropdown() {
        const rowSelect = document.getElementById('patch-row-dropdown');
        if (!rowSelect || !patchData) return;
        rowSelect.innerHTML = '';

        if (currentPatchMode === 'token') {
            const tokenGrid = patchData.grid.token_grid;
            for (let r = 0; r < tokenGrid.rows; r++) {
                const opt = document.createElement('option');
                opt.value = r;
                opt.textContent = `Token Row ${r} (Tokens #${r * tokenGrid.cols}..#${(r + 1) * tokenGrid.cols - 1})`;
                rowSelect.appendChild(opt);
            }
        } else {
            const baseGrid = patchData.grid.base_grid;
            for (let r = 0; r < baseGrid.rows; r++) {
                const opt = document.createElement('option');
                opt.value = r;
                opt.textContent = `Base Row ${r} (Patches #${r * baseGrid.cols}..#${(r + 1) * baseGrid.cols - 1})`;
                rowSelect.appendChild(opt);
            }
        }
    }

    function setupPatchMainCanvas() {
        const canvas = document.getElementById('patch-main-canvas');
        if (!canvas || !patchSourceImg || !patchData) return;

        canvas.width = patchData.dimensions.resized_input.width;
        canvas.height = patchData.dimensions.resized_input.height;

        redrawMainCanvas();

        canvas.onmousemove = (e) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const cx = Math.floor((e.clientX - rect.left) * scaleX);
            const cy = Math.floor((e.clientY - rect.top) * scaleY);

            if (cx < 0 || cx >= canvas.width || cy < 0 || cy >= canvas.height) return;

            const basePs = patchData.patch_dimensions?.base_patch_size?.[0] || 14;
            const baseGrid = patchData.grid?.base_grid;
            if (!baseGrid) return;

            const r = Math.min(baseGrid.rows - 1, Math.floor(cy / basePs));
            const c = Math.min(baseGrid.cols - 1, Math.floor(cx / basePs));

            selectPatch(r, c, false);
        };

        canvas.onclick = (e) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const cx = Math.floor((e.clientX - rect.left) * scaleX);
            const cy = Math.floor((e.clientY - rect.top) * scaleY);

            if (cx < 0 || cx >= canvas.width || cy < 0 || cy >= canvas.height) return;

            const basePs = patchData.patch_dimensions?.base_patch_size?.[0] || 14;
            const baseGrid = patchData.grid?.base_grid;
            if (!baseGrid) return;

            const r = Math.min(baseGrid.rows - 1, Math.floor(cy / basePs));
            const c = Math.min(baseGrid.cols - 1, Math.floor(cx / basePs));

            selectPatch(r, c, true);
        };
    }

    function selectPatch(r, c, syncRow = false) {
        if (!patchData || !patchSourceImg) return;
        const baseGrid = patchData.grid.base_grid;
        const tokenGrid = patchData.grid.token_grid;
        const basePs = patchData.patch_dimensions.base_patch_size[0];
        const tokenPs = patchData.patch_dimensions.merged_token_size[0];
        const mergeSize = patchData.patch_dimensions.merge_size || 2;

        const patchId = r * baseGrid.cols + c;
        const tr = Math.floor(r / mergeSize);
        const tc = Math.floor(c / mergeSize);
        const tokenId = tr * tokenGrid.cols + tc;

        // 4 constituent base patch indices
        const r0 = tr * mergeSize;
        const r1 = r0 + 1;
        const c0 = tc * mergeSize;
        const c1 = c0 + 1;

        const pTL = r0 * baseGrid.cols + c0;
        const pTR = r0 * baseGrid.cols + c1;
        const pBL = r1 * baseGrid.cols + c0;
        const pBR = r1 * baseGrid.cols + c1;

        const subY = r % mergeSize;
        const subX = c % mergeSize;
        const subIdx = subY * 2 + subX; // 0=TL, 1=TR, 2=BL, 3=BR

        activePatch = { r, c, patchId, tr, tc, tokenId, pTL, pTR, pBL, pBR, subIdx };

        // Update Text details
        const activeTitle = document.getElementById('active-patch-title');
        if (activeTitle) {
            activeTitle.textContent = (currentPatchMode === 'token') ? `Merged Token #${tokenId}` : `Base Patch #${patchId}`;
        }

        const hoverCoords = document.getElementById('patch-hover-coords');
        if (hoverCoords) {
            hoverCoords.textContent = `Token [${tr}, ${tc}] (#${tokenId}) ◀ Fused from Patches [#${pTL}, #${pTR}, #${pBL}, #${pBR}]`;
        }

        const mergerLabel = document.getElementById('merger-token-label');
        if (mergerLabel) mergerLabel.textContent = `Token #${tokenId}`;

        const metaToken = document.getElementById('meta-token-id');
        if (metaToken) metaToken.textContent = `#${tokenId} of ${tokenGrid.total_tokens.toLocaleString()}`;

        const metaGrid = document.getElementById('meta-grid-coords');
        if (metaGrid) metaGrid.textContent = `Token Row ${tr}, Col ${tc} (RoPE 2D)`;

        const metaBox = document.getElementById('meta-pixel-box');
        if (metaBox) {
            metaBox.textContent = `[x:${tc * tokenPs}, y:${tr * tokenPs}, w:${tokenPs}, h:${tokenPs}]`;
        }

        const metaPatches = document.getElementById('meta-base-patch-ids');
        if (metaPatches) {
            metaPatches.textContent = `#${pTL} (TL), #${pTR} (TR), #${pBL} (BL), #${pBR} (BR)`;
        }

        // Draw Single Unified Merged Token (28x28) onto Magnifier Canvas
        const magCanvas = document.getElementById('patch-magnifier-canvas');
        if (magCanvas) {
            magCanvas.width = tokenPs;
            magCanvas.height = tokenPs;
            const magCtx = magCanvas.getContext('2d');
            magCtx.imageSmoothingEnabled = false;
            magCtx.drawImage(patchSourceImg, tc * tokenPs, tr * tokenPs, tokenPs, tokenPs, 0, 0, tokenPs, tokenPs);
        }

        // Draw same 28x28 crop onto Step 4 Multimodal Fusion Mini Canvas
        const fusionCanvas = document.getElementById('fusion-slot-canvas');
        if (fusionCanvas) {
            fusionCanvas.width = tokenPs;
            fusionCanvas.height = tokenPs;
            const fCtx = fusionCanvas.getContext('2d');
            fCtx.imageSmoothingEnabled = false;
            fCtx.drawImage(patchSourceImg, tc * tokenPs, tr * tokenPs, tokenPs, tokenPs, 0, 0, tokenPs, tokenPs);
        }

        // Update Step 4 Multimodal Fusion details
        const prefixLen = patchData.multimodal_fusion?.prefix_length || 15;
        const totalSeq = patchData.multimodal_fusion?.total_sequence_length || 2516;
        const seqIdx = prefixLen + tokenId;

        const seqSlot = document.getElementById('fusion-seq-slot');
        if (seqSlot) seqSlot.textContent = `#${seqIdx} of ${totalSeq.toLocaleString()}`;

        const visTag = document.getElementById('fusion-visual-tag');
        if (visTag) visTag.textContent = `Visual Vector v_${tokenId}`;

        const mropeVal = document.getElementById('fusion-mrope-val');
        if (mropeVal) mropeVal.textContent = `[0, ${tr}, ${tc}]`;

        // Update active marker in sequence stream bar
        const marker = document.getElementById('stream-active-marker');
        if (marker && tokenGrid.total_tokens > 0) {
            const pct = (tokenId / tokenGrid.total_tokens) * 100;
            marker.style.left = `${pct}%`;
        }

        // Highlight chip in sequence stream
        document.querySelectorAll('.fusion-token-chip').forEach(el => el.classList.remove('active-chip'));
        const activeChip = document.getElementById(`seq-chip-${seqIdx}`);
        if (activeChip) {
            activeChip.classList.add('active-chip');
        }

        redrawMainCanvas();

        const activeRowIndex = (currentPatchMode === 'token') ? tr : r;
        if (syncRow && currentViewRow !== activeRowIndex) {
            currentViewRow = activeRowIndex;
            const rowSelect = document.getElementById('patch-row-dropdown');
            if (rowSelect) rowSelect.value = activeRowIndex;
            renderRowPatches(activeRowIndex);
        }
    }

    function redrawMainCanvas() {
        const canvas = document.getElementById('patch-main-canvas');
        if (!canvas || !patchSourceImg || !patchData) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw base photograph
        ctx.drawImage(patchSourceImg, 0, 0);

        const basePs = patchData.patch_dimensions?.base_patch_size?.[0] || 14;
        const tokenPs = patchData.patch_dimensions?.merged_token_size?.[0] || 28;

        // Draw Grid Lines if enabled
        if (showGridLines) {
            if (currentPatchMode === 'token') {
                // Clean solid lines for 28px Merged Tokens (1 Single Image per cell)
                ctx.save();
                ctx.beginPath();
                ctx.strokeStyle = gridLineColor;
                ctx.lineWidth = 1.25;
                ctx.globalAlpha = 0.7;
                for (let x = 0; x <= canvas.width; x += tokenPs) {
                    ctx.moveTo(x + 0.5, 0);
                    ctx.lineTo(x + 0.5, canvas.height);
                }
                for (let y = 0; y <= canvas.height; y += tokenPs) {
                    ctx.moveTo(0, y + 0.5);
                    ctx.lineTo(canvas.width, y + 0.5);
                }
                ctx.stroke();
                ctx.restore();
            } else {
                // Base 14px lines
                ctx.save();
                ctx.beginPath();
                ctx.strokeStyle = gridLineColor;
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.4;
                for (let x = 0; x <= canvas.width; x += basePs) {
                    ctx.moveTo(x + 0.5, 0);
                    ctx.lineTo(x + 0.5, canvas.height);
                }
                for (let y = 0; y <= canvas.height; y += basePs) {
                    ctx.moveTo(0, y + 0.5);
                    ctx.lineTo(canvas.width, y + 0.5);
                }
                ctx.stroke();
                ctx.restore();
            }
        }

        // Highlight active merged token as one single unified image
        if (activePatch) {
            ctx.save();
            if (currentPatchMode === 'token') {
                const tx = activePatch.tc * tokenPs;
                const ty = activePatch.tr * tokenPs;

                // Tint the whole 28x28 merged token as ONE single unified image
                ctx.fillStyle = "rgba(0, 255, 163, 0.35)";
                ctx.fillRect(tx, ty, tokenPs, tokenPs);

                // Thick glowing neon border around 28x28 token
                ctx.strokeStyle = gridLineColor;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(tx, ty, tokenPs, tokenPs);
            } else {
                const hx = activePatch.c * basePs;
                const hy = activePatch.r * basePs;
                ctx.fillStyle = "rgba(0, 255, 163, 0.4)";
                ctx.fillRect(hx, hy, basePs, basePs);
                ctx.strokeStyle = gridLineColor;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(hx, hy, basePs, basePs);
            }
            ctx.restore();
        }
    }

    function renderRowPatches(rowIndex) {
        const strip = document.getElementById('patch-row-strip');
        if (!strip || !patchSourceImg || !patchData) return;
        strip.innerHTML = '';

        const isToken = (currentPatchMode === 'token');
        const grid = isToken ? patchData.grid.token_grid : patchData.grid.base_grid;
        const tileSize = isToken ? patchData.patch_dimensions.merged_token_size[0] : patchData.patch_dimensions.base_patch_size[0];

        for (let col = 0; col < grid.cols; col++) {
            const id = rowIndex * grid.cols + col;
            const item = document.createElement('div');
            item.className = 'patch-tile-item' + (isToken ? ' token-tile' : '');

            const isActive = isToken
                ? (activePatch.tr === rowIndex && activePatch.tc === col)
                : (activePatch.r === rowIndex && activePatch.c === col);
            if (isActive) item.classList.add('active');

            const canvas = document.createElement('canvas');
            canvas.width = tileSize;
            canvas.height = tileSize;
            canvas.className = 'patch-tile-canvas';
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(patchSourceImg, col * tileSize, rowIndex * tileSize, tileSize, tileSize, 0, 0, tileSize, tileSize);

            const label = document.createElement('span');
            label.className = 'patch-tile-label';
            label.textContent = isToken ? `T#${id}` : `P#${id}`;

            item.appendChild(canvas);
            item.appendChild(label);

            item.onclick = () => {
                if (isToken) {
                    selectPatch(rowIndex * 2, col * 2, false);
                } else {
                    selectPatch(rowIndex, col, false);
                }
                document.querySelectorAll('#patch-row-strip .patch-tile-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
            };

            strip.appendChild(item);
        }
    }

    function renderBatchPatches(batchIdx) {
        const grid = document.getElementById('patch-batch-grid');
        if (!grid || !patchSourceImg || !patchData) return;
        grid.innerHTML = '';

        const isToken = (currentPatchMode === 'token');
        const gridObj = isToken ? patchData.grid.token_grid : patchData.grid.base_grid;
        const tileSize = isToken ? patchData.patch_dimensions.merged_token_size[0] : patchData.patch_dimensions.base_patch_size[0];
        const total = isToken ? gridObj.total_tokens : gridObj.total_patches;
        const totalBatches = Math.ceil(total / BATCH_SIZE);

        currentBatchPage = Math.max(0, Math.min(totalBatches - 1, batchIdx));

        const batchLabel = document.getElementById('patch-batch-label');
        if (batchLabel) {
            const unit = isToken ? 'Tokens' : 'Patches';
            const prefix = isToken ? 'T#' : 'P#';
            batchLabel.textContent = `Batch ${currentBatchPage + 1} of ${totalBatches} (${unit} ${prefix}${currentBatchPage * BATCH_SIZE}..${prefix}${Math.min(total - 1, (currentBatchPage + 1) * BATCH_SIZE - 1)})`;
        }

        const startIdx = currentBatchPage * BATCH_SIZE;
        const endIdx = Math.min(total, startIdx + BATCH_SIZE);

        for (let idx = startIdx; idx < endIdx; idx++) {
            const r = Math.floor(idx / gridObj.cols);
            const c = idx % gridObj.cols;

            const item = document.createElement('div');
            item.className = 'patch-tile-item' + (isToken ? ' token-tile' : '');

            const isActive = isToken
                ? (activePatch.tokenId === idx)
                : (activePatch.patchId === idx);
            if (isActive) item.classList.add('active');

            const canvas = document.createElement('canvas');
            canvas.width = tileSize;
            canvas.height = tileSize;
            canvas.className = 'patch-tile-canvas';
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(patchSourceImg, c * tileSize, r * tileSize, tileSize, tileSize, 0, 0, tileSize, tileSize);

            const label = document.createElement('span');
            label.className = 'patch-tile-label';
            label.textContent = isToken ? `T#${idx}` : `P#${idx}`;

            item.appendChild(canvas);
            item.appendChild(label);

            item.onclick = () => {
                if (isToken) {
                    selectPatch(r * 2, c * 2, true);
                } else {
                    selectPatch(r, c, true);
                }
                document.querySelectorAll('#patch-batch-grid .patch-tile-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
            };

            grid.appendChild(item);
        }
    }

    // Window controls
    window.setPatchMode = function(mode) {
        currentPatchMode = mode;
        const btnBase = document.getElementById('btn-mode-base-patch');
        const btnToken = document.getElementById('btn-mode-token-patch');
        const heading = document.getElementById('inspector-card-heading');
        const galleryTitle = document.getElementById('gallery-output-title');
        const btnRender = document.getElementById('btn-render-all-tiles');

        if (mode === 'token') {
            if (btnToken) btnToken.classList.add('active');
            if (btnBase) btnBase.classList.remove('active');
            if (heading) heading.textContent = 'Post-Merger Token Anatomy (4-to-1 Fusion)';
            if (galleryTitle) galleryTitle.textContent = 'Merged Tokens Output:';
            if (btnRender) btnRender.textContent = '🖼️ Render All Merged Tokens Sheet';
        } else {
            if (btnBase) btnBase.classList.add('active');
            if (btnToken) btnToken.classList.remove('active');
            if (heading) heading.textContent = 'Pre-Merger Base Patch Inspector (14×14)';
            if (galleryTitle) galleryTitle.textContent = 'Base Patches Output:';
            if (btnRender) btnRender.textContent = '🖼️ Render All Base Patches Sheet';
        }

        updateRowDropdown();
        currentViewRow = 0;
        currentBatchPage = 0;
        if (activePatch) {
            selectPatch(activePatch.r, activePatch.c, false);
        }
        renderRowPatches(0);
        renderBatchPatches(0);
    };

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function setupMultimodalFusion(data) {
        const fusion = data.multimodal_fusion || {};
        const totalSeq = fusion.total_sequence_length || 2516;
        const totalVisual = fusion.visual_tokens_count || (data.grid?.token_grid?.total_tokens || 2451);
        const prefixCount = fusion.prefix_length || 15;

        const summaryBadge = document.getElementById('fusion-sequence-summary');
        if (summaryBadge) {
            summaryBadge.textContent = `${totalSeq.toLocaleString()} Total Tokens in Sequence`;
        }

        const midLabel = document.getElementById('fusion-stream-midpoint');
        if (midLabel) {
            midLabel.textContent = `Token #${prefixCount}..#${prefixCount + totalVisual - 1} (${totalVisual.toLocaleString()} Visual Vectors)`;
        }

        const endLabel = document.getElementById('fusion-stream-endpoint');
        if (endLabel) {
            endLabel.textContent = `Token #${totalSeq - 1} (Urdu Prompt End)`;
        }

        const streamVisualLabel = document.getElementById('stream-visual-label');
        if (streamVisualLabel) {
            streamVisualLabel.textContent = `${totalVisual.toLocaleString()} Merged Visual Vectors (replacing <|image_pad|>)`;
        }

        // Render Token Stream Chips
        const streamContainer = document.getElementById('fusion-tokens-stream');
        if (!streamContainer) return;
        streamContainer.innerHTML = '';

        // 1. Prefix Tokens
        const prefixTokens = fusion.prefix_tokens || [];
        prefixTokens.forEach(tok => {
            const chip = document.createElement('div');
            chip.className = 'fusion-token-chip chip-prefix';
            chip.id = `seq-chip-${tok.seq_idx}`;
            chip.innerHTML = `
                <span class="chip-seq-num">#${tok.seq_idx}</span>
                <span style="font-weight:700;">${escapeHtml(tok.text.replace(/\n/g, '\\n'))}</span>
                <span style="font-size:0.55rem; color:#94a3b8;">ID: ${tok.id}</span>
            `;
            chip.title = `Prefix Text Token #${tok.seq_idx} (ID: ${tok.id}): "${tok.text}"`;
            streamContainer.appendChild(chip);
        });

        // 2. Visual Span Sample / Representative Chips
        const visualSpanStart = prefixCount;
        for (let i = 0; i < totalVisual; i++) {
            if (i < 12 || i > totalVisual - 13 || (activePatch && Math.abs(i - activePatch.tokenId) <= 3)) {
                const seqIdx = visualSpanStart + i;
                const chip = document.createElement('div');
                chip.className = 'fusion-token-chip chip-visual';
                chip.id = `seq-chip-${seqIdx}`;
                chip.innerHTML = `
                    <span class="chip-seq-num">#${seqIdx}</span>
                    <span style="font-weight:700;">v_${i}</span>
                    <span style="font-size:0.55rem; color:#00E5FF;">pad: 151655</span>
                `;
                chip.title = `Merged Visual Vector v_${i} (spliced into sequence #${seqIdx})`;
                chip.onclick = () => {
                    if (data.grid?.token_grid) {
                        const tr = Math.floor(i / data.grid.token_grid.cols);
                        const tc = i % data.grid.token_grid.cols;
                        selectPatch(tr * 2, tc * 2, true);
                    }
                };
                streamContainer.appendChild(chip);
            } else if (i === 12) {
                const elip = document.createElement('div');
                elip.className = 'fusion-token-chip';
                elip.style.opacity = '0.6';
                elip.innerHTML = `<span style="font-size:0.75rem; color:#94a3b8;">... (${totalVisual - 24} more visual vectors) ...</span>`;
                streamContainer.appendChild(elip);
            }
        }

        // 3. Postfix Tokens (Urdu Prompt)
        const postfixTokens = fusion.postfix_tokens || [];
        postfixTokens.forEach(tok => {
            const chip = document.createElement('div');
            chip.className = 'fusion-token-chip chip-postfix';
            chip.id = `seq-chip-${tok.seq_idx}`;
            chip.innerHTML = `
                <span class="chip-seq-num">#${tok.seq_idx}</span>
                <span style="font-weight:700; direction:rtl;">${escapeHtml(tok.text.replace(/\n/g, '\\n'))}</span>
                <span style="font-size:0.55rem; color:#94a3b8;">ID: ${tok.id}</span>
            `;
            chip.title = `Urdu Prompt Token #${tok.seq_idx} (ID: ${tok.id}): "${tok.text}"`;
            streamContainer.appendChild(chip);
        });
    }

    window.jumpSequence = function(type) {
        const streamContainer = document.getElementById('fusion-tokens-stream');
        if (!streamContainer || !patchData) return;
        if (type === 'prefix') {
            streamContainer.scrollTo({ left: 0, behavior: 'smooth' });
        } else if (type === 'postfix') {
            streamContainer.scrollTo({ left: streamContainer.scrollWidth, behavior: 'smooth' });
        } else if (type === 'active' && activePatch) {
            const prefixLen = patchData.multimodal_fusion?.prefix_length || 15;
            const chip = document.getElementById(`seq-chip-${prefixLen + activePatch.tokenId}`);
            if (chip) chip.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }
    };

    window.changeGridColor = function(color) {
        gridLineColor = color;
        redrawMainCanvas();
    };

    window.toggleGridLines = function() {
        showGridLines = !showGridLines;
        const btn = document.getElementById('btn-toggle-grid');
        if (btn) btn.textContent = showGridLines ? 'Hide Grid' : 'Show Grid';
        redrawMainCanvas();
    };

    window.switchPatchViewType = function(type) {
        const rowCont = document.getElementById('patch-row-view-container');
        const batchCont = document.getElementById('patch-batch-view-container');
        if (type === 'row') {
            if (rowCont) rowCont.classList.remove('hidden');
            if (batchCont) batchCont.classList.add('hidden');
            renderRowPatches(currentViewRow);
        } else {
            if (rowCont) rowCont.classList.add('hidden');
            if (batchCont) batchCont.classList.remove('hidden');
            renderBatchPatches(currentBatchPage);
        }
    };

    window.stepRow = function(delta) {
        if (!patchData) return;
        const isToken = (currentPatchMode === 'token');
        const totalRows = isToken ? patchData.grid.token_grid.rows : patchData.grid.base_grid.rows;
        currentViewRow = Math.max(0, Math.min(totalRows - 1, currentViewRow + delta));
        const rowSelect = document.getElementById('patch-row-dropdown');
        if (rowSelect) rowSelect.value = currentViewRow;

        if (isToken) {
            selectPatch(currentViewRow * 2, 0, false);
        } else {
            selectPatch(currentViewRow, 0, false);
        }
        renderRowPatches(currentViewRow);
    };

    window.selectRow = function(r) {
        currentViewRow = r;
        if (currentPatchMode === 'token') {
            selectPatch(r * 2, 0, false);
        } else {
            selectPatch(r, 0, false);
        }
        renderRowPatches(r);
    };

    window.stepBatch = function(delta) {
        renderBatchPatches(currentBatchPage + delta);
    };

    window.renderAllPatchesTiledCanvas = function() {
        if (!patchData || !patchSourceImg) return;
        const isToken = (currentPatchMode === 'token');
        const gridObj = isToken ? patchData.grid.token_grid : patchData.grid.base_grid;
        const tileSize = isToken ? patchData.patch_dimensions.merged_token_size[0] : patchData.patch_dimensions.base_patch_size[0];
        const unitName = isToken ? "Merged Tokens (28x28 px)" : "Base Patches (14x14 px)";

        const cols = gridObj.cols;
        const rows = gridObj.rows;
        const tileGap = 3;
        const tileW = tileSize + tileGap;
        const tileH = tileSize + tileGap;

        const tileCanvas = document.createElement('canvas');
        tileCanvas.width = cols * tileW;
        tileCanvas.height = rows * tileH;
        const tCtx = tileCanvas.getContext('2d');
        tCtx.fillStyle = "#05070a";
        tCtx.fillRect(0, 0, tileCanvas.width, tileCanvas.height);
        tCtx.imageSmoothingEnabled = false;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                tCtx.drawImage(
                    patchSourceImg,
                    c * tileSize, r * tileSize, tileSize, tileSize,
                    c * tileW, r * tileH, tileSize, tileSize
                );
            }
        }

        const dataUrl = tileCanvas.toDataURL('image/png');
        const win = window.open();
        if (win) {
            win.document.write(`<html><head><title>All ${gridObj.total_tokens ? gridObj.total_tokens.toLocaleString() : gridObj.total_patches.toLocaleString()} ${unitName}</title><style>body{margin:0;background:#080b10;color:#fff;font-family:sans-serif;padding:20px;text-align:center;}img{border:2px solid #00FFA3;image-rendering:pixelated;max-width:98%;box-shadow:0 0 20px rgba(0,255,163,0.3);}</style></head><body><h2>All ${isToken ? gridObj.total_tokens.toLocaleString() : gridObj.total_patches.toLocaleString()} ${unitName} (In-Memory Sheet)</h2><p style="color:#94a3b8;">${isToken ? 'Post-Merger 2D-RoPE tokens fed into the LLM' : 'Pre-Merger ViT convolution patches'} — 0 files written to disk</p><img src="${dataUrl}"></body></html>`);
        }
    };

    function renderStageCReview(manifest) {
        progressSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.add('hidden');
        if (pipelineResultsSection) pipelineResultsSection.classList.add('hidden');
        pipelineReviewSection.classList.remove('hidden');
        pipelineReviewSection.classList.add('fade-in');

        if (pipelineJobIdBadge) pipelineJobIdBadge.textContent = `Job: ${activeJobId}`;
        const adapterName = manifest.model_info?.actual_adapter || "Qwen2.5-VL-7B";
        if (pipelineModelBadge) pipelineModelBadge.textContent = `Adapter: ${adapterName}`;

        const rows = manifest.rows || [];
        if (pipelineRowsCountBadge) pipelineRowsCountBadge.textContent = `Proposed Rows: ${rows.length}`;
        if (reviewActiveCount) reviewActiveCount.textContent = rows.length;

        // Load Canonical Image
        reviewCanonicalImg.src = `${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/canonical_page.png?t=${Date.now()}`;
        reviewCanonicalImg.onload = () => {
            redrawOverlays(rows);
        };

        // Render Row Cards
        rowsScrollContainer.innerHTML = '';
        rows.forEach((r, idx) => {
            const card = document.createElement('div');
            card.className = 'row-item-card glass';
            card.id = `card-row-${r.row_id}`;

            const flagsHtml = (r.validation_flags || []).map(f => `<span class="badge-flag">${f}</span>`).join(' ');
            let statusBadge = `<span class="badge" id="status-badge-${r.row_id}">${r.status || 'proposed'}</span>`;
            if (r.is_manually_edited) {
                statusBadge = `<span class="badge badge-info" id="status-badge-${r.row_id}">Edited</span>`;
            } else if (r.is_refined_stage_b) {
                statusBadge = `<span class="badge badge-success" id="status-badge-${r.row_id}">Stage B Refined</span>`;
            }

            const isLast = (idx === rows.length - 1);
            const nextRowId = !isLast ? rows[idx + 1].row_id : null;

            card.innerHTML = `
                <div class="row-item-header">
                    <div class="row-id-title">
                        <span>#${idx + 1} ${r.row_id}</span>
                        ${statusBadge}
                    </div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <label style="font-size:0.75rem; color:var(--text-muted);">Type:</label>
                        <select class="type-select" id="select-type-${r.row_id}" onchange="saveCropEdit('${r.row_id}')">
                            <option value="transaction" ${r.row_type === 'transaction' ? 'selected' : ''}>Transaction</option>
                            <option value="section_heading" ${r.row_type === 'section_heading' ? 'selected' : ''}>Section Heading</option>
                            <option value="subtotal" ${r.row_type === 'subtotal' ? 'selected' : ''}>Subtotal</option>
                            <option value="total" ${r.row_type === 'total' ? 'selected' : ''}>Total</option>
                            <option value="note" ${r.row_type === 'note' ? 'selected' : ''}>Note</option>
                            <option value="unknown" ${r.row_type === 'unknown' ? 'selected' : ''}>Unknown</option>
                        </select>
                    </div>
                </div>

                ${flagsHtml ? `<div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px;">${flagsHtml}</div>` : ''}

                <div class="row-crop-preview-box">
                    <img id="crop-img-${r.row_id}" src="${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/crops/${r.crop_filename}?t=${Date.now()}" alt="${r.row_id}">
                </div>

                <div class="row-controls">
                    <div class="bound-group">
                        <span>Top:</span>
                        <input type="number" class="bound-input" id="input-top-${r.row_id}" value="${r.current_bounds.top}">
                    </div>
                    <div class="bound-group">
                        <span>Bot:</span>
                        <input type="number" class="bound-input" id="input-bot-${r.row_id}" value="${r.current_bounds.bottom}">
                    </div>
                    <button class="btn-ctrl btn-ctrl-primary" onclick="saveCropEdit('${r.row_id}')">💾 Save Bounds</button>
                    <button class="btn-ctrl" onclick="restoreCropProposal('${r.row_id}')">↺ Restore</button>
                    ${nextRowId ? `<button class="btn-ctrl" onclick="refineRowPair('${r.row_id}', '${nextRowId}')">🔍 Refine with Next (Stage B)</button>` : ''}
                    <a href="${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/crops/${r.crop_filename}" target="_blank" class="btn-ctrl" style="text-decoration:none;">↗ Open Crop</a>
                </div>
            `;

            // Hover card highlights SVG row
            card.addEventListener('mouseenter', () => {
                const rect = document.getElementById(`svg-rect-${r.row_id}`);
                if (rect) rect.classList.add('active');
            });
            card.addEventListener('mouseleave', () => {
                const rect = document.getElementById(`svg-rect-${r.row_id}`);
                if (rect) rect.classList.remove('active');
            });

            rowsScrollContainer.appendChild(card);
        });
    }

    function redrawOverlays(rows) {
        if (!reviewCanonicalImg || !reviewOverlaySvg) return;
        const natW = reviewCanonicalImg.naturalWidth;
        const natH = reviewCanonicalImg.naturalHeight;
        if (!natW || !natH) return;

        const dispW = reviewCanonicalImg.clientWidth;
        const dispH = reviewCanonicalImg.clientHeight;

        reviewOverlaySvg.setAttribute('viewBox', `0 0 ${dispW} ${dispH}`);
        reviewOverlaySvg.innerHTML = '';

        const scaleY = dispH / float(natH);

        rows.forEach(r => {
            const topCan = r.current_bounds.top;
            const botCan = r.current_bounds.bottom;
            const y = topCan * scaleY;
            const h = Math.max(4, (botCan - topCan) * scaleY);

            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("id", `svg-rect-${r.row_id}`);
            rect.setAttribute("class", "svg-row-rect");
            rect.setAttribute("x", "0");
            rect.setAttribute("y", y);
            rect.setAttribute("width", dispW);
            rect.setAttribute("height", h);

            rect.addEventListener("click", () => {
                const card = document.getElementById(`card-row-${r.row_id}`);
                if (card) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    card.classList.add('highlighted');
                    setTimeout(() => card.classList.remove('highlighted'), 1500);
                }
            });

            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("class", "svg-row-label");
            text.setAttribute("x", "12");
            text.setAttribute("y", Math.max(14, y + 14));
            text.textContent = r.row_id;

            reviewOverlaySvg.appendChild(rect);
            reviewOverlaySvg.appendChild(text);
        });
    }

    window.saveCropEdit = async function(rowId) {
        hideError();
        const topVal = parseInt(document.getElementById(`input-top-${rowId}`).value, 10);
        const botVal = parseInt(document.getElementById(`input-bot-${rowId}`).value, 10);
        const typeSelect = document.getElementById(`select-type-${rowId}`);
        const rowType = typeSelect ? typeSelect.value : null;

        if (isNaN(topVal) || isNaN(botVal) || topVal >= botVal) {
            showError(`Invalid boundaries for ${rowId}: Top (${topVal}) must be strictly less than Bottom (${botVal})`);
            return;
        }

        try {
            const resp = await fetch(`${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/crop_edit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    row_id: rowId,
                    new_top: topVal,
                    new_bottom: botVal,
                    new_row_type: rowType
                })
            });
            const data = await parseJsonResponse(resp);
            activeManifest = data.manifest;

            // Refresh crop image with timestamp
            const cropImg = document.getElementById(`crop-img-${rowId}`);
            if (cropImg) {
                cropImg.src = `${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/crops/${data.row.crop_filename}?t=${Date.now()}`;
            }

            const badge = document.getElementById(`status-badge-${rowId}`);
            if (badge) {
                badge.className = "badge badge-info";
                badge.textContent = "Edited";
            }

            redrawOverlays(activeManifest.rows);
        } catch (err) {
            console.error(err);
            showError(`Failed to save edit for ${rowId}: ` + err.message);
        }
    };

    window.restoreCropProposal = async function(rowId) {
        hideError();
        try {
            const resp = await fetch(`${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/restore_proposal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ row_id: rowId })
            });
            const data = await parseJsonResponse(resp);
            activeManifest = data.manifest;

            const topInp = document.getElementById(`input-top-${rowId}`);
            const botInp = document.getElementById(`input-bot-${rowId}`);
            if (topInp) topInp.value = data.row.current_bounds.top;
            if (botInp) botInp.value = data.row.current_bounds.bottom;

            const cropImg = document.getElementById(`crop-img-${rowId}`);
            if (cropImg) {
                cropImg.src = `${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/crops/${data.row.crop_filename}?t=${Date.now()}`;
            }

            const badge = document.getElementById(`status-badge-${rowId}`);
            if (badge) {
                badge.className = "badge";
                badge.textContent = "Proposed";
            }

            redrawOverlays(activeManifest.rows);
        } catch (err) {
            console.error(err);
            showError(`Failed to restore proposal for ${rowId}: ` + err.message);
        }
    };

    window.refineRowPair = async function(rowAId, rowBId) {
        hideError();
        if (stageBModal) {
            stageBModal.classList.remove('hidden');
            if (stageBModalTitle) stageBModalTitle.textContent = `Refining Context: ${rowAId} & ${rowBId}...`;
            if (stageBNotes) stageBNotes.innerHTML = `<em>Running Stage B refinement with Qwen2.5-VL on GPU...</em>`;
            if (stageBContextImg) stageBContextImg.src = '';
        }

        try {
            const resp = await fetch(`${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/refine_pair`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    row_a_id: rowAId,
                    row_b_id: rowBId,
                    padding_y: 28
                })
            });
            const data = await parseJsonResponse(resp);
            activeManifest = data.manifest;

            const contextFilename = `context_${rowAId}_${rowBId}.png`;
            if (stageBContextImg) {
                stageBContextImg.src = `${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/context_crops/${contextFilename}?t=${Date.now()}`;
            }

            const notes = data.refined?.separation_notes || "Refinement completed successfully.";
            const conflict = data.refined?.conflict_detected ? "<strong style='color:#ef4444;'>⚠️ Conflict detected in full-width cut</strong>" : "<strong style='color:#00FFA3;'>✅ Boundary safely separated</strong>";
            if (stageBNotes) {
                stageBNotes.innerHTML = `<div>${conflict}</div><p style="margin-top:6px;">${notes}</p>`;
            }

            renderStageCReview(activeManifest);
        } catch (err) {
            console.error(err);
            if (stageBNotes) stageBNotes.innerHTML = `<span style="color:#ef4444;">Refinement error: ${err.message}</span>`;
            showError("Stage B refinement failed: " + err.message);
        }
    };

    window.closeStageBModal = function() {
        if (stageBModal) stageBModal.classList.add('hidden');
    };

    window.transcribeReviewedCrops = async function() {
        hideError();
        if (!activeJobId) {
            showError("No active job to transcribe.");
            return;
        }

        if (pipelineReviewSection) pipelineReviewSection.classList.add('hidden');
        progressSection.classList.remove('hidden');

        if (step1Title) step1Title.textContent = "Crop Verification & Loading";
        if (step1Desc) step1Desc.textContent = "Preparing reviewed rectangular crops from canonical page";
        if (step2Title) step2Title.textContent = "Stage D: Faithful Row Transcription";
        if (step2Desc) step2Desc.textContent = "Transcribing handwritten Urdu, numbers, and ditto marks verbatim";
        if (step3Title) step3Title.textContent = "Stage E: Field Assignment & Excel Generation";
        if (step3Desc) step3Desc.textContent = "Mapping content into columns and exporting genuine XLSX";

        resetSteps();
        updateStep(1, 'active');

        try {
            updateStep(1, 'completed');
            updateStep(2, 'active');

            const resp = await fetch(`${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/transcribe`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixel_budget: 100352 })
            });

            updateStep(2, 'completed');
            updateStep(3, 'active');

            const data = await parseJsonResponse(resp);
            activeManifest = data.manifest;

            updateStep(3, 'completed');
            await sleep(350);

            renderStageFResults(data.manifest);
        } catch (err) {
            console.error(err);
            showError("Recognition & field assignment failed: " + err.message);
            if (pipelineReviewSection) pipelineReviewSection.classList.remove('hidden');
            progressSection.classList.add('hidden');
        }
    };

    function renderStageFResults(manifest) {
        progressSection.classList.add('hidden');
        if (pipelineReviewSection) pipelineReviewSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.add('hidden');
        pipelineResultsSection.classList.remove('hidden');
        pipelineResultsSection.classList.add('fade-in');

        const rows = manifest.rows || [];
        const columns = manifest.columns || [];

        const statsElem = document.getElementById('pipeline-results-stats');
        if (statsElem) {
            const dTime = (manifest.stage_timings?.stage_d_total_seconds || 0).toFixed(1);
            const eTime = (manifest.stage_timings?.stage_e_total_seconds || 0).toFixed(1);
            statsElem.textContent = `(${rows.length} Rows Extracted | Stage D: ${dTime}s, Stage E: ${eTime}s)`;
        }

        if (btnDownloadXlsx) {
            btnDownloadXlsx.href = `${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/export/xlsx`;
        }
        if (btnDownloadJson) {
            btnDownloadJson.href = `${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/export/json`;
        }

        // Build Table Header
        pipelineThead.innerHTML = '';
        const trHead = document.createElement('tr');
        trHead.innerHTML = `
            <th>Row ID</th>
            <th>Type</th>
            ${columns.map(c => `<th>${c.label || c.col_id}</th>`).join('')}
            <th>Raw Qwen Transcription</th>
            <th>Review Flags</th>
            <th>Clean Crop</th>
        `;
        pipelineThead.appendChild(trHead);

        // Build Table Body
        pipelineTbody.innerHTML = '';
        rows.forEach(r => {
            const tr = document.createElement('tr');
            const stageE = r.stage_e_fields || {};
            const cells = stageE.cells || {};
            const stageD = r.stage_d_transcription || {};
            const rawText = stageD.raw_text || '—';

            const colCellsHtml = columns.map(c => {
                const val = cells[c.col_id] || '';
                const isUrdu = /[\u0600-\u06FF]/.test(val);
                return `<td class="${isUrdu ? 'urdu-desc' : ''}">${val !== '' ? val : '<span class="empty-dash">—</span>'}</td>`;
            }).join('');

            const flags = (r.validation_flags || []).concat(stageE.review_flags || []);
            const flagsHtml = flags.length ? flags.map(f => `<span class="badge-flag">${f}</span>`).join(' ') : '<span style="color:#059669; font-size:0.8rem;">Clean</span>';

            tr.innerHTML = `
                <td style="font-weight:600; color:var(--accent);">${r.row_id}</td>
                <td><span class="badge">${r.row_type || 'transaction'}</span></td>
                ${colCellsHtml}
                <td class="urdu-desc" style="max-width:240px; font-size:0.9rem;">${rawText}</td>
                <td>${flagsHtml}</td>
                <td>
                    <a href="${BACKEND_BASE}/api/qwen_pipeline/job/${activeJobId}/artifact/crops/${r.crop_filename}" target="_blank" class="btn-ctrl" style="text-decoration:none;">View Crop</a>
                </td>
            `;
            pipelineTbody.appendChild(tr);
        });
    }

    window.addEventListener('resize', () => {
        if (activeManifest && !pipelineReviewSection.classList.contains('hidden')) {
            redrawOverlays(activeManifest.rows || []);
        }
    });

    window.resetApp = function() {
        if (resultsSection) resultsSection.classList.add('hidden');
        if (progressSection) progressSection.classList.add('hidden');
        if (pipelineReviewSection) pipelineReviewSection.classList.add('hidden');
        if (pipelineResultsSection) pipelineResultsSection.classList.add('hidden');
        if (krakenSection) krakenSection.classList.add('hidden');
        if (benchmarkScorecardSection) benchmarkScorecardSection.classList.add('hidden');
        if (stageBModal) stageBModal.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        uploadSection.classList.add('fade-in');
        fileInput.value = '';
        patchData = null;
        patchSourceImg = null;
        krakenData = null;
        krakenSourceImg = null;
        activeKrakenLineIdx = 0;
        hoveredKrakenLineIdx = -1;
        const mainCanvas = document.getElementById('patch-main-canvas');
        if (mainCanvas) {
            const ctx = mainCanvas.getContext('2d');
            ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
        }
        if (krakenMainCanvas) {
            const kCtx = krakenMainCanvas.getContext('2d');
            kCtx.clearRect(0, 0, krakenMainCanvas.width, krakenMainCanvas.height);
        }
        if (krakenLinesList) krakenLinesList.innerHTML = '';
        resetSteps();
    };


    // =========================================================================
    // Multi-Engine Document Segmentation Benchmark Suite
    // (Kraken BLLA, OpenCV Rulings, Table Transformer, YOLO, PaddleOCR)
    // =========================================================================
    let currentBenchmarkEngine = 'kraken';
    let lastBenchmarkInput = { file: null, sampleName: null };
    let cachedBenchmarkResults = null;
    let krakenData = null;
    let krakenSourceImg = null;
    let activeKrakenLineIdx = 0;
    let hoveredKrakenLineIdx = -1;
    let showKrakenBaselines = true;
    let showKrakenPolygons = true;
    let showKrakenBBoxes = true;
    let showKrakenLabels = true;

    function updateEnginePillsActive(engineKey) {
        document.querySelectorAll('.btn-engine-pill').forEach(btn => {
            if (btn.dataset.engine === engineKey) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    window.selectBenchmarkEngine = function(engineKey) {
        currentBenchmarkEngine = engineKey;
        updateEnginePillsActive(engineKey);

        if (lastBenchmarkInput.file || lastBenchmarkInput.sampleName) {
            if (engineKey === 'all') {
                if (cachedBenchmarkResults && cachedBenchmarkResults.scorecard) {
                    renderScorecardMatrix(cachedBenchmarkResults);
                } else {
                    startBenchmarkSegmentation(lastBenchmarkInput.file, lastBenchmarkInput.sampleName, 'all');
                }
            } else if (cachedBenchmarkResults && cachedBenchmarkResults.results && cachedBenchmarkResults.results[engineKey]) {
                renderBenchmarkResults(cachedBenchmarkResults.results[engineKey]);
            } else {
                startBenchmarkSegmentation(lastBenchmarkInput.file, lastBenchmarkInput.sampleName, engineKey);
            }
        }
    };

    window.switchBenchmarkEngine = function(engineKey) {
        selectBenchmarkEngine(engineKey);
    };

    window.inspectEngineFromScorecard = function(engineKey) {
        selectBenchmarkEngine(engineKey);
    };

    async function startBenchmarkSegmentation(file, sampleName, engineKey = 'kraken') {
        lastBenchmarkInput = { file, sampleName };
        currentBenchmarkEngine = engineKey;
        updateEnginePillsActive(engineKey);

        startPipelineUI();
        updateStep(1, 'active');

        const formData = new FormData();
        if (file) {
            formData.append('image', file);
        } else if (sampleName) {
            formData.append('sample_name', sampleName);
        }
        formData.append('engine', engineKey);
        formData.append('direction', 'horizontal-rl');

        try {
            updateStep(1, 'completed');
            updateStep(2, 'active');
            if (step2Title) {
                step2Title.textContent = (engineKey === 'all')
                    ? "Running 5-Engine Comparative Benchmark"
                    : `Executing ${engineKey.replace('_', ' ').toUpperCase()} Engine`;
            }

            const response = await fetch(`${BACKEND_BASE}/api/qwen_pipeline/benchmark_segment`, {
                method: 'POST',
                body: formData
            });
            const data = await parseJsonResponse(response);

            updateStep(2, 'completed');
            updateStep(3, 'active');
            await sleep(300);
            updateStep(3, 'completed');

            if (engineKey === 'all' || data.scorecard) {
                cachedBenchmarkResults = data;
                renderScorecardMatrix(data);
            } else {
                renderBenchmarkResults(data);
            }
        } catch (err) {
            console.error(err);
            showError("Segmentation benchmark failed: " + err.message);
            resetApp();
        }
    }

    // Backwards-compatible wrapper
    async function startKrakenSegmentation(file, sampleName) {
        await startBenchmarkSegmentation(file, sampleName, currentBenchmarkEngine);
    }

    function renderScorecardMatrix(data) {
        progressSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.add('hidden');
        if (pipelineReviewSection) pipelineReviewSection.classList.add('hidden');
        if (pipelineResultsSection) pipelineResultsSection.classList.add('hidden');
        if (uploadSection) uploadSection.classList.add('hidden');

        if (krakenSection) {
            krakenSection.classList.remove('hidden');
            krakenSection.classList.add('fade-in');
        }

        if (benchmarkScorecardSection) {
            benchmarkScorecardSection.classList.remove('hidden');
        }
        if (benchmarkSingleHeader) {
            benchmarkSingleHeader.classList.add('hidden');
        }
        if (benchmarkReviewGrid) {
            benchmarkReviewGrid.classList.add('hidden');
        }

        if (scorecardTotalTime) {
            scorecardTotalTime.textContent = `Total Benchmark Latency: ${data.total_latency_seconds ? data.total_latency_seconds.toFixed(2) + 's' : 'N/A'}`;
        }

        if (benchmarkScorecardTbody && data.scorecard) {
            benchmarkScorecardTbody.innerHTML = '';
            data.scorecard.forEach(item => {
                const tr = document.createElement('tr');
                const isGpu = item.device && item.device.toLowerCase().includes('gpu');
                const devBadgeClass = isGpu ? 'badge-gpu' : 'badge-cpu';

                const lat = item.latency_seconds || 0;
                let latClass = 'badge-fast';
                if (lat > 2.0) latClass = 'badge-moderate';

                tr.innerHTML = `
                    <td>
                        <strong style="color:#fff; font-size:0.9rem;">${item.name}</strong>
                    </td>
                    <td style="color:var(--text-secondary); max-width:240px; font-size:0.8rem;">
                        ${item.objective || ''}
                    </td>
                    <td>
                        <span style="font-family:monospace; font-size:0.75rem; color:#00E5FF;">${item.output_type || ''}</span>
                    </td>
                    <td>
                        <span class="scorecard-badge ${devBadgeClass}">${item.items_detected} items</span>
                    </td>
                    <td>
                        <span class="${latClass}">${lat.toFixed(2)}s</span>
                    </td>
                    <td>
                        <span class="scorecard-badge ${devBadgeClass}">${item.device}</span>
                    </td>
                    <td style="font-size:0.75rem; color:var(--text-muted); max-width:280px;">
                        ${item.notes || 'Executed successfully'}
                    </td>
                    <td>
                        <button type="button" class="btn-scorecard-action" onclick="inspectEngineFromScorecard('${item.engine_key}')">
                            🔍 Inspect Crops
                        </button>
                    </td>
                `;
                benchmarkScorecardTbody.appendChild(tr);
            });
        }
    }

    function renderBenchmarkResults(data) {
        krakenData = data;
        activeKrakenLineIdx = 0;
        hoveredKrakenLineIdx = -1;

        progressSection.classList.add('hidden');
        if (resultsSection) resultsSection.classList.add('hidden');
        if (pipelineReviewSection) pipelineReviewSection.classList.add('hidden');
        if (pipelineResultsSection) pipelineResultsSection.classList.add('hidden');
        if (uploadSection) uploadSection.classList.add('hidden');

        if (krakenSection) {
            krakenSection.classList.remove('hidden');
            krakenSection.classList.add('fade-in');
        }
        if (benchmarkScorecardSection) {
            benchmarkScorecardSection.classList.add('hidden');
        }
        if (benchmarkSingleHeader) {
            benchmarkSingleHeader.classList.remove('hidden');
        }
        if (benchmarkReviewGrid) {
            benchmarkReviewGrid.classList.remove('hidden');
        }

        const lines = data.lines || [];
        const meta = data.metadata || {};

        if (benchmarkEngineTitle) {
            benchmarkEngineTitle.textContent = meta.title || data.model_name || 'Document Segmentation Preprocessor';
        }
        if (krakenModelBadge) {
            krakenModelBadge.textContent = `Model: ${data.model_name || meta.name || 'Segmenter'}`;
        }
        if (krakenLinesBadge) {
            krakenLinesBadge.textContent = `Detected: ${data.line_count || lines.length} items`;
        }
        if (krakenDeviceBadge) {
            krakenDeviceBadge.textContent = `Device: ${data.device || 'cuda:0'}`;
        }
        if (krakenLatencyBadge) {
            krakenLatencyBadge.textContent = `Latency: ${data.latency_seconds ? data.latency_seconds.toFixed(2) + 's' : '0.00s'}`;
        }
        if (krakenDirBadge) {
            krakenDirBadge.textContent = `Type: ${meta.output_type || 'Segments'}`;
        }
        if (krakenGalleryCount) {
            krakenGalleryCount.textContent = (data.line_count || lines.length);
        }

        if (benchmarkEngineNote) {
            let noteHtml = `<strong>${meta.title || 'Engine'}:</strong> ${meta.objective || ''}`;
            if (meta.strengths) {
                noteHtml += `<br><span style="color:#00FFA3;">✓ Strengths:</span> ${meta.strengths}`;
            }
            if (meta.limitations) {
                noteHtml += `<br><span style="color:#FBBF24;">⚠ Limitations:</span> ${meta.limitations}`;
            }
            if (data.notes) {
                noteHtml += `<br><span style="color:#00E5FF;">ℹ Runtime Note:</span> ${data.notes}`;
            }
            benchmarkEngineNote.innerHTML = noteHtml;
        }

        krakenSourceImg = new Image();
        krakenSourceImg.onload = () => {
            if (krakenMainCanvas) {
                krakenMainCanvas.width = krakenSourceImg.naturalWidth;
                krakenMainCanvas.height = krakenSourceImg.naturalHeight;
            }
            drawKrakenCanvas();
            renderKrakenGallery();
            if (lines.length > 0) {
                selectKrakenLine(0);
            }
        };
        krakenSourceImg.src = data.full_image_data_url;
    }

    // Alias for backward compatibility
    function renderKrakenResults(data) {
        renderBenchmarkResults(data);
    }

    function drawKrakenCanvas() {
        if (!krakenMainCanvas || !krakenSourceImg || !krakenData) return;
        const ctx = krakenMainCanvas.getContext('2d');
        const lines = krakenData.lines || [];
        const engineKey = krakenData.engine_key || 'kraken';

        // 1. Draw base photograph
        ctx.clearRect(0, 0, krakenMainCanvas.width, krakenMainCanvas.height);
        ctx.drawImage(krakenSourceImg, 0, 0);

        // 2. Draw boundary polygons / masks
        if (showKrakenPolygons) {
            lines.forEach((line) => {
                const isSelected = (line.line_idx === activeKrakenLineIdx);
                const isHovered = (line.line_idx === hoveredKrakenLineIdx);
                const poly = line.boundary || [];

                if (poly.length >= 3) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        ctx.lineTo(poly[i][0], poly[i][1]);
                    }
                    ctx.closePath();

                    if (isSelected) {
                        ctx.fillStyle = 'rgba(255, 215, 0, 0.35)';
                        ctx.strokeStyle = '#FFD700';
                        ctx.lineWidth = 3;
                    } else if (isHovered) {
                        ctx.fillStyle = 'rgba(0, 229, 255, 0.28)';
                        ctx.strokeStyle = '#00E5FF';
                        ctx.lineWidth = 2.5;
                    } else {
                        if (engineKey === 'paddleocr') {
                            ctx.fillStyle = 'rgba(132, 204, 22, 0.2)';
                            ctx.strokeStyle = 'rgba(132, 204, 22, 0.85)';
                        } else if (engineKey === 'opencv_rulings') {
                            ctx.fillStyle = 'rgba(245, 158, 11, 0.12)';
                            ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
                        } else if (line.category === 'table') {
                            ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
                            ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
                            ctx.lineWidth = 3;
                        } else {
                            ctx.fillStyle = 'rgba(168, 85, 247, 0.18)';
                            ctx.strokeStyle = 'rgba(168, 85, 247, 0.85)';
                        }
                        ctx.lineWidth = 1.5;
                    }
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                }
            });
        }

        // 3. Draw bounding boxes
        if (showKrakenBBoxes) {
            lines.forEach((line) => {
                const isSelected = (line.line_idx === activeKrakenLineIdx);
                const isHovered = (line.line_idx === hoveredKrakenLineIdx);
                const bbox = line.bbox;
                if (!bbox || bbox.length < 4) return;
                const [minX, minY, maxX, maxY] = bbox;
                const bw = maxX - minX;
                const bh = maxY - minY;

                ctx.save();
                if (isSelected) {
                    ctx.strokeStyle = '#FFD700';
                    ctx.lineWidth = 3;
                    ctx.setLineDash([6, 3]);
                } else if (isHovered) {
                    ctx.strokeStyle = '#00E5FF';
                    ctx.lineWidth = 2.5;
                    ctx.setLineDash([4, 3]);
                } else {
                    if (line.category === 'table_row') {
                        ctx.strokeStyle = 'rgba(236, 72, 153, 0.7)';
                    } else if (line.category === 'table_column') {
                        ctx.strokeStyle = 'rgba(6, 182, 212, 0.7)';
                    } else if (engineKey === 'yolo') {
                        ctx.strokeStyle = 'rgba(245, 158, 11, 0.8)';
                    } else if (engineKey === 'opencv_rulings') {
                        ctx.strokeStyle = 'rgba(234, 179, 8, 0.7)';
                    } else {
                        ctx.strokeStyle = 'rgba(0, 229, 255, 0.5)';
                    }
                    ctx.lineWidth = 1.2;
                    ctx.setLineDash([3, 3]);
                }
                ctx.strokeRect(minX, minY, bw, bh);
                ctx.restore();
            });
        }

        // 4. Draw Baselines / Rulings
        if (showKrakenBaselines) {
            // For Kraken: neural baselines polylines
            lines.forEach((line) => {
                const isSelected = (line.line_idx === activeKrakenLineIdx);
                const isHovered = (line.line_idx === hoveredKrakenLineIdx);
                const bl = line.baseline || [];
                if (bl.length >= 2) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(bl[0][0], bl[0][1]);
                    for (let i = 1; i < bl.length; i++) {
                        ctx.lineTo(bl[i][0], bl[i][1]);
                    }

                    if (isSelected) {
                        ctx.strokeStyle = '#00FFA3';
                        ctx.lineWidth = 4.5;
                        ctx.shadowColor = '#00FFA3';
                        ctx.shadowBlur = 10;
                    } else if (isHovered) {
                        ctx.strokeStyle = '#00FFA3';
                        ctx.lineWidth = 3.5;
                        ctx.shadowColor = '#00FFA3';
                        ctx.shadowBlur = 6;
                    } else {
                        ctx.strokeStyle = '#00FFA3';
                        ctx.lineWidth = 2.2;
                    }
                    ctx.stroke();
                    ctx.restore();

                    // Draw vertices along baseline
                    ctx.fillStyle = isSelected ? '#FFD700' : '#00FFA3';
                    const r = isSelected ? 3.5 : 2.5;
                    for (let i = 0; i < bl.length; i++) {
                        ctx.beginPath();
                        ctx.arc(bl[i][0], bl[i][1], r, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            });

            // For OpenCV Rulings: horizontal ruling lines across the page
            if (krakenData.rulings && Array.isArray(krakenData.rulings)) {
                ctx.save();
                ctx.strokeStyle = '#3B82F6';
                ctx.lineWidth = 2;
                ctx.shadowColor = '#60A5FA';
                ctx.shadowBlur = 8;
                krakenData.rulings.forEach((ry) => {
                    ctx.beginPath();
                    ctx.moveTo(0, ry);
                    ctx.lineTo(krakenMainCanvas.width, ry);
                    ctx.stroke();
                });
                ctx.restore();
            }
        }

        // 5. Draw item badges/labels
        if (showKrakenLabels) {
            lines.forEach((line) => {
                const isSelected = (line.line_idx === activeKrakenLineIdx);
                const isHovered = (line.line_idx === hoveredKrakenLineIdx);
                const bbox = line.bbox || [0, 0, 10, 10];
                const x = bbox[0];
                const y = Math.max(16, bbox[1] - 4);

                let labelText = `#${line.line_idx + 1}`;
                if (engineKey === 'opencv_rulings') {
                    labelText = `Row #${line.line_idx + 1}`;
                } else if (engineKey === 'table_transformer') {
                    const cat = line.category === 'table_row' ? 'Row' : (line.category === 'table_column' ? 'Col' : (line.category === 'table' ? 'Table' : 'Header'));
                    labelText = `${cat} #${line.line_idx + 1}`;
                } else if (engineKey === 'yolo') {
                    labelText = `${line.category || 'layout'} (${Math.round((line.confidence || 0) * 100)}%)`;
                } else if (engineKey === 'paddleocr') {
                    labelText = `Text #${line.line_idx + 1}`;
                }

                ctx.save();
                ctx.font = 'bold 11px monospace';
                const metrics = ctx.measureText(labelText);
                const pw = metrics.width + 10;
                const ph = 18;

                // Background pill
                ctx.fillStyle = isSelected ? '#FFD700' : (isHovered ? '#00E5FF' : '#0d1117');
                ctx.beginPath();
                ctx.roundRect(x, y - ph + 4, pw, ph, 4);
                ctx.fill();

                // Text
                ctx.fillStyle = (isSelected || isHovered) ? '#000000' : '#00FFA3';
                ctx.fillText(labelText, x + 5, y);
                ctx.restore();
            });
        }
    }

    function setupKrakenCanvasEvents() {
        if (!krakenMainCanvas) return;

        function getCanvasCoords(e) {
            const rect = krakenMainCanvas.getBoundingClientRect();
            const scaleX = krakenMainCanvas.width / rect.width;
            const scaleY = krakenMainCanvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        function findLineAtCoords(x, y) {
            if (!krakenData || !krakenData.lines) return -1;
            for (let i = 0; i < krakenData.lines.length; i++) {
                const line = krakenData.lines[i];
                const bbox = line.bbox;
                if (!bbox) continue;
                if (x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3]) {
                    return line.line_idx;
                }
            }
            return -1;
        }

        krakenMainCanvas.addEventListener('mousemove', (e) => {
            const pt = getCanvasCoords(e);
            const foundIdx = findLineAtCoords(pt.x, pt.y);
            if (foundIdx !== hoveredKrakenLineIdx) {
                hoveredKrakenLineIdx = foundIdx;
                drawKrakenCanvas();
            }

            if (hoveredKrakenLineIdx !== -1) {
                krakenMainCanvas.style.cursor = 'pointer';
                if (krakenHoverCoords) {
                    const line = krakenData.lines.find(l => l.line_idx === hoveredKrakenLineIdx);
                    krakenHoverCoords.textContent = `Inspecting Item #${hoveredKrakenLineIdx + 1} (${line ? line.width + '×' + line.height + ' px' : ''})`;
                }
            } else {
                krakenMainCanvas.style.cursor = 'crosshair';
                if (krakenHoverCoords) {
                    krakenHoverCoords.textContent = `Coords: (${Math.round(pt.x)}, ${Math.round(pt.y)})`;
                }
            }
        });

        krakenMainCanvas.addEventListener('mouseleave', () => {
            if (hoveredKrakenLineIdx !== -1) {
                hoveredKrakenLineIdx = -1;
                drawKrakenCanvas();
            }
            if (krakenHoverCoords) {
                krakenHoverCoords.textContent = "Hover or click an item to inspect";
            }
        });

        krakenMainCanvas.addEventListener('click', (e) => {
            const pt = getCanvasCoords(e);
            const foundIdx = findLineAtCoords(pt.x, pt.y);
            if (foundIdx !== -1) {
                selectKrakenLine(foundIdx);
            }
        });
    }

    function selectKrakenLine(lineIdx) {
        if (!krakenData || !krakenData.lines) return;
        const line = krakenData.lines.find(l => l.line_idx === lineIdx);
        if (!line) return;

        activeKrakenLineIdx = lineIdx;
        drawKrakenCanvas();

        const engineKey = krakenData.engine_key || 'kraken';

        // Update active item inspector
        if (krakenActiveLineTitle) krakenActiveLineTitle.textContent = `Item #${line.line_idx + 1}`;
        if (krakenActiveCrop) krakenActiveCrop.src = line.crop_data_url || '';
        if (krakenMetaLineId) krakenMetaLineId.textContent = `#${line.line_idx + 1} (${line.id})`;
        if (krakenMetaBbox) krakenMetaBbox.textContent = `[x:${line.x}, y:${line.y}, ${line.width} × ${line.height} px]`;
        
        if (krakenMetaBaselinePts) {
            if (engineKey === 'kraken') {
                krakenMetaBaselinePts.textContent = `${line.baseline_points_count || 0} vertices`;
            } else if (engineKey === 'opencv_rulings') {
                krakenMetaBaselinePts.textContent = `Ruled band y: ${line.y} to ${line.y + line.height}`;
            } else {
                krakenMetaBaselinePts.textContent = 'N/A';
            }
        }

        if (krakenMetaBoundaryPts) {
            const bpts = (line.boundary || []).length;
            krakenMetaBoundaryPts.textContent = bpts > 0 ? `${bpts} vertices` : '4 vertices (rectangle)';
        }

        if (krakenMetaTags) {
            const t = (line.tags && line.tags.length > 0) ? line.tags.join(', ') : (line.category || 'default');
            krakenMetaTags.textContent = t;
        }

        // Update gallery card highlights and scroll into view
        document.querySelectorAll('.kraken-line-card').forEach(card => {
            const idx = parseInt(card.dataset.lineIdx, 10);
            if (idx === lineIdx) {
                card.classList.add('active');
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                card.classList.remove('active');
            }
        });
    }

    function renderKrakenGallery() {
        if (!krakenLinesList || !krakenData) return;
        krakenLinesList.innerHTML = '';
        const lines = krakenData.lines || [];
        const engineKey = krakenData.engine_key || 'kraken';

        lines.forEach(line => {
            const card = document.createElement('div');
            card.className = `kraken-line-card ${line.line_idx === activeKrakenLineIdx ? 'active' : ''}`;
            card.dataset.lineIdx = line.line_idx;
            card.onclick = () => selectKrakenLine(line.line_idx);

            const badge = document.createElement('span');
            badge.className = 'kraken-line-badge';
            badge.textContent = `#${line.line_idx + 1}`;

            const thumbWrap = document.createElement('div');
            thumbWrap.className = 'kraken-line-thumb-wrap';
            const img = document.createElement('img');
            img.className = 'kraken-line-thumb';
            img.src = line.crop_data_url || '';
            img.alt = `Item #${line.line_idx + 1}`;
            thumbWrap.appendChild(img);

            let labelExtra = '';
            if (engineKey === 'opencv_rulings') labelExtra = ' • Ruled Row Band';
            else if (engineKey === 'table_transformer') labelExtra = ` • ${line.category || 'box'}`;
            else if (engineKey === 'yolo') labelExtra = ` • ${line.category || 'layout'}`;
            else if (engineKey === 'paddleocr') labelExtra = ' • Text Polygon';
            else labelExtra = ` • ${line.baseline_points_count || 0} pts`;

            const info = document.createElement('div');
            info.className = 'kraken-line-info';
            info.innerHTML = `
                <span style="font-size:0.8rem; font-weight:600; color:#fff;">Item #${line.line_idx + 1}</span>
                <span class="kraken-line-dim">${line.width} × ${line.height} px${labelExtra}</span>
            `;

            card.appendChild(badge);
            card.appendChild(thumbWrap);
            card.appendChild(info);
            krakenLinesList.appendChild(card);
        });
    }

    window.toggleKrakenBaselines = function() {
        showKrakenBaselines = !showKrakenBaselines;
        if (btnToggleBaselines) {
            btnToggleBaselines.classList.toggle('active', showKrakenBaselines);
            btnToggleBaselines.textContent = `🟢 Baselines/Rulings: ${showKrakenBaselines ? 'ON' : 'OFF'}`;
        }
        drawKrakenCanvas();
    };

    window.toggleKrakenPolygons = function() {
        showKrakenPolygons = !showKrakenPolygons;
        if (btnTogglePolygons) {
            btnTogglePolygons.classList.toggle('active', showKrakenPolygons);
            btnTogglePolygons.textContent = `🟣 Polygons/Masks: ${showKrakenPolygons ? 'ON' : 'OFF'}`;
        }
        drawKrakenCanvas();
    };

    window.toggleKrakenBBoxes = function() {
        showKrakenBBoxes = !showKrakenBBoxes;
        if (btnToggleBBoxes) {
            btnToggleBBoxes.classList.toggle('active', showKrakenBBoxes);
            btnToggleBBoxes.textContent = `🔵 BBoxes/Rows: ${showKrakenBBoxes ? 'ON' : 'OFF'}`;
        }
        drawKrakenCanvas();
    };

    window.toggleKrakenLabels = function() {
        showKrakenLabels = !showKrakenLabels;
        if (btnToggleLabels) {
            btnToggleLabels.classList.toggle('active', showKrakenLabels);
            btnToggleLabels.textContent = `🏷️ Labels: ${showKrakenLabels ? 'ON' : 'OFF'}`;
        }
        drawKrakenCanvas();
    };

    setupKrakenCanvasEvents();

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
