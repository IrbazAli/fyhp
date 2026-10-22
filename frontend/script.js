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
    const uploadTitle = document.getElementById('upload-title');
    const uploadSubtitle = document.getElementById('upload-subtitle');
    const samplesLedger = document.getElementById('samples-ledger');
    const samplesWord = document.getElementById('samples-word');

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

    // Mode state: 'ledger' (default) or 'word'
    let currentMode = 'ledger';

    // Auto-detect backend: if opened from file:// or external port, route to local GPU port 5000
    const isPort5000 = (window.location.protocol.startsWith('http') && window.location.port === '5000');
    const BACKEND_BASE = isPort5000 ? '' : 'http://localhost:5000';

    let currentTableData = { columns: [], rows: [] };

    // Mode switching function
    window.switchMode = function(mode) {
        currentMode = (mode === 'word') ? 'word' : 'ledger';

        if (currentMode === 'word') {
            if (btnModeWord) btnModeWord.classList.add('active');
            if (btnModeLedger) btnModeLedger.classList.remove('active');
            if (uploadTitle) uploadTitle.textContent = "Upload Single Line / Word Crop";
            if (uploadSubtitle) uploadSubtitle.textContent = "Drag & drop cropped handwritten Urdu image or browse local files";
            if (samplesLedger) samplesLedger.classList.add('hidden');
            if (samplesWord) samplesWord.classList.remove('hidden');
        } else {
            if (btnModeLedger) btnModeLedger.classList.add('active');
            if (btnModeWord) btnModeWord.classList.remove('active');
            if (uploadTitle) uploadTitle.textContent = "Upload Ledger Image";
            if (uploadSubtitle) uploadSubtitle.textContent = "Drag & drop raw photograph or browse local files";
            if (samplesLedger) samplesLedger.classList.remove('hidden');
            if (samplesWord) samplesWord.classList.add('hidden');
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

    window.resetApp = function() {
        resultsSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        uploadSection.classList.add('fade-in');
        fileInput.value = '';
        resetSteps();
    };

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
