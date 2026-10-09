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

    // Auto-detect backend: if opened from file:// or external port, route to local GPU port 5000
    const isPort5000 = (window.location.protocol.startsWith('http') && window.location.port === '5000');
    const BACKEND_BASE = isPort5000 ? '' : 'http://localhost:5000';

    let currentTableData = { columns: [], rows: [] };

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

            const url = `${BACKEND_BASE}/api/sample?name=${encodeURIComponent(sampleName)}`;
            const response = await fetch(url);
            const data = await parseJsonResponse(response);

            updateStep(2, 'completed');
            updateStep(3, 'active');
            await sleep(400);
            updateStep(3, 'completed');

            showResults(data);
        } catch (err) {
            console.error(err);
            showError("Failed to process ledger sample: " + err.message);
            resetApp();
        }
    };

    async function startUpload(file) {
        hideError();
        startPipelineUI();
        updateStep(1, 'active');

        const formData = new FormData();
        formData.append('image', file);

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
            showError("Failed to transcribe uploaded ledger: " + error.message);
            resetApp();
        }
    }

    function startPipelineUI() {
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        progressSection.classList.add('fade-in');
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
