document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');
    
    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                startPipeline(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    async function startPipeline(file) {
        // Hide upload, show progress
        uploadSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        progressSection.classList.add('fade-in');

        // Simulate step 1
        updateStep(1, 'active');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('image', file);

        try {
            // Send to Flask Backend using relative URL to avoid CORS issues
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Backend failed');
            }
            
            // Show step 1 completed once the server responds
            updateStep(1, 'completed');
            updateStep(2, 'active');
            await sleep(500); // Visual pause

            const data = await response.json();
            
            updateStep(2, 'completed');
            updateStep(3, 'active');
            await sleep(500); // Visual pause
            
            updateStep(3, 'completed');
            
            // Show the results
            showResults(data.table, data.grid_image);

        } catch (error) {
            console.error('Pipeline Error:', error);
            alert("Error processing image: " + error.message);
            resetApp();
        }
    }

    function updateStep(stepNum, status) {
        const step = document.getElementById(`step-${stepNum}`);
        const statusText = step.querySelector('.step-status');
        
        // Remove existing classes
        step.classList.remove('active', 'completed');
        
        if (status === 'active') {
            step.classList.add('active');
            statusText.textContent = 'In Progress...';
        } else if (status === 'completed') {
            step.classList.add('completed');
            statusText.textContent = 'Completed';
            step.querySelector('.step-indicator').innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;
        }
    }

    function showResults(tableData, gridImageSrc) {
        progressSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('fade-in');

        // Display the grid image
        const gridContainer = document.getElementById('grid-visualization');
        const gridImg = document.getElementById('grid-img-display');
        if (gridImageSrc) {
            // Append a timestamp to bypass browser caching for the same filename
            gridImg.src = gridImageSrc + "?t=" + new Date().getTime();
            gridContainer.classList.remove('hidden');
        } else {
            gridContainer.classList.add('hidden');
        }

        const tbody = document.getElementById('results-body');
        tbody.innerHTML = '';

        tableData.forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Expose reset globally
    window.resetApp = function() {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        uploadSection.classList.add('fade-in');
        fileInput.value = '';
        
        // Reset steps
        [1, 2, 3].forEach(i => {
            const step = document.getElementById(`step-${i}`);
            step.classList.remove('active', 'completed');
            step.querySelector('.step-status').textContent = 'Pending';
            step.querySelector('.step-indicator').textContent = i;
        });
    };
});
