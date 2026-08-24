const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('video-upload');
const loadingState = document.getElementById('loading-state');
const uploadContent = document.querySelector('.upload-content');
const resultsSection = document.getElementById('results-section');
const jiraBoard = document.getElementById('jira-board');
const statsDiv = document.getElementById('processing-stats');

// Event Listeners for Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

async function handleFile(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please upload a video file.');
        return;
    }

    // Show loading
    uploadContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    jiraBoard.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/v1/analyze-video', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            renderResults(data);
        } else {
            alert(`Error: ${data.detail || 'Processing failed'}`);
        }
    } catch (error) {
        alert(`Network Error: ${error.message}`);
    } finally {
        // Reset upload zone
        uploadContent.classList.remove('hidden');
        loadingState.classList.add('hidden');
        fileInput.value = '';
    }
}

function renderResults(data) {
    resultsSection.classList.remove('hidden');
    
    // Render Stats
    if (data.processing_stats) {
        const s = data.processing_stats;
        statsDiv.innerHTML = `Extracted <strong>${s.total_frames}</strong> keyframes (${s.blurry_frames} blurry skipped)`;
    }

    // Render Jira Tickets
    if (data.jira_tickets && data.jira_tickets.length > 0) {
        data.jira_tickets.forEach((ticket, index) => {
            const card = document.createElement('div');
            card.className = 'jira-card';
            
            // Format description by replacing *text* with bold
            let formattedDesc = ticket.description || '';
            // Basic markdown bold to HTML
            formattedDesc = formattedDesc.replace(/\\*(.*?)\\*/g, '<strong>$1</strong>');
            
            card.innerHTML = `
                <div class="jira-tag">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M5 12l5 5L20 7"></path></svg>
                    STORY-${index + 1}
                </div>
                <h3 class="jira-title">${ticket.title || 'Generated User Story'}</h3>
                <div class="jira-desc">${formattedDesc}</div>
            `;
            jiraBoard.appendChild(card);
        });
    } else {
        jiraBoard.innerHTML = '<p style="color:var(--text-secondary)">No User Stories could be generated.</p>';
    }
}
