// Fetch and display stats
async function fetchStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        const statsText = document.getElementById('stats-text');
        if (data.status === 'Not initialized') {
            statsText.textContent = 'RAG system not initialized. Upload documents to get started.';
        } else {
            statsText.textContent = `Status: ${data.status} | Documents: ${data.total_documents || 0} | Chunks: ${data.total_chunks || 0}`;
        }
    } catch (error) {
        document.getElementById('stats-text').textContent = 'Could not fetch stats';
    }
}

// Upload form handler
document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const fileInput = this.querySelector('input[type="file"]');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        alert(data.message || 'Upload complete');
        fetchStats();
    } catch (error) {
        alert('Upload error: ' + error.message);
    }
});

// Delete form handler
document.getElementById('delete-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    if (!confirm("Are you sure you want to delete all documents?")) return;

    try {
        const response = await fetch('/delete', { method: 'POST' });
        const data = await response.json();
        alert(data.message || 'Deletion complete');
        fetchStats();
    } catch (error) {
        alert('Deletion error: ' + error.message);
    }
});

// Ask form handler
document.getElementById('ask-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const question = this.querySelector('textarea[name="question"]').value;
    const answerDiv = document.querySelector('.answer');
    const answerText = document.getElementById('answer-text');

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        answerText.textContent = data.answer || data.response || 'No response';
        answerDiv.style.display = 'block';
    } catch (error) {
        answerText.textContent = 'Error: ' + error.message;
        answerDiv.style.display = 'block';
    }
});

// Load stats on page load
fetchStats();
