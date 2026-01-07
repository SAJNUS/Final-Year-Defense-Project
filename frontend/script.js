// State
let currentTask = 'sentiment';

// DOM Elements
const taskButtons = document.querySelectorAll('.task-btn');
const inputText = document.getElementById('inputText');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const results = document.getElementById('results');
const banglabertCard = document.getElementById('banglabertCard');
const metaCard = document.getElementById('metaCard');

// Event Listeners
taskButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        taskButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTask = btn.dataset.task;
        clearResults();
    });
});

inputText.addEventListener('input', () => {
    const length = inputText.value.length;
    if (length > 0) {
        clearBtn.style.display = 'flex';
    } else {
        clearBtn.style.display = 'none';
    }
});

clearBtn.addEventListener('click', () => {
    inputText.value = '';
    clearBtn.style.display = 'none';
    clearResults();
});

analyzeBtn.addEventListener('click', handleAnalyze);

// Allow Ctrl+Enter to submit
inputText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleAnalyze();
    }
});

// Main Functions
async function handleAnalyze() {
    const text = inputText.value.trim();
    
    if (!text) {
        showError('Please enter some Bangla text');
        return;
    }

    showLoading();
    hideError();
    hideResults();
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                task: currentTask
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        hideLoading();
        analyzeBtn.disabled = false;
    }
}

function showLoading() {
    loading.style.display = 'flex';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'flex';
}

function hideError() {
    error.style.display = 'none';
}

function showResults() {
    results.style.display = 'block';
}

function hideResults() {
    results.style.display = 'none';
}

function clearResults() {
    banglabertCard.innerHTML = '';
    metaCard.innerHTML = '';
    hideResults();
    hideError();
}

function displayResults(data) {
    banglabertCard.innerHTML = createModelCard(data.banglabert, 'Baseline (BanglaBERT)');
    metaCard.innerHTML = createModelCard(data.meta_learning, 'Meta Learning (CPN)');
    showResults();
}

function createModelCard(result, title) {
    if (result.error) {
        return `
            <div class="model-card">
                <div class="card-content">
                    <h3 class="card-title">${title}</h3>
                    <div class="error-box">
                        <svg viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                        </svg>
                        <div>
                            <h3>Error</h3>
                            <p>${result.error}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    const confidence = result.confidence;
    const percentage = (confidence * 100).toFixed(1);
    
    // Determine confidence class
    let confidenceClass = 'high';
    if (confidence <= 0.6) confidenceClass = 'low';
    else if (confidence <= 0.8) confidenceClass = 'medium';

    // Sort probabilities
    const sortedProbs = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1]);

    let probsHtml = '';
    sortedProbs.forEach(([label, prob], idx) => {
        const probPercentage = (prob * 100).toFixed(1);
        const isTop = idx === 0;
        
        probsHtml += `
            <div class="prob-item">
                <div class="prob-header">
                    <span class="prob-label">${label}</span>
                    <span class="prob-value">${probPercentage}%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill ${isTop ? '' : 'secondary'}" style="width: ${probPercentage}%"></div>
                </div>
            </div>
        `;
    });

    return `
        <div class="model-card">
            <div class="card-content">
                <h3 class="card-title">${title}</h3>
                
                <div class="prediction-box">
                    <div class="prediction-label">Predicted Label</div>
                    <div class="prediction-value">${result.prediction}</div>
                </div>

                <div class="confidence-section">
                    <div class="confidence-header">
                        <span class="confidence-label">Confidence Score</span>
                        <span class="confidence-value">${percentage}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" style="width: ${percentage}%"></div>
                    </div>
                </div>

                <div class="probabilities-section">
                    <div class="probabilities-title">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                        </svg>
                        Class Probabilities
                    </div>
                    ${probsHtml}
                </div>

                <div class="model-type">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
                    </svg>
                    <span>${title.includes('Baseline') ? 'BanglaBERT' : 'CPN'}</span>
                </div>
            </div>
        </div>
    `;
}
