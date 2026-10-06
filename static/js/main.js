// Main JavaScript for Brain Tumor Detection App

document.addEventListener('DOMContentLoaded', function() {
    console.log('Brain Tumor Detection App Loaded');
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Active nav link
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentLocation) {
            link.classList.add('active');
        }
    });
});

// Prediction Page Functions
if (document.getElementById('imageUpload')) {
    const uploadArea = document.getElementById('uploadArea');
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const resultSection = document.getElementById('resultSection');
    const spinnerContainer = document.getElementById('spinnerContainer');
    
    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        });
    });
    
    uploadArea.addEventListener('drop', handleDrop);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        imageUpload.files = files;
        handleFiles(files);
    }
    
    // File input change
    imageUpload.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                displayPreview(file);
                predictBtn.disabled = false;
            } else {
                alert('Please upload an image file (JPG, JPEG, PNG)');
            }
        }
    }
    
    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.innerHTML = `
                <img src="${e.target.result}" class="preview-image" alt="Preview">
            `;
        };
        reader.readAsDataURL(file);
    }
    
    // Predict button
    if (predictBtn) {
        predictBtn.addEventListener('click', makePrediction);
    }
    
    async function makePrediction() {
        const file = imageUpload.files[0];
        const model = document.getElementById('modelSelect').value;
        
        if (!file) {
            alert('Please upload an image first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', model);
        
        // Show spinner
        spinnerContainer.style.display = 'block';
        resultSection.style.display = 'none';
        predictBtn.disabled = true;
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                displayResult(result);
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Prediction error:', error);
        } finally {
            spinnerContainer.style.display = 'none';
            predictBtn.disabled = false;
        }
    }
    
    function displayResult(result) {
        const hasTumor = result.has_tumor;
        const confidence = result.confidence;
        const badgeClass = hasTumor ? 'bg-danger' : 'bg-success';
        const icon = hasTumor ? 'fa-exclamation-circle' : 'fa-check-circle';
        
        resultSection.innerHTML = `
            <div class="result-card animate-in">
                <div class="text-center mb-4">
                    <span class="result-badge badge ${badgeClass}">
                        <i class="fas ${icon} me-2"></i>${result.prediction}
                    </span>
                </div>
                
                <div class="row mb-4">
                    <div class="col-md-6">
                        <h5><i class="fas fa-robot me-2"></i>Model Used</h5>
                        <p class="text-muted">${result.model}</p>
                    </div>
                    <div class="col-md-6">
                        <h5><i class="fas fa-clock me-2"></i>Analysis Time</h5>
                        <p class="text-muted">${result.timestamp}</p>
                    </div>
                </div>
                
                <div class="mb-4">
                    <h5><i class="fas fa-percentage me-2"></i>Confidence Level</h5>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%">
                            ${confidence.toFixed(2)}%
                        </div>
                    </div>
                </div>
                
                ${result.visualization ? `
                    <div class="mt-4">
                        <h5><i class="fas fa-chart-bar me-2"></i>Visualization</h5>
                        <img src="data:image/png;base64,${result.visualization}" 
                             class="img-fluid rounded" alt="Visualization">
                    </div>
                ` : ''}
                
                <div class="text-center mt-4">
                    <button class="btn btn-primary" onclick="window.location.reload()">
                        <i class="fas fa-redo me-2"></i>Analyze Another Image
                    </button>
                </div>
            </div>
        `;
        
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Visualization Page Charts
if (document.getElementById('accuracyChart')) {
    loadModelComparison();
}

async function loadModelComparison() {
    try {
        const response = await fetch('/api/comparison');
        const data = await response.json();
        
        if (data.error) {
            console.error('No comparison data available');
            return;
        }
        
        createComparisonCharts(data);
    } catch (error) {
        console.error('Error loading comparison data:', error);
    }
}

function createComparisonCharts(data) {
    const models = data.map(d => d.model);
    const accuracy = data.map(d => d.accuracy * 100);
    const precision = data.map(d => d.precision * 100);
    const recall = data.map(d => d.recall * 100);
    const f1Score = data.map(d => d.f1_score * 100);
    
    // Accuracy Chart
    new Chart(document.getElementById('accuracyChart'), {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'Accuracy (%)',
                data: accuracy,
                backgroundColor: 'rgba(79, 70, 229, 0.8)',
                borderColor: 'rgba(79, 70, 229, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
    
    // Performance Metrics Chart
    new Chart(document.getElementById('metricsChart'), {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'Precision (%)',
                    data: precision,
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Recall (%)',
                    data: recall,
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderColor: 'rgba(245, 158, 11, 1)',
                    borderWidth: 2
                },
                {
                    label: 'F1-Score (%)',
                    data: f1Score,
                    backgroundColor: 'rgba(239, 68, 68, 0.7)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
    
    // Comparison Chart
    new Chart(document.getElementById('comparisonChart'), {
        type: 'radar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'Accuracy',
                    data: accuracy,
                    borderColor: 'rgba(79, 70, 229, 1)',
                    backgroundColor: 'rgba(79, 70, 229, 0.2)'
                },
                {
                    label: 'Precision',
                    data: precision,
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)'
                }
            ]
        },
        options: {
            responsive: true
        }
    });
    
    // Precision vs Recall Chart
    new Chart(document.getElementById('precisionRecallChart'), {
        type: 'scatter',
        data: {
            datasets: models.map((model, index) => ({
                label: model,
                data: [{
                    x: recall[index],
                    y: precision[index]
                }],
                backgroundColor: [
                    'rgba(79, 70, 229, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ][index % 4],
                borderColor: [
                    'rgba(79, 70, 229, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(239, 68, 68, 1)'
                ][index % 4],
                pointRadius: 10,
                pointHoverRadius: 12
            }))
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Recall (%)'
                    },
                    min: 0,
                    max: 100
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precision (%)'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}
